# This file is used to reduce the actual runtime in the training notebook, we precompute
# train and val dataset here
# 1. Find the most relevant passage for all row in all dataset
# 2. Precompute all tokenization
# 3. Train and evaluate


import os
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass

import faiss
import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from searcher import Searcher
from utils import clean_memory


@dataclass
class DataCollatorForMultipleChoice:
    """Data collator that will dynamically pad the inputs for multiple choice"""

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, batch):
        # labels do not play well with the rest, do this first
        labels = torch.tensor([row["labels"] for row in batch])
        # flatten: batch has multiple rows, each row's key-value has 5 choices
        # fmt: off
        batch = {
            "input_ids": [choice for row in batch for choice in row["input_ids"]],
            "token_type_ids": [choice for row in batch for choice in row["token_type_ids"]],
            "attention_mask": [choice for row in batch for choice in row["attention_mask"]],
        }
        # fmt: on
        batch = self.tokenizer.pad(batch, return_tensors="pt")
        batch = {k: v.view(-1, 5, v.size(-1)) for k, v in batch.items()}
        batch["labels"] = labels
        return batch


def pretokenize(row: dict, tokenizer: PreTrainedTokenizerBase):
    """Convert single row of data to 5 multiple choices and a label"""
    first_sentences = [row["context"]] * 5
    template = "Question: {}\nAnswer: {}"
    second_sentences = [template.format(row["prompt"], row[ans]) for ans in "ABCDE"]
    encoded = tokenizer(first_sentences, second_sentences, truncation="only_first")
    label2id = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    encoded["labels"] = label2id[row["answer"]]
    return encoded


def map_at_3(logits: np.ndarray, labels: np.ndarray) -> float:
    preds = logits.argsort(-1)[:, ::-1]
    maps = (
        (preds[:, 0] == labels) / 1
        + (preds[:, 1] == labels) / 2
        + (preds[:, 2] == labels) / 3
    )
    return maps.mean()


def compute_metrics(p):
    map3 = map_at_3(p.predictions, p.label_ids)
    return {"map@3": map3}


def make_answer(logits: np.ndarray) -> list[str]:
    preds = logits.argsort(-1)[:, ::-1][:, :3]
    choices = np.array(["A", "B", "C", "D", "E"])
    top3_choices = choices[preds]
    return [" ".join(row) for row in top3_choices]


def main(cfg: Namespace):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # load all data-related
    wiki = load_from_disk("input/llmse-paragraph-level-emb-faiss/wiki_stem_paragraph")
    if cfg.title_trick:
        print("Using title injection faiss embeddings")
        index = faiss.read_index(
            "input/llmse-paragraph-level-emb-faiss/wiki_trick.index"
        )
    else:
        print("Using standard faiss embeddings")
        index = faiss.read_index(
            "input/llmse-paragraph-level-emb-faiss/wiki_notrick.index"
        )
    res = faiss.StandardGpuResources()
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index)
    bi_encoder = SentenceTransformer(
        "./input/llmse-paragraph-level-emb-faiss/all-MiniLM-L6-v2"
    )
    searcher = Searcher(index_gpu, wiki, bi_encoder)

    # load all train val test
    train_df = pd.read_csv("input/llmse-science-or-not/train.csv")
    if cfg.quick_run:
        limit = 5000 if cfg.science_only else 300  # science is rarer
        train_df = train_df.iloc[:limit]
    if cfg.science_only:
        print("Filtering out non-science articles")
        train_df = train_df[train_df["is_science"]].reset_index(drop=True)
    train_df = train_df.drop(
        columns=["context", "source", "science_prob", "is_science"]
    )
    val_df = pd.read_csv("input/kaggle-llm-science-exam/train.csv").drop(columns="id")
    test_df = pd.read_csv("input/kaggle-llm-science-exam/test.csv").drop(columns="id")
    test_df["answer"] = "A"  # fake answer so that the columns are uniform
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    test_ds = Dataset.from_pandas(test_df)
    # save to disk so that future map operations are cached
    train_ds.save_to_disk("input/llmse-train-and-inference/train_ds")
    val_ds.save_to_disk("input/llmse-train-and-inference/val_ds")
    test_ds.save_to_disk("input/llmse-train-and-inference/test_ds")
    train_ds = load_from_disk("input/llmse-train-and-inference/train_ds")
    val_ds = load_from_disk("input/llmse-train-and-inference/val_ds")
    test_ds = load_from_disk("input/llmse-train-and-inference/test_ds")

    del train_df, val_df, test_df
    clean_memory()

    # adding new contexts
    print("Adding new contexts to train val test")
    train_ds = train_ds.add_column(
        "context", searcher.search_only(train_ds["prompt"], k=cfg.k_neighbours)
    )
    val_ds = val_ds.add_column(
        "context", searcher.search_only(val_ds["prompt"], k=cfg.k_neighbours)
    )
    test_ds = test_ds.add_column(
        "context", searcher.search_only(test_ds["prompt"], k=cfg.k_neighbours)
    )

    del searcher, bi_encoder, index_gpu, res, index, wiki
    clean_memory()

    # pretokenize train val test
    print(f"Pretokenize using max length {cfg.max_length}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.pretrained, model_max_length=cfg.max_length
    )
    unused = ["prompt", "A", "B", "C", "D", "E", "answer", "context"]
    train_ds = train_ds.map(
        lambda row: pretokenize(row, tokenizer), remove_columns=unused
    )
    val_ds = val_ds.map(lambda row: pretokenize(row, tokenizer), remove_columns=unused)
    test_ds = test_ds.map(
        lambda row: pretokenize(row, tokenizer), remove_columns=unused
    )
    print(train_ds)

    # load the main model
    model = AutoModelForMultipleChoice.from_pretrained(cfg.pretrained)
    # freeze embedding, always freeze for now
    print("Freezing embedding layers")
    for param in model.deberta.embeddings.parameters():
        param.requires_grad = False
    # freezing layers, this is for finetuning
    if cfg.freeze_layers is not None and cfg.freeze_layers > 0:
        print(f"Freezing {cfg.freeze_layers} encoder layers")
        for layer in model.deberta.encoder.layer[: cfg.freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False
    # lora, do it last because if changes the actual model
    if cfg.use_lora:
        print("Using LoRA")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            inference_mode=False,
            target_modules=["query_proj", "value_proj"],
            modules_to_save=["classifier", "pooler"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # train model
    wandb.init(
        project="kaggle-llmse",
        notes="",
        tags=[],
        config=cfg,
    )

    training_args = TrainingArguments(
        # administration
        output_dir="input/llmse-train-and-inference/checkpoints",
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        eval_steps=200,
        logging_strategy="steps",
        logging_steps=200,
        save_strategy="epoch",
        save_steps=200,
        report_to=["wandb"],
        load_best_model_at_end=False,
        # training
        remove_unused_columns=False,  # why did hf remove `token_type_ids`?
        fp16=True,
        dataloader_num_workers=1,
        num_train_epochs=1,
        per_device_eval_batch_size=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        label_smoothing_factor=0.0,
        dataloader_pin_memory=True,
        # optimizer
        lr_scheduler_type="linear",
        warmup_ratio=0.2,
        learning_rate=2e-5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()
    wandb.finish()  # this is for notebook

    # predict and submit
    output = trainer.predict(test_ds)
    preds = make_answer(output.predictions)
    sub_df = pd.read_csv("input/kaggle-llm-science-exam/sample_submission.csv")
    sub_df["prediction"] = preds
    sub_df.to_csv("input/llmse-train-and-inference/submission.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--freeze_layers", type=int)
    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--lora_dropout", type=float)
    parser.add_argument("--science_only", action="store_true")
    parser.add_argument("--title_trick", action="store_true")
    # only allow "no" for answer_trick (for now)
    parser.add_argument("--answer_trick", type=str, choices=["no"], required=True)
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--quick_run", action="store_true")
    cfg = parser.parse_args()
    if cfg.use_lora:
        print("LoRA will automatically freeze layers, overriding freeze_layers")
        cfg.freeze_layers = None
        assert all(
            item is not None for item in [cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout]
        )
    else:
        print("Standard finetuning, overriding LoRA settings")
        cfg.lora_r = None
        cfg.lora_alpha = None
        cfg.lora_dropout = None
        assert cfg.freeze_layers is not None
    print(cfg)
    main(cfg)
