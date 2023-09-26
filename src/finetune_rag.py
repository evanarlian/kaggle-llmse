# This is RAG training and inference
# 1. Find the most relevant passage for all row in all dataset
# 2. Precompute all tokenization
# 3. Train and evaluate


import os
from argparse import ArgumentParser, Namespace

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from datasets import Dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from sentence_transformers import SentenceTransformer
from torch import nn
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from searcher import Searcher
from utils import clean_memory


def pretokenize(row: dict, tokenizer: PreTrainedTokenizerBase):
    # in decoder only finetuning, there is only one input (and itself as label)
    # the reason we are separating context and q+a is to truncate the context
    # with built-in tokenizer feature
    context = row["context"]
    template = "Question: Based on the context above, {}\nAnswer: {}"
    qa = template.format(row["prompt"], row[row["answer"]])
    # data collator will clone input_ids to create labels
    encoded = tokenizer(
        context, qa, truncation="only_first", return_token_type_ids=False
    )
    return encoded


@torch.no_grad()
def manual_eval(
    model: nn.Module, tokenizer: PreTrainedTokenizerBase, ds: Dataset
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    losses = torch.zeros(len(ds), 5)
    for i, row in tqdm(enumerate(ds), total=len(ds)):
        # tokenize the same way as the training pretokenization, but in batches of 5
        context = row["context"]
        template = "Question: Based on the context above, {}\nAnswer: {}"
        # my machine cannot handle big batches
        for j, ans in enumerate("ABCDE"):
            qa = template.format(row["prompt"], row[ans])
            encoded = tokenizer(
                context,
                qa,
                truncation="only_first",
                return_tensors="pt",
                return_token_type_ids=False,
            )
            encoded = {k: v.cuda() for k, v in encoded.items()}
            # get losses
            out = model(**encoded)
            shift_logits = out["logits"][:, :-1]
            shift_labels = encoded["input_ids"][:, 1:]
            losses[i, j] = F.cross_entropy(
                shift_logits.transpose(-2, -1), shift_labels, reduction="mean"
            )
    losses = losses.numpy()
    labels = np.array(
        {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}[ans] for ans in ds["answer"]
    )
    return losses, labels


def map_at_3(losses: np.ndarray, labels: np.ndarray) -> float:
    preds = losses.argsort(-1)
    maps = (
        (preds[:, 0] == labels) / 1
        + (preds[:, 1] == labels) / 2
        + (preds[:, 2] == labels) / 3
    )
    return maps.mean()


def make_answer(losses: np.ndarray) -> list[str]:
    preds = losses.argsort(-1)[:, :3]
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
    # # save to disk so that future map operations are cached
    # train_ds.save_to_disk("input/llmse-train-and-inference/train_ds")
    # val_ds.save_to_disk("input/llmse-train-and-inference/val_ds")
    # test_ds.save_to_disk("input/llmse-train-and-inference/test_ds")
    # train_ds = load_from_disk("input/llmse-train-and-inference/train_ds")
    # val_ds = load_from_disk("input/llmse-train-and-inference/val_ds")
    # test_ds = load_from_disk("input/llmse-train-and-inference/test_ds")

    del train_df, val_df, test_df
    clean_memory()

    # adding new contexts
    print("Adding new contexts to train val test")
    train_ds = train_ds.add_column(
        "context", searcher.search_only(train_ds["prompt"], k=cfg.knn)
    )
    val_ds = val_ds.add_column(
        "context", searcher.search_only(val_ds["prompt"], k=cfg.knn)
    )
    test_ds = test_ds.add_column(
        "context", searcher.search_only(test_ds["prompt"], k=cfg.knn)
    )

    del searcher, bi_encoder, index_gpu, res, index, wiki
    clean_memory()

    # we don't pretokenize nor remove anything from val and test because we want to
    # compare loss, val and test are different from train
    print(f"Pretokenize train_ds only using max length {cfg.max_tokens}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.pretrained, model_max_length=cfg.max_tokens
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    unused = ["prompt", "A", "B", "C", "D", "E", "answer", "context"]
    train_ds = train_ds.map(
        lambda row: pretokenize(row, tokenizer), remove_columns=unused
    )
    print(train_ds)

    # load the main model
    # TODO below this is only optimized for phi 1.5, not for galactica
    model = AutoModelForCausalLM.from_pretrained(
        cfg.pretrained, trust_remote_code=True, torch_dtype="auto"
    )
    # freeze embedding, always freeze for now
    print("Freezing embedding layers")
    for param in model.layers[0].parameters():
        param.requires_grad = False
    # freezing layers, this is for finetuning
    if cfg.freeze_layers is not None and cfg.freeze_layers > 0:
        print(f"Freezing {cfg.freeze_layers} encoder layers")
        for layer in model.layers[1 : cfg.freeze_layers + 1]:
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
            target_modules=["out_proj"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # train model
    wandb.init(
        mode="disabled" if cfg.quick_run else None,
        project="kaggle-llmse",
        notes="",
        tags=["rag"],
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
        num_train_epochs=cfg.ep,
        per_device_train_batch_size=cfg.bs,
        per_device_eval_batch_size=cfg.bs,
        gradient_accumulation_steps=cfg.grad_acc,
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
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        # compute_metrics=compute_metrics,
    )

    trainer.train()
    # manual evaluation at the very end
    val_losses, val_labels = manual_eval(model, tokenizer, val_ds)
    wandb.log({"eval/map@3": map_at_3(val_losses, val_labels)})
    wandb.finish()  # this is for notebook

    # predict and submit
    # TODO manual prediction at hte end
    test_losses, _ = manual_eval(model, tokenizer, test_ds)
    preds = make_answer(test_losses)
    sub_df = pd.read_csv("input/kaggle-llm-science-exam/sample_submission.csv")
    sub_df["prediction"] = preds
    sub_df.to_csv("input/llmse-train-and-inference/submission.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    # required
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, required=True)
    parser.add_argument("--knn", type=int, required=True)
    parser.add_argument("--ep", type=float, required=True)
    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--grad_acc", type=int, required=True)
    # optionals
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--freeze_layers", type=int)
    parser.add_argument("--science_only", action="store_true")
    parser.add_argument("--title_trick", action="store_true")
    # enable this flag for fast run and disable wandb
    parser.add_argument("--quick_run", action="store_true")
    # DEPRECATED: only allow "no" for answer_trick
    parser.add_argument("--answer_trick", type=str, choices=["no"], default="no")
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
