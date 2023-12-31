# This is multiple choice training and inference
# 1. Find the most relevant passage for all row in all dataset
# 2. Precompute all tokenization
# 3. Train and evaluate


import os
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from sentence_transformers import SentenceTransformer
from torch import nn
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
        labels = torch.tensor([row.pop("labels") for row in batch])
        # flatten: batch has multiple rows, each row's key-value has 5 choices
        batch = {
            k: [choice for row in batch for choice in row[k]] for k in batch[0].keys()
        }
        batch = self.tokenizer.pad(batch, return_tensors="pt")
        batch = {k: v.view(-1, 5, v.size(-1)) for k, v in batch.items()}
        batch["labels"] = labels
        return batch


def pretokenize(row: dict, tokenizer: PreTrainedTokenizerBase):
    """Convert single row of data to 5 multiple choices and a label"""
    first_sentences = [row["context"]] * 5
    second_sentences = [f"{row['prompt']} {row[ans]}" for ans in "ABCDE"]
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


def load_all_data(cfg: Namespace) -> tuple[Dataset, Dataset, Dataset]:
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
        "./input/llmse-paragraph-level-emb-faiss/BAAI/bge-small-en-v1.5"
    )
    searcher = Searcher(index_gpu, wiki, bi_encoder)

    # load all train val test
    train_df = pd.read_csv("input/llmse-science-or-not/train.csv")
    if cfg.science_only:
        print("Filtering out non-science articles")
        train_df = train_df[train_df["is_science"]].reset_index(drop=True)
    if cfg.quick_run:
        train_df = train_df.iloc[:300]
    train_df = train_df.drop(columns=["source", "science_prob"])
    val_df = pd.concat(
        [
            pd.read_csv("input/kaggle-llm-science-exam/train.csv").drop(columns="id"),
            pd.read_csv("input/dataset-wiki-new-1/dataset_wiki_new_1_balanced.csv"),
        ]
    ).reset_index(drop=True)
    test_df = pd.read_csv("input/kaggle-llm-science-exam/test.csv").drop(columns="id")
    test_df["answer"] = "A"  # fake answer so that the columns are uniform

    # adding new contexts
    sci = train_df["is_science"]
    if cfg.answer_trick == "no":
        print("Adding contexts just from questions")
        if cfg.replace_ctx:
            # BUG? pandas .loc will consume a lot of memory, crashing kaggle kernel
            train_df.loc[sci, "context"] = searcher.search_only(
                train_df.loc[sci, "prompt"].to_list(), k=cfg.knn
            )
        val_df["context"] = searcher.search_only(val_df["prompt"].to_list(), k=cfg.knn)
        test_df["context"] = searcher.search_only(
            test_df["prompt"].to_list(), k=cfg.knn
        )
    elif cfg.answer_trick in {"standard", "shorten"}:
        print("Adding contexts from question and answers")
        if cfg.replace_ctx:
            train_df.loc[sci, "context"] = searcher.search_include_answer(
                train_df.loc[sci, "prompt"].to_list(),
                answers={ans: train_df.loc[sci, ans].to_list() for ans in "ABCDE"},
                k=cfg.knn,
                shorten_answer=cfg.answer_trick == "shorten",
            )
        val_df["context"] = searcher.search_include_answer(
            val_df["prompt"].to_list(),
            answers={ans: val_df[ans].to_list() for ans in "ABCDE"},
            k=cfg.knn,
            shorten_answer=cfg.answer_trick == "shorten",
        )
        test_df["context"] = searcher.search_include_answer(
            test_df["prompt"].to_list(),
            answers={ans: test_df[ans].to_list() for ans in "ABCDE"},
            k=cfg.knn,
            shorten_answer=cfg.answer_trick == "shorten",
        )
    else:
        assert False, f"Impossible case: {cfg.answer_trick}"

    # tidy up train_df a bit, and drop is_science
    train_df = train_df[["prompt", "A", "B", "C", "D", "E", "answer", "context"]]
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    test_ds = Dataset.from_pandas(test_df)
    return train_ds, val_ds, test_ds


def freeze(model: nn.Module, cfg: Namespace):
    # so far, the models only differ in the core layer name. Must do manual check
    # first before adding here
    if "deberta" in cfg.pretrained:
        layer_type = "deberta"
    elif "bge" in cfg.pretrained:
        layer_type = "bert"
    elif "longformer" in cfg.pretrained:
        layer_type = "longformer"
    else:
        raise ValueError(f"Freezing not supported: {cfg.pretrained}")
    # always freeze first embedding layers, for now
    print("Freezing embedding layers")
    for param in getattr(model, layer_type).embeddings.parameters():
        param.requires_grad = False
    # freeze encoder layers
    if cfg.freeze_layers is not None and cfg.freeze_layers > 0:
        print(f"Freezing {cfg.freeze_layers} {layer_type} encoder layers")
        for layer in getattr(model, layer_type).encoder.layer[: cfg.freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False


def main(cfg: Namespace):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # folder for everything
    _sci = "sci" if cfg.science_only else "nosci"
    _ttrick = "ttrick" if cfg.title_trick else "nottrick"
    _atrick = cfg.answer_trick
    _knn = f"knn{cfg.knn}"
    _replace = "replace" if cfg.replace_ctx else "keep"
    temp_folder = Path(
        f"input/llmse-finetune-mc/{_sci}_{_replace}_{_ttrick}_{_atrick}_{_knn}"
    )

    # load and cache data
    if cfg.quick_run:
        train_ds, val_ds, test_ds = load_all_data(cfg)
    else:
        try:
            train_ds = load_from_disk(temp_folder / "train_ds")
            val_ds = load_from_disk(temp_folder / "val_ds")
            test_ds = load_from_disk(temp_folder / "test_ds")
        except Exception as e:
            print(e)
            train_ds, val_ds, test_ds = load_all_data(cfg)
            # save to disk so that map operations are cached
            train_ds.save_to_disk(temp_folder / "train_ds")
            val_ds.save_to_disk(temp_folder / "val_ds")
            test_ds.save_to_disk(temp_folder / "test_ds")
            train_ds = load_from_disk(temp_folder / "train_ds")
            val_ds = load_from_disk(temp_folder / "val_ds")
            test_ds = load_from_disk(temp_folder / "test_ds")
    clean_memory()

    # pretokenize train val test
    print(f"Pretokenize using max length {cfg.max_tokens}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.pretrained, model_max_length=cfg.max_tokens
    )
    unused = ["prompt", "A", "B", "C", "D", "E", "answer", "context"]
    train_ds = train_ds.map(
        lambda row: pretokenize(row, tokenizer), remove_columns=unused, num_proc=6
    )
    val_ds = val_ds.map(
        lambda row: pretokenize(row, tokenizer), remove_columns=unused, num_proc=6
    )
    test_ds = test_ds.map(
        lambda row: pretokenize(row, tokenizer), remove_columns=unused, num_proc=6
    )
    print(train_ds)

    # load the main model
    model = AutoModelForMultipleChoice.from_pretrained(
        cfg.pretrained, ignore_mismatched_sizes=True
    )
    freeze(model, cfg)
    # ####
    # # torch.autograd.set_detect_anomaly(True)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=8e-6)
    # model.cuda()
    # model.gradient_checkpointing_enable()
    # from torch.utils.data import DataLoader
    # from torch.cuda.amp import autocast, GradScaler
    # dl = DataLoader(
    #     train_ds,
    #     batch_size=1,
    #     shuffle=False,
    #     collate_fn=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    # )
    # batch = next(iter(dl))
    # batch = {k: v.to("cuda") for k, v in batch.items()}

    # # fake train 1 time
    # scaler = GradScaler()
    # N = 10
    # for i in range(N):
    #     with autocast():
    #         out = model(**batch)
    #     # out.loss.backward()
    #     # optimizer.step()
    #     scaler.scale(out.loss).backward()
    #     scaler.step(optimizer)
    #     scaler.update()
    #     if i < N - 1:
    #         optimizer.zero_grad()

    # for name, param in model.named_parameters():
    #     print(param.grad.max().item() if param.grad is not None else "--", name)
    # from IPython import embed

    # embed()
    # ####
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
        mode="disabled" if cfg.quick_run else None,
        project="kaggle-llmse",
        notes="",
        tags=["mc"],
        config=cfg,
    )

    # NOTE: turns out the debertav3 authors have already provided us the reasonable
    # hparams for finetuning https://arxiv.org/abs/2111.09543, table 11
    training_args = TrainingArguments(
        # administration
        output_dir=temp_folder / "ckpt",
        overwrite_output_dir=True,
        save_total_limit=1,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="epoch",
        save_steps=100,
        report_to=["wandb"],
        load_best_model_at_end=False,
        # training
        fp16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        num_train_epochs=cfg.ep,
        per_device_train_batch_size=cfg.bs,
        per_device_eval_batch_size=cfg.bs,
        gradient_accumulation_steps=cfg.grad_acc,
        label_smoothing_factor=0.0,
        dataloader_pin_memory=True,
        # optimizer
        # optim="adamw_8bit",
        optim="adamw_torch",
        lr_scheduler_type="linear",
        adam_epsilon=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.999,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        learning_rate=cfg.lr,
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

    # save model for further testing
    result_path = temp_folder / "finished" / cfg.pretrained.split("/")[-1]
    trainer.save_model(result_path)
    tokenizer.save_pretrained(result_path)

    # predict and submit
    output = trainer.predict(test_ds)
    preds = make_answer(output.predictions)
    sub_df = pd.read_csv("input/kaggle-llm-science-exam/sample_submission.csv")
    sub_df["prediction"] = preds
    sub_df.to_csv("input/llmse-finetune-mc/submission.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    # required
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, required=True)
    parser.add_argument("--knn", type=int, required=True)
    parser.add_argument("--ep", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--grad_acc", type=int, required=True)
    parser.add_argument(
        "--answer_trick", type=str, choices=["no", "standard", "shorten"], required=True
    )
    # optionals
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--freeze_layers", type=int)
    parser.add_argument("--science_only", action="store_true")
    parser.add_argument("--title_trick", action="store_true")
    parser.add_argument("--replace_ctx", action="store_true")
    # enable this flag for fast run and disable wandb
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
