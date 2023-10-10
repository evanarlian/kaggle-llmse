from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import (
    DebertaV2ForSequenceClassification,
    DebertaV2Tokenizer,
)

from searcher import Searcher


def load_test() -> Dataset:
    # load data
    wiki = load_from_disk("input/llmse-paragraph-level-emb-faiss/wiki_stem_paragraph")
    index = faiss.read_index("input/llmse-paragraph-level-emb-faiss/wiki_trick.index")
    res = faiss.StandardGpuResources()
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index)
    bi_encoder = SentenceTransformer(
        "./input/llmse-paragraph-level-emb-faiss/BAAI/bge-small-en-v1.5"
    )
    searcher = Searcher(index_gpu, wiki, bi_encoder)

    # load the test set
    test_df = pd.read_csv("input/kaggle-llm-science-exam/test.csv").drop(columns="id")
    test_ds = Dataset.from_pandas(test_df)

    # populate context
    test_ds = test_ds.add_column(
        "context",
        searcher.search_include_answer(
            test_ds["prompt"],
            answers={ans: test_ds[ans] for ans in "ABCDE"},
            k=15,
            shorten_answer=False,
        ),
    )
    return test_ds


def predict(pretrained: str, test_ds: Dataset, num_labels: int) -> np.ndarray:
    model = DebertaV2ForSequenceClassification.from_pretrained(
        pretrained, num_labels=num_labels
    )
    model.cuda().eval()
    tokenizer = DebertaV2Tokenizer.from_pretrained(pretrained, model_max_length=512)

    logits_list = []
    for row in tqdm(test_ds):
        first_sentences = [row["context"]] * 5
        second_sentences = [f"{row['prompt']} {row[ans]}" for ans in "ABCDE"]
        encoded = tokenizer(
            first_sentences,
            second_sentences,
            truncation="only_first",
            padding=True,
            return_tensors="pt",
        )
        encoded = {k: v.cuda() for k, v in encoded.items()}
        with torch.no_grad():
            # for zero shot, logits is (5, 3): positive, neutral, negative
            # for mult choice, logits is (5, 1)
            logits = model(**encoded)["logits"]
        logits_list.append(logits[:, 0].cpu())  # pick first column regardless of task
    return torch.stack(logits_list).numpy()


def make_answer(logits: np.ndarray) -> list[str]:
    preds = logits.argsort(-1)[:, ::-1][:, :3]
    choices = np.array(["A", "B", "C", "D", "E"])
    top3_choices = choices[preds]
    return [" ".join(row) for row in top3_choices]


def main():
    test_ds = load_test()
    logits1 = predict("sileod/deberta-v3-large-tasksource-nli", test_ds, num_labels=3)
    logits2 = predict(
        "input/llmse-finetune-mc/nosci_keep_ttrick_standard_knn16/finished/deberta-v3-large-tasksource-nli",
        test_ds,
        num_labels=1,
    )
    logits_avg = (logits1 + logits2) / 2
    preds = make_answer(logits_avg)
    sub_df = pd.read_csv("input/kaggle-llm-science-exam/sample_submission.csv")
    sub_df["prediction"] = preds
    Path("input/llmse-ensemble/").mkdir(parents=True, exist_ok=True)
    sub_df.to_csv("input/llmse-ensemble/submission.csv", index=False)


if __name__ == "__main__":
    main()
