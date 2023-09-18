# This file is used for determining the "scienceness" of the document. General view
# on the steps on this file:
# 1. Load 60k data with context by @cdeotte
# 2. Filling null values from ABCDE with random wrong answer
# 3. Clean up prompts with "...wikipedia" string
# 4. Using science detector model from @nbroad to predict context columns scienceness

import os
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython.display import display
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def fillna_with_wrong_ans(df: pd.DataFrame) -> pd.DataFrame:
    df = df.fillna("<nan>")  # because the real nan is a pain
    choices = list("ABCDE")
    # i dont know how to do this in vectorized way
    for i, row in enumerate(df.itertuples()):
        wrong_ans = list(
            {row.A, row.B, row.C, row.D, row.E} - {getattr(row, row.answer), "<nan>"}
        )
        for choice in choices:
            if df.loc[i, choice] == "<nan>":
                df.loc[i, choice] = random.choice(wrong_ans)
    return df


def remove_wiki_string(text: str) -> str:
    # remove these:
    # According to the provided Wikipedia excerpt ...
    # ... based on the provided Wikipedia excerpt?
    # ... according to the provided Wikipedia excerpt?

    # has no "wikipedia"
    if not re.search(r"wikipedia", text, flags=re.I):
        return text
    # split texts by comma, delete substring with "wikipedia"
    splitted = [s.strip() for s in text.split(",")]
    splitted = [s for s in splitted if "wikipedia" not in s.lower()]
    if splitted != []:
        joined = ", ".join(splitted)
        joined = joined if joined.endswith("?") else joined + "?"
        return joined
    # text has "wikipedia", remove obvious pattern
    text = re.sub(
        r" (?:based on|according to|mentioned in|as described|in the provided).+(?:wikipedia|excerpt|provided)\?$",
        "",
        text,
        flags=re.I,
    )
    if "wikipedia" not in text.lower():
        text += "?"
        return text
    # lastly, these are too hard, just remove "wikipedia"
    text = re.sub(r" wikipedia", "", text, flags=re.I)
    alternatives = ["excerpt", "article", "text", "passage", "snippet", "segment"]
    text = re.sub(r"excerpt", random.choice(alternatives), text, flags=re.I)
    return text


class ScienceOrNotDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int = 512) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i) -> str:
        return self.texts[i]

    def collate_fn(self, batch: list[str]):
        encoded = self.tokenizer(
            batch,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )
        return encoded

    def create_dataloader(self, batch_size: int, num_workers: int):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
        )


@torch.no_grad()
def get_science_probs(model, loader, device) -> np.ndarray:
    science_probs = []
    for batch in tqdm(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        probs = model(**batch).logits.softmax(-1)
        science_probs.append(probs[:, 1].cpu())
    return torch.cat(science_probs).numpy()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    train_df = pd.read_csv("input/60k-data-with-context-v2/all_12_with_context2.csv")
    train_df = fillna_with_wrong_ans(train_df)
    train_df["prompt"] = train_df["prompt"].apply(remove_wiki_string)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = "nbroad/sciwiki-e5-sm"
    model = (
        AutoModelForSequenceClassification.from_pretrained(pretrained).to(device).eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    ds = ScienceOrNotDataset(train_df["context"].to_list(), tokenizer)
    loader = ds.create_dataloader(batch_size=64, num_workers=2)

    train_df["science_prob"] = get_science_probs(model, loader, device=device)
    train_df["is_science"] = train_df["science_prob"] > 0.95

    # sns.histplot(train_df, x="science_prob", bins=100)
    # plt.title("Science probability from 60k rows")
    # plt.show()

    # some datasets are pure science, but detected as not science
    # manually toggle them, see sources from here
    # https://www.kaggle.com/datasets/cdeotte/60k-data-with-context-v2
    save_path = Path("input/llmse-science-or-not")
    save_path.mkdir(parents=True, exist_ok=True)
    train_df.loc[train_df["source"].isin([6, 10, 11, 12]), "is_science"] = True
    train_df.to_csv(save_path / "train.csv")

    # sns.countplot(train_df, x="source", hue="is_science")
    # plt.title("Science amount from different sources (after manual fix)")
    # plt.show()


if __name__ == "__main__":
    main()
