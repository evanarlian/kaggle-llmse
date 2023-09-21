# This file is used to create my own paragraph-level faiss index. Some things this
# file do:
# 1. Load @nbroad wiki STEM and @mbanaei wiki STEM by Cohere, then merge
# 2. Clean unused paragraphs ("See also", "References", etc)
# 3. Create 2 embeddings by using title injection trick, and not
# 4. Make flat faiss index (for both embeddings) since we want perfect recall
# 5. Save the model that creates the embeddings


import re
from pathlib import Path

import faiss
import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from utils import clean_memory


def clean(text: str) -> str:
    text = text.rsplit("See also", maxsplit=1)[0]
    text = text.rsplit("Bibliography", maxsplit=1)[0]
    text = text.rsplit("References", maxsplit=1)[0]
    text = text.rsplit("External links", maxsplit=1)[0]
    text = text.rsplit("Further reading", maxsplit=1)[0]
    text = text.strip()
    return text


def load_wiki_stem() -> Dataset:
    folder = Path("input/wiki-20220301-en-sci")
    files = list(folder.glob("*.parquet"))
    files.sort(key=lambda x: int(x.name.split("_")[0]))
    files = [str(f) for f in files]
    # split="train" makes it a Dataset, not DatasetDict
    ds = load_dataset("parquet", data_files=files, split="train")
    df = ds.to_pandas()
    df["text"] = df["text"].apply(clean)
    # separate to paragraphs
    paragraphs = []
    titles = []
    unique_paragraphs = set()
    for i, text, url, title in df.itertuples():
        pars = [p for p in text.split("\n\n")]
        pars = [p for p in pars if not re.match("^[a-zA-Z ]*$", p)]
        pars = [p for p in pars if p not in unique_paragraphs]
        paragraphs += pars
        titles += [title] * len(pars)
        unique_paragraphs |= set(pars)
    # make hf_dataset
    ds = Dataset.from_dict(
        {
            "title": titles,
            "text": paragraphs,
        }
    )
    return ds


def load_wiki_cohere() -> Dataset:
    folder = "input/stem-wiki-cohere-no-emb"
    ds = load_from_disk(folder)
    ds = ds.select_columns(["title", "text"])
    return ds


@torch.no_grad()
def create_faiss(model: SentenceTransformer, ds: Dataset, title_trick: bool) -> None:
    # manual embeddings creation to bypass sbert's sorting and to utilize hf dataset
    # caching and mmap. Modified from: SentenceTransformers' encode method
    # fmt: off
    if title_trick:
        def _trick(row):
            if row["title"].lower() in row["text"].lower():
                return {"pars": row["text"]}
            else:
                return {"pars": f"{row['title']}. {row['text']}"}
        paragraphs = ds.map(_trick, remove_columns=["title", "text"])
    else:
        def _notrick(row):
            return {"pars": row["text"]}
        paragraphs = ds.map(_notrick, remove_columns=["title", "text"])
    # fmt: on
    # np concat consumes huge memory so we can't do that
    index = faiss.IndexFlatIP(384)
    for batch in tqdm(paragraphs.iter(batch_size=128), desc="Embed"):
        pars = batch["pars"]
        features = model.tokenize(pars)
        features = {k: v.cuda() for k, v in features.items()}
        out_features = model.forward(features)
        embeddings = out_features["sentence_embedding"]
        # no need to normalize embeddings since we want to use IndexFlatIP (dot prod)
        index.add(embeddings.cpu().numpy())
    savepath = f"input/llmse-paragraph-level-emb-faiss/wiki_{'trick' if title_trick else 'notrick'}.index"
    faiss.write_index(index, savepath)


def main():
    ds_stem = load_wiki_stem()
    ds_cohe = load_wiki_cohere()

    # find duplicates by title (there are *A LOT* of them)
    stem_titles = set(t.lower() for t in ds_stem["title"])  # n=131008
    cohe_titles = set(t.lower() for t in ds_cohe["title"])  # n=276824
    intersection = stem_titles & cohe_titles  # n=108116

    # we need to remove duplicates from wiki stem and merge
    stem_only = ds_stem.filter(lambda x: x["title"].lower() not in intersection)
    ds_combined = concatenate_datasets([ds_cohe, stem_only])
    ds_combined.save_to_disk(
        "input/llmse-paragraph-level-emb-faiss/wiki_stem_paragraph"
    )
    ds_combined = load_from_disk(
        "input/llmse-paragraph-level-emb-faiss/wiki_stem_paragraph"
    )  # force mmap?
    # ds_combined = ds_combined.select(list(range(1000)))  # quick check
    del ds_stem, ds_cohe, stem_titles, cohe_titles, intersection, stem_only
    clean_memory()

    # hparams
    SBERT_MODEL = "all-MiniLM-L6-v2"
    model = SentenceTransformer(SBERT_MODEL).cuda().eval()
    create_faiss(model, ds_combined, title_trick=False)
    clean_memory()
    create_faiss(model, ds_combined, title_trick=True)
    clean_memory()
    model.save(f"input/llmse-paragraph-level-emb-faiss/{SBERT_MODEL}")


if __name__ == "__main__":
    main()
