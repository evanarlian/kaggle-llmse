import re
from pathlib import Path

import blingfire as bf
import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from IPython import embed
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm.auto import tqdm

from searcher import search_and_rerank_context

# from utils import clean_memory

# TODO
# [1]. reconstruct wikipedia from combination of cohere, parsed cohere, and/or nbroad
# [2]. construct faiss index for about 300k wiki artickles using the first paragraphs
# 2b see some weird paragraph
# 3. for every prompt, find top N articles, just concat it visualize distribution
# 4. for every prompt and concatted article, find top K sentences using cross ecnoder with the question, then concat (this is context)
# 5. do the same with val


def clean(text: str) -> str:
    # clean unused sections
    text = text.rsplit("See also", maxsplit=1)[0]
    text = text.rsplit("Bibliography", maxsplit=1)[0]
    text = text.rsplit("References", maxsplit=1)[0]
    text = text.rsplit("External links", maxsplit=1)[0]
    text = text.rsplit("Further reading", maxsplit=1)[0]
    text = text.strip()
    # clean paragraphs
    pars = [p for p in text.split("\n\n")]
    pars = [p for p in pars if not re.match("^[a-zA-Z ]*$", p)]
    text = "\n\n".join(pars)
    return text


def load_wiki_stem() -> Dataset:
    folder = Path("input/wiki-20220301-en-sci")
    files = list(folder.glob("*.parquet"))
    files.sort(key=lambda x: int(x.name.split("_")[0]))
    files = [str(f) for f in files]
    # split="train" makes it a Dataset, not DatasetDict
    ds = load_dataset("parquet", data_files=files, split="train")
    ds = ds.map(lambda x: {"article": clean(x["text"])}, remove_columns=["url", "text"])
    return ds


def load_wiki_cohere() -> Dataset:
    folder = "input/stem-wiki-cohere-no-emb"
    ds = load_from_disk(folder)
    ds = ds.select_columns(["title", "text"]).sort("title")
    endpts = [0] + ds.to_pandas().groupby("title").size().to_numpy().cumsum().tolist()
    titles = []
    articles = []
    for start, end in zip(endpts[:-1], endpts[1:]):
        rows = ds[start:end]
        titles.append(rows["title"][0])
        articles.append("\n\n".join(rows["text"]))
    ds = Dataset.from_dict({"title": titles, "article": articles})
    return ds


def load_wiki_merged() -> Dataset:
    stem = load_wiki_stem()
    cohe = load_wiki_cohere()
    # find duplicates by title (there are *A LOT* of them)
    stem_titles = set(t.lower() for t in stem["title"])
    cohe_titles = set(t.lower() for t in cohe["title"])
    intersection = stem_titles & cohe_titles
    # we need to remove duplicates from wiki stem (prioritize cohere) and merge
    stem_only = stem.filter(lambda x: x["title"].lower() not in intersection)
    ds_combined = concatenate_datasets([cohe, stem_only])
    return ds_combined


def get_abstract(row: dict) -> dict:
    # abstract is title + few sentences from document
    title = row["title"]
    first_sents = bf.text_to_sentences(row["article"]).split("\n")[:2]
    return {"abstract": f"{title}. {' '.join(first_sents)}"}


@torch.no_grad()
def make_faiss_index(model: SentenceTransformer, wiki: Dataset) -> faiss.Index:
    embeddings = model.encode(
        wiki["abstract"],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


# hparams
ROOT = Path("input/llmse-context-filling/")
BI_ENCODER = "BAAI/bge-base-en-v1.5"
CROSS_ENCODER = "BAAI/bge-reranker-base"

# make faiss index for searching relevant wiki articles
# wiki = load_wiki_merged()
# wiki = wiki.map(get_abstract)
# save and load to trigger caching?
# wiki.save_to_disk(ROOT / "wiki_stem")
wiki = load_from_disk(ROOT / "wiki_stem")

# NOTE repeat this part during testing
embedder = SentenceTransformer(BI_ENCODER)
# index = make_faiss_index(embedder, wiki)
# embedder.save(str(ROOT / BI_ENCODER))
# faiss.write_index(index, str(ROOT / "wiki_stem.index"))


#
index = faiss.read_index(str(ROOT / "wiki_stem.index"))
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)


#
val_df = pd.read_csv("input/kaggle-llm-science-exam/train.csv")
val_ds = Dataset.from_pandas(val_df).select(range(10))


# reranker
# https://huggingface.co/BAAI/bge-reranker-base#using-huggingface-transformers-1
ranker = CrossEncoder(CROSS_ENCODER)

contexts, debug = search_and_rerank_context(
    wiki,
    index,
    embedder,
    ranker,
    val_ds,
    n_articles=5,
    k_pars=10,
    k_sents=2,
)
val_ds = val_ds.add_column("context", contexts)
embed()
