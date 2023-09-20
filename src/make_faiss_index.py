# This file is used to create my own paragraph-level faiss index. Some things this
# file do:
# 1. Load @nbroad wiki STEM and @mbanaei wiki STEM by Cohere, then merge
# 2. Clean unused paragraphs ("See also", "References", etc)
# 3. Create embedding by using title injection trick, hopefully better retrieval
# 4. Make faiss index, idk what index is the most suitable


import gc
import re
from pathlib import Path

import faiss
import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


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

    # # nb
    # del ds_stem, ds_cohe, stem_titles, cohe_titles, intersection
    # gc.collect()

    # hparams
    SBERT_MODEL = "all-MiniLM-L6-v2"
    TITLE_TRICK = True
    FAISS_INDEX = "Flat"  # NOTE we cannot tolerate < 100% recall

    # create embeddings but use title trick, we just check if the title is present
    # in the text body or not, if not then append to the front
    if TITLE_TRICK:
        paragraphs = []
        for row in tqdm(ds_combined, desc="Title trick"):
            if row["title"].lower() in row["text"].lower():
                paragraphs.append(row["text"])
            else:
                paragraphs.append(f"{row['title']}. {row['text']}")
    else:
        paragraphs = ds_combined["text"]

    # # nb
    # del ds_combined
    # gc.collect()

    model = SentenceTransformer(SBERT_MODEL)
    embs = model.encode(
        sentences=paragraphs,
        batch_size=128,
        show_progress_bar=True,
        device="cuda",
    )

    # # nb
    # del model, paragraphs
    # gc.collect()

    # just so that we don't repeat this ever again
    with open("input/llmse-paragraph-level-emb-faiss/embs.npy", "wb") as f:
        np.save(f, embs)
        print("Save npy complete")

    # make faiss
    index = faiss.index_factory(embs.shape[1], FAISS_INDEX)
    index.train(embs)
    index.add(embs)
    faiss.write_index(
        index, "input/llmse-paragraph-level-emb-faiss/wiki_stem_faiss.index"
    )


if __name__ == "__main__":
    main()
