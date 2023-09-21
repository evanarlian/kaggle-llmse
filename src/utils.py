import ctypes
import gc
import re

import numpy as np
import pandas as pd
from datasets import Dataset
from faiss import Index
from IPython.display import display
from sentence_transformers import SentenceTransformer


def wider(df: pd.DataFrame):
    with pd.option_context("display.max_colwidth", None):
        display(df)


def dirx(obj, pat=None):
    """dir() extended, can filter by regex."""
    members = dir(obj)
    if pat is not None:
        members = [m for m in members if re.search(pat, m)]
    return members


def clean_memory():
    # from https://www.kaggle.com/code/simjeg/platypus2-70b-with-wikipedia-rag
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    # torch.cuda.empty_cache()


def map_at_3(logits: np.ndarray, labels: np.ndarray) -> float:
    preds = logits.argsort(-1)[:, ::-1]
    maps = (
        (preds[:, 0] == labels) / 1
        + (preds[:, 1] == labels) / 2
        + (preds[:, 2] == labels) / 3
    )
    return maps.mean()


class Searcher:
    def __init__(self, index: Index, wiki: Dataset, bi_encoder: SentenceTransformer):
        self.index = index
        self.wiki = wiki
        self.bi_encoder = bi_encoder

    def remove_common_suffix_prefix(self, texts: list[str]) -> list[str]:
        splitted = [text.split() for text in texts]
        n_prefix = 0
        for elems in zip(*splitted):  # transpose
            if len(set(elems)) == 1:
                print(set(elems))
                n_prefix += 1
            else:
                break
        splitted_reversed = [s[::-1] for s in splitted]
        n_suffix = 0
        for elems in zip(*splitted_reversed):  # transpose
            if len(set(elems)) == 1:
                n_suffix += 1
            else:
                break
        # len(s) - n_suffix is used instead of [:-n_suffix]
        # because [:] is not the same as [:-0]
        return [" ".join(s[n_prefix : len(s) - n_suffix]) for s in splitted]

    def search_include_answer(
        self,
        questions: list[str],
        answers: dict[str, list[str]],
        k: int,
        shorten_answer: bool,
    ):
        # flatten questions + answers to a single list
        combined = []
        for i, ques in enumerate(questions):
            A = answers["A"][i]
            B = answers["B"][i]
            C = answers["C"][i]
            D = answers["D"][i]
            E = answers["E"][i]
            if shorten_answer:
                A, B, C, D, E = self.remove_common_suffix_prefix([A, B, C, D, E])
            combined += [f"{ques} {ans}" for ans in [A, B, C, D, E]]
        print(combined)  # (5*quest)
        D, I = self.search_only(combined, k=k)
        print(D, I, sep="\n")
        D = D.reshape(D.shape[0] // 5, -1)
        I = I.reshape(I.shape[0] // 5, -1)
        D_rank = D.argsort(-1)  # NOTE: assume smaller distance is better (L2 index)
        first_dim_idx = np.arange(len(D))[:, None]
        D = D[first_dim_idx, D_rank]
        I = I[first_dim_idx, D_rank]
        print(D, I, sep="\n")
        # get top-k from every row, i don't know the vectorized way of doing this
        topks = []
        for ii in I:
            topk = list(dict.fromkeys(ii))[:k]
            topks.append(topk)
        topks = np.stack(topks, axis=0)
        print(topks)

    def search_only(self, questions: list[str], k: int) -> list[str]:
        D = np.random.randn(len(questions), k)
        I = np.random.randint(0, 3, (len(questions), k))
        return D, I
        emb = self.bi_encoder.encode(
            questions, convert_to_tensor=True, show_progress_bar=True
        )
        D, I = self.faiss_index.search(emb, k=q)  # (B, q)
        retrieved = []  # (B,)
        # TODO use tqdm for batch??
        for distances, indices in zip(D.tolist(), I.tolist()):
            paragraphs = "\n".join(self.hf_ds[indices]["text"])
            retrieved.append(paragraphs)
        return retrieved


def main():
    # TODO WIP WIP this shit is so bad rn
    searcher = Searcher(None, None, None)
    questions = ["what is red?", "is water wet?"]
    answers = {
        "A": ["a1 hehe", "a2"],
        "B": ["b1 hehe", "b2"],
        "C": ["c1 hehe", "c2"],
        "D": ["d1 hehe", "d2"],
        "E": ["e1 hehe", "e2"],
    }
    # searcher.search_include_answer(questions, answers, k=2, shorten_answer=False)
    searcher.search_include_answer(questions, answers, k=2, shorten_answer=True)


if __name__ == "__main__":
    main()

D = np.random.randn(50, 4)
I = np.random.randint(0, 100, (50, 4))
