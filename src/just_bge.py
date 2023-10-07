import faiss
import numpy as np
import pandas as pd
from datasets import Dataset, load_from_disk
from sentence_transformers import SentenceTransformer

from searcher import Searcher


def map_at_3(logits: np.ndarray, labels: np.ndarray) -> float:
    preds = logits.argsort(-1)[:, ::-1]
    maps = (
        (preds[:, 0] == labels) / 1
        + (preds[:, 1] == labels) / 2
        + (preds[:, 2] == labels) / 3
    )
    return maps.mean()


def main():
    # load all data-related
    wiki = load_from_disk("input/llmse-paragraph-level-emb-faiss/wiki_stem_paragraph")
    index = faiss.read_index("input/llmse-paragraph-level-emb-faiss/wiki_trick.index")
    res = faiss.StandardGpuResources()
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index)
    bi_encoder = SentenceTransformer(
        "./input/llmse-paragraph-level-emb-faiss/BAAI/bge-small-en-v1.5"
    )
    searcher = Searcher(index_gpu, wiki, bi_encoder)

    # load the val set
    val_df = pd.concat(
        [
            pd.read_csv("input/kaggle-llm-science-exam/train.csv").drop(columns="id"),
            pd.read_csv("input/dataset-wiki-new-1/dataset_wiki_new_1_balanced.csv"),
        ]
    ).reset_index(drop=True)
    val_ds = Dataset.from_pandas(val_df)

    # populate context
    val_ds = val_ds.add_column(
        "context",
        searcher.search_include_answer(
            val_ds["prompt"],
            answers={ans: val_ds[ans] for ans in "ABCDE"},
            k=15,
            shorten_answer=False,
        ),
    )

    # extract contexts and reshaped-qa
    contexts = val_ds["context"]
    qas = []
    for line in val_ds:
        for ans in "ABCDE":
            template = f"{line['prompt']} {line[ans]}"
            qas.append(template)
    print(len(contexts), len(qas), "must be 5 times")

    # predict using bge large
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    cemb = model.encode(
        contexts,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    cemb = cemb.unsqueeze(1)  # (N, 1, dim)
    qaemb = model.encode(
        qas,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    qaemb = qaemb.view(-1, 5, qaemb.size(-1))  # (N, 5, dim)
    # what we want is (N, 5), similarity per ctx
    sim = qaemb @ cemb.transpose(-2, -1)  # (N, 5, 1)
    sim = sim.squeeze(-1)

    # calc metric
    mapper = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    labels = np.array([mapper[ans] for ans in val_ds["answer"]])
    logits = sim.cpu().numpy()
    result = map_at_3(logits, labels)
    print(result, "not that good :(")


if __name__ == "__main__":
    main()
