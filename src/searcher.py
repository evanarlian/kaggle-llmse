import numpy as np
from datasets import Dataset
from faiss import Index
from sentence_transformers import SentenceTransformer


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

    def get_paragraphs(self, arr: np.ndarray) -> list[str]:
        # arr shape is (n_quest, neighbour)
        paragraphs = []
        for row in arr:
            combined = " ".join(self.wiki[row]["text"])
            paragraphs.append(combined)
        return paragraphs

    def search_only(self, questions: list[str], k: int) -> list[str]:
        emb = self.bi_encoder.encode(questions, show_progress_bar=False, device="cuda")
        D, I = self.index.search(emb, k=k)  # (n_question, k)
        return self.get_paragraphs(I)

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
        emb = self.bi_encoder.encode(combined, show_progress_bar=False, device="cuda")
        D, I = self.index.search(emb, k=k)  # (n_question, k)
        D = D.reshape(D.shape[0] // 5, -1)
        I = I.reshape(I.shape[0] // 5, -1)
        D_rank = D.argsort(-1)  # NOTE: assume smaller distance is better (L2 index)
        first_dim_idx = np.arange(len(D))[:, None]
        D = D[first_dim_idx, D_rank]
        I = I[first_dim_idx, D_rank]
        # get top-k from every row, i don't know the vectorized way of doing this
        topks = []
        for ii in I:
            topk = list(dict.fromkeys(ii))[:k]
            topks.append(topk)
        topks = np.stack(topks, axis=0)
        return self.get_paragraphs(topks)


def main():
    import faiss
    from datasets import load_from_disk

    wiki = load_from_disk("input/llmse-paragraph-level-emb-faiss/wiki_stem_paragraph")
    index = faiss.read_index("input/llmse-paragraph-level-emb-faiss/wiki_trick.index")
    res = faiss.StandardGpuResources()
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index)
    bi_encoder = SentenceTransformer(
        "./input/llmse-paragraph-level-emb-faiss/all-MiniLM-L6-v2"
    )
    searcher = Searcher(index_gpu, wiki, bi_encoder)
    questions = ["What is La Nina?", "Who invented radio?"]  # spoiler alert: [B, E]
    answers = {
        "A": ["A delicious mexican food", "Alexander Graham Bell"],
        "B": ["A weather pattern", "Isaac Newton"],
        "C": ["A wild animal", "Leonhard Euler"],
        "D": ["A Spanish movie", "J. Robert Oppenheimer"],
        "E": ["A festival in Chile", "Guglielmo Marconi"],
    }
    print("\n🔥SEARCH ONLY ===========================")
    pars = searcher.search_only(questions, k=3)
    for par in pars:
        print("  >>>", par)
    print("\n🔥INCLUDE ANS, UNSHORTENED ===========================")
    pars = searcher.search_include_answer(questions, answers, k=4, shorten_answer=False)
    for par in pars:
        print("  >>>", par)
    print("\n🔥INCLUDE ANS, SHORTENED ===========================")
    pars = searcher.search_include_answer(questions, answers, k=2, shorten_answer=True)
    for par in pars:
        print("  >>>", par)

    # we can see that by injecting answer during search, the result will be bad


if __name__ == "__main__":
    main()
