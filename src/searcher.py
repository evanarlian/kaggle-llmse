import blingfire as bf
import faiss
import numpy as np
import torch
from datasets import Dataset
from faiss import Index
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm


@torch.no_grad()
def search_and_rerank_context(
    wiki: Dataset,
    index: faiss.Index,
    embedder: SentenceTransformer,
    ranker: CrossEncoder,
    ds: Dataset,
    n_articles: int,
    k_pars: int,
    k_sents: int,
):
    """
    Generate context for RAG.

    Args:
        wiki (Dataset): Knowledge base
        index (faiss.Index): The index of knowledge base
        embedder (SentenceTransformer): Bi encoder model
        ranker (CrossEncoder): Cross encoder model
        ds (Dataset): Competition dataset
        n_articles (int): Num wiki articles to fetch
        k_pars (int): Top k paragraphs from articles after reranking with questions
        k_sents (int): Top k sentences from articles after matching with answers
    """
    debug = {}  #  , distances, rerank score
    q_embs = embedder.encode(
        ds["prompt"], show_progress_bar=True, normalize_embeddings=True
    )
    D, I = index.search(q_embs, k=n_articles)
    debug["retrieved_article_ids"] = I
    debug["retrieved_article_scores"] = D
    articles: list[str] = []
    for indices in I:
        rows = wiki[indices]
        articles.append("\n\n".join(rows["article"]))
    # break apart to paragraphs for reranking   TODO this is too slow!
    debug["ranker_scores"] = []
    contexts: list[str] = []
    for quest, article in tqdm(zip(ds["prompt"], articles), total=len(articles)):
        paragraphs = article.split("\n\n")
        pairs = [[quest, par] for par in paragraphs]
        rank_score = ranker.predict(pairs, show_progress_bar=False)
        topk = rank_score.argsort()[::-1][:k_pars]
        ranker_pars = "\n".join([paragraphs[i] for i in topk])
        contexts.append(ranker_pars)
        debug["ranker_scores"].append(rank_score[topk])
    debug["ranker_pars"] = [c.split("\n") for c in contexts]
    # break apart to paragraphs for searching
    debug["embedder_scores"] = []
    contexts2: list[str] = []
    for q_emb, article in tqdm(zip(q_embs, articles), total=len(articles)):
        paragraphs = article.split("\n\n")
        par_embs = embedder.encode(paragraphs)
        embedder_score = q_emb @ par_embs.T
        topk = embedder_score.argsort()[::-1][:k_pars]
        embedder_pars = "\n".join([paragraphs[i] for i in topk])
        contexts2.append(embedder_pars)
        debug["embedder_scores"].append(embedder_score[topk])
    debug["embedder_pars"] = [c.split("\n") for c in contexts2]
    # break apart to sentences for tfidf
    debug["tfidf_sentences"] = []
    debug["tfidf_scores"] = []
    for i, (row, article) in tqdm(enumerate(zip(ds, articles)), total=len(articles)):
        sentences = bf.text_to_sentences(article).split("\n")
        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform(sentences)
        Y = tfidf.transform([row[ans] for ans in "ABCDE"])
        similarity = (Y @ X.T).toarray()
        # just select one max from the all answers
        idx = similarity.argmax()
        contexts[i] = "\n".join([sentences[idx % len(sentences)], contexts[i]])
        debug["tfidf_sentences"].append(sentences[idx % len(sentences)])
        debug["tfidf_scores"].append(similarity.flat[idx])
    return contexts2, Dataset.from_dict(debug)


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
