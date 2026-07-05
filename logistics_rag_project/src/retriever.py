from typing import List, Tuple, Dict
from pathlib import Path

from .embedder import LocalEmbedder, APIEmbedder
from .vector_store import VectorStore


class Retriever:
    """
    检索器，负责将用户查询编码为向量，在 FAISS 索引中检索 Top-K 文本块。
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder,
        top_k: int = 5,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[Dict, float]]:
        if top_k is None:
            top_k = self.top_k
        query_vec = self.embedder.encode_single(query, normalize=True)
        results = self.vector_store.search(query_vec, top_k=top_k)
        return results

    def retrieve_with_scores(self, query: str, top_k: int = None):
        results = self.retrieve(query, top_k=top_k)
        chunks = [r[0] for r in results]
        scores = [r[1] for r in results]
        return chunks, scores


def create_retriever(
    vector_store: VectorStore,
    embedder,
    top_k: int = 5,
) -> Retriever:
    return Retriever(
        vector_store=vector_store,
        embedder=embedder,
        top_k=top_k,
    )