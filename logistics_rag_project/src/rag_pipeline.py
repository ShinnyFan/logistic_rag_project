from typing import Dict, List, Optional

from .retriever import Retriever
from .qa_chain import QwenChain
from .prompt_templates import build_rag_messages


class RAGPipeline:
    """
    完整 RAG 流程编排：Retrieve → Build Prompt → Generate。
    """

    def __init__(
        self,
        retriever: Retriever,
        qwen_chain: QwenChain,
        top_k: int = 5,
    ):
        self.retriever = retriever
        self.qwen_chain = qwen_chain
        self.top_k = top_k

    def query(self, question: str, top_k: int = None, stream: bool = False) -> Dict:
        if top_k is None:
            top_k = self.top_k

        results = self.retriever.retrieve(question, top_k=top_k)
        chunks = [r[0] for r in results]
        scores = [r[1] for r in results]

        sources = []
        for chunk in chunks:
            sources.append({
                "text": chunk.get("text", ""),
                "source": chunk.get("source", "unknown"),
                "id": chunk.get("id", ""),
            })

        messages = build_rag_messages(question, chunks)
        answer = self.qwen_chain.generate(messages, stream=stream)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "scores": scores,
        }

    def ask(self, question: str, top_k: int = None) -> str:
        result = self.query(question, top_k=top_k, stream=False)
        return result["answer"]