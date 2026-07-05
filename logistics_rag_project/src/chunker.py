import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

from .pdf_loader import PDFDocument


class TextChunker:
    """
    文本切块器，使用 LangChain RecursiveCharacterTextSplitter 按语义切分。
    支持自定义 chunk_size 和 chunk_overlap。
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = None

    @property
    def splitter(self):
        if self._splitter is None:
            try:
                from langchain_text_splitters import RecursiveCharacterTextSplitter
            except ImportError:
                from langchain.text_splitter import RecursiveCharacterTextSplitter
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
                length_function=len,
            )
        return self._splitter

    def chunk_documents(self, documents: List[PDFDocument]) -> List[Dict]:
        all_chunks = []
        chunk_id = 0

        for doc in documents:
            doc_chunks = self.splitter.split_text(doc.full_text)
            for chunk_text in doc_chunks:
                text = chunk_text.strip()
                if not text or len(text) < 20:
                    continue
                all_chunks.append({
                    "id": str(chunk_id),
                    "text": text,
                    "source": doc.filename,
                    "char_count": len(text),
                })
                chunk_id += 1

        print(f"共切分为 {len(all_chunks)} 个文本块 (来自 {len(documents)} 个PDF)")
        return all_chunks

    def chunk_text(self, text: str, source: str = "unknown") -> List[Dict]:
        chunks = self.splitter.split_text(text)
        result = []
        for i, chunk_text in enumerate(chunks):
            text_content = chunk_text.strip()
            if not text_content or len(text_content) < 20:
                continue
            chunk_id = hashlib.md5(f"{source}_{i}".encode()).hexdigest()[:12]
            result.append({
                "id": chunk_id,
                "text": text_content,
                "source": source,
                "char_count": len(text_content),
            })
        return result


def chunk_documents(
    documents: List[PDFDocument],
    chunk_size: int = 512,
    chunk_overlap: int = 128,
) -> List[Dict]:
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_documents(documents)


def save_chunks(chunks: List[Dict], chunks_path: Path, metadata_path: Optional[Path] = None):
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"文本块已保存至: {chunks_path}")

    if metadata_path:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "total_chunks": len(chunks),
            "sources": list(set(c["source"] for c in chunks)),
            "avg_char_count": sum(c["char_count"] for c in chunks) / max(len(chunks), 1),
        }
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        print(f"元数据已保存至: {metadata_path}")


def load_chunks(chunks_path: Path) -> List[Dict]:
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)