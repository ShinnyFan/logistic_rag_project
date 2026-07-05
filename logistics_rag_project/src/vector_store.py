import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import faiss


class VectorStore:
    """
    FAISS 向量存储，使用 IndexFlatIP（内积）实现精确检索。
    向量需预先归一化，使得内积等价于余弦相似度。
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index: Optional[faiss.IndexFlatIP] = None
        self.id_map: Dict[int, Dict] = {}

    def build_index(self, chunks: List[Dict], embeddings: np.ndarray):
        n, d = embeddings.shape
        if d != self.dim:
            self.dim = d
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)

        self.id_map = {}
        for i, chunk in enumerate(chunks):
            self.id_map[i] = {
                "id": chunk["id"],
                "text": chunk["text"],
                "source": chunk.get("source", "unknown"),
            }

        print(f"FAISS索引构建完成: {self.index.ntotal} 条向量, 维度 {self.dim}")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        if self.index is None:
            raise RuntimeError("索引未构建或未加载")

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype(np.float32)
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            chunk_info = self.id_map.get(int(idx), {"id": "unknown", "text": "", "source": "unknown"})
            results.append((chunk_info, float(scores[0][i])))
        return results

    def save(self, index_path: Path, id_map_path: Path):
        if self.index is None:
            raise RuntimeError("索引为空，无法保存")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = faiss.serialize_index(self.index)
        with open(index_path, "wb") as f:
            f.write(serialized.tobytes())
        with open(id_map_path, "wb") as f:
            pickle.dump(self.id_map, f)
        print(f"索引已保存: {index_path}")

    def load(self, index_path: Path, id_map_path: Path):
        if not index_path.exists():
            raise FileNotFoundError(f"索引文件不存在: {index_path}")
        with open(index_path, "rb") as f:
            data = f.read()
        self.index = faiss.deserialize_index(np.frombuffer(data, dtype=np.uint8))
        self.dim = self.index.d
        with open(id_map_path, "rb") as f:
            self.id_map = pickle.load(f)
        print(f"索引已加载: {self.index.ntotal} 条向量, 维度 {self.dim}")

    @property
    def is_loaded(self) -> bool:
        return self.index is not None

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0