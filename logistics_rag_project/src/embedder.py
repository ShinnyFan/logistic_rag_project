import numpy as np
from typing import List, Optional
from pathlib import Path


class LocalEmbedder:
    """
    本地 Embedding 模型封装，使用 sentence-transformers。
    模型: paraphrase-multilingual-MiniLM-L12-v2 (384维)
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self.dim = 384

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self.dim = self._model.get_sentence_embedding_dimension()
            print(f"已加载本地Embedding模型: {self.model_name} (维度: {self.dim})")
        return self._model

    def encode(
        self,
        texts: List[str],
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
        )
        return embeddings.astype(np.float32)

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        return self.encode([text], show_progress=False, normalize=normalize)[0]


class APIEmbedder:
    """
    Qwen API Embedding 封装（备用方案）。
    使用阿里云 text-embedding-v3 模型。
    """

    def __init__(self, api_key: str, base_url: str, model: str = "text-embedding-v3"):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model
        self.dim = 1024  # text-embedding-v3 默认维度

    def encode(
        self,
        texts: List[str],
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        from openai import OpenAI
        from tqdm import tqdm

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        embeddings = []

        iterator = tqdm(texts, desc="API Embedding") if show_progress else texts
        for text in iterator:
            resp = client.embeddings.create(model=self.model_name, input=text)
            embeddings.append(resp.data[0].embedding)

        emb_array = np.array(embeddings).astype(np.float32)
        if normalize:
            from faiss import normalize_L2
            normalize_L2(emb_array)
        return emb_array

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        return self.encode([text], show_progress=False, normalize=normalize)[0]


def create_embedder(
    mode: str = "local",
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    device: str = "cpu",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_model: str = "text-embedding-v3",
):
    if mode == "local":
        return LocalEmbedder(model_name=model_name, device=device)
    elif mode == "api":
        return APIEmbedder(api_key=api_key, base_url=base_url, model=api_model)
    else:
        raise ValueError(f"不支持的Embedding模式: {mode}，可选: 'local' 或 'api'")