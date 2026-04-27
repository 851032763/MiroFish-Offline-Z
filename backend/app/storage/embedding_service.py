"""
EmbeddingService — 通过 Ollama API 生成本地嵌入

使用本地 nomic-embed-text 模型替代 Zep Cloud 的内置嵌入。
使用 Ollama 的 /api/embed 端点生成向量 (768 维)。
"""

import time
import logging
from typing import List, Optional
from functools import lru_cache

import requests

from ..config import Config

logger = logging.getLogger('mirofish.embedding')


class EmbeddingService:
    """使用本地 Ollama 服务器生成嵌入。"""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        self.model = model or Config.EMBEDDING_MODEL
        self.base_url = (base_url or Config.EMBEDDING_BASE_URL).rstrip('/')
        self.max_retries = max_retries
        self.timeout = timeout
        self._embed_url = f"{self.base_url}/api/embed"

        # 简单的内存缓存 (文本 -> 嵌入向量)
        # 使用字典而不是 lru_cache，因为列表不可哈希
        self._cache: dict[str, List[float]] = {}
        self._cache_max_size = 2000

    def embed(self, text: str) -> List[float]:
        """
        为单个文本生成嵌入。

        Args:
            text: 要嵌入的输入文本

        Returns:
            768 维浮点向量

        Raises:
            EmbeddingError: 如果 Ollama 请求在重试后失败
        """
        if not text or not text.strip():
            raise EmbeddingError("无法嵌入空文本")

        text = text.strip()

        # 检查缓存
        if text in self._cache:
            return self._cache[text]

        vectors = self._request_embeddings([text])
        vector = vectors[0]

        # 缓存结果
        self._cache_put(text, vector)

        return vector

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        为多个文本生成嵌入。

        批量处理以避免压垮 Ollama。

        Args:
            texts: 输入文本列表
            batch_size: 每个请求的文本数量

        Returns:
            嵌入向量列表 (与输入顺序相同)
        """
        if not texts:
            return []

        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        # 先检查缓存
        for i, text in enumerate(texts):
            text = text.strip() if text else ""
            if text in self._cache:
                results[i] = self._cache[text]
            elif text:
                uncached_indices.append(i)
                uncached_texts.append(text)
            else:
                # 空文本 — 零向量
                results[i] = [0.0] * 768

        # 批量嵌入未缓存的文本
        if uncached_texts:
            all_vectors: List[List[float]] = []
            for start in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[start:start + batch_size]
                vectors = self._request_embeddings(batch)
                all_vectors.extend(vectors)

            # 放置结果并缓存
            for idx, vec, text in zip(uncached_indices, all_vectors, uncached_texts):
                results[idx] = vec
                self._cache_put(text, vec)

        return results  # type: ignore

    def _request_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        向 Ollama /api/embed 端点发送 HTTP 请求，带重试机制。

        Args:
            texts: 要嵌入的文本列表 (Ollama 支持单个请求批量处理)

        Returns:
            嵌入向量列表
        """
        payload = {
            "model": self.model,
            "input": texts,
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self._embed_url,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()

                embeddings = data.get("embeddings", [])
                if len(embeddings) != len(texts):
                    raise EmbeddingError(
                        f"Expected {len(texts)} embeddings, got {len(embeddings)}"
                    )

                return embeddings

            except requests.exceptions.ConnectionError as e:
                last_error = e
                logger.warning(
                    f"Ollama 连接失败 (尝试 {attempt + 1}/{self.max_retries}): {e}"
                )
            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning(
                    f"Ollama 请求超时 (尝试 {attempt + 1}/{self.max_retries})"
                )
            except requests.exceptions.HTTPError as e:
                last_error = e
                logger.error(f"Ollama HTTP 错误: {e.response.status_code} - {e.response.text}")
                if e.response.status_code >= 500:
                    # 服务器错误 — 重试
                    pass
                else:
                    # 客户端错误 (4xx) — 不重试
                    raise EmbeddingError(f"Ollama 嵌入失败: {e}") from e
            except (KeyError, ValueError) as e:
                raise EmbeddingError(f"无效的 Ollama 响应: {e}") from e

            # 指数退避
            if attempt < self.max_retries - 1:
                wait = 2 ** attempt
                logger.info(f"在 {wait} 秒后重试...")
                time.sleep(wait)

        raise EmbeddingError(
            f"Ollama 嵌入在 {self.max_retries} 次重试后失败: {last_error}"
        )

    def _cache_put(self, text: str, vector: List[float]) -> None:
        """添加到缓存，如果已满则删除最旧的条目。"""
        if len(self._cache) >= self._cache_max_size:
            # 删除约 10% 的最旧条目
            keys_to_remove = list(self._cache.keys())[:self._cache_max_size // 10]
            for key in keys_to_remove:
                del self._cache[key]
        self._cache[text] = vector

    def health_check(self) -> bool:
        """检查 Ollama 嵌入端点是否可达。"""
        try:
            vec = self.embed("健康检查")
            return len(vec) > 0
        except Exception:
            return False


class EmbeddingError(Exception):
    """嵌入生成失败时抛出。"""
    pass
