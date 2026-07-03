"""嵌入后端抽象 —— 一键跨平台。

嵌入模型(bge-m3)：
- Apple Silicon → 用 **MLX**(苹果原生，吃满 GPU/ANE，快且省电)；
- Windows / Linux / GPU 服务器 → 用 **sentence-transformers**(torch，有 CUDA 自动上 GPU，否则 CPU)。

两个后端加载的是**同一个 bge-m3**、输出**同维度向量**，可互换。后端默认按平台自动选
(config.EMBED_BACKEND="auto")，也可用环境变量 EMBED_BACKEND 强制成 "mlx" / "st"。

对外只暴露一个函数：embed(texts) -> np.ndarray(float32, **未归一化**)。
归一化(faiss.normalize_L2)仍由调用方做，保持原有逻辑不变。

注意：换后端后建议在目标机器上重跑 ingest 建索引（建库/查询用同一后端，向量才对齐）。
"""
from __future__ import annotations

import logging
import platform

import numpy as np

import config

logger = logging.getLogger(__name__)

_backend = None      # "mlx" | "st"
_state = None        # mlx: (model, tokenizer)；st: SentenceTransformer 实例


def _pick_backend() -> str:
    b = (config.EMBED_BACKEND or "auto").lower()
    if b != "auto":
        return b
    # 自动：Apple Silicon → mlx；其余一律 → sentence-transformers(torch)
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx"
    return "st"


def _ensure_loaded():
    global _backend, _state
    if _state is not None:
        return
    _backend = _pick_backend()
    if _backend == "mlx":
        from mlx_embeddings import load                     # 仅 Apple 装了它
        logger.info(f"Loading embedder: MLX / {config.EMBED_MODEL}")
        _state = load(config.EMBED_MODEL)                   # (model, tokenizer)
    else:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedder: sentence-transformers / {config.EMBED_MODEL}")
        _state = SentenceTransformer(config.EMBED_MODEL)    # torch，自动挑 CUDA/CPU


def embed(texts: list[str]) -> np.ndarray:
    """一批文本 → float32 向量矩阵(未归一化)。两后端输出同一 bge-m3 向量。"""
    _ensure_loaded()
    if _backend == "mlx":
        from mlx_embeddings import generate
        model, tokenizer = _state
        out = generate(model, tokenizer, texts)
        return np.array(out.text_embeds, dtype=np.float32)
    vecs = _state.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
    return np.asarray(vecs, dtype=np.float32)


def backend() -> str:
    """返回当前生效的后端名（调试用）。"""
    _ensure_loaded()
    return _backend
