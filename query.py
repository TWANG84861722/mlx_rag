import json
import logging
import re
from pathlib import Path

import faiss
import numpy as np

from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from hgnc import expand_query

import config
import embedder
from config import (
    DB_DIR, RERANKER_MODEL,
    CANDIDATE_K,
)

logger = logging.getLogger(__name__)

# ----------------------------
# Load models and index
# ----------------------------

index_path    = DB_DIR / "index.faiss"
metadata_path = DB_DIR / "metadata.json"

if not index_path.exists():
    raise FileNotFoundError(f"Index not found at {index_path}. Run ingest.py first.")
if not metadata_path.exists():
    raise FileNotFoundError(f"Metadata not found at {metadata_path}. Run ingest.py first.")

logger.info(f"Embedder backend: {embedder.backend()}")

logger.info("Loading reranker...")
# max_length=512：把每个 (query, doc) 对截断到 512 token。交叉编码器的标准做法，
# 否则遇到超长 chunk(整页扫描/大图描述几千 token)，注意力矩阵 ~序列长² 会爆到几十 GiB、
# 在 MPS/GPU 上直接 OOM（"Invalid buffer size"）。截断不影响排序质量。
reranker = CrossEncoder(RERANKER_MODEL, max_length=512)

logger.info("Loading FAISS index...")
index = faiss.read_index(str(index_path))

with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)


# ----------------------------
# 小工具：文档标题 / 文本拼装 / 分词
# ----------------------------

def _doc_title(paper):
    return Path(paper).stem.replace("_", " ").replace("-", " ").strip()


def _tok(text):
    """简单分词：小写 ASCII 词 + 单个中文字。
    （英文/基因名按词；中文按单字切——BM25 够用，避免漏掉中文。想更精确可上 jieba。）"""
    return re.findall(r"[a-z0-9]+|[一-鿿]", text.lower())


def _bm25_text(c):
    """BM25 索引用：标题 + section + 正文 → 这样能按"入职/离职"等标题词区分文档。
    （IDF 会自动给常见词低权、罕见词高权，所以同质语料的标题词无害。）"""
    return f"{_doc_title(c['paper'])} {c.get('section', '')} {c['text']}"


def _rerank_text(c):
    """rerank 输入用：给正文带上「标题 + section」上下文，帮 cross-encoder 区分近乎相同的段。"""
    sec = c.get("section", "")
    head = f"[{_doc_title(c['paper'])}]" + (f" [{sec}]" if sec else "")
    return f"{head}\n{c['text']}"


logger.info("Building BM25 index...")
bm25 = BM25Okapi([_tok(_bm25_text(c)) for c in metadata])

logger.info(f"Index ready — {index.ntotal} vectors, {len(metadata)} chunks")


# ----------------------------
# 混合检索：向量 + BM25 → RRF 融合
# ----------------------------

def _hybrid_fuse(question, pool):
    """向量检索 + BM25 检索 → RRF 融合 → 返回前 pool 个 chunk 的下标。

    向量管语义、BM25 管精确词/字面区分；RRF 把两路排名公平地合成一个。
    """
    expanded = expand_query(question)

    # 向量路（query 和 doc 现在都是纯正文，对称）
    qv = embedder.embed([expanded])
    faiss.normalize_L2(qv)
    _, I = index.search(qv, pool)
    vec_ranked = [int(i) for i in I[0] if i >= 0]

    # BM25 路
    bm25_scores = bm25.get_scores(_tok(expanded))
    bm25_ranked = [int(i) for i in np.argsort(bm25_scores)[::-1][:pool]]

    # RRF 融合：score = Σ 1/(k + 名次)，k=60
    rrf = {}
    for ranks in (vec_ranked, bm25_ranked):
        for r, idx in enumerate(ranks):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (60 + r)
    return sorted(rrf, key=lambda i: rrf[i], reverse=True)[:pool]


def retrieve_page(question, offset, page_size=None):
    """取融合排名里 [offset : offset+page_size] 这一页，再 rerank。

    chat.py 的 map_reduce 靠它分页取候选；融合排名只算一次成本不高。
    """
    if page_size is None:
        page_size = CANDIDATE_K

    fused = _hybrid_fuse(question, offset + page_size)
    page = fused[offset : offset + page_size]
    candidates = [metadata[i] for i in page]
    if not candidates:
        return []

    pairs = [[question, _rerank_text(c)] for c in candidates]
    rerank_scores = reranker.predict(pairs)
    ranked = sorted(zip(rerank_scores, candidates), key=lambda x: x[0], reverse=True)
    logger.info(f"retrieve_page(offset={offset}, size={len(ranked)})")
    return [{"rerank_score": float(s), **c} for s, c in ranked]
