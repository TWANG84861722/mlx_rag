import bisect
import hashlib
import logging
import json
from pathlib import Path

import faiss
import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from config import (
    DATA_DIR, DB_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, BATCH_SIZE,
)
import embedder
import loaders

logger = logging.getLogger(__name__)

# ----------------------------
# Chunker（模块级：build_chunks 要用它）
# ----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", ".", ""],
    add_start_index=True,
)

# ----------------------------
# 格式分发表：扩展名 → 解析函数
# 加新格式：在 loaders 实现 parse_xxx，再在这里加一行即可，main() 不用动。
# ----------------------------

PARSERS = {
    ".pdf":  loaders.parse_pdf,
    ".txt":  loaders.parse_txt,
    ".docx": loaders.parse_docx,
    ".pptx": loaders.parse_pptx,
    ".xlsx": loaders.parse_xlsx,
    ".xls":  loaders.parse_xlsx,
    ".csv":  loaders.parse_csv,
}


# ════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════

def main():
    """扫描 data/ → 逐文件解析 → 切块 → 嵌入 → 建 FAISS 索引 → 存盘。"""
    DB_DIR.mkdir(exist_ok=True)

    logger.info(f"Embedder backend: {embedder.backend()}")   # 触发加载 + 打印 mlx/st

    # 载入已有索引 / 元数据（支持断点续传）
    index_path    = DB_DIR / "index.faiss"
    metadata_path = DB_DIR / "metadata.json"
    if index_path.exists() and metadata_path.exists():
        logger.info("Loading existing index and metadata...")
        index = faiss.read_index(str(index_path))
        with open(metadata_path, encoding="utf-8") as f:
            all_chunks = json.load(f)
        processed_papers = {c["paper"] for c in all_chunks}
        logger.info(f"Resuming — {len(processed_papers)} files already done, {len(all_chunks)} chunks loaded")
    else:
        index = None
        all_chunks = []
        processed_papers = set()

    files = sorted(p for p in DATA_DIR.rglob("*") if p.is_file())
    if not files:
        logger.warning(f"No files found in {DATA_DIR}")

    for file_path in files:
        parser = PARSERS.get(file_path.suffix.lower())
        if parser is None:
            # 有真实后缀、且非隐藏文件，才提示（.DS_Store / 隐藏文件不刷屏）
            if file_path.suffix and not file_path.name.startswith("."):
                logger.info(f"跳过不支持的格式: {file_path.name}")
            continue

        if file_path.name in processed_papers:
            logger.info(f"Skip {file_path.name} (already processed)")
            continue

        logger.info(f"Processing {file_path.name}")
        try:
            elements = parser(file_path)
        except Exception as e:
            logger.error(f"Parse failed for {file_path.name}: {e}")
            continue

        paper_chunks = build_chunks(file_path.name, elements)
        if not paper_chunks:
            logger.warning(f"No chunks extracted from {file_path.name}, skipping")
            continue

        # ── Embed ────────────────────────────────────────────
        logger.info(f"  Embedding {len(paper_chunks)} chunks...")
        # 用纯正文嵌入：保持向量对比度干净（标题/section 是主题词，盖在每段上会压低对比度）。
        # "按文档/章节区分"交给 BM25 + rerank（它们带上标题也不会压对比度）。
        texts = [c["text"] for c in paper_chunks]
        vecs_list = []
        for i in range(0, len(texts), BATCH_SIZE):
            vecs_list.append(embedder.embed(texts[i : i + BATCH_SIZE]))
        vecs = np.concatenate(vecs_list, axis=0)
        faiss.normalize_L2(vecs)

        # ── Add to index ─────────────────────────────────────
        if index is None:
            index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        all_chunks.extend(paper_chunks)

        # ── Save immediately ─────────────────────────────────
        faiss.write_index(index, str(index_path))
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"  Saved — {len(paper_chunks)} chunks | total {len(all_chunks)} | index {index.ntotal} vectors")

    logger.info(f"Done. Total chunks: {len(all_chunks)}")


# ════════════════════════════════════════════════════════════
#  main() 用到的零件
# ════════════════════════════════════════════════════════════

def doc_title(paper):
    """文件名(去后缀) → 干净的文档标题，如 "入职流程"/"2020 Perrin ..."。拼进 embedding 用。"""
    return Path(paper).stem.replace("_", " ").replace("-", " ").strip()


def build_chunks(paper, elements):
    """elements → 最终 chunks。

    table / figure 型是成品，原样保留；text 型先合并再按 chunk_size 切块，
    并用字符偏移把每个 chunk 回填到它所属的 page / section。
    最后给每个 chunk 打上 doc_id / chunk_id（稳定身份，便于引用、去重、以后增量更新）。
    """
    doc_id = hashlib.md5(paper.encode("utf-8")).hexdigest()[:10]   # 文档ID = 文件名哈希（紧凑稳定）
    chunks = []
    text_stream = []
    for el in elements:
        if el["type"] == "text":
            text_stream.append((el["page"], el["section"], el["text"]))
        else:  # table / figure：成品，直接带上来源文件名保留
            chunks.append({"paper": paper, **el})

    if text_stream:
        combined = ""
        boundaries = []
        for pn, sec, txt in text_stream:
            boundaries.append((len(combined), pn, sec))
            combined += txt + " "
        offsets = [b[0] for b in boundaries]

        for doc_chunk in splitter.create_documents([combined]):
            start = doc_chunk.metadata.get("start_index", 0)
            idx = bisect.bisect_right(offsets, start) - 1
            _, pn, sec = boundaries[max(0, idx)]
            chunks.append({
                "paper": paper,
                "page": pn,
                "section": sec,
                "type": "text",
                "text": doc_chunk.page_content,
            })

    for i, c in enumerate(chunks):
        c["doc_id"] = doc_id
        c["chunk_id"] = f"{doc_id}#{i:04d}"
    return chunks


if __name__ == "__main__":
    main()
