import logging

from query import retrieve_page
import config
from config import MAX_HISTORY_TURNS, MAX_TOKENS
import model_client

EARLY_STOP_MISSES = 3   # consecutive misses before stopping map phase
MIN_RERANK_SCORE  = 0.1  # chunks below this are counted as miss without LLM call

MAP_PROMPT_TMPL = """Does the following text excerpt directly answer or explicitly address the question?

Question: {question}

[{paper}  p.{page}  {section}]
{text}

Rules:
- If the excerpt explicitly contains facts that answer the question, extract only those facts concisely.
- If the excerpt does not directly address the question, respond with exactly: NONE
- Do not infer, generalize, or include loosely related content.
- NONE means zero direct relevance, not low relevance."""

REDUCE_PROMPT_TMPL = """The following were extracted from multiple scientific papers in response to:
"{question}"

Synthesize into a single comprehensive answer. Deduplicate and organize clearly.

Extracted findings:
{extractions}
"""

CONDENSE_PROMPT_TMPL = """Given the conversation history below, rewrite the LAST user question into a \
standalone, self-contained question: replace any references (it, the second one, that, these, etc.) with \
the specific names they refer to. If the question is already self-contained, return it unchanged. \
Keep it in the same language as the question. Output ONLY the rewritten question — no explanation, no quotes.

Conversation history:
{history}

Last user question: {question}

Rewritten question:"""

logger = logging.getLogger(__name__)

SYSTEM = """/no_think
Use the retrieved evidence first.

When answering, structure your response as:

FACTS:
- Only statements directly supported by the provided context.

INFERENCE:
- Reasoned conclusions based on evidence.
- Clearly distinguish from facts.

CONFIDENCE:
- High / Medium / Low

If evidence is insufficient, explicitly say so.
Do not fabricate facts.
"""

def map_phase(question, chunks):
    """Run map step on a list of chunks. Returns (extractions, sources, early_stopped)."""
    extractions = []
    sources = []
    consecutive_misses = 0

    for i, chunk in enumerate(chunks):
        prompt_text = MAP_PROMPT_TMPL.format(
            question=question,
            paper=chunk["paper"],
            page=chunk["page"],
            section=chunk.get("section", ""),
            text=chunk["text"],
        )
        messages = [
            {"role": "user", "content": f"/no_think\n\n{prompt_text}"},
        ]
        score = chunk.get("rerank_score", 0)
        if score < MIN_RERANK_SCORE:
            result = "NONE"
        else:
            result = model_client.chat(messages, max_tokens=150).strip()

        if result and result != "NONE":
            extractions.append(result)
            sources.append(chunk)
            consecutive_misses = 0
            print(f"    chunk {i+1}/{len(chunks)}  found   (rerank={chunk.get('rerank_score', 0):.3f})")
        else:
            consecutive_misses += 1
            reason = "low score" if score < MIN_RERANK_SCORE else "NONE"
            print(f"    chunk {i+1}/{len(chunks)}  skip [{reason}]  (rerank={score:.3f})  misses={consecutive_misses}")
            if consecutive_misses >= EARLY_STOP_MISSES:
                print(f"    Early stop.")
                return extractions, sources, True

    return extractions, sources, False


def map_reduce(question):
    all_extractions = []
    all_sources = []
    offset = 0
    page = 0

    while True:
        page += 1
        chunks = retrieve_page(question, offset=offset)
        if not chunks:
            break

        scores = [c["rerank_score"] for c in chunks]
        print(f"\n[Page {page} — RRF rank {offset+1}–{offset+len(chunks)} (FAISS+BM25, pre-rerank) → reranked here]"
              f"  rerank: max={max(scores):.3f}  median={sorted(scores)[len(scores)//2]:.3f}  min={min(scores):.3f}")
        extractions, sources, early_stopped = map_phase(question, chunks)
        all_extractions.extend(extractions)
        all_sources.extend(sources)
        offset += len(chunks)

        if early_stopped:
            break

        print(f"[No early stop in page {page} — fetching next {config.CANDIDATE_K} chunks...]")

    if not all_extractions:
        return "No relevant information found.", []

    all_findings = "\n\n".join(f"[{i+1}] {e}" for i, e in enumerate(all_extractions))
    reduce_prompt_text = REDUCE_PROMPT_TMPL.format(
        question=question, extractions=all_findings
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": reduce_prompt_text},
    ]
    print(f"\n[Reducing {len(all_extractions)} findings...]")
    return model_client.chat(messages, max_tokens=MAX_TOKENS), all_sources


def condense_question(question, history):
    """用对话历史把当前问题改写成"自足问题"（供检索用）。

    多轮 RAG 的关键：检索是拿"当前问题"这句话去匹配文档，而"它定位在哪?"这类带指代的话
    检不到东西。所以先让 LLM 结合【完整问答历史】把它改写成不依赖上下文的完整问题，再去检索。
    history 为空（第一问）时原样返回，不调 LLM。指代对象常在"上一轮回答"里，所以回答也要带上。
    """
    if not history:
        return question
    lines = []
    for m in history:
        who = "User" if m["role"] == "user" else "Assistant"
        content = m["content"]
        if m["role"] == "assistant" and len(content) > 500:
            content = content[:500] + " …"          # 回答可能很长 → 截断省 token（够解指代即可）
        lines.append(f"{who}: {content}")
    prompt = CONDENSE_PROMPT_TMPL.format(history="\n".join(lines), question=question)
    rewritten = model_client.chat([{"role": "user", "content": prompt}], max_tokens=200).strip()
    return rewritten or question


def main():
    history = []
    while True:
        try:
            question = input("\nQuestion: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if question.lower() in ["exit", "quit"]:
            break
        if not question.strip():
            continue

        # 多轮：先把带指代的追问改写成自足问题，再检索作答（第一问 history 空→原样）
        standalone = condense_question(question, history)
        if standalone != question:
            print(f"[改写为自足问题 → {standalone}]")
        answer, sources = map_reduce(standalone)
        if not sources:
            print("\nNo relevant documents found.")
            continue

        print("\n" + "=" * 80)
        print(answer)

        print("\nSources:")
        for i, hit in enumerate(sources, 1):
            section = hit.get("section", "").strip()
            chunk_type = hit.get("type", "text")
            section_str = f"  [{section}]" if section else ""
            if chunk_type == "figure":
                section_str += "  [figure]"
            elif chunk_type == "table":
                section_str += "  [table]"
            snippet = hit["text"][:120].replace("\n", " ").strip()
            print(
                f"[{i}] {hit['paper']}  p.{hit['page']}{section_str}"
                f"  (rerank={hit.get('rerank_score', 0):.3f})"
            )
            print(f"     {snippet}...")

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        if len(history) > MAX_HISTORY_TURNS * 2:
            history = history[-(MAX_HISTORY_TURNS * 2):]


if __name__ == "__main__":
    main()
