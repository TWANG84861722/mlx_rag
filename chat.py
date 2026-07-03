import logging

from query import retrieve_rounds
import config
from config import MAX_HISTORY_TURNS, MAX_TOKENS
import model_client

EARLY_STOP_MISSES = 3   # 连续多少个 chunk 被 LLM 判 NONE 就停止本轮 map（现在是唯一的停止机制）

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

CONDENSE_PROMPT_TMPL = """Rewrite the LAST user question into a standalone, self-contained question \
**in English**, for searching an English scientific-paper corpus.
- Translate to English if the question is in another language.
- Resolve any references (it, the second one, that, these, ...) using the conversation history, \
replacing them with the specific names they refer to.
- If a gene symbol or technical term looks garbled (e.g. from speech recognition), correct it to the \
most likely intended term given the context.
- If it is already a standalone English question, return it unchanged.
Output ONLY the rewritten English question — no explanation, no quotes.

Conversation history:
{history}

Last user question: {question}

Rewritten standalone English question:"""

logger = logging.getLogger(__name__)

SYSTEM = """Use the retrieved evidence first.

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
    """Run map step. Returns (extractions, sources, examined).

    examined = 这一轮实际看了多少个 chunk（早停时=停的位置，没停时=整批大小）。
    上层用它判断"是否翻过前 CANDIDATE_K 还没停"来决定要不要再加深一轮。
    """
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
            {"role": "user", "content": prompt_text},
        ]
        # 不再用 rerank 分数硬截断：每个 chunk 都交给 LLM 判"相不相关"。
        # 因为同质语料里相关 chunk 的分数本来就可能很低，硬截断会在 LLM 看到之前就把它误杀。
        result = model_client.chat(messages, max_tokens=150).strip()

        if result and result != "NONE":
            extractions.append(result)
            sources.append(chunk)
            consecutive_misses = 0
            print(f"    chunk {i+1}/{len(chunks)}  found   (rerank={chunk.get('rerank_score', 0):.3f})")
        else:
            consecutive_misses += 1
            print(f"    chunk {i+1}/{len(chunks)}  skip [LLM:NONE]  (rerank={chunk.get('rerank_score', 0):.3f})  misses={consecutive_misses}")
            if consecutive_misses >= EARLY_STOP_MISSES:
                print(f"    Early stop.")
                return extractions, sources, i + 1      # examined = 停在这，看了 i+1 个

    return extractions, sources, len(chunks)            # 整批看完都没停


def map_reduce(question):
    all_extractions = []
    all_sources = []

    for round_idx, chunks in enumerate(retrieve_rounds(question), 1):
        scores = [c["rerank_score"] for c in chunks]
        print(f"\n[Round {round_idx} — {len(chunks)} candidates (FAISS+BM25 union → reranked)]"
              f"  rerank: max={max(scores):.3f}  median={sorted(scores)[len(scores)//2]:.3f}  min={min(scores):.3f}")
        extractions, sources, examined = map_phase(question, chunks)
        all_extractions.extend(extractions)
        all_sources.extend(sources)

        # 续取规则：这一轮"翻过前 CANDIDATE_K 个还没停"(examined > K) → 货多，再加深一轮；
        # 早早就停(examined ≤ K) → 收工。
        if examined <= config.CANDIDATE_K:
            break
        print(f"[Round {round_idx} 翻过 {config.CANDIDATE_K} 仍高产(examined={examined}) → 再取下一轮]")

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
    """把问题（任何语言）规整成一个自足的【英文】问题——供检索用。

    一次 LLM 调用同时干三件事：
    1. 翻译成英文（语料是英文；BM25 只认字面，同语言才匹配得上，否则半条腿废掉）；
    2. 用对话历史解掉指代（"它/第二个" → 具体名称，指代对象常在上一轮回答里）；
    3. 顺手修语音识别在基因名/术语上的错（靠上下文猜回）。
    已是英文 且 无历史（无需解指代）时，直接返回、不调 LLM（省一次）。
    """
    if question.isascii() and not history:      # 已是英文、又没历史要解 → 原样，省一次 LLM
        return question
    hist = "(none)"
    if history:
        lines = []
        for m in history:
            who = "User" if m["role"] == "user" else "Assistant"
            content = m["content"]
            if m["role"] == "assistant" and len(content) > 500:
                content = content[:500] + " …"      # 回答可能很长 → 截断省 token（够解指代即可）
            lines.append(f"{who}: {content}")
        hist = "\n".join(lines)
    prompt = CONDENSE_PROMPT_TMPL.format(history=hist, question=question)
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
            print(f"[规整为英文检索问题 → {standalone}]")
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
