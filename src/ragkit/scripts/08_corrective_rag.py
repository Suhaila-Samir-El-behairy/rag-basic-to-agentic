"""Corrective RAG (CRAG), following Yan et al. (2024).

Paper: "Corrective Retrieval Augmented Generation", https://arxiv.org/abs/2401.15884

Pipeline (Section 4 of the paper):
  1. Retrieve, then score each document's relevance with a retrieval evaluator.
  2. Decide an action from the per-document scores:
       CORRECT    at least one document scores above the upper threshold
       INCORRECT  every document scores below the lower threshold
       AMBIGUOUS  otherwise
  3. Gather knowledge for the action:
       CORRECT    knowledge refinement of the retrieved documents
       INCORRECT  web search with a keyword-rewritten query (refined too)
       AMBIGUOUS  both, concatenated
  4. Generate from the gathered knowledge.

Knowledge refinement = decompose each document into short "strips", keep only
the strips the evaluator marks relevant, and recompose them in order.

Deviations from the paper, stated plainly:
  - The evaluator is the same LLM prompted to output a score, not a fine-tuned
    T5-large. Scores from a prompted LLM are coarser and less calibrated.
  - Strip filtering is done with one LLM call per document that selects relevant
    strip indices, rather than scoring every strip separately (cheaper on free tiers).
  - ``--rule avg`` keeps the older average-score rule for comparison/ablation.
"""

import argparse
import re
import sys
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ragkit.corpus import default_retriever
from ragkit.pipeline import RAGResult
from ragkit.utils import generate_answer, get_llm, web_search

UPPER_THRESHOLD = 0.7
LOWER_THRESHOLD = 0.3
Rule = Literal["paper", "avg"]


# ---------- Retrieval evaluator ----------
class DocumentGrade(BaseModel):
    score: float = Field(description="Relevance score 0.0 to 1.0")
    reasoning: str = Field(description="One-sentence explanation")


def build_doc_grader():
    prompt = ChatPromptTemplate.from_template(
        """Grade the relevance of this document chunk to the question.

Score 0.0 (irrelevant) to 1.0 (highly relevant).
- 0.0-0.3:  not relevant
- 0.4-0.6:  tangentially related
- 0.7-1.0:  directly addresses the question

Question: {question}
Document: {document}

Respond with JSON: {{"score": 0.X, "reasoning": "..."}}"""
    )
    return prompt | get_llm().with_structured_output(DocumentGrade)


def grade_documents(grader, question: str, docs) -> list[tuple[float, str]]:
    """Score every document; a failed or malformed grade counts as 0.0 (irrelevant)."""
    outputs = grader.batch(
        [{"question": question, "document": d.page_content[:2000]} for d in docs],
        {"max_concurrency": 4},
        return_exceptions=True,
    )
    grades = []
    for out in outputs:
        if isinstance(out, Exception) or out is None or out.score is None:
            reason = f"grader error: {out}" if isinstance(out, Exception) else "no grade"
            grades.append((0.0, reason))
        else:
            grades.append((min(max(float(out.score), 0.0), 1.0), out.reasoning))
    return grades


# ---------- Action trigger ----------
def t_correct_decision(scores, upper=UPPER_THRESHOLD, lower=LOWER_THRESHOLD, rule: Rule = "paper"):
    """Decide CORRECT / INCORRECT / AMBIGUOUS from per-document scores.

    rule="paper": CORRECT if any score >= upper, INCORRECT if all scores <= lower
                  (Yan et al. 2024). Returns (decision, max score).
    rule="avg":   compare the mean score to the thresholds. Returns (decision, mean).

    Scores are clamped to [0, 1]. No retrieved docs counts as INCORRECT.
    """
    if not scores:
        return "INCORRECT", 0.0
    clamped = [min(max(s, 0.0), 1.0) for s in scores]

    if rule == "avg":
        avg = sum(clamped) / len(clamped)
        if avg >= upper:
            return "CORRECT", avg
        if avg <= lower:
            return "INCORRECT", avg
        return "AMBIGUOUS", avg

    if rule != "paper":
        raise ValueError(f"Unknown rule: {rule}")
    best = max(clamped)
    if best >= upper:
        return "CORRECT", best
    if all(s <= lower for s in clamped):
        return "INCORRECT", best
    return "AMBIGUOUS", best


# ---------- Knowledge refinement: decompose -> filter -> recompose ----------
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def decompose(text: str, sentences_per_strip: int = 2) -> list[str]:
    """Split a document into short strips of a few sentences each."""
    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s and s.strip()]
    return [
        " ".join(sentences[i : i + sentences_per_strip])
        for i in range(0, len(sentences), sentences_per_strip)
    ]


class StripSelection(BaseModel):
    relevant: list[int] = Field(description="Indices of strips relevant to the question")


def build_strip_selector():
    prompt = ChatPromptTemplate.from_template(
        """Below are numbered text strips. Select the strips that contain information
useful for answering the question. Return their indices (possibly none).

Question: {question}

Strips:
{strips}"""
    )
    return prompt | get_llm().with_structured_output(StripSelection)


def refine_knowledge(selector, question: str, texts: list[str]) -> str:
    """Keep only the relevant strips of each text, in their original order."""
    strip_lists = [decompose(t) for t in texts]
    selections = selector.batch(
        [
            {"question": question, "strips": "\n".join(f"[{i}] {s}" for i, s in enumerate(strips))}
            for strips in strip_lists
        ],
        {"max_concurrency": 4},
        return_exceptions=True,
    )
    kept: list[str] = []
    for strips, sel in zip(strip_lists, selections, strict=True):
        if isinstance(sel, Exception) or sel is None:
            kept.extend(strips)  # refinement failed: keep the text rather than lose it
            continue
        kept.extend(strips[i] for i in sorted(set(sel.relevant)) if 0 <= i < len(strips))
    return "\n".join(kept)


# ---------- Web search with a rewritten query ----------
def build_query_rewriter():
    prompt = ChatPromptTemplate.from_template(
        """Rewrite the question below as a short web search query made of the
most important keywords (at most 8 words).

Question: {question}

Output ONLY the search query."""
    )
    return prompt | get_llm() | StrOutputParser()


# ---------- Pipeline ----------
def run(
    question: str,
    upper: float = UPPER_THRESHOLD,
    lower: float = LOWER_THRESHOLD,
    rule: Rule = "paper",
    refine: bool = True,
    retriever=None,
) -> RAGResult:
    docs = (retriever or default_retriever()).invoke(question)
    grades = grade_documents(build_doc_grader(), question, docs)
    scores = [s for s, _ in grades]
    decision, agg = t_correct_decision(scores, upper, lower, rule)
    for i, (s, why) in enumerate(grades):
        print(f"  Doc {i + 1}: score={s:.2f} - {why[:80]}", file=sys.stderr)
    print(f"[t-correct] rule={rule} score={agg:.2f} -> {decision}", file=sys.stderr)

    selector = build_strip_selector() if refine else None

    def internal() -> list[str]:
        # Discard documents the evaluator judged irrelevant before refining.
        relevant = [d.page_content for d, (s, _) in zip(docs, grades, strict=True) if s > lower]
        if selector is None:
            return relevant
        return [refine_knowledge(selector, question, relevant)] if relevant else []

    def external() -> list[str]:
        query = build_query_rewriter().invoke({"question": question}).strip()
        print(f"  Web search query: {query}", file=sys.stderr)
        results = web_search(query, max_results=4)
        if selector is None or not results:
            return [results] if results else []
        return [refine_knowledge(selector, question, [results])]

    if decision == "CORRECT":
        contexts = internal()
    elif decision == "INCORRECT":
        contexts = external()
    else:
        contexts = internal() + external()

    contexts = [c for c in contexts if c.strip()]
    return RAGResult(
        answer=generate_answer(question, "\n\n".join(contexts)),
        contexts=contexts,
        tool_calls=[] if decision == "CORRECT" else ["web_search"],
        metadata={"decision": decision, "score": round(agg, 3), "doc_scores": scores},
    )


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Corrective RAG (CRAG), Yan et al. 2024")
    parser.add_argument("--question", required=True)
    parser.add_argument("--upper", type=float, default=UPPER_THRESHOLD, help="default 0.7")
    parser.add_argument("--lower", type=float, default=LOWER_THRESHOLD, help="default 0.3")
    parser.add_argument(
        "--rule",
        choices=["paper", "avg"],
        default="paper",
        help="paper: any-above-upper / all-below-lower (default); avg: mean score",
    )
    parser.add_argument(
        "--no-refine", action="store_true", help="Skip decompose-filter-recompose refinement"
    )
    args = parser.parse_args()

    result = run(args.question, args.upper, args.lower, args.rule, refine=not args.no_refine)
    print("=" * 60)
    print(result.answer)
    print("=" * 60)
    print(f"\n[trace] {result.metadata}", file=sys.stderr)


if __name__ == "__main__":
    main()
