"""Offline tests for the evaluation harness: metrics, statistics, runner, report."""

import json

import pytest

from ragkit.eval import metrics, report
from ragkit.eval.cli import missing_evidence
from ragkit.eval.datasets import DEFAULT_DATASET, EvalExample, load_jsonl
from ragkit.eval.judge import Judge
from ragkit.eval.runner import load_rows, run_eval, score
from ragkit.pipeline import RAGResult, run_pipeline
from ragkit.tracking import track_usage

from .conftest import StructuredFakeLLM


# ---------- dataset ----------
def test_bundled_dataset_is_valid_and_balanced():
    examples = load_jsonl(DEFAULT_DATASET)
    assert len(examples) >= 40
    cats = {e.category for e in examples}
    assert cats == {"single_hop", "multi_hop", "out_of_corpus", "unanswerable"}
    for e in examples:
        if e.category in ("single_hop", "multi_hop"):
            assert e.gold_evidence, f"{e.id} needs gold evidence"


def test_missing_evidence_reports_absent_phrases():
    ex = [EvalExample("a", "q", "g", "single_hop", gold_evidence=["ReAct", "nope"])]
    assert missing_evidence(ex, "the  react   framework") == [("a", "nope")]


# ---------- retrieval metrics ----------
def test_recall_and_mrr():
    contexts = ["nothing here", "uses ReAct", "and Reflexion"]
    assert metrics.recall_at_k(contexts, ["react", "reflexion"]) == 1.0
    assert metrics.recall_at_k(contexts, ["react", "reflexion"], k=2) == 0.5
    assert metrics.reciprocal_rank(contexts, ["reflexion", "react"]) == 0.5
    assert metrics.reciprocal_rank(contexts, ["absent"]) == 0.0
    assert metrics.recall_at_k(contexts, []) is None


@pytest.mark.parametrize(
    ("answer", "abstained"),
    [
        ("I don't know.", True),
        ("The context does not contain that.", True),
        ("I don't have enough information.", True),
        ("", True),
        ("ReAct combines reasoning and acting.", False),
    ],
)
def test_abstention_detection(answer, abstained):
    assert metrics.is_abstention(answer) is abstained


def test_tool_selection_maps_routes_to_tools():
    assert metrics.tool_selection_correct(["vectorstore"], ["vector_search"])
    assert metrics.tool_selection_correct(["web_search", "arxiv_search"], ["arxiv_search"])
    assert not metrics.tool_selection_correct(["web_search"], ["vector_search"])
    assert metrics.tool_selection_correct([], [])


# ---------- statistics ----------
def test_bootstrap_ci_contains_mean_and_ignores_none():
    m, lo, hi = metrics.bootstrap_ci([1, 0, 1, 1, None, 0, 1, 1])
    assert m == pytest.approx(5 / 7)
    assert lo <= m <= hi
    assert metrics.bootstrap_ci([None]) == (None, None, None)


def test_paired_bootstrap_detects_a_clear_win():
    better = [1] * 18 + [0] * 2
    worse = [0] * 18 + [1] * 2
    result = metrics.paired_bootstrap(better, worse)
    assert result["diff"] == pytest.approx(0.8)
    assert result["lo"] > 0
    assert result["p"] < 0.01


def test_paired_bootstrap_no_difference():
    same = [1, 0] * 10
    assert metrics.paired_bootstrap(same, same)["p"] == 1.0


def test_cohen_kappa_known_values():
    assert metrics.cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    # observed 0.5, expected 0.5 -> kappa 0
    assert metrics.cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    # textbook example: 20 items, po = 0.7, pe = 0.5 -> 0.4
    a = [True] * 10 + [False] * 10
    b = [True] * 7 + [False] * 3 + [True] * 3 + [False] * 7
    assert metrics.cohen_kappa(a, b) == pytest.approx(0.4)


# ---------- tracking ----------
def test_run_pipeline_counts_llm_calls_and_survives_errors():
    llm = StructuredFakeLLM(responses=["a", "b"])

    def two_calls(q):
        llm.invoke(q)
        llm.invoke(q)
        return RAGResult(answer="x")

    ok = run_pipeline(two_calls, "q")
    assert ok.llm_calls == 2 and ok.error is None

    def crash(q):
        raise RuntimeError("rate limited")

    bad = run_pipeline(crash, "q")
    assert bad.error == "RuntimeError: rate limited" and bad.answer == ""


def test_track_usage_is_scoped():
    llm = StructuredFakeLLM(responses=["a"] * 3)
    with track_usage() as outer:
        llm.invoke("1")
        with track_usage() as inner:
            llm.invoke("2")
    llm.invoke("3")
    assert inner.llm_calls == 1
    assert outer.llm_calls >= 1


# ---------- judge ----------
class CountingJudgeLLM(StructuredFakeLLM):
    calls: int = 0


def _judge(tmp_path, correct=True, faithful=True):
    from ragkit.eval.judge import CorrectnessVerdict, FaithfulnessVerdict

    llm = CountingJudgeLLM(responses=["unused"])

    def counted(value):
        def fn(_):
            llm.calls += 1
            return value

        return fn

    llm.structured = {
        "CorrectnessVerdict": counted(CorrectnessVerdict(reasoning="r", correct=correct)),
        "FaithfulnessVerdict": counted(FaithfulnessVerdict(reasoning="r", faithful=faithful)),
    }
    return Judge(llm=llm, cache_path=tmp_path / "judge.jsonl", model="fake"), llm


def test_judge_caches_verdicts_on_disk(tmp_path):
    judge, llm = _judge(tmp_path)
    assert judge.correctness("q", "gold", "ans")["correct"] is True
    judge.correctness("q", "gold", "ans")
    assert llm.calls == 1

    reloaded, llm2 = _judge(tmp_path)
    reloaded.correctness("q", "gold", "ans")
    assert llm2.calls == 0, "a new Judge should reuse the on-disk cache"


def test_score_rules(tmp_path):
    judge, llm = _judge(tmp_path, correct=True)
    answerable = EvalExample("a", "q", "g", "single_hop", gold_evidence=["react"])
    unanswerable = EvalExample("u", "q", "none", "unanswerable")

    m = score(answerable, RAGResult(answer="ReAct does X", contexts=["react text"]), judge)
    assert m["correct"] is True and m["faithful"] is True and m["recall@4"] == 1.0

    m = score(answerable, RAGResult(answer="I don't know."), judge)
    assert m["correct"] is False and m["faithful"] is None

    m = score(unanswerable, RAGResult(answer="I don't know."), judge)
    assert m["correct"] is True

    m = score(unanswerable, RAGResult(answer="It is 42.", contexts=["c"]), judge)
    assert m["correct"] is False, "answering an unanswerable question is wrong"

    m = score(answerable, RAGResult(answer="", error="boom"), judge)
    assert m["correct"] is False and m["error"] is True

    tool_ex = EvalExample("t", "q", "g", "single_hop", expected_tools=["vector_search"])
    m = score(tool_ex, RAGResult(answer="x", tool_calls=["vectorstore"]), None, "03:routing")
    assert m["tool_correct"] is True


# ---------- runner + report, end to end ----------
def test_run_eval_end_to_end_with_resume(tmp_path):
    examples = [
        EvalExample("e1", "What is ReAct?", "reasoning + acting", "single_hop", ["react"]),
        EvalExample("e2", "Pet name?", "not stated", "unanswerable"),
        EvalExample("e3", "Capital of Australia?", "Canberra", "out_of_corpus", [], ["web_search"]),
    ]
    calls = []

    def good(q):
        calls.append(("good", q))
        if "Pet" in q:
            return RAGResult(answer="I don't know.")
        return RAGResult(answer="An answer", contexts=["ReAct text"])

    def baseline(q):
        calls.append(("base", q))
        return RAGResult(answer="I don't know.")

    fns = {"01:basic": baseline, "X:good": good}
    judge, _ = _judge(tmp_path)
    run_dir = tmp_path / "run"
    rows = run_eval(fns, examples, run_dir, judge, pipeline_fns=fns)
    assert len(rows) == 6

    n_calls = len(calls)
    run_eval(fns, examples, run_dir, judge, pipeline_fns=fns)
    assert len(calls) == n_calls, "completed pairs must not be re-run"

    summary = report.summarize(load_rows(run_dir))
    assert summary["X:good"]["correct"][0] == pytest.approx(1.0)
    assert summary["01:basic"]["correct"][0] == pytest.approx(1 / 3)

    (run_dir / report.HUMAN_LABELS_FILE).write_text(
        json.dumps({"pipeline": "X:good", "example_id": "e1", "correct": True}) + "\n"
    )
    out = report.write_report(run_dir, load_rows(run_dir))
    md = out.read_text()
    assert "`X:good`" in md and "paired bootstrap" in md and "Cohen" in md
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["n_examples"] == 3
