"""Benchmark harness: run every pipeline on a labelled QA set and score it.

Modules:
  datasets  load the QA set (questions, gold answers, gold evidence, expected tools)
  metrics   retrieval recall/MRR, abstention, tool selection, bootstrap CIs, Cohen's kappa
  judge     LLM-as-judge for answer correctness and faithfulness (cached on disk)
  runner    run pipelines x questions with resume, scoring each row
  report    aggregate rows into a markdown table + quality-vs-cost plot
  cli       `ragkit eval`, `ragkit eval-report`, `ragkit eval-label`
"""
