"""CLI dispatcher — run any script by number, or the evaluation harness."""

import importlib
import sys

SCRIPTS = {
    "01": "ragkit.scripts.01_basic_rag",
    "02": "ragkit.scripts.02_query_transformations",
    "03": "ragkit.scripts.03_routing",
    "04": "ragkit.scripts.04_indexing",
    "05": "ragkit.scripts.05_retrieval",
    "06": "ragkit.scripts.06_self_reflection",
    "07": "ragkit.scripts.07_agentic_rag",
    "08": "ragkit.scripts.08_corrective_rag",
    "09": "ragkit.scripts.09_graph_rag",
}
EVAL_COMMANDS = ("eval", "eval-report", "eval-label", "eval-check")


def _usage() -> None:
    print(f"Usage: ragkit <{min(SCRIPTS)}-{max(SCRIPTS)}> --question 'your question' [options]")
    print("       ragkit eval [--pipelines ...] [--limit N]   (see: ragkit eval --help)")
    print("       ragkit eval-report --run results/<run_id>")
    print("       ragkit eval-label  --run results/<run_id> [--n 30]")
    print("       ragkit eval-check  (checks gold evidence against the corpus)")
    print("\nAvailable scripts:")
    for num, module in SCRIPTS.items():
        print(f"  {num}: {module.split('.')[-1]}")


def main():
    # Unicode output (arrows, bullets) on consoles that default to a legacy code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    command = sys.argv[1] if len(sys.argv) > 1 else None
    if command in EVAL_COMMANDS:
        from ragkit.eval import cli as eval_cli

        sys.exit(eval_cli.main(command, sys.argv[2:]))

    if command not in SCRIPTS:
        _usage()
        sys.exit(1)

    module = importlib.import_module(SCRIPTS[command])
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    module.main()


if __name__ == "__main__":
    main()
