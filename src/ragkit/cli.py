"""CLI dispatcher — run any script by number."""

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


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SCRIPTS:
        print(f"Usage: ragkit <{min(SCRIPTS)}-{max(SCRIPTS)}> --question 'your question' [options]")
        print("\nAvailable scripts:")
        for num, module in SCRIPTS.items():
            print(f"  {num}: {module.split('.')[-1]}")
        sys.exit(1)

    module = importlib.import_module(SCRIPTS[sys.argv[1]])
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    module.main()


if __name__ == "__main__":
    main()
