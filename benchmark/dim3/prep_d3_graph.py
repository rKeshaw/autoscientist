import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import prepare_shared_graph


def main():
    print("============================================================")
    print("Preparing Shared Brain for Dimension 3 Tests")
    print("============================================================")
    prepare_shared_graph("dim3")
    print("Done. Tests can now reuse benchmark/dim3/shared/")


if __name__ == "__main__":
    main()
