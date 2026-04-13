import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import prepare_shared_graph


def main():
    print("============================================================")
    print("Preparing Enriched Brain for Dimension 4 Tests")
    print("============================================================")
    prepare_shared_graph("dim4")
    print("Done. Tests can now reuse benchmark/dim4/shared/")


if __name__ == "__main__":
    main()
