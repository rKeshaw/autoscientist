"""
Generate the aggregated Dimension 5 benchmark report.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmark" / "dim5" / "results"

TESTS = [
    ("Retrieval Relevance",     "d5_retrieval_relevance.json"),
    ("Extraction SNR",          "d5_extraction_snr.json"),
    ("Resolution Rate",         "d5_resolution_rate.json"),
    ("Reading List Quality",    "d5_reading_list.json"),
    ("Index Freshness",         "d5_index_freshness.json"),
    ("Predictive Processing",   "d5_predictive_processing.json"),
    ("Dedup Accuracy",          "d5_dedup_accuracy.json"),
    ("Thinker vs Observer",     "d5_thinker_vs_observer.json"),
]


def main():
    lines = [
        "# Dimension 5: Research & Reading Acquisition Benchmark Report\n",
        "This report aggregates the Dimension 5 suite for retrieval relevance,",
        "extraction quality, resolution rate, reading list quality, index",
        "freshness, predictive processing, and dedup accuracy.\n",
        "## Summary\n",
    ]

    pass_count = 0
    total = 0
    skipped_count = 0
    table_rows = []
    detail_sections = []

    for name, filename in TESTS:
        fpath = RESULTS / filename
        if not fpath.exists():
            table_rows.append(f"| {name} | MISSING | — |")
            continue

        with open(fpath) as f:
            data = json.load(f)

        summary = data.get("summary", {})
        passed = summary.get("PASS", False)
        skipped = summary.get("skipped", False)
        total += 1

        if skipped:
            skipped_count += 1
            table_rows.append(f"| {name} | SKIPPED | — |")
            continue

        if passed:
            pass_count += 1

        status = "PASS" if passed else "FAIL"

        # Find a key metric
        key_metric = ""
        for k, v in summary.items():
            if k in ("PASS", "skipped"):
                continue
            if isinstance(v, (int, float)):
                if isinstance(v, float):
                    key_metric = f"{k}: {v:.3f}"
                else:
                    key_metric = f"{k}: {v}"
                break

        table_rows.append(f"| {name} | {status} | {key_metric} |")

        # Detail section
        detail = [f"### {name}", f"**Verdict:** {status}\n"]
        for k, v in summary.items():
            if k == "PASS":
                continue
            detail.append(f"- **{k}**: {v}")
        detail.append("\n---\n")
        detail_sections.append("\n".join(detail))

    total_for_display = max(total, 1)
    lines.append(
        f"**Overall Pipeline Status:** {pass_count} / {total_for_display} Tests Passed"
        f" ({skipped_count} skipped)\n"
    )
    lines.append("| Test Component | Status | Key Metric |")
    lines.append("| -------------- | ------ | ---------- |")
    lines.extend(table_rows)
    lines.append("\n## Detailed Results\n")
    lines.extend(detail_sections)

    report = "\n".join(lines)
    out = RESULTS / "report_d5_summary.md"
    with open(out, "w") as f:
        f.write(report)

    print(report)
    print(f"\nReport saved to: {out}")


if __name__ == "__main__":
    main()
