"""
Dimension 3 - Reporting Script
==============================
Aggregates the JSON results from the Dimension 3 suite into a single markdown
summary report.
"""

import json
from pathlib import Path


D3_TESTS = [
    ("d3_pattern_selection.json", "Pattern Selection Accuracy"),
    ("d3_reductive_sufficiency.json", "Reductive Sub-question Sufficiency"),
    ("d3_actionability.json", "Insight Actionability"),
    ("d3_cross_round_coherence.json", "Cross-round Coherence"),
    ("d3_specificity_lift.json", "Mission-answer Specificity Lift"),
    ("d3_subquestion_utility.json", "Sub-question Utility"),
]


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _key_metric(filename: str, summary: dict) -> str:
    if filename == "d3_pattern_selection.json":
        return f"Accuracy: {summary.get('accuracy', 0) * 100:.1f}%"
    if filename == "d3_reductive_sufficiency.json":
        return f"Sufficient fraction: {summary.get('sufficient_fraction', 0) * 100:.1f}%"
    if filename == "d3_actionability.json":
        return f"Actionable fraction: {summary.get('actionable_fraction', 0) * 100:.1f}%"
    if filename == "d3_cross_round_coherence.json":
        return f"Coherent fraction: {summary.get('coherent_fraction', 0) * 100:.1f}%"
    if filename == "d3_specificity_lift.json":
        return f"Mean lift: {summary.get('mean_lift', 0):+.3f}"
    if filename == "d3_subquestion_utility.json":
        return f"Useful fraction: {summary.get('useful_fraction', 0) * 100:.1f}%"
    return "N/A"


def main():
    results_dir = Path("benchmark/dim3/results")
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    report_lines = [
        "# Dimension 3: Thinker / Structured Reasoning Benchmark Report",
        "",
        "This report aggregates the Dimension 3 suite for deliberate reasoning,",
        "pattern selection, decomposition quality, and multi-round mission-oriented",
        "thinking behavior.",
        "",
        "## Summary",
        "",
    ]

    all_results = []
    passed_tests = 0
    missing_tests = 0

    for filename, label in D3_TESTS:
        data = _load_json(results_dir / filename)
        if data is None:
            all_results.append((filename, label, None))
            missing_tests += 1
            continue
        summary = data.get("summary", {})
        if summary.get("PASS"):
            passed_tests += 1
        all_results.append((filename, label, data))

    report_lines.append(
        f"**Overall Pipeline Status:** {passed_tests} / {len(D3_TESTS)} Tests Passed"
    )
    report_lines.append("")
    report_lines.append("| Test Component | Status | Key Metric |")
    report_lines.append("| -------------- | ------ | ---------- |")

    for filename, label, data in all_results:
        if data is None:
            report_lines.append(f"| {label} | MISSING | No result file found |")
            continue
        status = "PASS" if data.get("summary", {}).get("PASS") else "FAIL"
        metric = _key_metric(filename, data.get("summary", {}))
        report_lines.append(f"| {label} | {status} | {metric} |")

    report_lines.append("")
    report_lines.append("## Detailed Results")
    report_lines.append("")

    for _, label, data in all_results:
        report_lines.append(f"### {label}")
        if data is None:
            report_lines.append("Result file missing.")
            report_lines.append("")
            continue
        status = "PASS" if data.get("summary", {}).get("PASS") else "FAIL"
        report_lines.append(f"**Verdict:** {status}")
        report_lines.append("")
        for key, value in data.get("summary", {}).items():
            if key == "PASS":
                continue
            if isinstance(value, dict):
                report_lines.append(f"- **{key}**:")
                for sub_key, sub_value in value.items():
                    report_lines.append(f"  - {sub_key}: {sub_value}")
            else:
                report_lines.append(f"- **{key}**: {value}")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")

    out_path = results_dir / "report_d3_summary.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"Dimension 3 aggregate report generated at: {out_path}")
    if missing_tests:
        print(f"Missing result files: {missing_tests}")


if __name__ == "__main__":
    main()
