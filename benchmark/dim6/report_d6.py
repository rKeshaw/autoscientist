"""
Dimension 6 - Reporting Script
==============================
Aggregates the JSON results from the Dimension 6 suite into a single markdown
summary report.
"""

import json
from pathlib import Path


D6_TESTS = [
    ("d6_synthesis_genuineness.json", "Synthesis Genuineness"),
    ("d6_abstraction_quality.json", "Abstraction Quality"),
    ("d6_gap_inference.json", "Gap Inference Accuracy"),
    ("d6_contradiction_maintenance.json", "Contradiction Maintenance"),
    ("d6_decay_calibration.json", "Decay Calibration"),
    ("d6_delayed_insight_promotion.json", "Delayed Insight Promotion"),
    ("d6_time_to_promotion.json", "Time-to-Promotion"),
]


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _key_metric(filename: str, summary: dict) -> str:
    if filename == "d6_synthesis_genuineness.json":
        return (
            "Pos genuine: "
            f"{summary.get('positive_genuine_rate', 0) * 100:.1f}% | "
            "Incoherent abstain: "
            f"{summary.get('incoherent_negative_abstain_rate', 0) * 100:.1f}% | "
            "Boundary safe: "
            f"{summary.get('boundary_safe_rate', 0) * 100:.1f}%"
        )
    if filename == "d6_abstraction_quality.json":
        return (
            "Pos pass: "
            f"{summary.get('positive_pass_rate', 0) * 100:.1f}% | "
            "Neg pass: "
            f"{summary.get('negative_pass_rate', 0) * 100:.1f}%"
        )
    if filename == "d6_gap_inference.json":
        return (
            "Pos pass: "
            f"{summary.get('positive_pass_rate', 0) * 100:.1f}% | "
            "Neg pass: "
            f"{summary.get('negative_pass_rate', 0) * 100:.1f}%"
        )
    if filename == "d6_contradiction_maintenance.json":
        return f"Updated contradictions: {summary.get('contradictions_updated', 0)}"
    if filename == "d6_decay_calibration.json":
        return (
            f"Pruned weak edge: {summary.get('weak_edge_pruned', False)} | "
            f"Medium edge calibrated: {summary.get('medium_edge_decayed_but_preserved', False)}"
        )
    if filename == "d6_delayed_insight_promotion.json":
        return f"Promoted: {summary.get('promoted_count', 0)}"
    if filename == "d6_time_to_promotion.json":
        return f"Pruned: {summary.get('pruned_count', 0)}"
    return "N/A"


def main():
    results_dir = Path("benchmark/dim6/results")
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    report_lines = [
        "# Dimension 6: Consolidation & Insight Buffer Benchmark Report",
        "",
        "This report aggregates the Dimension 6 suite for synthesis,",
        "abstraction, gap inference, contradiction upkeep, decay, and delayed",
        "insight promotion.",
        "",
        "## Summary",
        "",
    ]

    all_results = []
    passed_tests = 0
    missing_tests = 0

    for filename, label in D6_TESTS:
        data = _load_json(results_dir / filename)
        if data is None:
            all_results.append((filename, label, None))
            missing_tests += 1
            continue
        if data.get("summary", {}).get("PASS"):
            passed_tests += 1
        all_results.append((filename, label, data))

    report_lines.append(
        f"**Overall Pipeline Status:** {passed_tests} / {len(D6_TESTS)} Tests Passed"
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
            report_lines.append(f"- **{key}**: {value}")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")

    out_path = results_dir / "report_d6_summary.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"Dimension 6 aggregate report generated at: {out_path}")
    if missing_tests:
        print(f"Missing result files: {missing_tests}")


if __name__ == "__main__":
    main()
