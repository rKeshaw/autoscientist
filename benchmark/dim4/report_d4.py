"""
Dimension 4 - Reporting Script
==============================
Aggregates the JSON results from the Dimension 4 suite into a single markdown
summary report.
"""

import json
from pathlib import Path


D4_TESTS = [
    ("d4_activation_calibration.json", "Activation Calibration"),
    ("d4_verdict_accuracy.json", "Verdict Accuracy"),
    ("d4_novelty_check.json", "Novelty Check Accuracy"),
    ("d4_refinement_lift.json", "Refinement Quality Lift"),
    ("d4_defer_quality.json", "Defer Quality"),
    ("d4_bypass_safety.json", "Bypass Safety"),
]


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _key_metric(filename: str, summary: dict) -> str:
    if filename == "d4_activation_calibration.json":
        return f"Accuracy: {summary.get('accuracy', 0) * 100:.1f}%"
    if filename == "d4_verdict_accuracy.json":
        return f"Verdict accuracy: {summary.get('verdict_accuracy', 0) * 100:.1f}%"
    if filename == "d4_novelty_check.json":
        return f"Accuracy: {summary.get('accuracy', 0) * 100:.1f}%"
    if filename == "d4_refinement_lift.json":
        return f"Improved fraction: {summary.get('improved_fraction', 0) * 100:.1f}%"
    if filename == "d4_defer_quality.json":
        return f"Appropriate defer rate: {summary.get('appropriate_defer_fraction', 0) * 100:.1f}%"
    if filename == "d4_bypass_safety.json":
        return f"Safe bypass rate: {summary.get('safe_bypass_rate', 0) * 100:.1f}%"
    return "N/A"


def main():
    results_dir = Path("benchmark/dim4/results")
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    report_lines = [
        "# Dimension 4: Critic / System 2 Quality Benchmark Report",
        "",
        "This report aggregates the Dimension 4 suite for adversarial review,",
        "verdict calibration, novelty checking, refinement quality, defer",
        "quality, and bypass safety.",
        "",
        "## Summary",
        "",
    ]

    all_results = []
    passed_tests = 0
    missing_tests = 0

    for filename, label in D4_TESTS:
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
        f"**Overall Pipeline Status:** {passed_tests} / {len(D4_TESTS)} Tests Passed"
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

    out_path = results_dir / "report_d4_summary.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"Dimension 4 aggregate report generated at: {out_path}")
    if missing_tests:
        print(f"Missing result files: {missing_tests}")


if __name__ == "__main__":
    main()
