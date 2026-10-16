"""
Formatting and exporting utilities for evaluation benchmark results.
"""

from typing import Dict, Any, List, Optional
import json
import pandas as pd

from .benchmark_harness import EvaluationScenarioResult


def format_benchmark_markdown(results: Dict[str, EvaluationScenarioResult]) -> str:
    """
    Renders benchmark results as a structured Markdown report.
    """
    lines = [
        "## PhantomRoad Benchmark Evaluation Summary",
        "",
        "| Scenario | Total Frames | Initial Degraded | Recovered | Recovery Rate | Human Escalations | Avg Gain | Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for sc_name, r in results.items():
        rec_pct = f"{r.recovery_rate * 100:.1f}%"
        gain_str = f"{r.avg_quality_gain:+.3f}"
        lines.append(
            f"| **{sc_name}** | {r.total_frames} | {r.initial_unreliable_frames} | "
            f"{r.recovered_to_reliable_frames} | **{rec_pct}** | {r.human_escalations} | "
            f"{gain_str} | {r.mean_latency_ms:.1f}ms |"
        )

    lines.append("")
    lines.append(f"_Evaluated {len(results)} distinct test conditions with recorded transform parameters._")
    return "\n".join(lines)


def export_benchmark_summary(
    results: Dict[str, EvaluationScenarioResult],
    json_path: Optional[str] = None,
    csv_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Exports benchmark results to structured JSON and CSV files.
    """
    serializable = {name: res.to_dict() for name, res in results.items()}

    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    if csv_path:
        rows = [res.to_dict() for res in results.values()]
        df = pd.DataFrame(rows)
        # Drop nested dict for flat CSV
        if "transformation_spec" in df.columns:
            df["transformation_spec"] = df["transformation_spec"].apply(json.dumps)
        df.to_csv(csv_path, index=False)

    return serializable
