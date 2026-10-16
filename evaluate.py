"""
Standalone CLI Benchmark Evaluation Runner for PhantomRoad.
OpenCV AI Competition 2026.

Usage:
    python evaluate.py --frames 25 --output reports/benchmark.json
"""

import argparse
import os
import sys

from src.evaluation.benchmark_harness import BenchmarkHarness
from src.evaluation.metrics_reporter import format_benchmark_markdown, export_benchmark_summary


def main():
    parser = argparse.ArgumentParser(description="PhantomRoad Benchmark Evaluation Suite")
    parser.add_argument("--frames", type=int, default=25, help="Number of frames per scenario (default: 25)")
    parser.add_argument("--scenarios", nargs="+", default=None, help="Specific scenarios to evaluate")
    parser.add_argument("--json", type=str, default="reports/evaluation_results.json", help="Path for JSON output")
    parser.add_argument("--csv", type=str, default="reports/evaluation_results.csv", help="Path for CSV output")
    args = parser.parse_args()

    os.makedirs("reports", exist_ok=True)

    print("=" * 70)
    print("PhantomRoad: Executing Reproducible Vision Reliability Benchmark...")
    print(f"Frames per scenario: {args.frames}")
    print("=" * 70)

    harness = BenchmarkHarness()
    results = harness.run_full_suite(scenarios=args.scenarios, num_frames=args.frames)

    md_report = format_benchmark_markdown(results)
    print("\n" + md_report + "\n")

    export_benchmark_summary(results, json_path=args.json, csv_path=args.csv)
    print(f"[REPORT SAVED] JSON: {os.path.abspath(args.json)}")
    print(f"[REPORT SAVED] CSV:  {os.path.abspath(args.csv)}")


if __name__ == "__main__":
    main()
