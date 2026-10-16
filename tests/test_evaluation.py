"""
Unit tests for the evaluation benchmark harness.
"""

from src.evaluation.benchmark_harness import BenchmarkHarness, EvaluationScenarioResult
from src.evaluation.metrics_reporter import format_benchmark_markdown


def test_benchmark_harness_single_scenario():
    harness = BenchmarkHarness()
    res = harness.run_scenario("clear_day", num_frames=6, fps=20)

    assert isinstance(res, EvaluationScenarioResult)
    assert res.total_frames == 6
    assert res.scenario_name == "clear_day"
    assert res.mean_latency_ms > 0
    assert res.false_alarm_rate >= 0.0


def test_benchmark_markdown_formatting():
    harness = BenchmarkHarness()
    res = harness.run_scenario("dense_fog", num_frames=6, fps=20)
    md = format_benchmark_markdown({"dense_fog": res})

    assert "## PhantomRoad Benchmark Evaluation Summary" in md
    assert "dense_fog" in md
