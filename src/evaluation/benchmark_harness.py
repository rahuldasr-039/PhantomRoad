"""
Benchmark harness for evaluating reliability and adaptive recovery across conditions.
Provides reproducible measurements without synthetic fabrication.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import time
import os
import tempfile
import numpy as np

from ..agent.action_executor import AdaptiveReliabilityAgent
from ..agent.state import ReliabilityState, AgentFrameDecision
from ..video.frame_extractor import VideoLoader
from ..video.synthetic_generator import generate_synthetic_road_video, AVAILABLE_SCENARIOS


@dataclass
class EvaluationScenarioResult:
    """
    Empirical benchmark metrics for a single test condition.
    """
    scenario_name: str
    total_frames: int
    initial_unreliable_frames: int
    recovered_to_reliable_frames: int
    recovery_rate: float
    false_alarm_frames: int
    false_alarm_rate: float
    human_escalations: int
    escalation_rate: float
    mean_latency_ms: float
    avg_quality_before: float
    avg_quality_after: float
    avg_quality_gain: float
    transformation_spec: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BenchmarkHarness:
    """
    Runs automated comparative evaluations between clear baseline and degraded clips.
    """

    def __init__(self, agent: Optional[AdaptiveReliabilityAgent] = None):
        self.agent = agent or AdaptiveReliabilityAgent(max_attempts=3)

    def run_scenario(
        self,
        scenario: str,
        num_frames: int = 30,
        fps: int = 20,
        temp_dir: Optional[str] = None
    ) -> EvaluationScenarioResult:
        """
        Generates reproducible clip for a scenario and evaluates the agent on every frame.
        """
        owns_dir = False
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="phantomroad_bench_")
            owns_dir = True

        clip_path = os.path.join(temp_dir, f"{scenario}.mp4")

        # Record exact transformation specification for scientific reproducibility
        spec = {
            "scenario": scenario,
            "description": AVAILABLE_SCENARIOS.get(scenario, ""),
            "num_frames": num_frames,
            "fps": fps,
            "resolution": "640x360",
            "fourcc": "mp4v"
        }

        # 1. Generate Clip
        generate_synthetic_road_video(
            scenario=scenario,
            output_path=clip_path,
            num_frames=num_frames,
            fps=fps,
            width=640,
            height=360
        )

        # 2. Evaluate with Agent
        self.agent.reset_temporal_state()

        latencies_ms: List[float] = []
        scores_before: List[float] = []
        scores_after: List[float] = []
        initial_unreliable = 0
        recovered = 0
        false_alarms = 0
        escalations = 0

        with VideoLoader(clip_path) as loader:
            for idx, timestamp, frame in loader.iterate_frames():
                t0 = time.perf_counter()
                decision = self.agent.process_frame(
                    frame_bgr=frame,
                    frame_index=idx,
                    timestamp_sec=timestamp,
                    video_loader=loader
                )
                t1 = time.perf_counter()

                latency = (t1 - t0) * 1000.0
                latencies_ms.append(latency)

                q_init = decision.initial_metrics.composite_quality_score
                q_final = decision.final_metrics.composite_quality_score
                scores_before.append(q_init)
                scores_after.append(q_final)

                is_init_unreliable = decision.initial_state != ReliabilityState.RELIABLE
                if is_init_unreliable:
                    initial_unreliable += 1

                # If clear baseline, check for false alarms
                if scenario == "clear_day" and is_init_unreliable:
                    false_alarms += 1

                # Recovery outcome
                if is_init_unreliable and decision.final_state == ReliabilityState.RELIABLE:
                    recovered += 1

                if decision.human_review_required:
                    escalations += 1

        # Clean up temporary video if harness created directory
        if owns_dir:
            try:
                if os.path.exists(clip_path):
                    os.remove(clip_path)
                os.rmdir(temp_dir)
            except OSError:
                pass

        total = len(latencies_ms)
        recovery_rate = float(recovered / max(1, initial_unreliable)) if initial_unreliable > 0 else 1.0
        false_alarm_rate = float(false_alarms / max(1, total)) if scenario == "clear_day" else 0.0
        escalation_rate = float(escalations / max(1, total))

        mean_q_before = float(np.mean(scores_before)) if scores_before else 0.0
        mean_q_after = float(np.mean(scores_after)) if scores_after else 0.0
        gain = float(mean_q_after - mean_q_before)

        return EvaluationScenarioResult(
            scenario_name=scenario,
            total_frames=total,
            initial_unreliable_frames=initial_unreliable,
            recovered_to_reliable_frames=recovered,
            recovery_rate=round(recovery_rate, 4),
            false_alarm_frames=false_alarms,
            false_alarm_rate=round(false_alarm_rate, 4),
            human_escalations=escalations,
            escalation_rate=round(escalation_rate, 4),
            mean_latency_ms=round(float(np.mean(latencies_ms)), 2),
            avg_quality_before=round(mean_q_before, 4),
            avg_quality_after=round(mean_q_after, 4),
            avg_quality_gain=round(gain, 4),
            transformation_spec=spec
        )

    def run_full_suite(
        self,
        scenarios: Optional[List[str]] = None,
        num_frames: int = 30
    ) -> Dict[str, EvaluationScenarioResult]:
        """
        Runs the benchmark suite across all or selected scenarios.
        """
        target_scenarios = scenarios or list(AVAILABLE_SCENARIOS.keys())
        results = {}

        for sc in target_scenarios:
            res = self.run_scenario(scenario=sc, num_frames=num_frames)
            results[sc] = res

        return results
