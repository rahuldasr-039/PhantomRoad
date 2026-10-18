# PhantomRoad — Adaptive Vision Reliability Agent
### OpenCV AI Competition 2026 Entry

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/opencv-5.0.0-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.66.0-red.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/pytest-passing-brightgreen.svg)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Executive Summary

Autonomous vehicles and Advanced Driver Assistance Systems (ADAS) depend on computer vision to detect road hazards, lane boundaries, vehicles, and pedestrians. However, real-world road environments frequently encounter **optical degradation**—such as motion blur, low-light darkness, low contrast, dense fog/spray, specular glare, and camera lens contamination.

Typical systems either passively display weather warnings or naively feed degraded pixels into deep neural networks, risking catastrophic perception failure or hallucinated safety.

**PhantomRoad** introduces an **Adaptive Vision Reliability Agent**:
- **Continuously Measures Optical Degradation**: Employs documented image processing methods (Laplacian variance, 2D-FFT frequency distribution, RMS contrast, Dark Channel Prior transmission, and spatial entropy).
- **Distinguishes Measurable Degradation from Inferred Causes**: Separates physical sensor measurements from speculative environmental hypotheses.
- **Executes Closed-Loop Corrective Actions**: Intelligently selects targeted enhancements (CLAHE, non-linear Gamma tone-curves, Dark Channel dehazing, unsharp masking, or temporal neighbor frame scrubbing).
- **Empirical Before/After Verification**: Re-evaluates quality indicators post-action and **accepts modifications only if quantitative metrics prove quality improved without artifact injection**—rolling back failed attempts.
- **Safety Escalation**: Stops after a strict configurable retry limit ($N \le 3$) and escalates persistent obstruction or unresolvable uncertainty to human review.
- **Reliability-Aware Downstream Perception**: Annotates downstream detections as `TRUSTED`, `UNVERIFIED_DEGRADED`, or `SUPPRESSED_OBSTRUCTED`. Poor image quality never falsely claims an obstacle is absent.

---

## 2. Core Computer Vision Formulations

### A. Blur & Sharpness Estimation
- **Pech-Pacheco Laplacian Variance**:
  $$\text{Var}(\nabla^2 I) = \frac{1}{N} \sum_{x, y} (\nabla^2 I(x, y) - \mu)^2$$
- **2D-FFT High-Frequency Energy Ratio**:
  Calculates the proportion of energy outside the central DC radius in the 2D Fourier magnitude spectrum.

### B. Illumination & Contrast Profiling
- **Mean Luminance**: Average intensity in perceptual luminance space ($Y$ channel of $YCrCb$).
- **RMS Contrast**: Standard deviation of pixel intensities normalized to $[0.0, 1.0]$.
- **Dynamic Range**: Intensity difference between the 95th and 5th percentiles.
- **Exposure Clipping Ratios**: Quantitative percentage of pixels in deep shadow ($I < 20$) vs specular clipping ($I > 235$).

### C. Visibility & Atmospheric Attenuation (Dark Channel Prior)
Based on He et al. (IEEE TPAMI 2011):
- **Dark Channel**:
  $$J^{dark}(x) = \min_{c \in \{r,g,b\}} \left( \min_{y \in \Omega(x)} (I^c(y)) \right)$$
- **Transmission Map**:
  $$t(x) = 1 - \omega \min_c \left( \min_{y \in \Omega(x)} \frac{I^c(y)}{A^c} \right)$$
- Evaluates transmission primarily in the road surface region to prevent false positives from bright, open skies.

### D. Multi-Indicator Obstruction & Temporal Persistence
- **Spatial Grid Entropy & Edge Sparsity**: Partitions frame into an $R \times C$ grid to detect dead/lifeless patches.
- **Temporal Persistence Filter**: Tracks obstruction candidates across a sliding multi-frame window:
  - Ephemeral occlusions (windshield wiper sweep, falling leaf) resolve in 1–2 frames.
  - Foreign matter on camera lens (mud, smudge, lens tape) persists across $K \ge 3$ consecutive frames, triggering `CRITICAL_OBSTRUCTION`.

---

## 3. Architecture & Modular Project Structure

```
PhantomRoad/
├── app.py                           # Master Streamlit Operations Dashboard
├── evaluate.py                      # Standalone CLI Benchmark Suite
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variables template
├── .gitignore                       # Git ignore configuration
├── README.md                        # Documentation and guide
├── models/
│   └── download_models.py           # 1-click MobileNet-SSD weights downloader
├── src/
│   ├── video/
│   │   ├── frame_extractor.py       # VideoLoader and temporal frame buffering
│   │   └── synthetic_generator.py   # 6 standard reproducible scenario generators
│   ├── metrics/
│   │   ├── blur.py                  # Laplacian variance & FFT frequency metrics
│   │   ├── illumination.py          # Brightness, RMS contrast, clipping fractions
│   │   ├── visibility.py            # Dark Channel Prior & glare analysis
│   │   ├── obstruction.py           # Multi-indicator spatial & temporal persistence
│   │   └── quality_assessor.py      # Telemetry aggregator & hypothesis separator
│   ├── agent/
│   │   ├── state.py                 # Reliability states & audit structures
│   │   ├── decision_engine.py       # State determination & action selection
│   │   └── action_executor.py       # Closed-loop action loop, rollback, & escalation
│   ├── enhancement/
│   │   ├── contrast_clahe.py        # LAB color space CLAHE
│   │   ├── illumination_gamma.py    # Precomputed LUT Gamma correction
│   │   ├── sharpness_unsharp.py     # Adaptive unsharp masking
│   │   ├── dehaze_dcp.py            # Guided filter dark channel dehazing
│   │   ├── denoise.py               # Edge-preserving bilateral filtering
│   │   └── verification.py          # Quantitative before/after improvement verifier
│   ├── detector/
│   │   ├── base.py                  # DetectionBox & Detector interface
│   │   ├── opencv_dnn.py            # OpenCV MobileNet-SSD / ONNX detector
│   │   └── reliability_wrapper.py   # Reliability decorator & HUD renderer
│   ├── evaluation/
│   │   ├── benchmark_harness.py     # Systematic comparative benchmark engine
│   │   └── metrics_reporter.py      # Markdown & CSV/JSON report generators
│   ├── reporting/
│   │   └── audit_report.py          # Complete JSON compliance and CSV exporter
│   ├── cloud/
│   │   ├── aws_integration.py       # Enterprise S3 storage, presigned URLs, and IAM role auth
│   │   └── cloud_pipeline.py        # Cloud event schemas, batch job dispatching & serverless templates
│   └── ui/
│       ├── styles.py                # Operations Dark Theme CSS & badges
│       └── components.py            # Reusable Streamlit telemetry widgets
├── tests/
│   ├── test_metrics.py              # Tests for all low-level CV metrics
│   ├── test_enhancements.py         # Tests for OpenCV enhancement algorithms
│   ├── test_agent.py                # Tests for agent state transitions & rollback
│   ├── test_video.py                # Tests for video decoding & synthetic generator
│   ├── test_detector.py             # Tests for detector wrapper & safety flags
│   ├── test_evaluation.py           # Tests for benchmark evaluation harness
│   ├── test_stage1.py               # Comprehensive Stage 1 ingestion & telemetry test suite
│   ├── test_stage2.py               # Comprehensive Stage 2 closed-loop decision test suite
│   ├── test_stage3.py               # Comprehensive Stage 3 downstream perception test suite
│   ├── test_stage4.py               # Comprehensive Stage 4 AWS cloud & S3 mocked test suite
│   └── test_reporting_and_cloud.py  # Tests for JSON/CSV audit & offline AWS mode
└── reports/                         # Default directory for exported audit reports
```

---

## 4. Quickstart Guide (Windows PowerShell)

### Step 1: Install Dependencies
Open Windows PowerShell in the project root:
```powershell
python -m pip install -r requirements.txt
```

### Step 2: Run Automated Tests
Verify that all 23 unit and integration tests pass:
```powershell
python -m pytest tests/ -v
```

### Step 3: Launch the Interactive Operations Dashboard
```powershell
python -m streamlit run app.py
```
Streamlit will automatically launch in your default browser at `http://localhost:8501`.

---

## 5. Using the Interactive Dashboard

1. **Preset Scenarios or Video Upload**:
   - In the sidebar, select **Preset Scenario** to test any of the 6 standard reproducible conditions (`Clear Day`, `Motion Blur`, `Night Low Light`, `Dense Fog`, `Oncoming Glare`, `Lens Obstruction`) with a single click.
   - Or upload your own road dashcam video (`.mp4`, `.avi`, `.mov`).
2. **Live Frame Operations Tab**:
   - Scrub through video frames with instant telemetry updates.
   - View the side-by-side comparison of original vs agent-corrected frames.
   - Inspect the pixel intensity difference heatmap.
   - Read the step-by-step action history explaining why enhancements were accepted or rolled back.
3. **Full Video Timeline & Export Tab**:
   - Run batch processing over the full video clip.
   - Inspect the interactive Altair timeline of reliability states.
   - Download verifiable JSON and CSV audit reports.
4. **Benchmark & Evaluation Tab**:
   - Run the automated benchmark suite across all conditions.
   - View empirical recovery rates and processing latencies.
5. **Human Escalation Console Tab**:
   - Review frames flagged for human intervention, inspect reasons, and submit certified operator review decisions.

---

## 6. Downstream Road-Scene Detection Setup

PhantomRoad integrates with **OpenCV DNN** (`cv2.dnn`) to detect vehicles, cyclists, and pedestrians.

- **Offline Mode (Default)**: If model weights are not present, PhantomRoad displays an honest setup message. **It never simulates or fabricates detections.**
- **Enabling Live Object Detection**:
  Run the automated downloader to fetch the official MobileNet-SSD model (~22MB):
  ```powershell
  python models/download_models.py
  ```
  Once downloaded, OpenCV DNN object detection activates automatically in the dashboard!

---

## 7. Running the Evaluation Benchmark via CLI

To run the scientific benchmark harness from the command line:
```powershell
python evaluate.py --frames 25 --json reports/evaluation_results.json --csv reports/evaluation_results.csv
```
This generates a full markdown performance report including recovery rates, false alarm rates, and average quality gains.

---

## 8. Stage 4: Optional AWS Cloud Deployment & S3 Storage

PhantomRoad runs **100% locally by default with zero network calls and zero cloud charges**. 

For enterprise fleet operations and compliance archival, **Stage 4** introduces an optional hybrid AWS deployment:
- **boto3 S3 Integration**: Automated archival of degraded video snippets, full-resolution frame snapshots, and verifiable audit reports (`audit_report.json`, `audit_report.csv`).
- **Cryptographic Audit Manifest**: Every synced run automatically generates an immutable `manifest.json` containing run metadata and SHA-256 integrity checksums.
- **Enterprise Security & Encryption**: Enforces Server-Side Encryption (`SSE-S3` / `AES256` or AWS KMS) and generates time-bounded presigned URLs ($TTL \le 3600\text{s}$) for secure sharing without exposing public S3 buckets.
- **IAM Role & STS AssumeRole Support**: Supports EC2 Instance Profiles, ECS Task Roles, and STS `assume_role` for credential-less cloud execution alongside standard environment variables.
- **100% Mocked Automated Testing**: Includes 22 automated test cases (`tests/test_stage4.py`) executing offline without network access or cloud charges.

### Enabling Optional AWS Cloud Sync
1. Copy `.env.example` to `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Enable `PHANTOMROAD_STORAGE_MODE=aws_s3` and specify your bucket:
   ```env
   PHANTOMROAD_STORAGE_MODE=aws_s3
   PHANTOMROAD_S3_BUCKET=your-bucket-name
   PHANTOMROAD_AWS_REGION=us-east-1
   PHANTOMROAD_S3_ENCRYPTION=AES256
   ```
3. Use the **☁️ Stage 4: Optional AWS Cloud Deployment & S3 Sync** tab in the dashboard to verify connectivity, dispatch jobs, and sync batch audit bundles.

### Stage 4 Architecture & Operations Guides
- [Cloud Processing Architecture](file:///c:/I-AIML/PROJECTS_2025-26/OPEN%20CV/PhantomRoad/docs/cloud_architecture.md): Hybrid edge-cloud data flow, compute tiers (Lambda, ECS Fargate), and Mermaid diagrams.
- [AWS Deployment Guide](file:///c:/I-AIML/PROJECTS_2025-26/OPEN%20CV/PhantomRoad/docs/aws_deployment_guide.md): Step-by-step setup, least-privilege IAM policies, and TLS bucket policies.
- [Cost & Security Checklist](file:///c:/I-AIML/PROJECTS_2025-26/OPEN%20CV/PhantomRoad/docs/cost_and_security_checklist.md): Financial estimation matrix ($0.00 local vs $0.35/mo), S3 lifecycle rules, and security guardrails.

---

## 9. Troubleshooting Common Issues

| Issue | Resolution |
| :--- | :--- |
| `streamlit : The term 'streamlit' is not recognized` | Run via python module: `python -m streamlit run app.py` |
| `No module named 'cv2'` | Run `python -m pip install opencv-python-headless>=4.8.0` (or `opencv-python>=4.8.0` locally) |
| Video file fails to open | Ensure video codec is supported (H.264 / MP4V / MPEG4). Use MP4 format for best compatibility. |
| Detector shows "Offline" | Run `python models/download_models.py` to download weights. |
| Port 8501 is already in use | Run `python -m streamlit run app.py --server.port 8502` |

---

## 10. License

Developed for the **OpenCV AI Competition 2026**. Distributed under the MIT License.
