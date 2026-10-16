# Stage 3: Downstream Road-Scene Object Detector & Reliability Integration

## 1. Overview & Architecture

In autonomous driving and advanced driver-assistance systems (ADAS), downstream perception models (e.g. object detectors) assume that input camera frames are visually reliable. When adverse environmental conditions (motion blur, low light, atmospheric fog, or lens obstructions) degrade the optical evidence, traditional detectors either:
1. **Miss obstacles entirely (False Negatives)** due to feature attenuation.
2. **Hallucinate boundaries (False Positives)** due to sensor noise and specular glare.
3. **Report falsely confident detections** without communicating that the visual evidence was compromised.

**PhantomRoad** introduces an **Adaptive Reliability Wrapper** around downstream road-scene detectors. Rather than modifying or retraining the detector, our agent assesses optical quality independently and **decorates downstream detections with explicit reliability status**:

| Reliability Verdict | Detection Status | Downstream Planner Action |
|---|---|---|
| `RELIABLE` | `[TRUSTED]` | Bounding box is verified against nominal optical quality; safe for trajectory planning. |
| `DEGRADED_CORRECTABLE` / `DEGRADED_UNCERTAIN` | `[UNVERIFIED_DEGRADED]` | Optical evidence is degraded (blur, low light, fog, low contrast); downstream planner must increase safety margins and assume undetected obstacles may exist. |
| `CRITICAL_OBSTRUCTION` / `HUMAN_REVIEW_REQUIRED` | `[SUPPRESSED_OBSTRUCTED]` | Sensor lens is physically obstructed or unresolvable; perception stream is untrusted, and immediate human takeover or sensor cleaning is required. |

---

## 2. Model Selection: MobileNet-SSD (Caffe / OpenCV DNN)

### Why MobileNet-SSD?
- **Zero Heavy Dependencies**: Runs directly via OpenCV's built-in deep neural network module (`cv2.dnn.readNetFromCaffe`), requiring no PyTorch, TensorFlow, CUDA, or paid cloud APIs.
- **Lightweight (~22 MB)**: Small footprint suitable for edge ADAS hardware and fast local execution.
- **Real-Time CPU Inference**: Operates at 25–35 FPS on standard laptop processors.
- **Relevant Road Classes**: Trained on Pascal VOC with detection for **cars, buses, motorbikes, bicycles, persons, and trains**.

---

## 3. How to Obtain and Configure the Model

### Step 1: Automated Download
Run the local downloader script from the repository root:
```powershell
python models/download_models.py
```
This utility:
1. Downloads `MobileNetSSD_deploy.prototxt` (architecture definition, ~44 KB).
2. Downloads `MobileNetSSD_deploy.caffemodel` (pretrained weights, ~22.1 MB).
3. Verifies file integrity and file size.

### Step 2: Manual Acquisition (Alternative)
If you prefer to download files manually, place both files into the `models/` directory:
- Prototxt: `models/MobileNetSSD_deploy.prototxt`
- Caffe Model: `models/MobileNetSSD_deploy.caffemodel`

### Step 3: Python Configuration
```python
from src.detector.opencv_dnn import OpenCVDNNDetector
from src.detector.reliability_wrapper import ReliableRoadSceneAnalyzer

# Initialize detector (automatically detects files in models/)
detector = OpenCVDNNDetector(models_dir="models")

# Wrap with PhantomRoad reliability engine
analyzer = ReliableRoadSceneAnalyzer(detector)

# Check availability
if detector.is_available():
    print("Detector online and ready for local inference.")
else:
    print(detector.get_setup_message())
```

---

## 4. Offline Fallback Guarantee & Honesty Policy

> [!IMPORTANT]
> **PhantomRoad NEVER invents, hallucinates, or fabricates detections.**
> If model weights are missing or unreadable:
> - `detector.is_available()` returns `False`.
> - `detect()` returns an empty detection list (`detections = []`).
> - An actionable setup notice is displayed in the dashboard and audit logs.
> - **All other features remain 100% operational**: Video playback, frame extraction, Laplacian blur measurement, illumination histograms, decision engine analysis, and adaptive enhancements continue functioning normally.

---

## 5. Documented Limitations

1. **Class Scope**:
   MobileNet-SSD detects 6 road classes (`car`, `bus`, `motorbike`, `bicycle`, `person`, `train`). It does not detect specialized road assets such as traffic cones, lane dividers, or potholes (these are monitored via our Stage 1 spatial metrics).
2. **Fixed Input Resolution ($300 \times 300$)**:
   The network resizes input frames to a $300 \times 300$ blob. Consequently, tiny or distant vehicles ($< 25\text{px}$ in 1080p footage) have diminished recall.
3. **Degradation Sensitivity**:
   In dense atmospheric fog ($T < 0.30$) or heavy motion blur ($\text{Var}(\nabla^2 I) < 30$), convolution feature maps are attenuated, causing false negatives. PhantomRoad's reliability wrapper is specifically engineered to mitigate this hazard by warning downstream planners that absence of detections does not guarantee an empty road.
4. **Single-Frame Temporal Decoupling**:
   Inference is currently evaluated frame-by-frame. Downstream production deployments should couple this output with a multi-object Kalman filter or SORT tracker.
