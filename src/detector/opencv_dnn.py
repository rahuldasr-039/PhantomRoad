"""
OpenCV DNN implementation of road-scene object detector.
Supports MobileNet-SSD (Caffe / ONNX) with honest setup reporting when weights are absent.
"""

from typing import List, Tuple, Optional
import os
import cv2
import numpy as np

from .base import BaseRoadSceneDetector, DetectionBox, DetectionResult

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

ROAD_CLASSES = {"car", "bus", "motorbike", "bicycle", "person", "train"}


class OpenCVDNNDetector(BaseRoadSceneDetector):
    """
    Road-scene detector utilizing OpenCV's native deep neural network module (cv2.dnn).
    """

    def __init__(
        self,
        prototxt_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        models_dir: str = "models"
    ):
        base_dir = os.path.abspath(models_dir)
        self.prototxt_path = prototxt_path or os.path.join(base_dir, "MobileNetSSD_deploy.prototxt")
        self.weights_path = weights_path or os.path.join(base_dir, "MobileNetSSD_deploy.caffemodel")
        self.model_name = "MobileNet-SSD (OpenCV DNN)"
        self.net = None

        has_files = bool(
            os.path.exists(self.prototxt_path) and
            os.path.exists(self.weights_path) and
            os.path.getsize(self.weights_path) > 1000000  # at least ~1MB
        )

        if has_files:
            try:
                if hasattr(cv2.dnn, "readNetFromCaffe"):
                    self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.weights_path)
                elif hasattr(cv2.dnn, "readNet"):
                    self.net = cv2.dnn.readNet(self.weights_path, self.prototxt_path)
                if self.net is not None:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            except Exception:
                self.net = None

    def is_available(self) -> bool:
        """
        Verifies that actual model architecture and weight files exist on disk and net loaded successfully.
        """
        return bool(
            os.path.exists(self.prototxt_path) and
            os.path.exists(self.weights_path) and
            os.path.getsize(self.weights_path) > 1000000 and
            self.net is not None
        )

    def get_setup_message(self) -> str:
        """
        Provides honest, actionable instructions for acquiring model weights.
        """
        return (
            "**Road-Scene Detector Offline**: MobileNet-SSD weight files were not detected in `models/`.\n\n"
            "To enable live road object detection without external APIs:\n"
            "1. Run the local download utility in your terminal:\n"
            "   ```powershell\n"
            "   python models/download_models.py\n"
            "   ```\n"
            "2. Or manually place `MobileNetSSD_deploy.prototxt` and `MobileNetSSD_deploy.caffemodel` into the `models/` directory.\n\n"
            "_Note: PhantomRoad never simulates or fabricates detections when models are offline._"
        )

    def detect(self, image_bgr: np.ndarray, conf_threshold: float = 0.40) -> DetectionResult:
        """
        Executes inference if model weights are present. Otherwise returns clean setup status.
        """
        if not self.is_available() or self.net is None:
            return DetectionResult(
                detections=[],
                model_name=self.model_name,
                model_available=False,
                setup_message=self.get_setup_message(),
                frame_reliability_state="UNKNOWN"
            )

        h, w = image_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image_bgr, (300, 300)),
            scalefactor=0.007843,
            size=(300, 300),
            mean=(127.5, 127.5, 127.5),
            swapRB=False
        )

        self.net.setInput(blob)
        raw_detections = self.net.forward()

        boxes: List[DetectionBox] = []
        for i in range(raw_detections.shape[2]):
            confidence = float(raw_detections[0, 0, i, 2])
            if confidence >= conf_threshold:
                class_id = int(raw_detections[0, 0, i, 1])
                if class_id < len(VOC_CLASSES):
                    label = VOC_CLASSES[class_id]
                    if label in ROAD_CLASSES:
                        x1 = max(0, int(raw_detections[0, 0, i, 3] * w))
                        y1 = max(0, int(raw_detections[0, 0, i, 4] * h))
                        x2 = min(w, int(raw_detections[0, 0, i, 5] * w))
                        y2 = min(h, int(raw_detections[0, 0, i, 6] * h))

                        boxes.append(DetectionBox(
                            label=label,
                            confidence=confidence,
                            box_xyxy=(x1, y1, x2, y2),
                            is_reliable=True,
                            reliability_status="TRUSTED"
                        ))

        return DetectionResult(
            detections=boxes,
            model_name=self.model_name,
            model_available=True,
            setup_message=None,
            frame_reliability_state="RELIABLE"
        )
