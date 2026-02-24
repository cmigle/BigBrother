"""
phone_detector.py — YOLOv8-based phone detection module for BigBrother.

Uses Ultralytics YOLOv8 to detect cell phones (COCO class 67) in webcam
frames. Tracks detection persistence so we only trigger scare events when
a phone has been visible for a configurable duration.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# COCO class ID for cell phone
CELL_PHONE_CLASS_ID = 67

# Maximum camera indices to probe when listing available cameras
_CAMERA_SCAN_LIMIT = 10


@dataclass
class PhoneDetectionResult:
    """Result from a single frame of phone detection."""
    detected: bool = False
    confidence: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    annotated_frame: Optional[np.ndarray] = None
    detection_duration: float = 0.0


class PhoneDetector:
    """
    Real-time phone detector using YOLOv8.

    Captures frames from the webcam, runs YOLOv8 inference filtered to
    COCO class 67 (cell phone), and tracks how long a phone has been
    continuously visible.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.5,
        camera_index: int = 0,
        roi: Optional[Tuple[int, int, int, int]] = None,
        calibration: Optional[Dict] = None,
    ):
        # Lazy-import ultralytics so we fail fast with a clear message
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for phone detection. "
                "Install it with: pip install ultralytics"
            )

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.camera_index = camera_index
        self.roi = roi  # (x1, y1, x2, y2) or None
        # calibration keys: width, height, fps, brightness, contrast, exposure
        self.calibration: Dict = calibration or {}

        self._detection_start: Optional[float] = None
        self._phone_detected: bool = False
        self._latest_result: Optional[PhoneDetectionResult] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._cap: Optional[cv2.VideoCapture] = None

        logger.info(
            "PhoneDetector initialized — model=%s  conf=%.2f  cam=%d",
            model_path, confidence, camera_index,
        )

    # ------------------------------------------------------------------
    # Camera enumeration
    # ------------------------------------------------------------------

    @staticmethod
    def list_cameras(max_index: int = _CAMERA_SCAN_LIMIT) -> List[Dict]:
        """
        Probe camera indices 0..max_index-1 and return a list of dicts
        describing each available camera::

            [{"index": 0, "width": 1280, "height": 720, "fps": 30.0}, ...]
        """
        found: List[Dict] = []
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    info = {
                        "index": idx,
                        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        "fps": cap.get(cv2.CAP_PROP_FPS),
                    }
                    found.append(info)
                    logger.info(
                        "Camera %d found: %dx%d @ %.0f fps",
                        idx, info["width"], info["height"], info["fps"],
                    )
                cap.release()
        if not found:
            logger.warning("No cameras detected during scan.")
        return found

    # ------------------------------------------------------------------
    # Camera management
    # ------------------------------------------------------------------

    def open_camera(self) -> bool:
        """Open the webcam and apply calibration settings. Returns True on success."""
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            logger.error("Could not open camera index %d", self.camera_index)
            return False

        # Apply calibration / capture properties
        cal = self.calibration
        if cal.get("width") and cal.get("height"):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cal["width"])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cal["height"])
        if cal.get("fps"):
            self._cap.set(cv2.CAP_PROP_FPS, cal["fps"])
        if cal.get("brightness") is not None:
            self._cap.set(cv2.CAP_PROP_BRIGHTNESS, cal["brightness"])
        if cal.get("contrast") is not None:
            self._cap.set(cv2.CAP_PROP_CONTRAST, cal["contrast"])
        if cal.get("exposure") is not None:
            # Disable auto-exposure first, then set manual value
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual
            self._cap.set(cv2.CAP_PROP_EXPOSURE, cal["exposure"])

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "Camera %d opened — %dx%d @ %.0f fps",
            self.camera_index, actual_w, actual_h, actual_fps,
        )
        return True

    def release_camera(self):
        """Release the webcam resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released")

    def grab_frame(self) -> Optional[np.ndarray]:
        """Read a single frame from the webcam. Returns None on failure."""
        if self._cap is None or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> PhoneDetectionResult:
        """
        Run YOLOv8 inference on *frame*.

        Filters detections to COCO class 67 (cell phone).  If an ROI is
        configured, only detections whose centre falls inside the ROI are
        kept.

        Returns a PhoneDetectionResult with confidence, bounding box, and
        annotated frame.
        """
        result = PhoneDetectionResult()

        # Run inference — only look for cell phones
        preds = self.model.predict(
            frame,
            classes=[CELL_PHONE_CLASS_ID],
            conf=self.confidence,
            verbose=False,
        )

        annotated = frame.copy()
        best_conf = 0.0
        best_box = None

        for det in preds:
            boxes = det.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # ROI filter — is the centre of the box inside the ROI?
                if self.roi is not None:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    rx1, ry1, rx2, ry2 = self.roi
                    if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                        continue

                if conf > best_conf:
                    best_conf = conf
                    best_box = (x1, y1, x2, y2)

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"Phone {conf:.2f}"
                cv2.putText(
                    annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                )

        # Draw ROI if set
        if self.roi is not None:
            rx1, ry1, rx2, ry2 = self.roi
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)

        # Update detection state
        now = time.time()
        detected = best_conf > 0

        if detected:
            if not self._phone_detected:
                self._detection_start = now
            self._phone_detected = True
        else:
            self._phone_detected = False
            self._detection_start = None

        duration = 0.0
        if self._phone_detected and self._detection_start is not None:
            duration = now - self._detection_start

        result.detected = detected
        result.confidence = best_conf
        result.bbox = best_box
        result.annotated_frame = annotated
        result.detection_duration = duration

        self._latest_result = result
        self._latest_frame = annotated

        return result

    def is_phone_persistent(self, min_duration: float = 2.0) -> bool:
        """
        Returns True if a phone has been continuously detected for at
        least *min_duration* seconds.
        """
        if self._latest_result is None:
            return False
        return (
            self._latest_result.detected
            and self._latest_result.detection_duration >= min_duration
        )

    def get_annotated_frame(self) -> Optional[np.ndarray]:
        """Return the latest frame with YOLO bounding boxes drawn."""
        return self._latest_frame

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def update_settings(
        self,
        confidence: Optional[float] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ):
        """Hot-update detection settings without recreating the detector."""
        if confidence is not None:
            self.confidence = confidence
            logger.info("Confidence threshold updated to %.2f", confidence)
        if roi is not None:
            self.roi = roi
            logger.info("ROI updated to %s", roi)
