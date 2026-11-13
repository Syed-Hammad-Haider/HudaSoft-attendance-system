"""
Person detector built on Ultralytics YOLO.

Design decision: this module does ONLY detection (find people in a single
frame), not tracking (linking people across frames). That responsibility
lives in person_tracking/. Keeping them separate means:
  - swapping YOLOv8 -> YOLOv11 never touches tracking code
  - swapping ByteTrack -> another tracker never touches detection code

Default model: YOLOv8n ("nano"), the smallest/fastest YOLOv8 variant --
more than accurate enough for "is there a person in frame" at doorway
distances, and cheap enough to run live on a CPU.

RUNTIME ACCELERATION (OpenVINO) -- the key to a smooth feed on an Intel
laptop with no dedicated GPU:
This detector automatically prefers an OpenVINO version of the model if
one has been exported next to the weights (e.g. 'yolov8n_openvino_model/'
beside 'yolov8n.pt'). OpenVINO runs the EXACT SAME YOLO weights on Intel's
integrated GPU / NPU, which is typically several times faster than
PyTorch-on-CPU -- with NO change to what gets detected (it's the identical
model, only a faster runtime). To turn it on, in the project's Python
environment run, once:

    pip install openvino
    yolo export model=yolov8n.pt format=openvino half=True imgsz=384

(Match imgsz to DetectionConfig.inference_size.) Then restart -- this
class picks the exported model up on its own.

EVERYTHING DEGRADES SAFELY, so shipping this never risks a working setup:
  - No OpenVINO model exported, or the 'openvino' package isn't installed,
    or the exported model fails to load -> fall back to the PyTorch .pt
    model and behave exactly as before.
  - OpenVINO model loads but the Intel GPU can't run it -> fall back to the
    Intel CPU plugin, then to the .pt model.
None of this changes detection results or thresholds; it only changes
WHERE the same math runs.
"""

import logging
import os
from typing import List

import numpy as np
from ultralytics import YOLO

from config.settings import DetectionConfig
from person_detection.detection import Detection

logger = logging.getLogger(__name__)


def _openvino_dir_for(model_path: str) -> str:
    """Ultralytics' default OpenVINO export folder for a .pt model:
    'yolov8n.pt' -> 'yolov8n_openvino_model'."""
    base = model_path[:-3] if model_path.lower().endswith(".pt") else model_path
    return base + "_openvino_model"


class PersonDetector:
    def __init__(self, config: DetectionConfig):
        self._config = config
        self._model, self._device = self._load_model()

    def _load_model(self):
        """
        Load the fastest available runtime for the SAME model, newest-first,
        with safe fallbacks (see module docstring). Returns (model, device).
        """
        ov_dir = _openvino_dir_for(self._config.model_path)
        if os.path.isdir(ov_dir):
            for dev in ("intel:gpu", "intel:npu", "intel:cpu"):
                try:
                    model = YOLO(ov_dir, task="detect")
                    # Warm-up proves this device can actually run inference
                    # before we commit to it for the live loop.
                    model.predict(
                        source=np.zeros((64, 64, 3), dtype=np.uint8),
                        device=dev, imgsz=self._config.inference_size, verbose=False,
                    )
                    logger.info("Detection running via OpenVINO on '%s' (model: %s).", dev, ov_dir)
                    return model, dev
                except Exception:
                    logger.warning("OpenVINO device '%s' not usable; trying next option.", dev)
            logger.warning(
                "An OpenVINO model is present but no OpenVINO device worked "
                "(is the 'openvino' package installed?) -- using the PyTorch weights instead."
            )

        # PyTorch path (original behavior) + the CUDA/MPS -> CPU warm-up
        # fallback from before.
        device = self._config.device
        logger.info("Loading detection model '%s' on device '%s'...", self._config.model_path, device)
        model = YOLO(self._config.model_path)
        if device != "cpu":
            try:
                model.predict(
                    source=np.zeros((64, 64, 3), dtype=np.uint8),
                    device=device, imgsz=self._config.inference_size, verbose=False,
                )
                logger.info("Detection running on accelerated device '%s'.", device)
            except Exception:
                logger.exception("Device '%s' could not run YOLO -- falling back to CPU.", device)
                device = "cpu"
        return model, device

    def detect(self, frame) -> List[Detection]:
        """
        Run detection on a single BGR frame and return only 'person'
        detections above the configured confidence threshold.
        """
        results = self._model.predict(
            source=frame,
            classes=[self._config.person_class_id],
            conf=self._config.confidence_threshold,
            device=self._device,
            imgsz=self._config.inference_size,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
            detections.append(Detection(bbox=bbox, confidence=conf))

        return detections
