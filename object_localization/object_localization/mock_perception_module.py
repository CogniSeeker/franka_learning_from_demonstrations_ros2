"""Minimal stand-in for the external Perception Module."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class MockPerceptionModel:
    """Return deterministic detections without inspecting the camera input."""

    def inference(self, bgr_image: Any, camera_info: Any) -> list[Any]:
        del bgr_image, camera_info
        return [
            SimpleNamespace(
                detection_id=1,
                layout_id="button_1",
                classified_class="button",
                yolo_class="button",
                state=SimpleNamespace(name="unpressed"),
                confidence=1.0,
                pose=SimpleNamespace(
                    position=(0.3, 0.0, 0.0),
                    orientation=(0.0, 0.0, 0.0, 1.0),
                ),
                pose_valid=True,
            ),
            SimpleNamespace(
                detection_id=2,
                layout_id="cup_1",
                classified_class="cup",
                yolo_class="cup",
                state=None,
                confidence=0.9,
                pose=SimpleNamespace(
                    position=(0.15, -0.1, 0.5),
                    orientation=(0.0, 0.0, 0.0, 1.0),
                ),
                pose_valid=True,
            ),
        ]


def load_model(weights_path: str) -> MockPerceptionModel:
    """Match the real module's loader signature; mock weights are unused."""
    del weights_path
    return MockPerceptionModel()
