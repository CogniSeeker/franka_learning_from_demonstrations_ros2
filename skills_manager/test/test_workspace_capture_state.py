from types import SimpleNamespace
import threading

import numpy as np
import pytest


pytest.importorskip("rclpy")
pytest.importorskip("cv_bridge")

from skills_manager.workspace_capture import CaptureRequest, WorkspaceCapture  # noqa: E402


def _message(stamp_ns: int):
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(
                sec=stamp_ns // 1_000_000_000,
                nanosec=stamp_ns % 1_000_000_000,
            )
        )
    )


class _Logger:
    def __init__(self):
        self.warnings = []
        self.errors = []
        self.infos = []

    def warning(self, message):
        self.warnings.append(message)

    def error(self, message):
        self.errors.append(message)

    def info(self, message):
        self.infos.append(message)


class _CaptureHarness:
    request_capture = WorkspaceCapture.request_capture
    request_early_stop = WorkspaceCapture.request_early_stop
    _collect_image_pairs_locked = WorkspaceCapture._collect_image_pairs_locked
    _capture_tick = WorkspaceCapture._capture_tick
    _fail_request = WorkspaceCapture._fail_request

    def __init__(self):
        self._lock = threading.Lock()
        self._capture_enabled = True
        self._stop_requested = False
        self._busy = False
        self._request = None
        self._camera = {}
        self.capture_button = "check"
        self.capture_timeout_s = 2.0
        self.sync_slop_ns = 10
        self.completed_steps = 0
        self.requested_steps = 3
        self.completion_reason = "running"
        self.end = False
        self.statuses = []
        self.logger = _Logger()

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=100))

    def get_logger(self):
        return self.logger

    def _set_status(self, state, step, color):
        self.statuses.append((state, step, color))


def test_second_button_press_is_ignored_while_capture_is_busy():
    capture = _CaptureHarness()

    capture.request_capture()
    first_request = capture._request
    capture.request_capture()

    assert capture._busy is True
    assert capture._request is first_request
    assert capture.completed_steps == 0
    assert capture.statuses[0][0:2] == ("BUTTON PRESSED", 1)
    assert len(capture.logger.warnings) == 1


def test_multi_camera_request_completes_only_after_every_pair_arrives():
    capture = _CaptureHarness()
    capture._request = CaptureRequest(button_stamp_ns=100, deadline_monotonic=999)
    capture._camera = {
        "camera3": SimpleNamespace(
            colors=[_message(110)], depths=[_message(112)]
        ),
        "camera4": SimpleNamespace(colors=[_message(111)], depths=[]),
    }

    capture._collect_image_pairs_locked()
    assert set(capture._request.images) == {"camera3"}

    capture._camera["camera4"].depths.append(_message(113))
    capture._collect_image_pairs_locked()
    assert set(capture._request.images) == {"camera3", "camera4"}


def test_timeout_rejects_capture_without_incrementing_step():
    capture = _CaptureHarness()
    capture._busy = True
    capture._request = CaptureRequest(button_stamp_ns=100, deadline_monotonic=-1)

    capture._capture_tick()

    assert capture._request is None
    assert capture._busy is False
    assert capture.completed_steps == 0
    assert capture.logger.errors


def test_circle_waits_for_inflight_request_then_stops_after_rejection():
    capture = _CaptureHarness()
    capture._busy = True
    capture._request = CaptureRequest(button_stamp_ns=100, deadline_monotonic=999)

    capture.request_early_stop()
    assert capture.end is False

    capture._fail_request("test rejection")
    assert capture.end is True
    assert capture.completion_reason == "circle_early_stop"


def test_circle_without_inflight_request_stops_immediately():
    capture = _CaptureHarness()

    capture.request_early_stop()

    assert capture.end is True
    assert capture.completion_reason == "circle_early_stop"


def test_raw_depth_conversion_preserves_uint16_values_and_rejects_float():
    source = np.array([[0, 1, 65535], [42, 1000, 4096]], dtype=np.uint16)
    capture = SimpleNamespace(
        bridge=SimpleNamespace(imgmsg_to_cv2=lambda message, desired_encoding: source)
    )
    message = SimpleNamespace(encoding="16UC1")

    converted = WorkspaceCapture._depth_array(capture, message)
    np.testing.assert_array_equal(converted, source)
    assert converted.dtype == np.uint16

    capture.bridge.imgmsg_to_cv2 = lambda message, desired_encoding: source.astype(
        np.float32
    )
    with pytest.raises(ValueError, match="uint16"):
        WorkspaceCapture._depth_array(capture, message)
