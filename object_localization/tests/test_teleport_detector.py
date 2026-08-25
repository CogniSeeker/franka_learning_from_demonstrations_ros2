"""Timing checks for the teleport detector camera warm-up."""

from object_localization import teleport_detector


def test_warmup_starts_when_first_image_arrives(monkeypatch):
    node = object.__new__(teleport_detector.TeleportDetectionService)
    node._camera_warmup_until = None
    node._camera_info = None
    logger = type("Logger", (), {"info": lambda self, message: None})()
    monkeypatch.setattr(node, "get_logger", lambda: logger)

    now = 10.0
    monkeypatch.setattr(teleport_detector.time, "monotonic", lambda: now)

    node._camera_info_callback(object())
    assert node._camera_info is None

    node._image_callback(object())
    assert node._camera_warmup_until == 13.0

    now = 12.0
    node._camera_info_callback(object())
    node._image_callback(object())
    assert node._camera_info is None
    assert node._camera_warmup_until == 13.0
