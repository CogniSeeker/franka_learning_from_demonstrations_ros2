"""Round-trip checks for object_localization.geometry.

The risk this file exists to catch: a sign, transpose or frame-convention error
in pixel_to_base would still put every object exactly on the table (z_plane is
forced), so the output looks plausible while being in the wrong place. A round
trip through base -> pixel -> base cannot be fooled that way.

Run: python3 -m pytest tests/test_geometry.py
  or python3 tests/test_geometry.py
"""

import numpy as np
import pytest
import tf_transformations as tft

from object_localization.geometry import (
    base_to_pixel,
    normalised_ray,
    pixel_to_base,
    yaw_in_base,
)

# Realsense-ish 1280x720 colour intrinsics; crop values in cfg/*/params.yaml go
# up to 1176 x 711, which is what fixes this resolution.
K = np.array([
    [900.0,   0.0, 640.0],
    [  0.0, 900.0, 360.0],
    [  0.0,   0.0,   1.0],
])
NO_DISTORTION = np.zeros(5)

# The real rig, read off object_localization/config/camera_transform.yaml and a
# template capture pose in cfg/cube/params.yaml.
CAMERA_IN_HAND = (
    tft.translation_matrix([0.142099, -0.00746304, -0.166843])
    @ tft.quaternion_matrix([0.09425513, -0.0874602, -0.69987041, 0.70260095])
)
HAND_IN_BASE = (
    tft.translation_matrix([0.3989376013739807, 8.768434537130787e-05, 0.4005559512224378])
    @ tft.quaternion_matrix([0.9999995224144351, 5.751080394252929e-05,
                             -0.0009468870446352185, -0.00023509218151233552])
)
T_BASE_CAM = HAND_IN_BASE @ CAMERA_IN_HAND


def test_camera_looks_down():
    """Premise for every other test: the optical axis descends toward z=0."""
    assert T_BASE_CAM[2, 3] > 0.4, "camera should be above the table"
    assert T_BASE_CAM[2, 2] < -0.9, "optical +z should point down in base"


@pytest.mark.parametrize("p_base", [
    [0.5414, 0.0075, 0.0],    # directly under the camera
    [0.45, -0.12, 0.0],
    [0.65, 0.18, 0.0],
    [0.30, 0.00, 0.0],
    [0.75, -0.25, 0.0],
])
def test_roundtrip_on_the_plane(p_base):
    """base -> pixel -> base must return the original point."""
    p_base = np.array(p_base)
    uv = base_to_pixel(K, T_BASE_CAM, p_base)
    recovered = pixel_to_base(K, NO_DISTORTION, T_BASE_CAM, uv, z_plane=0.0)
    assert recovered is not None
    assert np.allclose(recovered, p_base, atol=1e-9), f"{recovered} != {p_base}"


def test_roundtrip_at_frame_corners():
    """Wide crops (coffeemachine spans x 157..1176) live near the edges."""
    for uv in [(157.0, 180.0), (1176.0, 708.0), (157.0, 708.0), (1176.0, 180.0)]:
        p = pixel_to_base(K, NO_DISTORTION, T_BASE_CAM, uv, z_plane=0.0)
        assert p is not None
        assert abs(p[2]) < 1e-12, "must land exactly on the requested plane"
        assert np.allclose(base_to_pixel(K, T_BASE_CAM, p), uv, atol=1e-6)


def test_lands_on_requested_plane():
    for z_plane in (0.0, 0.04, 0.25):
        p = pixel_to_base(K, NO_DISTORTION, T_BASE_CAM, (640.0, 360.0), z_plane)
        assert p is not None
        assert abs(p[2] - z_plane) < 1e-12


@pytest.mark.parametrize("uv", [
    (640.0, 360.0),     # principal point -> ray along the optical axis, 14.9 deg tilt
    (631.2, 599.1),     # ray straight down under the camera, ~0 deg tilt
    (1176.0, 180.0),    # frame corner, largest tilt
])
def test_camera_height_error_costs_lateral_offset(uv):
    """A camera-height calibration error shifts x,y by height_error * tan(tilt).

    This is the cost of getting z_plane or the extrinsics' z wrong, and it is
    NOT uniform across the frame: a ray pointing straight down is immune, while
    a tilted ray slides along the plane. The 79 mm discrepancy between
    params.yaml depth and the camera-to-table range lands here if it turns out
    to live in the extrinsics.
    """
    height_error = 0.070

    truth = pixel_to_base(K, NO_DISTORTION, T_BASE_CAM, uv, z_plane=0.0)
    shifted = T_BASE_CAM.copy()
    shifted[2, 3] += height_error
    biased = pixel_to_base(K, NO_DISTORTION, shifted, uv, z_plane=0.0)
    assert truth is not None and biased is not None

    ray = T_BASE_CAM[0:3, 0:3] @ normalised_ray(K, NO_DISTORTION, uv)
    ray /= np.linalg.norm(ray)
    tilt = np.arccos(abs(ray[2]))

    lateral = np.linalg.norm(biased[:2] - truth[:2])
    assert lateral == pytest.approx(height_error * np.tan(tilt), abs=1e-9)


def test_optical_axis_sensitivity_is_19mm():
    """Pin the headline number quoted when z_plane=0 was chosen."""
    uv = (K[0, 2], K[1, 2])
    truth = pixel_to_base(K, NO_DISTORTION, T_BASE_CAM, uv, z_plane=0.0)
    shifted = T_BASE_CAM.copy()
    shifted[2, 3] += 0.070
    biased = pixel_to_base(K, NO_DISTORTION, shifted, uv, z_plane=0.0)

    lateral = np.linalg.norm(biased[:2] - truth[:2])
    assert 0.017 < lateral < 0.020, f"expected ~18.6 mm, got {1000 * lateral:.1f} mm"


def test_distortion_is_applied():
    """Non-zero d must change the answer, and zeros must not."""
    uv = (1100.0, 650.0)  # off-centre, where distortion actually bites
    plain = pixel_to_base(K, NO_DISTORTION, T_BASE_CAM, uv, z_plane=0.0)
    none_given = pixel_to_base(K, None, T_BASE_CAM, uv, z_plane=0.0)
    assert np.allclose(plain, none_given, atol=1e-12), "zero d must equal no d"

    barrel = pixel_to_base(K, np.array([-0.35, 0.12, 0.0, 0.0, 0.0]),
                           T_BASE_CAM, uv, z_plane=0.0)
    assert not np.allclose(plain, barrel, atol=1e-4), "d was ignored"


def test_unreachable_plane_returns_none():
    # plane above the camera: the downward ray can never reach it
    assert pixel_to_base(K, NO_DISTORTION, T_BASE_CAM, (640.0, 360.0), z_plane=5.0) is None

    # camera rotated to look horizontally: ray is parallel to the plane
    sideways = T_BASE_CAM.copy()
    sideways[0:3, 0:3] = tft.euler_matrix(0.0, np.pi / 2, 0.0)[0:3, 0:3]
    assert pixel_to_base(K, NO_DISTORTION, sideways, (640.0, 360.0), z_plane=0.0) is None


def test_affine_recovers_scale_and_yaw():
    """The two quantities measure_object() pulls out of estimateAffinePartial2D."""
    angle, scale = np.deg2rad(17.0), 1.24
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    A = np.hstack([scale * rotation, np.array([[31.0], [-12.0]])])

    assert np.isclose(np.sqrt(abs(np.linalg.det(A[0:2, 0:2]))), scale, atol=1e-12)
    assert np.isclose(np.arctan2(A[1, 0], A[0, 0]), angle, atol=1e-12)


def test_yaw_sign_flips_when_camera_faces_down():
    """Optical axis pointing down inverts an image rotation about base z."""
    assert yaw_in_base(T_BASE_CAM, 0.4) == pytest.approx(-0.4)

    upward = T_BASE_CAM.copy()
    upward[0:3, 0:3] = np.eye(3)  # optical +z along base +z
    assert yaw_in_base(upward, 0.4) == pytest.approx(0.4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
