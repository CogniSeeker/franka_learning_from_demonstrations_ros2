"""Pinhole geometry for turning a matched image pixel into a base-frame point.

Pure math: no ROS, no node, no robot. Everything here is testable offline,
which matters because a sign or transpose error in this file would produce
positions that look plausible (objects land on the table by construction) while
being in the wrong place.

Frame conventions used throughout:

* ``K`` is the 3x3 intrinsic matrix (``camera_info.k`` reshaped).
* ``D`` is the distortion vector (``camera_info.d``); may be None or all zeros.
* ``T_base_cam`` is the 4x4 homogeneous transform of the *camera optical frame*
  expressed in the base frame, i.e. it maps camera coordinates to base
  coordinates. Its translation column is the camera origin in base, and its
  third rotation column is the optical +z axis in base.
* The camera optical frame is the usual OpenCV one: +x right, +y down, +z into
  the scene along the optical axis.
"""

import cv2
import numpy as np

# A ray needs to actually descend toward the plane. Anything flatter than this
# either misses entirely or hits so obliquely that the result is meaningless.
MIN_RAY_COMPONENT = 1e-3


def normalised_ray(K, D, uv):
    """Un-project a pixel to a direction in the camera optical frame.

    Returns a non-normalised vector whose z component is exactly 1.0, so the
    plane solve below can treat the parameter as a straight ratio.
    """
    pixel = np.array([[[float(uv[0]), float(uv[1])]]], dtype=np.float64)
    coefficients = None if D is None else np.asarray(D, dtype=np.float64).reshape(1, -1)
    if coefficients is not None and not np.any(coefficients):
        coefficients = None  # all-zero distortion: skip the call entirely
    undistorted = cv2.undistortPoints(pixel, np.asarray(K, dtype=np.float64), coefficients)
    return np.array([undistorted[0, 0, 0], undistorted[0, 0, 1], 1.0])


def pixel_to_base(K, D, T_base_cam, uv, z_plane):
    """Intersect the ray through pixel `uv` with the horizontal plane z=z_plane.

    Returns the base-frame point as a length-3 array, or None when the ray
    cannot reach the plane (pointing away from it, or the intersection lands
    behind the camera).
    """
    T_base_cam = np.asarray(T_base_cam, dtype=np.float64)
    origin = T_base_cam[0:3, 3]
    ray = T_base_cam[0:3, 0:3] @ normalised_ray(K, D, uv)

    gap = z_plane - origin[2]
    # Ray and plane must be on speaking terms: the z component has to point
    # from the camera toward the plane, not parallel to it or away from it.
    if abs(ray[2]) < MIN_RAY_COMPONENT or (gap / ray[2]) <= 0.0:
        return None

    return origin + (gap / ray[2]) * ray


def base_to_pixel(K, T_base_cam, p_base):
    """Project a base-frame point to a pixel. Inverse of pixel_to_base.

    Ignores distortion on purpose -- this exists to generate test inputs and to
    let callers sanity-check a round trip, not to model the lens.
    """
    T_base_cam = np.asarray(T_base_cam, dtype=np.float64)
    T_cam_base = np.linalg.inv(T_base_cam)
    point_cam = T_cam_base @ np.append(np.asarray(p_base, dtype=np.float64), 1.0)
    if point_cam[2] <= 0.0:
        raise ValueError("point is behind the camera")
    K = np.asarray(K, dtype=np.float64)
    return (
        K[0, 0] * point_cam[0] / point_cam[2] + K[0, 2],
        K[1, 1] * point_cam[1] / point_cam[2] + K[1, 2],
    )


def yaw_in_base(T_base_cam, yaw_image):
    """Map an image-plane rotation onto a rotation about the base z axis.

    `yaw_image` is the rotation the SIFT affine measured, in image coordinates
    (+x right, +y down). That is a rotation about the camera optical +z axis.
    Expressing it about base z means accounting for both the camera's in-plane
    rotation relative to base and the sign flip when the optical axis points
    down rather than up.

    IMPORTANT: the result is relative to the *template capture* orientation,
    which is recorded nowhere. Zero means "rotated the same as when the
    template image was taken", not "aligned with the base frame".
    """
    T_base_cam = np.asarray(T_base_cam, dtype=np.float64)
    # Sign of the optical axis' base-z component says whether a positive
    # rotation about the optical axis reads as positive about base z.
    sense = -1.0 if T_base_cam[2, 2] < 0.0 else 1.0
    return sense * float(yaw_image)
