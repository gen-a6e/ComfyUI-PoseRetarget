"""3D bone-length retargeting for SAM 3D Body's MHR70 output.

This module deliberately has no dependency on SAM 3D Body or torch.  It
consumes the ``SAM3D_OUTPUT`` dictionaries emitted by ComfyUI-SAM3DBody and
keeps all geometry operations in numpy so the integration remains light.
"""

from __future__ import annotations

import numpy as np


EPS = 1e-8
MHR70_COUNT = 70
MHR_KEYPOINT_COUNT = 308

# MHR70 indices used by SAM 3D Body.
NOSE = 0
LEFT_EYE, RIGHT_EYE = 1, 2
LEFT_EAR, RIGHT_EAR = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_HIP, RIGHT_HIP = 9, 10
LEFT_KNEE, RIGHT_KNEE = 11, 12
LEFT_ANKLE, RIGHT_ANKLE = 13, 14
LEFT_HEEL, RIGHT_HEEL = 17, 20
RIGHT_WRIST, LEFT_WRIST = 41, 62
NECK = 69


# The synthetic root is the midpoint of the hips.  Parents always precede
# children, including the proximal-to-distal order of every finger.
BODY_EDGES = (
    (NECK, None, "torso"),
    (LEFT_HIP, None, "body"),
    (RIGHT_HIP, None, "body"),
    (LEFT_SHOULDER, NECK, "body"),
    (RIGHT_SHOULDER, NECK, "body"),
    (LEFT_ELBOW, LEFT_SHOULDER, "arm"),
    (LEFT_WRIST, LEFT_ELBOW, "arm"),
    (RIGHT_ELBOW, RIGHT_SHOULDER, "arm"),
    (RIGHT_WRIST, RIGHT_ELBOW, "arm"),
    (LEFT_KNEE, LEFT_HIP, "leg"),
    (LEFT_ANKLE, LEFT_KNEE, "leg"),
    (RIGHT_KNEE, RIGHT_HIP, "leg"),
    (RIGHT_ANKLE, RIGHT_KNEE, "leg"),
    (15, LEFT_ANKLE, "foot"),
    (16, LEFT_ANKLE, "foot"),
    (17, LEFT_ANKLE, "foot"),
    (18, RIGHT_ANKLE, "foot"),
    (19, RIGHT_ANKLE, "foot"),
    (20, RIGHT_ANKLE, "foot"),
    (NOSE, NECK, "head"),
    (LEFT_EYE, NOSE, "head"),
    (RIGHT_EYE, NOSE, "head"),
    (LEFT_EAR, LEFT_EYE, "head"),
    (RIGHT_EAR, RIGHT_EYE, "head"),
)


def _finger_edges(wrist, chains):
    edges = []
    for chain in chains:
        parent = wrist
        for child in chain:
            edges.append((child, parent, "hand"))
            parent = child
    return tuple(edges)


RIGHT_HAND_CHAINS = (
    (24, 23, 22, 21),
    (28, 27, 26, 25),
    (32, 31, 30, 29),
    (36, 35, 34, 33),
    (40, 39, 38, 37),
)
LEFT_HAND_CHAINS = (
    (45, 44, 43, 42),
    (49, 48, 47, 46),
    (53, 52, 51, 50),
    (57, 56, 55, 54),
    (61, 60, 59, 58),
)
HAND_EDGES = (
    _finger_edges(RIGHT_WRIST, RIGHT_HAND_CHAINS)
    + _finger_edges(LEFT_WRIST, LEFT_HAND_CHAINS)
)
RETARGET_EDGES = BODY_EDGES + HAND_EDGES


# Corresponding left/right children.  Each child uniquely identifies its
# incoming edge, so this also mirrors the matching bone.
MIRROR_CHILD = {
    LEFT_HIP: RIGHT_HIP,
    LEFT_SHOULDER: RIGHT_SHOULDER,
    LEFT_ELBOW: RIGHT_ELBOW,
    LEFT_WRIST: RIGHT_WRIST,
    LEFT_KNEE: RIGHT_KNEE,
    LEFT_ANKLE: RIGHT_ANKLE,
    15: 18,
    16: 19,
    17: 20,
    LEFT_EYE: RIGHT_EYE,
    LEFT_EAR: RIGHT_EAR,
}
for left_chain, right_chain in zip(LEFT_HAND_CHAINS, RIGHT_HAND_CHAINS):
    MIRROR_CHILD.update(zip(left_chain, right_chain))
MIRROR_CHILD.update({right: left for left, right in tuple(MIRROR_CHILD.items())})


COCO18_FROM_MHR70 = (
    NOSE,
    NECK,
    RIGHT_SHOULDER,
    RIGHT_ELBOW,
    RIGHT_WRIST,
    LEFT_SHOULDER,
    LEFT_ELBOW,
    LEFT_WRIST,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_ANKLE,
    RIGHT_EYE,
    LEFT_EYE,
    RIGHT_EAR,
    LEFT_EAR,
)

RIGHT_HAND_FROM_MHR70 = (
    RIGHT_WRIST,
    24, 23, 22, 21,
    28, 27, 26, 25,
    32, 31, 30, 29,
    36, 35, 34, 33,
    40, 39, 38, 37,
)
LEFT_HAND_FROM_MHR70 = (
    LEFT_WRIST,
    45, 44, 43, 42,
    49, 48, 47, 46,
    53, 52, 51, 50,
    57, 56, 55, 54,
    61, 60, 59, 58,
)


def as_numpy(value, name):
    """Convert numpy/torch-like values without importing torch."""
    if value is None:
        raise ValueError(f"SAM3D output is missing {name}")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SAM3D {name} is not a numeric array") from exc


def extract_mhr70(output):
    """Read the 70 camera-space joints from a SAM3D_OUTPUT dictionary."""
    if not isinstance(output, dict):
        raise ValueError("SAM3D input must be a SAM3D_OUTPUT dictionary")
    value = output.get("joints")
    if value is None:
        raw = output.get("raw_output") or {}
        value = raw.get("pred_keypoints_3d")
    points = as_numpy(value, "joints")
    while points.ndim > 2 and points.shape[0] == 1:
        points = points[0]
    if points.ndim != 2 or points.shape[0] < MHR70_COUNT or points.shape[1] < 3:
        raise ValueError(
            f"SAM3D joints must have shape (70, 3); received {points.shape}")
    points = points[:MHR70_COUNT, :3].copy()
    if not np.all(np.isfinite(points)):
        raise ValueError("SAM3D joints contain NaN or infinite values")
    return points


def extract_head_top(output, mhr70=None):
    """Return the uppermost dense MHR head keypoint and its full-set index."""
    if not isinstance(output, dict):
        raise ValueError("SAM3D input must be a SAM3D_OUTPUT dictionary")
    value = output.get("keypoints_3d_full")
    if value is None:
        raw = output.get("raw_output") or {}
        value = raw.get("pred_keypoints_3d_full")
    if value is None:
        raise ValueError(
            "head_to_heel requires full MHR keypoints from the updated "
            "ComfyUI-SAM3DBody; rerun SAM 3D Body: Process Image"
        )

    points = as_numpy(value, "full MHR keypoints")
    while points.ndim > 2 and points.shape[0] == 1:
        points = points[0]
    if (
        points.ndim != 2
        or points.shape[0] < MHR_KEYPOINT_COUNT
        or points.shape[1] < 3
    ):
        raise ValueError(
            "SAM3D full MHR keypoints must have shape (308, 3); "
            f"received {points.shape}"
        )
    points = points[:MHR_KEYPOINT_COUNT, :3]

    body = extract_mhr70(output) if mhr70 is None else np.asarray(
        mhr70, dtype=np.float64
    )
    if body.shape != (MHR70_COUNT, 3):
        raise ValueError("mhr70 must have shape (70, 3)")

    # Points 70..307 are dense head/face landmarks. Select the landmark
    # furthest along the neck-to-head axis, so raised hands can never be
    # mistaken for the crown and a tilted head remains supported.
    head_center = np.mean(
        body[[NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR]], axis=0
    )
    head_axis = head_center - body[NECK]
    axis_length = np.linalg.norm(head_axis)
    if not np.isfinite(axis_length) or axis_length <= EPS:
        raise ValueError("SAM3D skeleton has no usable neck-to-head direction")
    head_axis /= axis_length

    dense = points[MHR70_COUNT:]
    valid = np.all(np.isfinite(dense), axis=1)
    dense = dense[valid]
    dense_indices = np.arange(MHR70_COUNT, MHR_KEYPOINT_COUNT)[valid]
    if not dense.size or np.max(np.ptp(dense, axis=0)) <= EPS:
        raise ValueError("SAM3D full MHR head keypoints are empty")

    selected = int(np.argmax((dense - body[NECK]) @ head_axis))
    return dense[selected].copy(), int(dense_indices[selected])


def hip_center(points):
    return (points[LEFT_HIP] + points[RIGHT_HIP]) * 0.5


def shoulder_center(points):
    return (points[LEFT_SHOULDER] + points[RIGHT_SHOULDER]) * 0.5


def _edge_delta(points, child, parent):
    origin = hip_center(points) if parent is None else points[parent]
    return points[child] - origin


def body_unit(points, mode="torso", head_top=None):
    """Return a pose-resistant 3D size unit for proportion transfer."""
    torso = np.linalg.norm(points[NECK] - hip_center(points))
    shoulder = np.linalg.norm(points[LEFT_SHOULDER] - points[RIGHT_SHOULDER])
    body_height = (
        torso
        + np.linalg.norm(points[NOSE] - shoulder_center(points))
        + 0.5 * (
            np.linalg.norm(points[LEFT_KNEE] - points[LEFT_HIP])
            + np.linalg.norm(points[RIGHT_KNEE] - points[RIGHT_HIP]))
        + 0.5 * (
            np.linalg.norm(points[LEFT_ANKLE] - points[LEFT_KNEE])
            + np.linalg.norm(points[RIGHT_ANKLE] - points[RIGHT_KNEE]))
    )
    head_to_heel = None
    if head_top is not None:
        head_top = np.asarray(head_top, dtype=np.float64).reshape(-1)
        if head_top.size < 3 or not np.all(np.isfinite(head_top[:3])):
            raise ValueError("head_top must contain three finite coordinates")
        head_to_heel = (
            body_height
            + np.linalg.norm(head_top[:3] - points[NOSE])
            + 0.5 * (
                np.linalg.norm(points[LEFT_HEEL] - points[LEFT_ANKLE])
                + np.linalg.norm(points[RIGHT_HEEL] - points[RIGHT_ANKLE])
            )
        )
    elif mode == "head_to_heel":
        raise ValueError("head_to_heel requires a full-MHR head-top keypoint")

    candidates = {
        "torso": (torso, shoulder, body_height),
        "shoulder_width": (shoulder, torso, body_height),
        "body_height": (body_height, torso, shoulder),
        "head_to_heel": (head_to_heel, body_height, torso, shoulder),
    }.get(mode, (torso, shoulder, body_height))
    for value in candidates:
        if np.isfinite(value) and value > EPS:
            return float(value)
    raise ValueError("SAM3D skeleton has no usable body-size measurement")


def reference_lengths(points, symmetry="average"):
    lengths = {
        child: float(np.linalg.norm(_edge_delta(points, child, parent)))
        for child, parent, _ in RETARGET_EDGES
    }
    if symmetry == "off":
        return lengths
    if symmetry != "average":
        raise ValueError(f"unknown reference symmetry mode: {symmetry}")

    visited = set()
    for child, mirror in MIRROR_CHILD.items():
        pair = tuple(sorted((child, mirror)))
        if pair in visited or child not in lengths or mirror not in lengths:
            continue
        visited.add(pair)
        left, right = lengths[child], lengths[mirror]
        value = (left + right) * 0.5
        lengths[child] = value
        lengths[mirror] = value
    return lengths


def body_measurements(points):
    """Return the principal 3D lengths shown in the retarget report."""
    return {
        "torso": float(np.linalg.norm(points[NECK] - hip_center(points))),
        "shoulder_width": float(np.linalg.norm(
            points[LEFT_SHOULDER] - points[RIGHT_SHOULDER])),
        "hip_width": float(np.linalg.norm(
            points[LEFT_HIP] - points[RIGHT_HIP])),
        "shoulder_to_nose": float(np.linalg.norm(
            points[NOSE] - shoulder_center(points))),
        "upper_arm": 0.5 * (
            float(np.linalg.norm(points[LEFT_ELBOW] - points[LEFT_SHOULDER]))
            + float(np.linalg.norm(
                points[RIGHT_ELBOW] - points[RIGHT_SHOULDER]))),
        "forearm": 0.5 * (
            float(np.linalg.norm(points[LEFT_WRIST] - points[LEFT_ELBOW]))
            + float(np.linalg.norm(
                points[RIGHT_WRIST] - points[RIGHT_ELBOW]))),
        "thigh": 0.5 * (
            float(np.linalg.norm(points[LEFT_KNEE] - points[LEFT_HIP]))
            + float(np.linalg.norm(points[RIGHT_KNEE] - points[RIGHT_HIP]))),
        "shin": 0.5 * (
            float(np.linalg.norm(points[LEFT_ANKLE] - points[LEFT_KNEE]))
            + float(np.linalg.norm(
                points[RIGHT_ANKLE] - points[RIGHT_KNEE]))),
    }


def normalized_body_proportions(points, unit):
    """Normalize report measurements by one explicit body-size unit."""
    if not np.isfinite(unit) or unit <= EPS:
        raise ValueError("body-size unit must be positive")
    return {
        name: value / float(unit)
        for name, value in body_measurements(points).items()
    }


def _unit_direction(primary, fallback):
    """Return a stable unit vector, preferring the driving-pose direction."""
    for value in (primary, fallback):
        norm = float(np.linalg.norm(value))
        if np.isfinite(norm) and norm > EPS:
            return value / norm
    return np.zeros(3, dtype=np.float64)


def _place_edge(output, driving, reference, child, parent, target_length):
    direction = _unit_direction(
        driving[child] - driving[parent],
        reference[child] - reference[parent],
    )
    output[child] = output[parent] + direction * target_length


def retarget_mhr70(reference, driving, size_reference="torso",
                   reference_symmetry="average", uniform_scale=1.0,
                   leg_scale=1.0, arm_scale=1.0, head_scale=1.0,
                   hand_scale=1.0, torso_scale=1.0,
                   shoulder_width_scale=1.0, hip_width_scale=1.0,
                   neck_scale=1.0, upper_arm_scale=1.0,
                   forearm_scale=1.0, thigh_scale=1.0,
                   shin_scale=1.0, reference_head_top=None,
                   driving_head_top=None):
    """Combine reference 3D bone lengths with driving 3D directions."""
    reference = np.asarray(reference, dtype=np.float64)
    driving = np.asarray(driving, dtype=np.float64)
    if reference.shape != (MHR70_COUNT, 3) or driving.shape != (MHR70_COUNT, 3):
        raise ValueError("reference and driving joints must both have shape (70, 3)")

    ref_unit = body_unit(reference, size_reference, reference_head_top)
    drv_unit = body_unit(driving, size_reference, driving_head_top)
    base_scale = drv_unit / ref_unit * float(uniform_scale)
    target_unit = drv_unit * float(uniform_scale)
    length_ratios = {
        child: length / ref_unit
        for child, length in reference_lengths(
            reference, reference_symmetry).items()
    }
    reference_proportions = normalized_body_proportions(reference, ref_unit)

    output = driving.copy()
    root = hip_center(driving)

    # Build the central body explicitly. This makes widths exact instead of
    # treating the left and right halves as unrelated bones.
    hip_axis = _unit_direction(
        driving[LEFT_HIP] - driving[RIGHT_HIP],
        reference[LEFT_HIP] - reference[RIGHT_HIP],
    )
    hip_width = (
        reference_proportions["hip_width"] * target_unit
        * float(hip_width_scale)
    )
    output[LEFT_HIP] = root + hip_axis * hip_width * 0.5
    output[RIGHT_HIP] = root - hip_axis * hip_width * 0.5

    torso_direction = _unit_direction(
        driving[NECK] - hip_center(driving),
        reference[NECK] - hip_center(reference),
    )
    torso_length = (
        reference_proportions["torso"] * target_unit
        * float(torso_scale)
    )
    output[NECK] = root + torso_direction * torso_length

    shoulder_axis = _unit_direction(
        driving[LEFT_SHOULDER] - driving[RIGHT_SHOULDER],
        reference[LEFT_SHOULDER] - reference[RIGHT_SHOULDER],
    )
    shoulder_width = (
        reference_proportions["shoulder_width"] * target_unit
        * float(shoulder_width_scale)
    )
    driving_shoulder_offset = shoulder_center(driving) - driving[NECK]
    output_shoulder_center = (
        output[NECK]
        + driving_shoulder_offset / drv_unit * target_unit
    )
    output[LEFT_SHOULDER] = (
        output_shoulder_center + shoulder_axis * shoulder_width * 0.5)
    output[RIGHT_SHOULDER] = (
        output_shoulder_center - shoulder_axis * shoulder_width * 0.5)

    edge_scales = (
        (LEFT_ELBOW, LEFT_SHOULDER,
         float(arm_scale) * float(upper_arm_scale)),
        (RIGHT_ELBOW, RIGHT_SHOULDER,
         float(arm_scale) * float(upper_arm_scale)),
        (LEFT_WRIST, LEFT_ELBOW,
         float(arm_scale) * float(forearm_scale)),
        (RIGHT_WRIST, RIGHT_ELBOW,
         float(arm_scale) * float(forearm_scale)),
        (LEFT_KNEE, LEFT_HIP,
         float(leg_scale) * float(thigh_scale)),
        (RIGHT_KNEE, RIGHT_HIP,
         float(leg_scale) * float(thigh_scale)),
        (LEFT_ANKLE, LEFT_KNEE,
         float(leg_scale) * float(shin_scale)),
        (RIGHT_ANKLE, RIGHT_KNEE,
         float(leg_scale) * float(shin_scale)),
    )
    for child, parent, part_scale in edge_scales:
        _place_edge(
            output, driving, reference, child, parent,
            length_ratios[child] * target_unit * part_scale,
        )

    for child, parent, _ in BODY_EDGES:
        if parent not in (LEFT_ANKLE, RIGHT_ANKLE):
            continue
        _place_edge(
            output, driving, reference, child, parent,
            length_ratios[child] * target_unit * float(leg_scale),
        )

    # Shoulder center -> nose is a direct reference proportion. Anchoring it
    # explicitly avoids accumulating shoulder/neck estimation offsets.
    driving_shoulders = shoulder_center(driving)
    reference_shoulders = shoulder_center(reference)
    nose_direction = _unit_direction(
        driving[NOSE] - driving_shoulders,
        reference[NOSE] - reference_shoulders,
    )
    neck_length = (
        reference_proportions["shoulder_to_nose"] * target_unit
        * float(head_scale) * float(neck_scale)
    )
    output[NOSE] = shoulder_center(output) + nose_direction * neck_length

    for child, parent in (
            (LEFT_EYE, NOSE), (RIGHT_EYE, NOSE),
            (LEFT_EAR, LEFT_EYE), (RIGHT_EAR, RIGHT_EYE)):
        _place_edge(
            output, driving, reference, child, parent,
            length_ratios[child] * target_unit * float(head_scale),
        )

    for child, parent, _ in HAND_EDGES:
        _place_edge(
            output, driving, reference, child, parent,
            length_ratios[child] * target_unit
            * float(arm_scale) * float(hand_scale),
        )

    details = {
        "reference_unit": ref_unit,
        "driving_unit": drv_unit,
        "base_scale": base_scale,
        "size_reference": size_reference,
        "reference_proportions": reference_proportions,
        "generated_proportions": normalized_body_proportions(
            output, target_unit),
    }
    return output, details


def extract_camera(output):
    if not isinstance(output, dict):
        raise ValueError("driving SAM3D input must be a dictionary")
    raw = output.get("raw_output") or {}
    camera = output.get("camera")
    if camera is None:
        camera = raw.get("pred_cam_t")
    focal = output.get("focal_length")
    if focal is None:
        focal = raw.get("focal_length")

    camera = as_numpy(camera, "camera translation").reshape(-1)
    focal = as_numpy(focal, "focal length").reshape(-1)
    if camera.size < 3 or focal.size < 1:
        raise ValueError("SAM3D camera data has an invalid shape")
    if not np.all(np.isfinite(camera[:3])) or not np.all(np.isfinite(focal)):
        raise ValueError("SAM3D camera data contains invalid values")
    fx = float(focal[0])
    fy = float(focal[1] if focal.size > 1 else focal[0])
    if fx <= EPS or fy <= EPS:
        raise ValueError("SAM3D focal length must be positive")
    return camera[:3], np.array([fx, fy], dtype=np.float64)


def project_mhr70(points, camera, focal_xy, width, height):
    """Use SAM 3D Body's perspective-camera convention."""
    camera_points = np.asarray(points, dtype=np.float64) + camera[None, :]
    depth = camera_points[:, 2]
    valid = np.isfinite(camera_points).all(axis=1) & (depth > EPS)
    projected = np.zeros((len(points), 2), dtype=np.float64)
    projected[valid, 0] = (
        focal_xy[0] * camera_points[valid, 0] / depth[valid] + width * 0.5)
    projected[valid, 1] = (
        focal_xy[1] * camera_points[valid, 1] / depth[valid] + height * 0.5)
    return projected, valid, depth


def fit_projected(points, valid, width, height, mode="shrink_to_fit", margin=16):
    """Fit valid projected points while preserving aspect ratio."""
    points = np.asarray(points, dtype=np.float64).copy()
    if mode == "off" or not np.any(valid):
        return points, 1.0
    if mode not in {"shrink_to_fit", "fit_exactly"}:
        raise ValueError(f"unknown fit mode: {mode}")

    subset = points[valid]
    lo = subset.min(axis=0)
    hi = subset.max(axis=0)
    if mode == "shrink_to_fit" and (
            lo[0] >= 0 and lo[1] >= 0 and hi[0] <= width and hi[1] <= height):
        return points, 1.0

    max_margin = max(0.0, min(width, height) * 0.5 - 1.0)
    margin = min(max(float(margin), 0.0), max_margin)
    available = np.array([width - 2 * margin, height - 2 * margin], dtype=np.float64)
    span = np.maximum(hi - lo, EPS)
    scale = float(np.min(available / span))
    if mode == "shrink_to_fit":
        scale = min(1.0, scale)
    center = (lo + hi) * 0.5
    target_center = np.array([width * 0.5, height * 0.5], dtype=np.float64)
    points[valid] = (points[valid] - center) * scale + target_center
    return points, scale


def _openpose_field(projected, valid, indices):
    out = np.zeros((len(indices), 3), dtype=np.float64)
    for output_index, mhr_index in enumerate(indices):
        if valid[mhr_index]:
            out[output_index, 0] = projected[mhr_index, 0]
            out[output_index, 1] = projected[mhr_index, 1]
            out[output_index, 2] = 1.0
    return [round(float(value), 6) for value in out.reshape(-1)]


def to_pose_keypoint(projected, valid, width, height):
    """Convert MHR70 body/hands to absolute-pixel OpenPose fields."""
    person = {
        "pose_keypoints_2d": _openpose_field(
            projected, valid, COCO18_FROM_MHR70),
        "face_keypoints_2d": [0.0] * (70 * 3),
        "hand_left_keypoints_2d": _openpose_field(
            projected, valid, LEFT_HAND_FROM_MHR70),
        "hand_right_keypoints_2d": _openpose_field(
            projected, valid, RIGHT_HAND_FROM_MHR70),
    }
    return [{
        "canvas_width": int(width),
        "canvas_height": int(height),
        "people": [person],
    }]


def image_size(image):
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) < 3:
        raise ValueError("driving_image must be a ComfyUI IMAGE tensor")
    height, width = int(shape[-3]), int(shape[-2])
    if width <= 0 or height <= 0:
        raise ValueError("driving_image has an invalid size")
    return width, height
