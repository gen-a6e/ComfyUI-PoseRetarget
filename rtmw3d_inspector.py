"""RTMW3D inference helpers and inspection artifact generation.

This module is intentionally not imported by the ComfyUI node package.  The
optional RTMW3D dependencies are only needed when the inspector CLI is used.
The inference and coordinate conversion functions live here so a future
ComfyUI RTMW3D node can reuse them without depending on the CLI.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


RTMW3D_MODEL_URL = (
    "https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/"
    "rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx"
)

# RTMW3D encodes X/Y on its 288x384 model-input plane and Z on a separate
# 288-bin SimCC depth axis.  The decoded Z range is ±2.1744869 metres around
# the root.  Keeping these native coordinates avoids pretending that an
# uncalibrated source image has known camera intrinsics.
MODEL_INPUT_WIDTH = 288.0
MODEL_INPUT_HEIGHT = 384.0
MODEL_DEPTH_SIZE = 288.0
MODEL_Z_RANGE_M = 2.1744869
MODEL_DEPTH_PIXELS_PER_M = MODEL_DEPTH_SIZE / (2.0 * MODEL_Z_RANGE_M)
HIP_INDICES = (11, 12)
KEYPOINT_COUNT = 133


BODY_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
]


def _hand_names(side: str) -> List[str]:
    names = [f"{side}_hand_root"]
    for finger in ("thumb", "forefinger", "middle_finger", "ring_finger",
                   "pinky_finger"):
        names.extend(f"{side}_{finger}{joint}" for joint in range(1, 5))
    return names


KEYPOINT_NAMES = (
    BODY_NAMES
    + [f"face-{index}" for index in range(68)]
    + _hand_names("left")
    + _hand_names("right")
)


BODY_EDGES = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
    (15, 17), (15, 18), (15, 19),
    (16, 20), (16, 21), (16, 22),
]


def _chain(indices: Sequence[int], closed: bool = False) -> List[Tuple[int, int]]:
    edges = list(zip(indices, indices[1:]))
    if closed and len(indices) > 2:
        edges.append((indices[-1], indices[0]))
    return edges


def _face_edges() -> List[Tuple[int, int]]:
    offset = 23
    parts = [
        (range(0, 17), False),
        (range(17, 22), False),
        (range(22, 27), False),
        (range(27, 31), False),
        (range(31, 36), False),
        (range(36, 42), True),
        (range(42, 48), True),
        (range(48, 60), True),
        (range(60, 68), True),
    ]
    edges: List[Tuple[int, int]] = []
    for indices, closed in parts:
        edges.extend(_chain([offset + value for value in indices], closed))
    return edges


def _hand_edges(root: int, wrist: int) -> List[Tuple[int, int]]:
    edges = [(wrist, root)]
    for start in range(root + 1, root + 21, 4):
        edges.extend(_chain([root, start, start + 1, start + 2, start + 3]))
    return edges


FACE_EDGES = _face_edges()
LEFT_HAND_EDGES = _hand_edges(91, 9)
RIGHT_HAND_EDGES = _hand_edges(112, 10)
ALL_EDGES = BODY_EDGES + FACE_EDGES + LEFT_HAND_EDGES + RIGHT_HAND_EDGES


def keypoint_group(index: int) -> str:
    if index < 23:
        return "body"
    if index < 91:
        return "face"
    if index < 112:
        return "left_hand"
    return "right_hand"


def edge_group(a: int, b: int) -> str:
    groups = {keypoint_group(a), keypoint_group(b)}
    if "left_hand" in groups:
        return "left_hand"
    if "right_hand" in groups:
        return "right_hand"
    if "face" in groups:
        return "face"
    return "body"


def _scores_2d(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    while values.ndim > 2 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError(f"unexpected score shape: {values.shape}")
    return values


def normalize_inference_arrays(
        keypoints: np.ndarray,
        scores: np.ndarray,
        keypoints_2d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize rtmlib outputs to N x 133 arrays."""
    points_3d = np.asarray(keypoints, dtype=np.float32)
    points_2d = np.asarray(keypoints_2d, dtype=np.float32)
    confidences = _scores_2d(scores)

    if points_3d.ndim == 2:
        points_3d = points_3d[None, ...]
    if points_2d.ndim == 2:
        points_2d = points_2d[None, ...]
    if points_3d.ndim != 3 or points_3d.shape[-1] < 3:
        raise ValueError(f"unexpected 3D keypoint shape: {points_3d.shape}")
    if points_2d.ndim != 3 or points_2d.shape[-1] < 2:
        raise ValueError(f"unexpected 2D keypoint shape: {points_2d.shape}")
    if not (points_3d.shape[:2] == points_2d.shape[:2]
            == confidences.shape[:2]):
        raise ValueError(
            "RTMW3D output shapes disagree: "
            f"3d={points_3d.shape}, 2d={points_2d.shape}, "
            f"scores={confidences.shape}")
    if points_3d.shape[1] < KEYPOINT_COUNT:
        raise ValueError(
            f"RTMW3D returned {points_3d.shape[1]} keypoints; "
            f"expected at least {KEYPOINT_COUNT}")

    return (points_3d[:, :KEYPOINT_COUNT, :3],
            confidences[:, :KEYPOINT_COUNT],
            points_2d[:, :KEYPOINT_COUNT, :2])


def resolve_device(requested: str) -> Tuple[str, List[str]]:
    """Resolve ``auto`` and return the selected device and ORT providers."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is missing. Install requirements-inspector.txt "
            "or onnxruntime-gpu in the inspector virtual environment.") from exc

    providers = list(ort.get_available_providers())
    if requested == "auto":
        if "CUDAExecutionProvider" in providers:
            return "cuda", providers
        if ("CoreMLExecutionProvider" in providers
                or "MPSExecutionProvider" in providers):
            return "mps", providers
        return "cpu", providers

    base = requested.split(":", 1)[0]
    required = {
        "cuda": "CUDAExecutionProvider",
        "mps": "CoreMLExecutionProvider",
        "rocm": "ROCMExecutionProvider",
    }.get(base)
    if required and required not in providers:
        raise RuntimeError(
            f"device {requested!r} requires {required}, but available "
            f"providers are: {', '.join(providers)}")
    return requested, providers


def load_image(path: Path) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required by the inspector") from exc

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"could not read image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        color = image[:, :, :3].astype(np.float32)
        image = np.clip(color * alpha + 255.0 * (1.0 - alpha), 0, 255)
        image = image.astype(np.uint8)
    elif image.shape[2] != 3:
        raise ValueError(f"unsupported image shape: {image.shape}")
    return image


def run_inference(
        image: np.ndarray,
        device: str = "auto",
        bbox_mode: str = "auto",
        model: Optional[str] = None,
        detector: Optional[str] = None,
        cache_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Run RTMW3D-X via rtmlib and return normalized arrays."""
    if cache_dir is not None:
        os.environ["XDG_CACHE_HOME"] = str(cache_dir.expanduser().resolve())

    try:
        from rtmlib import RTMPose3d, Wholebody3d
    except ImportError as exc:
        raise RuntimeError(
            "rtmlib is missing. Install requirements-inspector.txt in a "
            "Python 3.10+ virtual environment.") from exc

    selected_device, providers = resolve_device(device)
    pose_model = model or RTMW3D_MODEL_URL
    common = {
        "backend": "onnxruntime",
        "device": selected_device,
        "pose_input_size": (288, 384),
    }

    if bbox_mode == "full-image":
        estimator = RTMPose3d(
            pose_model,
            model_input_size=(288, 384),
            backend="onnxruntime",
            device=selected_device,
        )
        keypoints, scores, _, keypoints_2d = estimator(image)
    elif bbox_mode == "auto":
        if detector:
            common["det"] = detector
            common["det_input_size"] = (640, 640)
        estimator = Wholebody3d(pose=pose_model, **common)
        # RTMPose3d deliberately falls back to a full-image box if YOLOX
        # finds no person, which is useful for stylized characters.
        keypoints, scores, _, keypoints_2d = estimator(image)
    else:
        raise ValueError("bbox_mode must be 'auto' or 'full-image'")

    points_3d, confidences, points_2d = normalize_inference_arrays(
        keypoints, scores, keypoints_2d)
    metadata = {
        "device": selected_device,
        "onnxruntime_providers": providers,
        "bbox_mode": bbox_mode,
        "model": pose_model,
    }
    return points_3d, confidences, points_2d, metadata


def center_relative_depth(
        relative_z: np.ndarray,
        scores: np.ndarray,
        confidence_threshold: float,
) -> Tuple[np.ndarray, float, str]:
    """Center depth at the hip midpoint, with a body-median fallback."""
    z = np.asarray(relative_z, dtype=np.float32).copy()
    confidence = np.asarray(scores, dtype=np.float32)
    usable_hips = [index for index in HIP_INDICES
                   if confidence[index] >= confidence_threshold
                   and np.isfinite(z[index])]
    if usable_hips:
        root = float(np.mean(z[usable_hips]))
        method = "hip_midpoint"
    else:
        body_mask = ((confidence[:23] >= confidence_threshold)
                     & np.isfinite(z[:23]))
        if np.any(body_mask):
            root = float(np.median(z[:23][body_mask]))
            method = "visible_body_median"
        else:
            finite = z[np.isfinite(z)]
            root = float(np.median(finite)) if finite.size else 0.0
            method = "all_keypoint_median"
    return z - root, root, method


def relative_depth_to_model_pixels(relative_z: np.ndarray) -> np.ndarray:
    """Convert decoded root-relative Z metres back to RTMW3D depth pixels.

    This is the inverse of MMPose's ``SimCC3DLabel.decode`` mapping after the
    root depth has been subtracted.  The returned values are offsets from the
    hip plane; add ``MODEL_DEPTH_SIZE / 2`` to recover absolute SimCC bins.
    """
    z = np.asarray(relative_z, dtype=np.float32)
    return z * MODEL_DEPTH_PIXELS_PER_M


def depth_label(z_value: float, neutral_band: float = 0.02) -> str:
    if z_value < -neutral_band:
        return "nearer"
    if z_value > neutral_band:
        return "farther"
    return "root_plane"


def build_payload(
        image_path: Path,
        image_width: int,
        image_height: int,
        keypoints: np.ndarray,
        scores: np.ndarray,
        keypoints_2d: np.ndarray,
        person_index: int = 0,
        confidence_threshold: float = 0.3,
        inference_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    points_3d, confidences, points_2d = normalize_inference_arrays(
        keypoints, scores, keypoints_2d)
    if not 0 <= person_index < len(points_3d):
        raise IndexError(
            f"person index {person_index} is outside 0..{len(points_3d) - 1}")

    selected_scores = confidences[person_index]
    selected_2d = points_2d[person_index]
    relative_z, raw_root_z, z_center_method = center_relative_depth(
        points_3d[person_index, :, 2], selected_scores,
        confidence_threshold)
    selected_model = points_3d[person_index]
    model_depth = relative_depth_to_model_pixels(relative_z)

    records: List[Dict[str, Any]] = []
    for index, name in enumerate(KEYPOINT_NAMES):
        score = float(selected_scores[index])
        records.append({
            "index": index,
            "name": name,
            "group": keypoint_group(index),
            "x_px": float(selected_2d[index, 0]),
            "y_px": float(selected_2d[index, 1]),
            "model_x_px": float(selected_model[index, 0]),
            "model_y_px": float(selected_model[index, 1]),
            "z_relative_m": float(relative_z[index]),
            "z_model_px": float(model_depth[index]),
            "score": score,
            "visible": bool(score >= confidence_threshold),
            "depth": depth_label(float(relative_z[index])),
        })

    visible = [record for record in records if record["visible"]]
    nearest = sorted(visible, key=lambda item: item["z_relative_m"])[:5]
    farthest = sorted(visible, key=lambda item: item["z_relative_m"],
                      reverse=True)[:5]
    warnings = [
        "Z is relative depth: negative is nearer to the camera; positive is farther.",
        "Model X/Y and Z px are RTMW3D input-grid coordinates, not calibrated camera-space metres.",
        "Changing the viewer depth scale only magnifies the display; exported Z values do not change.",
    ]
    if z_center_method != "hip_midpoint":
        warnings.append(
            "Hip confidence was low; relative Z was centered using "
            f"{z_center_method.replace('_', ' ')}.")

    return {
        "schema": "rtmw3d-inspector-v2",
        "image": {
            "path": str(image_path),
            "width": int(image_width),
            "height": int(image_height),
        },
        "person_index": int(person_index),
        "person_count": int(len(points_3d)),
        "confidence_threshold": float(confidence_threshold),
        "coordinate_system": {
            "x_px": "image pixels, positive right",
            "y_px": "image pixels, positive down",
            "model_x_px": "RTMW3D 288px input-grid X, positive right",
            "model_y_px": "RTMW3D 384px input-grid Y, positive down",
            "z_relative_m": "hip-centered relative depth; negative nearer, positive farther",
            "z_model_px": "hip-centered RTMW3D depth-grid pixels; negative nearer, positive farther",
            "model_input_size": [MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT],
            "model_depth_size": MODEL_DEPTH_SIZE,
            "model_z_range_m": MODEL_Z_RANGE_M,
            "model_depth_pixels_per_m": MODEL_DEPTH_PIXELS_PER_M,
        },
        "centering": {
            "raw_model_root_z": raw_root_z,
            "z_method": z_center_method,
        },
        "inference": inference_metadata or {},
        "summary": {
            "visible_count": len(visible),
            "low_confidence_count": KEYPOINT_COUNT - len(visible),
            "nearest": [
                {"index": point["index"], "name": point["name"],
                 "z_relative_m": point["z_relative_m"],
                 "z_model_px": point["z_model_px"]}
                for point in nearest
            ],
            "farthest": [
                {"index": point["index"], "name": point["name"],
                 "z_relative_m": point["z_relative_m"],
                 "z_model_px": point["z_model_px"]}
                for point in farthest
            ],
        },
        "warnings": warnings,
        "keypoints": records,
        "edges": [
            {"a": a, "b": b, "group": edge_group(a, b)}
            for a, b in ALL_EDGES
        ],
    }


def _depth_bgr(z_value: float, max_abs_z: float) -> Tuple[int, int, int]:
    amount = min(abs(z_value) / max(max_abs_z, 1e-6), 1.0)
    neutral = np.array([80.0, 220.0, 255.0])
    target = (np.array([60.0, 60.0, 255.0]) if z_value < 0
              else np.array([255.0, 110.0, 40.0]))
    color = neutral * (1.0 - amount) + target * amount
    return tuple(int(value) for value in color)


def draw_overlay(
        image: np.ndarray,
        payload: Dict[str, Any],
        confidence_threshold: float,
) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required by the inspector") from exc

    output = image.copy()
    points = payload["keypoints"]
    visible_z = [abs(point["z_relative_m"]) for point in points
                 if point["score"] >= confidence_threshold]
    max_abs_z = float(np.percentile(visible_z, 95)) if visible_z else 1.0
    edge_colors = {
        "body": (120, 230, 120),
        "face": (210, 210, 210),
        "left_hand": (90, 220, 255),
        "right_hand": (255, 170, 90),
    }

    for edge in payload["edges"]:
        a, b = points[edge["a"]], points[edge["b"]]
        if (a["score"] < confidence_threshold
                or b["score"] < confidence_threshold):
            continue
        cv2.line(
            output,
            (int(round(a["x_px"])), int(round(a["y_px"]))),
            (int(round(b["x_px"])), int(round(b["y_px"]))),
            edge_colors[edge["group"]], 2, cv2.LINE_AA)

    for point in points:
        if point["score"] < confidence_threshold:
            continue
        position = (int(round(point["x_px"])), int(round(point["y_px"])))
        color = _depth_bgr(point["z_relative_m"], max_abs_z)
        radius = 5 if point["group"] == "body" else 2
        cv2.circle(output, position, radius, color, -1, cv2.LINE_AA)
        if point["group"] == "body":
            cv2.putText(output, str(point["index"]),
                        (position[0] + 6, position[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 255, 255), 1, cv2.LINE_AA)

    legend_y = 24
    for text, color in (("nearer (-Z)", (60, 60, 255)),
                        ("root plane", (80, 220, 255)),
                        ("farther (+Z)", (255, 110, 40))):
        cv2.circle(output, (16, legend_y - 5), 5, color, -1, cv2.LINE_AA)
        cv2.putText(output, text, (28, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(output, text, (28, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (30, 30, 30), 1, cv2.LINE_AA)
        legend_y += 22
    return output


def _json_for_html(payload: Dict[str, Any]) -> str:
    # Prevent a path containing ``</script>`` from terminating the data tag.
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


def write_artifacts(
        output_dir: Path,
        image: np.ndarray,
        payload: Dict[str, Any],
        viewer_template: Path,
) -> Dict[str, Path]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required by the inspector") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "keypoints.json"
    csv_path = output_dir / "keypoints.csv"
    overlay_path = output_dir / "overlay.png"
    viewer_path = output_dir / "viewer.html"

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    fieldnames = [
        "index", "name", "group", "x_px", "y_px", "model_x_px",
        "model_y_px", "z_relative_m", "z_model_px", "score", "visible",
        "depth",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {field: point[field] for field in fieldnames}
            for point in payload["keypoints"])

    overlay = draw_overlay(
        image, payload, payload["confidence_threshold"])
    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError(f"failed to write overlay image: {overlay_path}")

    template = viewer_template.read_text(encoding="utf-8")
    if "__RTMW3D_DATA__" not in template:
        raise ValueError("viewer template has no __RTMW3D_DATA__ placeholder")
    viewer_path.write_text(
        template.replace("__RTMW3D_DATA__", _json_for_html(payload)),
        encoding="utf-8")

    return {
        "overlay": overlay_path,
        "json": json_path,
        "csv": csv_path,
        "viewer": viewer_path,
    }


def output_directory(base: Path, image_path: Path) -> Path:
    safe_stem = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in image_path.stem)
    return base / (safe_stem or "image")


assert len(KEYPOINT_NAMES) == KEYPOINT_COUNT
