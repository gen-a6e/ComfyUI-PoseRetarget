"""
Bone-length retargeting for OpenPose COCO-18 keypoints.

Takes joint ANGLES from a driving pose and BONE LENGTHS from a reference pose,
so the resulting skeleton has the reference character's body proportions while
striking the driving pose.

Pure numpy + PIL. No cv2, no torch beyond what ComfyUI already provides.
"""

import math
import numpy as np

# ---------------------------------------------------------------- topology

# COCO-18 order used by comfyui_controlnet_aux / DWPose:
#  0 nose        1 neck        2 R-shoulder  3 R-elbow   4 R-wrist
#  5 L-shoulder  6 L-elbow     7 L-wrist     8 R-hip     9 R-knee
# 10 R-ankle    11 L-hip      12 L-knee     13 L-ankle  14 R-eye
# 15 L-eye      16 R-ear      17 L-ear

ROOT = 1

PARENT = {
    0: 1,
    14: 0, 15: 0, 16: 14, 17: 15,
    2: 1, 3: 2, 4: 3,
    5: 1, 6: 5, 7: 6,
    8: 1, 9: 8, 10: 9,
    11: 1, 12: 11, 13: 12,
}

# parents always appear before their children
TRAVERSAL = [1, 0, 14, 15, 16, 17, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

R_HIP, L_HIP = 8, 11
R_ELBOW, L_ELBOW = 3, 6
R_SHO, L_SHO = 2, 5
R_WRIST, L_WRIST = 4, 7
NOSE = 0
R_EYE, L_EYE = 14, 15
R_EAR, L_EAR = 16, 17
HEAD_JOINTS = (NOSE, R_EYE, L_EYE, R_EAR, L_EAR)

LIMB_SEQ = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9],
    [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16],
    [0, 15], [15, 17],
]

COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85],
]

EPS = 1e-6

# Approximate un-foreshortened bone lengths, expressed in torso units
# (torso = neck -> hip centre). Used only as a fallback when symmetry
# cannot tell us how long a bone "should" look.
CANONICAL = {
    0: 0.24,
    14: 0.10, 15: 0.10, 16: 0.13, 17: 0.13,
    2: 0.36, 5: 0.36,
    3: 0.64, 6: 0.64,
    4: 0.50, 7: 0.50,
    8: 1.05, 11: 1.05,
    9: 0.84, 12: 0.84,
    10: 0.85, 13: 0.85,
}

# Bones that mirror each other; either one is evidence for the other's
# true length when the limb is pointing toward or away from the camera.
MIRROR = {
    2: 5, 5: 2, 3: 6, 6: 3, 4: 7, 7: 4,
    8: 11, 11: 8, 9: 12, 12: 9, 10: 13, 13: 10,
    14: 15, 15: 14, 16: 17, 17: 16,
}



# ---------------------------------------------------------------- parsing


def _as_frame_list(pose_keypoint):
    """POSE_KEYPOINT is a list of per-frame dicts, but be forgiving."""
    if pose_keypoint is None:
        return []
    if isinstance(pose_keypoint, dict):
        return [pose_keypoint]
    return list(pose_keypoint)


def _canvas(frame):
    w = frame.get("canvas_width") or frame.get("width") or 512
    h = frame.get("canvas_height") or frame.get("height") or 768
    return float(w), float(h)


def _is_normalized(frames):
    """controlnet_aux emits 0..1 coords; some packs emit pixels. Detect."""
    hi = 0.0
    for frame in frames:
        for person in frame.get("people", []) or []:
            flat = person.get("pose_keypoints_2d") or []
            arr = np.asarray(flat, dtype=np.float32).reshape(-1, 3)
            valid = arr[arr[:, 2] > 0]
            if valid.size:
                hi = max(hi, float(np.abs(valid[:, :2]).max()))
    return hi <= 1.5


def _to_pixels(flat, w, h, normalized):
    if not flat:
        return None
    arr = np.asarray(flat, dtype=np.float32).reshape(-1, 3).copy()
    if normalized:
        arr[:, 0] *= w
        arr[:, 1] *= h
    return arr


def _to_flat(arr, w, h, normalized):
    out = arr.copy()
    if normalized:
        out[:, 0] /= max(w, EPS)
        out[:, 1] /= max(h, EPS)
    out[out[:, 2] <= 0] = 0.0
    return [round(float(v), 6) for v in out.reshape(-1)]


def _valid(arr, i):
    return i < len(arr) and arr[i, 2] > 0


def _midpoint(arr, a, b):
    if _valid(arr, a) and _valid(arr, b):
        return (arr[a, :2] + arr[b, :2]) / 2.0
    if _valid(arr, a):
        return arr[a, :2].copy()
    if _valid(arr, b):
        return arr[b, :2].copy()
    return None


# ---------------------------------------------------------------- measuring


def _bone_lengths(arr):
    out = {}
    for child, parent in PARENT.items():
        if _valid(arr, child) and _valid(arr, parent):
            out[child] = float(np.linalg.norm(arr[child, :2] - arr[parent, :2]))
    return out


JOINT_NAMES = {
    2: "R-shoulder", 5: "L-shoulder", 3: "R-upper-arm", 6: "L-upper-arm",
    4: "R-forearm", 7: "L-forearm", 8: "R-hip", 11: "L-hip",
    9: "R-upper-leg", 12: "L-upper-leg", 10: "R-lower-leg", 13: "L-lower-leg",
}


def symmetrize_lengths(ref_len, mode="longer_side", warn_below=0.85):
    """A bent or foreshortened limb in the REFERENCE image measures short,
    and that wrong length would then be baked into every output pose.
    Left and right bones are the same length on a real body, so borrow from
    whichever side was measured cleanly."""
    if mode == "off":
        return dict(ref_len), []

    out = dict(ref_len)
    warnings = []
    done = set()

    for a, b in MIRROR.items():
        key = tuple(sorted((a, b)))
        if key in done:
            continue
        done.add(key)
        if a not in ref_len or b not in ref_len:
            continue

        la, lb = ref_len[a], ref_len[b]
        longer = max(la, lb)
        if longer <= EPS:
            continue

        if min(la, lb) / longer < warn_below:
            short = a if la < lb else b
            warnings.append(
                f"{JOINT_NAMES.get(short, short)} measured "
                f"{min(la, lb) / longer:.0%} of its mirror")

        value = longer if mode == "longer_side" else (la + lb) / 2.0
        out[a] = value
        out[b] = value

    return out, warnings


def _body_unit(arr, mode):
    """A canvas-independent size measure, so proportions transfer but the
    figure keeps the driving pose's on-screen size."""
    order = {
        "torso": ["torso", "shoulder", "head"],
        "shoulder_width": ["shoulder", "torso", "head"],
        "head_size": ["head", "torso", "shoulder"],
    }.get(mode, ["torso", "shoulder", "head"])

    for kind in order:
        if kind == "torso":
            hip = _midpoint(arr, R_HIP, L_HIP)
            if hip is not None and _valid(arr, ROOT):
                d = float(np.linalg.norm(arr[ROOT, :2] - hip))
                if d > EPS:
                    return d
        elif kind == "shoulder":
            if _valid(arr, R_SHO) and _valid(arr, L_SHO):
                d = float(np.linalg.norm(arr[R_SHO, :2] - arr[L_SHO, :2]))
                if d > EPS:
                    return d
        elif kind == "head":
            if _valid(arr, ROOT) and _valid(arr, NOSE):
                d = float(np.linalg.norm(arr[ROOT, :2] - arr[NOSE, :2]))
                if d > EPS:
                    return d
    return None


def _face_extent(arr):
    """Return a robust scalar size for dense face landmarks.

    DWPose face landmarks describe the visible face rather than the hair or
    skull.  The diagonal of their 5th--95th percentile box is still a much
    better proxy for the ControlNet face footprint than neck-to-nose alone.
    """
    if arr is None or arr.size == 0:
        return None
    mask = ((arr[:, 2] > 0)
            & np.isfinite(arr[:, 0])
            & np.isfinite(arr[:, 1]))
    points = arr[mask, :2]
    if len(points) < 8:
        return None
    lo = np.percentile(points, 5.0, axis=0)
    hi = np.percentile(points, 95.0, axis=0)
    width, height = hi - lo
    size = float(np.hypot(width, height))
    return size if size > EPS else None


def _joint_span(arr, a, b):
    if arr is None or not (_valid(arr, a) and _valid(arr, b)):
        return None
    size = float(np.linalg.norm(arr[a, :2] - arr[b, :2]))
    return size if size > EPS else None


def _matched_head_sizes(source_body, source_face, current_body, current_face):
    """Measure source and current heads with the same available metric.

    Mixing a dense-face measurement on one side with an eye-span measurement
    on the other would introduce an arbitrary conversion factor.  Select the
    first metric that is valid for both sides instead.
    """
    source_size = _face_extent(source_face)
    current_size = _face_extent(current_face)
    if source_size is not None and current_size is not None:
        return source_size, current_size, "face_landmarks"

    for a, b, name in (
            (R_EAR, L_EAR, "ear_span"),
            (R_EYE, L_EYE, "eye_span"),
            (ROOT, NOSE, "neck_to_nose")):
        source_size = _joint_span(source_body, a, b)
        current_size = _joint_span(current_body, a, b)
        if source_size is not None and current_size is not None:
            return source_size, current_size, name
    return None, None, "unavailable"


def head_correction_factor(source_body, source_face, current_body,
                           current_face, size_mode, manual_scale=1.0):
    """Scale the current head to the source's head-to-body proportion."""
    source_unit = _body_unit(source_body, size_mode)
    current_unit = _body_unit(current_body, size_mode)
    source_size, current_size, method = _matched_head_sizes(
        source_body, source_face, current_body, current_face)

    if (source_unit is None or current_unit is None
            or source_size is None or current_size is None
            or source_unit <= EPS or current_size <= EPS):
        return float(manual_scale), method, None

    target_size = source_size / source_unit * current_unit * manual_scale
    factor = float(target_size / current_size)
    details = {
        "source_ratio": float(source_size / source_unit),
        "current_size": float(current_size),
        "target_size": float(target_size),
    }
    return factor, method, details


def scale_head_keypoints(arr, factor, include_neck_to_nose=False):
    """Scale face joints without lengthening the neck-to-nose segment.

    A face-size correction should normally leave the nose where retargeting
    placed it and scale eyes/ears around that nose.  Only the final
    neck-to-nose fallback lacks a real face span, so it must scale the whole
    head chain around the neck.
    """
    out = arr.copy()
    if include_neck_to_nose:
        if not _valid(out, ROOT):
            return out
        pivot = out[ROOT, :2].copy()
        joints = HEAD_JOINTS
    else:
        if not _valid(out, NOSE):
            return out
        pivot = out[NOSE, :2].copy()
        joints = (R_EYE, L_EYE, R_EAR, L_EAR)
    for joint in joints:
        if _valid(out, joint):
            out[joint, :2] = pivot + (out[joint, :2] - pivot) * factor
    return out


# ---------------------------------------------------------------- core


def foreshortening(drv, drv_len, drv_unit, mode, floor,
                   canonical_trigger=0.75):
    """How much shorter each bone LOOKS than it should, in 0..1.

    A limb pointing at the camera projects short. We estimate the length it
    would have had in the picture plane, then keep the ratio so the same
    out-of-plane angle survives retargeting.
    """
    factors = {}
    if mode == "off":
        return factors

    for child in PARENT:
        # Head direction is already represented by the driving face.  The
        # limb heuristic would misread a small/stylised face as foreshortened
        # and shrink it a second time.
        if child in HEAD_JOINTS:
            continue
        observed = drv_len.get(child)
        if observed is None or observed <= EPS:
            continue

        candidates = [observed]

        twin = MIRROR.get(child)
        if twin is not None and twin in drv_len:
            candidates.append(drv_len[twin])

        # The canonical table is a weaker signal: a stylised figure with
        # genuinely short arms looks identical to one pointing at the camera.
        # Only trust it when the bone is much shorter than any real body
        # proportion would explain.
        if mode == "symmetry_and_canonical" and drv_unit:
            canon = CANONICAL.get(child)
            if canon:
                expected_canon = canon * drv_unit
                if observed < expected_canon * canonical_trigger:
                    candidates.append(expected_canon)

        expected = max(candidates)
        if expected <= EPS:
            continue
        factors[child] = float(min(1.0, max(floor, observed / expected)))

    return factors


def retarget_body(ref, drv, ref_len, ref_unit, size_mode, uniform_scale,
                  leg_scale, arm_scale, head_scale,
                  foreshorten_mode="symmetry", foreshorten_floor=0.15,
                  canonical_trigger=0.75):
    """Directions from `drv`, lengths from `ref`. Returns new (18,3) array."""
    out = np.zeros_like(drv)
    out[:, 2] = 0.0

    drv_unit = _body_unit(drv, size_mode)
    if ref_unit is None or drv_unit is None or ref_unit <= EPS:
        ratio = 1.0
    else:
        ratio = drv_unit / ref_unit

    drv_len = _bone_lengths(drv)
    fore = foreshortening(drv, drv_len, drv_unit, foreshorten_mode,
                          foreshorten_floor, canonical_trigger)

    if not _valid(drv, ROOT):
        # No neck to root the tree on: nothing sensible to do.
        return drv.copy()


    out[ROOT] = drv[ROOT]

    LEGS = {8, 9, 10, 11, 12, 13}
    ARMS = {2, 3, 4, 5, 6, 7}
    HEAD = set(HEAD_JOINTS)

    for joint in TRAVERSAL:
        if joint == ROOT:
            continue
        parent = PARENT[joint]
        if not (_valid(drv, joint) and _valid(drv, parent) and out[parent, 2] > 0):
            continue

        direction = drv[joint, :2] - drv[parent, :2]
        norm = float(np.linalg.norm(direction))
        if norm < EPS:
            out[joint, :2] = out[parent, :2]
            out[joint, 2] = drv[joint, 2]
            continue

        if joint in ref_len:
            length = ref_len[joint] * ratio * fore.get(joint, 1.0)
        else:
            length = norm  # no reference for this bone: keep the driving one

        length *= uniform_scale
        if joint in LEGS:
            length *= leg_scale
        elif joint in ARMS:
            length *= arm_scale
        elif joint in HEAD:
            length *= head_scale

        out[joint, :2] = out[parent, :2] + direction / norm * length
        out[joint, 2] = drv[joint, 2]

    return out


def _anchor_shift(out, drv, anchor):
    if anchor == "neck":
        return np.zeros(2, dtype=np.float32)
    if anchor == "hips":
        a, b = _midpoint(out, R_HIP, L_HIP), _midpoint(drv, R_HIP, L_HIP)
    elif anchor == "feet":
        a, b = _midpoint(out, 10, 13), _midpoint(drv, 10, 13)
    elif anchor == "bbox_center":
        va, vb = out[out[:, 2] > 0][:, :2], drv[drv[:, 2] > 0][:, :2]
        if va.size == 0 or vb.size == 0:
            return np.zeros(2, dtype=np.float32)
        a = (va.min(0) + va.max(0)) / 2.0
        b = (vb.min(0) + vb.max(0)) / 2.0
    else:
        return np.zeros(2, dtype=np.float32)
    if a is None or b is None:
        return np.zeros(2, dtype=np.float32)
    return (b - a).astype(np.float32)


def _fit_points(points, w, h, margin, mode):
    """Scale about the bbox centre so the figure sits inside the margin box.

    `shrink_to_fit` only ever shrinks, so a figure that already fits is left
    alone -- which means canvas_margin does nothing for it. `fit_exactly`
    always scales to the margin box, so the margin is guaranteed.
    """
    if points.size == 0 or mode == "off":
        return 1.0, np.zeros(2, dtype=np.float32)

    lo, hi = points.min(0), points.max(0)
    span = np.maximum(hi - lo, EPS)
    avail = np.maximum(
        np.array([w - 2.0 * margin, h - 2.0 * margin], dtype=np.float32), EPS)

    s = float(np.min(avail / span))
    if mode == "shrink_to_fit":
        s = min(1.0, s)

    centre = (lo + hi) / 2.0
    target = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    return s, (target - centre * s).astype(np.float32)


def _apply_affine(arr, scale, shift, pivot=None):
    out = arr.copy()
    mask = out[:, 2] > 0
    if pivot is None:
        out[mask, :2] = out[mask, :2] * scale + shift
    else:
        out[mask, :2] = (out[mask, :2] - pivot) * scale + pivot + shift
    return out


def _move_extra(flat, w, h, normalized, delta, scale, pivot):
    """Face / hand keypoints ride along with the joint they hang off."""
    arr = _to_pixels(flat, w, h, normalized)
    if arr is None or arr.size == 0:
        return flat
    mask = arr[:, 2] > 0
    if not mask.any():
        return flat
    arr[mask, :2] = (arr[mask, :2] - pivot) * scale + pivot + delta
    return _to_flat(arr, w, h, normalized)


# ---------------------------------------------------------------- rendering


def _ellipse_poly(cx, cy, ax, ay, angle_deg, steps=24):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    pts = []
    for i in range(steps):
        t = 2.0 * math.pi * i / steps
        x, y = ax * math.cos(t), ay * math.sin(t)
        pts.append((cx + x * ca - y * sa, cy + x * sa + y * ca))
    return pts


def render_pose(frames, normalized, width=None, height=None, stick_width=4,
                point_radius=4, draw_face=True, draw_hands=True):
    from PIL import Image, ImageDraw

    images = []
    for frame in frames:
        w, h = _canvas(frame)
        w = int(width or w)
        h = int(height or h)
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        for person in frame.get("people", []) or []:
            arr = _to_pixels(person.get("pose_keypoints_2d"), w, h, normalized)
            if arr is None:
                continue

            # limbs, blended the way the reference implementation does
            layer = Image.fromarray(canvas)
            draw = ImageDraw.Draw(layer)
            for idx, (a, b) in enumerate(LIMB_SEQ):
                if not (_valid(arr, a) and _valid(arr, b)):
                    continue
                p, q = arr[a, :2], arr[b, :2]
                mid = (p + q) / 2.0
                length = float(np.linalg.norm(p - q))
                angle = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))
                poly = _ellipse_poly(mid[0], mid[1], length / 2.0,
                                     stick_width, angle)
                draw.polygon(poly, fill=tuple(COLORS[idx]))
            blended = (canvas.astype(np.float32) * 0.4
                       + np.asarray(layer, dtype=np.float32) * 0.6)
            canvas = np.clip(blended, 0, 255).astype(np.uint8)

            img = Image.fromarray(canvas)
            draw = ImageDraw.Draw(img)
            for i in range(min(18, len(arr))):
                if not _valid(arr, i):
                    continue
                x, y = float(arr[i, 0]), float(arr[i, 1])
                draw.ellipse(
                    [x - point_radius, y - point_radius,
                     x + point_radius, y + point_radius],
                    fill=tuple(COLORS[i]),
                )

            if draw_face:
                face = _to_pixels(person.get("face_keypoints_2d"), w, h, normalized)
                if face is not None:
                    for i in range(len(face)):
                        if face[i, 2] <= 0:
                            continue
                        x, y = float(face[i, 0]), float(face[i, 1])
                        draw.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5],
                                     fill=(255, 255, 255))

            if draw_hands:
                for key in ("hand_left_keypoints_2d", "hand_right_keypoints_2d"):
                    hand = _to_pixels(person.get(key), w, h, normalized)
                    if hand is None:
                        continue
                    for i in range(len(hand)):
                        if hand[i, 2] <= 0:
                            continue
                        x, y = float(hand[i, 0]), float(hand[i, 1])
                        draw.ellipse([x - 2, y - 2, x + 2, y + 2],
                                     fill=(0, 0, 255))

            canvas = np.asarray(img, dtype=np.uint8)

        images.append(canvas.astype(np.float32) / 255.0)

    if not images:
        images = [np.zeros((512, 512, 3), dtype=np.float32)]
    return np.stack(images, axis=0)
