import copy

import numpy as np

from .pose_retarget import (
    ROOT as ROOT_JOINT,
    EPS, L_WRIST, NOSE, R_WRIST, L_ELBOW, R_ELBOW,
    _anchor_shift, _apply_affine, _bone_lengths,
    _body_unit, _canvas, _as_frame_list, _is_normalized,
    _to_flat, _to_pixels, _valid, retarget_body, _fit_points,
    enforce_neck_ratio, head_correction_factor, scale_head_keypoints,
    symmetrize_lengths,
)


SDPOSE_REQUIRED_KEYPOINTS = {
    "face_keypoints_2d": 70,
    "hand_left_keypoints_2d": 21,
    "hand_right_keypoints_2d": 21,
}


def _normalize_optional_keypoints(flat, count):
    """Return a fixed-size, zero-padded field accepted by SDPose Draw."""
    out = np.zeros((count, 3), dtype=np.float32)
    if flat is None:
        return out.reshape(-1).tolist()

    try:
        raw = np.asarray(flat, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return out.reshape(-1).tolist()

    complete_points = raw.size // 3
    if complete_points:
        source = raw[:complete_points * 3].reshape(-1, 3)
        take = min(count, len(source))
        source = source[:take]
        valid_rows = np.all(np.isfinite(source), axis=1)
        valid_indices = np.flatnonzero(valid_rows)
        out[valid_indices] = source[valid_indices]

    return [round(float(v), 6) for v in out.reshape(-1)]


class PoseRetargetProportions:
    """Joint angles from the driving pose, bone lengths from the reference."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_pose": ("POSE_KEYPOINT",),
                "driving_pose": ("POSE_KEYPOINT",),
                "size_reference": (["torso", "shoulder_width", "head_size"],),
                "reference_symmetry": (["longer_side", "average", "off"],),
                "anchor": (["hips", "neck", "feet", "bbox_center"],),
                "uniform_scale": ("FLOAT", {"default": 1.0, "min": 0.1,
                                            "max": 3.0, "step": 0.01}),
                "leg_scale": ("FLOAT", {"default": 1.0, "min": 0.1,
                                        "max": 3.0, "step": 0.01}),
                "arm_scale": ("FLOAT", {"default": 1.0, "min": 0.1,
                                        "max": 3.0, "step": 0.01}),
                "head_scale": ("FLOAT", {"default": 1.0, "min": 0.1,
                                         "max": 3.0, "step": 0.01}),
                "foreshorten_mode": (["symmetry", "symmetry_and_canonical", "off"],),
                "foreshorten_floor": ("FLOAT", {"default": 0.15, "min": 0.0,
                                                "max": 1.0, "step": 0.01}),
                "canonical_trigger": ("FLOAT", {"default": 0.75, "min": 0.1,
                                                "max": 1.0, "step": 0.01}),
                "fit_to_canvas": (["shrink_to_fit", "fit_exactly", "off"],),
                "canvas_margin": ("INT", {"default": 16, "min": 0,
                                          "max": 512, "step": 1}),
            }
        }

    RETURN_TYPES = ("POSE_KEYPOINT", "STRING")
    RETURN_NAMES = ("pose_keypoint", "report")
    FUNCTION = "run"
    CATEGORY = "pose-retarget"

    def run(self, reference_pose, driving_pose, size_reference,
            reference_symmetry, anchor,
            uniform_scale, leg_scale, arm_scale, head_scale,
            foreshorten_mode, foreshorten_floor, canonical_trigger,
            fit_to_canvas, canvas_margin):

        ref_frames = _as_frame_list(reference_pose)
        drv_frames = _as_frame_list(driving_pose)
        if not ref_frames or not drv_frames:
            return (copy.deepcopy(driving_pose), "empty input; passed through")

        ref_norm = _is_normalized(ref_frames)
        drv_norm = _is_normalized(drv_frames)

        rw, rh = _canvas(ref_frames[0])
        ref_people = ref_frames[0].get("people", []) or []
        if not ref_people:
            return (copy.deepcopy(driving_pose),
                    "no person in reference pose; passed through")
        ref_person = ref_people[0]
        ref = _to_pixels(ref_person.get("pose_keypoints_2d"),
                         rw, rh, ref_norm)
        if ref is None:
            return (copy.deepcopy(driving_pose),
                    "reference has no body keypoints; passed through")
        ref_face = _to_pixels(ref_person.get("face_keypoints_2d"),
                              rw, rh, ref_norm)

        ref_len = _bone_lengths(ref)
        ref_len, sym_warnings = symmetrize_lengths(ref_len, reference_symmetry)
        ref_unit = _body_unit(ref, size_reference)

        out_frames = []
        touched = 0
        head_notes = []
        neck_notes = []
        ignored_driving_people = 0
        for frame in drv_frames:
            w, h = _canvas(frame)
            new_frame = copy.deepcopy(frame)
            people = new_frame.get("people", []) or []
            if not people:
                out_frames.append(new_frame)
                continue

            ignored_driving_people += max(0, len(people) - 1)
            person = people[0]
            new_frame["people"] = [person]
            for key, count in SDPOSE_REQUIRED_KEYPOINTS.items():
                person[key] = _normalize_optional_keypoints(
                    person.get(key), count)
            drv = _to_pixels(person.get("pose_keypoints_2d"), w, h, drv_norm)
            if drv is not None:
                out = retarget_body(ref, drv, ref_len, ref_unit, size_reference,
                                    uniform_scale, leg_scale, arm_scale,
                                    1.0, foreshorten_mode,
                                    foreshorten_floor, canonical_trigger)
                out = _apply_affine(out, 1.0,
                                    _anchor_shift(out, drv, anchor))

                # face and hands ride along with the joint they hang off,
                # scaled by how much that limb changed
                extras = {}
                for key, joint, mate in (
                        ("face_keypoints_2d", NOSE, ROOT_JOINT),
                        ("hand_left_keypoints_2d", L_WRIST, L_ELBOW),
                        ("hand_right_keypoints_2d", R_WRIST, R_ELBOW)):
                    ex = _to_pixels(person.get(key), w, h, drv_norm)
                    if ex is None or ex.size == 0:
                        continue
                    if not (_valid(drv, joint) and _valid(out, joint)):
                        continue
                    ratio = 1.0
                    if _valid(drv, mate) and _valid(out, mate):
                        a = float(np.linalg.norm(drv[joint, :2] - drv[mate, :2]))
                        b = float(np.linalg.norm(out[joint, :2] - out[mate, :2]))
                        if a > EPS:
                            ratio = b / a
                    pivot = drv[joint, :2]
                    delta = out[joint, :2] - pivot
                    m = ex[:, 2] > 0
                    ex[m, :2] = (ex[m, :2] - pivot) * ratio + pivot + delta
                    extras[key] = ex

                # Transfer the reference's actual face-to-body ratio.
                # Keep the driving landmark layout (pose/expression), but
                # scale its footprint to the reference proportion.
                face = extras.get("face_keypoints_2d")
                correction, method, details = head_correction_factor(
                    ref, ref_face, out, face,
                    size_reference, head_scale)
                use_neck_pivot = method == "neck_to_nose"
                out = scale_head_keypoints(
                    out, correction,
                    include_neck_to_nose=use_neck_pivot)
                pivot_joint = ROOT_JOINT if use_neck_pivot else NOSE
                if face is not None and _valid(out, pivot_joint):
                    extras["face_keypoints_2d"] = _apply_affine(
                        face, correction, np.zeros(2, dtype=np.float32),
                        out[pivot_joint, :2])
                if details:
                    head_notes.append(
                        f"head_source=reference, metric={method}, "
                        f"correction={correction:.3f}, "
                        f"source_ratio={details['source_ratio']:.3f}")
                else:
                    head_notes.append(
                        f"head metric {method}; "
                        f"manual head_scale={correction:.3f} only")

                # Re-measure the body that was actually produced. Rebuilding
                # the hips/shoulders can change its final size relative to the
                # driving pose used by retarget_body's first estimate.
                out, neck_details = enforce_neck_ratio(
                    ref, out, size_reference, drv)
                if neck_details:
                    neck_shift = neck_details["shift"]
                    face = extras.get("face_keypoints_2d")
                    if face is not None:
                        extras["face_keypoints_2d"] = _apply_affine(
                            face, 1.0, neck_shift)
                    neck_notes.append(
                        "neck shoulder_center_to_nose/"
                        f"{size_reference}: "
                        f"reference={neck_details['source_ratio']:.3f}, "
                        f"projection={neck_details['projection_factor']:.3f}, "
                        f"target={neck_details['target_ratio']:.3f}, "
                        f"before={neck_details['before_ratio']:.3f}, "
                        f"after={neck_details['after_ratio']:.3f}")
                    if (neck_details["raw_projection_factor"]
                            > neck_details["projection_factor"] + EPS):
                        neck_notes[-1] += (
                            " (projection capped from "
                            f"{neck_details['raw_projection_factor']:.3f})")
                else:
                    neck_notes.append(
                        "neck shoulder_center_to_nose unavailable; "
                        "kept initial retarget")

                # fit the WHOLE figure, face and hands included, so they
                # cannot be clipped at the edge
                if fit_to_canvas != "off":
                    pts = [out[out[:, 2] > 0][:, :2]]
                    pts += [e[e[:, 2] > 0][:, :2] for e in extras.values()]
                    pts = [p for p in pts if p.size]
                    if pts:
                        scale, fshift = _fit_points(np.concatenate(pts), w, h,
                                                    canvas_margin, fit_to_canvas)
                        out = _apply_affine(out, scale, fshift)
                        for k in extras:
                            extras[k] = _apply_affine(extras[k], scale, fshift)

                for key, ex in extras.items():
                    person[key] = _to_flat(ex, w, h, drv_norm)

                person["pose_keypoints_2d"] = _to_flat(out, w, h, drv_norm)
                touched += 1

            out_frames.append(new_frame)

        ratio_note = []
        for name, joint in (("upper_leg", 9), ("lower_leg", 10),
                            ("upper_arm", 3), ("forearm", 4)):
            if joint in ref_len and ref_unit:
                ratio_note.append(f"{name}={ref_len[joint] / ref_unit:.2f}")
        report = (f"retargeted {touched} pose(s) over {len(out_frames)} frame(s); "
                  f"foreshortening={foreshorten_mode}; "
                  f"reference bone/{size_reference} ratios: "
                  + ", ".join(ratio_note))
        if sym_warnings:
            report += ("\nWARNING: the reference pose looks bent or "
                       "foreshortened -- " + "; ".join(sym_warnings)
                       + ". A straight standing reference gives better "
                         "measurements.")
        if head_notes:
            report += "\n" + "\n".join(head_notes)
        if neck_notes:
            report += "\n" + "\n".join(neck_notes)
        ignored_reference_people = max(0, len(ref_people) - 1)
        if ignored_reference_people or ignored_driving_people:
            report += ("\nWARNING: single-person node; ignored "
                       f"{ignored_reference_people} extra reference person(s) "
                       f"and {ignored_driving_people} extra driving person(s).")
        return (out_frames, report)


NODE_CLASS_MAPPINGS = {
    "PoseRetargetProportions": PoseRetargetProportions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseRetargetProportions": "Pose Retarget (keep body proportions)",
}
