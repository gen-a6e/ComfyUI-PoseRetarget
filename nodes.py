from .sam3d_debug import SAM3DBodySkeletonDebug
from .sam3d_retarget import (
    extract_camera,
    extract_head_top,
    extract_mhr70,
    extract_mhr70_2d,
    estimated_height,
    fit_projected,
    image_size,
    project_mhr70,
    projected_difference,
    LEFT_HAND_FROM_MHR70,
    RIGHT_HAND_FROM_MHR70,
    retarget_mhr70,
    to_pose_keypoint,
)


def _height_report(sam3d, points, side, warnings):
    """補足の身長計測だけを省略できるようにし、骨格の必須検証とは分離する。"""
    try:
        head_top, index = extract_head_top(sam3d)
        height = estimated_height(points, head_top)
    except ValueError as exc:
        warnings.append(f"{side}_height unavailable: {exc}")
        return "unavailable", "unavailable"
    return f"{height:.3f} m", f"R{index}"


class SAM3DBodyPoseRetarget:
    """referenceの体型とdrivingのポーズを3D空間で合成するComfyUIノード。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 2枚の画像をSAM 3D Bodyで解析した結果と、出力サイズの基準画像。
                "reference_sam3d": ("SAM3D_OUTPUT",),
                "driving_sam3d": ("SAM3D_OUTPUT",),
                "driving_image": ("IMAGE",),
                # referenceの左右差を骨長へどう反映するか。
                "reference_symmetry": (["average", "off"],),
                "uniform_scale": ("FLOAT", {"default": 1.0, "min": 0.1,
                                            "max": 3.0, "step": 0.01}),
                "leg_scale": ("FLOAT", {"default": 1.0, "min": 0.1,
                                        "max": 3.0, "step": 0.01}),
                "arm_scale": ("FLOAT", {"default": 1.0, "min": 0.1,
                                        "max": 3.0, "step": 0.01}),
                "head_scale": ("FLOAT", {"default": 1.0, "min": 0.1,
                                         "max": 3.0, "step": 0.01}),
                "hand_scale": ("FLOAT", {"default": 1.0, "min": 0.1,
                                         "max": 3.0, "step": 0.01}),
                "fit_to_canvas": (["shrink_to_fit", "fit_exactly", "off"],),
                "canvas_margin": ("INT", {"default": 16, "min": 0,
                                          "max": 512, "step": 1}),
                "torso_scale": ("FLOAT", {"default": 1.0, "min": 0.1,
                                            "max": 3.0, "step": 0.01}),
                "shoulder_width_scale": (
                    "FLOAT", {"default": 1.0, "min": 0.1,
                              "max": 3.0, "step": 0.01}),
                "hip_width_scale": (
                    "FLOAT", {"default": 1.0, "min": 0.1,
                              "max": 3.0, "step": 0.01}),
                "neck_scale": ("FLOAT", {"default": 1.0, "min": 0.1,
                                           "max": 3.0, "step": 0.01}),
                "upper_arm_scale": (
                    "FLOAT", {"default": 1.0, "min": 0.1,
                              "max": 3.0, "step": 0.01}),
                "forearm_scale": (
                    "FLOAT", {"default": 1.0, "min": 0.1,
                              "max": 3.0, "step": 0.01}),
                "thigh_scale": ("FLOAT", {"default": 1.0, "min": 0.1,
                                            "max": 3.0, "step": 0.01}),
                "shin_scale": ("FLOAT", {"default": 1.0, "min": 0.1,
                                           "max": 3.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = (
        "POSE_KEYPOINT", "POSE_KEYPOINT", "STRING", "POSE_KEYPOINT"
    )
    RETURN_NAMES = (
        "pose_keypoint",
        "driving_pose_keypoint",
        "report",
        "sam_raw_driving_pose_keypoint",
    )
    FUNCTION = "run"
    CATEGORY = "pose-retarget"

    def run(self, reference_sam3d, driving_sam3d, driving_image,
            reference_symmetry, uniform_scale, leg_scale, arm_scale,
            head_scale, hand_scale,
            fit_to_canvas, canvas_margin, torso_scale=1.0,
            shoulder_width_scale=1.0, hip_width_scale=1.0,
            neck_scale=1.0, upper_arm_scale=1.0, forearm_scale=1.0,
            thigh_scale=1.0, shin_scale=1.0):
        # 1. SAM3D_OUTPUTから、体・足・手を含むMHR70の3D座標を取り出す。
        reference = extract_mhr70(reference_sam3d)
        driving = extract_mhr70(driving_sam3d)

        # 身長は補足情報。R126が欠ける・使用不能でも骨長転送は続行する。
        warnings = []
        reference_height_note, reference_head_top_index = _height_report(
            reference_sam3d, reference, "reference", warnings
        )
        driving_height_note, driving_head_top_index = _height_report(
            driving_sam3d, driving, "driving", warnings
        )

        # 2. referenceの各骨長とdrivingの各ボーン方向を合成する。
        # ここではまだ3D座標のままで、カメラ投影やcanvas調整は行わない。
        retargeted, details = retarget_mhr70(
            reference,
            driving,
            reference_symmetry=reference_symmetry,
            uniform_scale=uniform_scale,
            leg_scale=leg_scale,
            arm_scale=arm_scale,
            head_scale=head_scale,
            hand_scale=hand_scale,
            torso_scale=torso_scale,
            shoulder_width_scale=shoulder_width_scale,
            hip_width_scale=hip_width_scale,
            neck_scale=neck_scale,
            upper_arm_scale=upper_arm_scale,
            forearm_scale=forearm_scale,
            thigh_scale=thigh_scale,
            shin_scale=shin_scale,
        )

        # 3. driving画像と同じカメラ・焦点距離・canvasサイズで2Dへ投影する。
        # fit_to_canvasは合成後の骨格だけに適用し、必要なら全身が収まるよう移動・縮小する。
        width, height = image_size(driving_image)
        camera, focal_xy = extract_camera(driving_sam3d)
        projected, valid, depth = project_mhr70(
            retargeted, camera, focal_xy, width, height)
        projected, fit_scale = fit_projected(
            projected, valid, width, height,
            mode=fit_to_canvas, margin=canvas_margin)
        output = to_pose_keypoint(projected, valid, width, height)

        # 4. 比較用として、加工前のdriving骨格も同じカメラで直接投影する。
        # scale・retarget・fitを一切適用しないため、SAM 3D Bodyの元結果を確認できる。
        driving_projected, driving_valid, _ = project_mhr70(
            driving, camera, focal_xy, width, height)
        driving_output = to_pose_keypoint(
            driving_projected, driving_valid, width, height)

        # SAM内部で計算済みの2D点も再投影せず出力する。3D再投影との差を診断する用途。
        # 診断不能時もcanvas情報は保持するが、人物・座標は捏造しない。
        raw_driving_output = [{
            "canvas_width": width, "canvas_height": height, "people": []
        }]
        right_hand_difference = left_hand_difference = {"count": 0}
        try:
            raw_driving_projected, raw_driving_valid = extract_mhr70_2d(
                driving_sam3d
            )
            if not raw_driving_valid.any():
                raise ValueError("SAM raw 2D keypoints contain no finite points")
        except ValueError as exc:
            warnings.append(f"sam_raw_driving_pose_keypoint unavailable: {exc}")
        else:
            raw_driving_valid &= driving_valid
            raw_driving_output = to_pose_keypoint(
                raw_driving_projected, raw_driving_valid, width, height
            )
            right_hand_difference = projected_difference(
                raw_driving_projected,
                raw_driving_valid,
                driving_projected,
                driving_valid,
                RIGHT_HAND_FROM_MHR70,
            )
            left_hand_difference = projected_difference(
                raw_driving_projected,
                raw_driving_valid,
                driving_projected,
                driving_valid,
                LEFT_HAND_FROM_MHR70,
            )

        # 5. 概算身長、倍率、生成前後の実骨長を確認できる文字列にまとめる。
        valid_depth = depth[valid]
        depth_note = "unavailable"
        if valid_depth.size:
            depth_note = f"{valid_depth.min():.3f}..{valid_depth.max():.3f} m"
        length_note = ", ".join(
            f"{name}:{details['reference_measurements'][name]:.3f}"
            f"->{details['generated_measurements'][name]:.3f}"
            for name in details["reference_measurements"]
        )
        report = (
            "SAM 3D Body retargeted one MHR70 skeleton; "
            f"reference_height={reference_height_note}; "
            f"driving_height={driving_height_note}; "
            f"size_source={details['size_source']}; "
            f"scale={details['base_scale']:.3f}; "
            f"fit_scale={fit_scale:.3f}; "
            f"camera_depth={depth_note}."
            f" Lengths reference->generated (m): {length_note}."
        )
        for side, difference in (
            ("right", right_hand_difference),
            ("left", left_hand_difference),
        ):
            if difference["count"]:
                report += (
                    f" {side}_hand_raw2d_vs_reprojected: "
                    f"rms={difference['rms']:.3f} px, "
                    f"max={difference['max']:.3f} px, "
                    f"points={difference['count']}."
                )
            else:
                report += (
                    f" {side}_hand_raw2d_vs_reprojected: unavailable."
                )
        # 頭頂の出典は内部リグR126。MHR70/308の番号とは区別する。
        report += (
            " Head-top internal-rig indices "
            f"reference={reference_head_top_index}, "
            f"driving={driving_head_top_index}."
        )
        for warning in warnings:
            report += f" WARNING: {warning}."
        invalid_count = int((~valid).sum())
        if invalid_count:
            # カメラより後ろにある点は2Dへ投影できず、confidence 0になる。
            report += f" WARNING: {invalid_count} point(s) were behind the camera."
        driving_invalid_count = int((~driving_valid).sum())
        if driving_invalid_count:
            report += (
                f" WARNING: {driving_invalid_count} original driving point(s) "
                "were behind the camera."
            )
        report += (
            " Dense face landmarks are unavailable in MHR70; "
            "face_keypoints_2d is zero-confidence."
        )

        # 既存3出力の順序を維持し、SAM内部2Dの診断出力を末尾へ追加する。
        return (output, driving_output, report, raw_driving_output)


NODE_CLASS_MAPPINGS = {
    "SAM3DBodyPoseRetarget": SAM3DBodyPoseRetarget,
    "SAM3DBodySkeletonDebug": SAM3DBodySkeletonDebug,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBodyPoseRetarget": "SAM 3D Body Pose Retarget",
    "SAM3DBodySkeletonDebug": "SAM 3D Body Skeleton Debug",
}
