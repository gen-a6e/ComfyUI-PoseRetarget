from .sam3d_retarget import (
    extract_camera,
    extract_head_top,
    extract_mhr70,
    fit_projected,
    image_size,
    project_mhr70,
    retarget_mhr70,
    to_pose_keypoint,
)


class SAM3DBodyPoseRetarget:
    """Reference proportions plus driving directions in true 3D."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_sam3d": ("SAM3D_OUTPUT",),
                "driving_sam3d": ("SAM3D_OUTPUT",),
                "driving_image": ("IMAGE",),
                "size_reference": ([
                    "torso", "shoulder_width", "body_height", "head_to_heel"
                ],),
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

    RETURN_TYPES = ("POSE_KEYPOINT", "STRING")
    RETURN_NAMES = ("pose_keypoint", "report")
    FUNCTION = "run"
    CATEGORY = "pose-retarget"

    def run(self, reference_sam3d, driving_sam3d, driving_image,
            size_reference, reference_symmetry,
            uniform_scale, leg_scale, arm_scale, head_scale, hand_scale,
            fit_to_canvas, canvas_margin, torso_scale=1.0,
            shoulder_width_scale=1.0, hip_width_scale=1.0,
            neck_scale=1.0, upper_arm_scale=1.0, forearm_scale=1.0,
            thigh_scale=1.0, shin_scale=1.0):
        reference = extract_mhr70(reference_sam3d)
        driving = extract_mhr70(driving_sam3d)
        reference_head_top = None
        driving_head_top = None
        reference_head_top_index = None
        driving_head_top_index = None
        if size_reference == "head_to_heel":
            reference_head_top, reference_head_top_index = extract_head_top(
                reference_sam3d, reference
            )
            driving_head_top, driving_head_top_index = extract_head_top(
                driving_sam3d, driving
            )
        retargeted, details = retarget_mhr70(
            reference,
            driving,
            size_reference=size_reference,
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
            reference_head_top=reference_head_top,
            driving_head_top=driving_head_top,
        )

        width, height = image_size(driving_image)
        camera, focal_xy = extract_camera(driving_sam3d)
        projected, valid, depth = project_mhr70(
            retargeted, camera, focal_xy, width, height)
        projected, fit_scale = fit_projected(
            projected, valid, width, height,
            mode=fit_to_canvas, margin=canvas_margin)
        output = to_pose_keypoint(projected, valid, width, height)

        valid_depth = depth[valid]
        depth_note = "unavailable"
        if valid_depth.size:
            depth_note = f"{valid_depth.min():.3f}..{valid_depth.max():.3f} m"
        ratio_note = ", ".join(
            f"{name}:{details['reference_proportions'][name]:.3f}"
            f"->{details['generated_proportions'][name]:.3f}"
            for name in details["reference_proportions"]
        )
        report = (
            "SAM 3D Body retargeted one MHR70 skeleton; "
            f"reference_unit={details['reference_unit']:.3f} m; "
            f"driving_unit={details['driving_unit']:.3f} m; "
            f"scale={details['base_scale']:.3f}; "
            f"fit_scale={fit_scale:.3f}; "
            f"camera_depth={depth_note}."
            f" Ratios reference->generated "
            f"(normalized by {details['size_reference']}): {ratio_note}."
        )
        if size_reference == "head_to_heel":
            report += (
                " Head-top full-MHR indices "
                f"reference={reference_head_top_index}, "
                f"driving={driving_head_top_index}."
            )
        invalid_count = int((~valid).sum())
        if invalid_count:
            report += f" WARNING: {invalid_count} point(s) were behind the camera."
        report += (
            " Dense face landmarks are unavailable in MHR70; "
            "face_keypoints_2d is zero-confidence."
        )
        return (output, report)


NODE_CLASS_MAPPINGS = {
    "SAM3DBodyPoseRetarget": SAM3DBodyPoseRetarget,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBodyPoseRetarget": "SAM 3D Body Pose Retarget",
}
