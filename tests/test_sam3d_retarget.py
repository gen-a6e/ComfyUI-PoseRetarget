import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sam3d_retarget as sr  # noqa: E402


def load_package():
    package_name = "comfyui_pose_retarget_sam3d_test_package"
    spec = importlib.util.spec_from_file_location(
        package_name, ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    return package


def skeleton():
    points = np.zeros((70, 3), dtype=np.float64)
    points[sr.LEFT_HIP] = (0.15, 0.0, 0.0)
    points[sr.RIGHT_HIP] = (-0.15, 0.0, 0.0)
    points[sr.NECK] = (0.0, -0.60, 0.0)
    points[sr.LEFT_SHOULDER] = (0.25, -0.58, 0.0)
    points[sr.RIGHT_SHOULDER] = (-0.25, -0.58, 0.0)
    points[sr.LEFT_ELBOW] = (0.55, -0.52, 0.0)
    points[sr.RIGHT_ELBOW] = (-0.55, -0.52, 0.0)
    points[sr.LEFT_WRIST] = (0.82, -0.46, 0.0)
    points[sr.RIGHT_WRIST] = (-0.82, -0.46, 0.0)
    points[sr.LEFT_KNEE] = (0.16, 0.55, 0.0)
    points[sr.RIGHT_KNEE] = (-0.16, 0.55, 0.0)
    points[sr.LEFT_ANKLE] = (0.16, 1.08, 0.0)
    points[sr.RIGHT_ANKLE] = (-0.16, 1.08, 0.0)
    points[15], points[16], points[17] = (
        (0.20, 1.18, -0.12), (0.12, 1.18, -0.10), (0.16, 1.13, 0.10))
    points[18], points[19], points[20] = (
        (-0.12, 1.18, -0.10), (-0.20, 1.18, -0.12), (-0.16, 1.13, 0.10))
    points[sr.NOSE] = (0.0, -0.80, -0.02)
    points[sr.LEFT_EYE] = (0.04, -0.84, -0.03)
    points[sr.RIGHT_EYE] = (-0.04, -0.84, -0.03)
    points[sr.LEFT_EAR] = (0.09, -0.82, 0.0)
    points[sr.RIGHT_EAR] = (-0.09, -0.82, 0.0)

    for wrist, chains, sign in (
            (sr.LEFT_WRIST, sr.LEFT_HAND_CHAINS, 1.0),
            (sr.RIGHT_WRIST, sr.RIGHT_HAND_CHAINS, -1.0)):
        for finger_index, chain in enumerate(chains):
            parent = points[wrist].copy()
            for segment_index, child in enumerate(chain):
                parent = parent + np.array(
                    [sign * 0.045, (finger_index - 2) * 0.006,
                     segment_index * 0.004])
                points[child] = parent

    points[63:69] = points[sr.LEFT_ELBOW]
    return points


def sam_output(points=None):
    return {
        "joints": skeleton() if points is None else points,
        "camera": np.array([0.0, 0.0, 5.0]),
        "focal_length": np.array([800.0]),
    }


def full_keypoints(points=None, head_top=(0.0, -1.02, 0.0), head_top_index=184):
    body = skeleton() if points is None else points
    full = np.repeat(body[sr.NOSE][None, :], sr.MHR_KEYPOINT_COUNT, axis=0)
    full[:sr.MHR70_COUNT] = body
    full[head_top_index] = head_top
    return full


class SAM3DRetargetTests(unittest.TestCase):
    def test_reference_bone_length_and_driving_3d_direction_are_combined(self):
        reference = skeleton()
        driving = skeleton()
        reference[sr.LEFT_ELBOW] = reference[sr.LEFT_SHOULDER] + (0.48, 0.0, 0.0)
        driving[sr.LEFT_ELBOW] = driving[sr.LEFT_SHOULDER] + (0.0, 0.0, -0.20)
        driving[sr.LEFT_WRIST] = driving[sr.LEFT_ELBOW] + (0.0, 0.0, -0.20)

        output, details = sr.retarget_mhr70(
            reference, driving, reference_symmetry="off")

        expected_length = 0.48 * details["base_scale"]
        actual = output[sr.LEFT_ELBOW] - output[sr.LEFT_SHOULDER]
        self.assertAlmostEqual(np.linalg.norm(actual), expected_length, places=7)
        np.testing.assert_allclose(actual / np.linalg.norm(actual), (0.0, 0.0, -1.0))

    def test_reference_proportions_survive_large_scale_and_side_rotation(self):
        reference = skeleton()
        reference[sr.LEFT_SHOULDER] = (0.34, -0.58, 0.0)
        reference[sr.RIGHT_SHOULDER] = (-0.34, -0.58, 0.0)
        reference[sr.LEFT_HIP] = (0.11, 0.0, 0.0)
        reference[sr.RIGHT_HIP] = (-0.11, 0.0, 0.0)

        # A much larger driving person viewed from the side. The rotation
        # puts lateral and forward-reaching bones primarily on the Z axis.
        angle = np.deg2rad(88.0)
        rotation = np.array([
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ])
        driving = (skeleton() * 2.75) @ rotation.T

        output, details = sr.retarget_mhr70(
            reference, driving, reference_symmetry="off")

        for name, expected in details["reference_proportions"].items():
            self.assertAlmostEqual(
                details["generated_proportions"][name], expected,
                places=7, msg=name)

        driving_direction = (
            driving[sr.LEFT_ELBOW] - driving[sr.LEFT_SHOULDER])
        output_direction = (
            output[sr.LEFT_ELBOW] - output[sr.LEFT_SHOULDER])
        np.testing.assert_allclose(
            output_direction / np.linalg.norm(output_direction),
            driving_direction / np.linalg.norm(driving_direction),
        )

    def test_each_size_reference_matches_driving_unit_by_default(self):
        reference = skeleton()
        driving = skeleton() * 1.85

        for mode in ("torso", "shoulder_width", "body_height"):
            with self.subTest(mode=mode):
                output, details = sr.retarget_mhr70(
                    reference, driving, size_reference=mode,
                    reference_symmetry="off")

                self.assertAlmostEqual(
                    sr.body_unit(output, mode),
                    details["driving_unit"],
                    places=7,
                )

    def test_full_mhr_head_top_is_selected_along_head_axis(self):
        body = skeleton()
        full = full_keypoints(body)
        # A point far to the side must not win over the anatomical crown.
        full[200] = (3.0, -0.80, 0.0)
        output = sam_output(body)
        output["keypoints_3d_full"] = full

        head_top, index = sr.extract_head_top(output, body)

        self.assertEqual(index, 184)
        np.testing.assert_allclose(head_top, (0.0, -1.02, 0.0))

    def test_head_to_heel_uses_crown_and_both_heel_segments(self):
        body = skeleton()
        head_top = np.array((0.0, -1.02, 0.0))
        expected = (
            sr.body_unit(body, "body_height")
            + np.linalg.norm(head_top - body[sr.NOSE])
            + 0.5 * (
                np.linalg.norm(body[sr.LEFT_HEEL] - body[sr.LEFT_ANKLE])
                + np.linalg.norm(body[sr.RIGHT_HEEL] - body[sr.RIGHT_ANKLE])
            )
        )

        self.assertAlmostEqual(
            sr.body_unit(body, "head_to_heel", head_top), expected, places=7
        )

    def test_head_to_heel_normalizes_reference_to_driving_full_height(self):
        reference = skeleton()
        driving = skeleton() * 1.85
        reference_head_top = np.array((0.0, -1.02, 0.0))
        driving_head_top = reference_head_top * 1.85

        _, details = sr.retarget_mhr70(
            reference,
            driving,
            size_reference="head_to_heel",
            reference_symmetry="off",
            reference_head_top=reference_head_top,
            driving_head_top=driving_head_top,
        )

        self.assertAlmostEqual(details["base_scale"], 1.85, places=7)
        self.assertAlmostEqual(
            details["driving_unit"], details["reference_unit"] * 1.85,
            places=7,
        )

    def test_head_to_heel_requires_updated_full_mhr_output(self):
        with self.assertRaisesRegex(ValueError, "requires full MHR keypoints"):
            sr.extract_head_top(sam_output())

    def test_detailed_scales_control_each_reported_proportion(self):
        reference = skeleton()
        driving = skeleton()

        _, details = sr.retarget_mhr70(
            reference,
            driving,
            reference_symmetry="off",
            arm_scale=1.10,
            leg_scale=0.90,
            head_scale=1.20,
            torso_scale=1.30,
            shoulder_width_scale=0.80,
            hip_width_scale=1.15,
            neck_scale=0.75,
            upper_arm_scale=1.40,
            forearm_scale=0.60,
            thigh_scale=1.25,
            shin_scale=0.70,
        )

        reference_ratios = details["reference_proportions"]
        generated_ratios = details["generated_proportions"]
        expected_scales = {
            "torso": 1.30,
            "shoulder_width": 0.80,
            "hip_width": 1.15,
            "shoulder_to_nose": 1.20 * 0.75,
            "upper_arm": 1.10 * 1.40,
            "forearm": 1.10 * 0.60,
            "thigh": 0.90 * 1.25,
            "shin": 0.90 * 0.70,
        }
        for name, multiplier in expected_scales.items():
            self.assertAlmostEqual(
                generated_ratios[name],
                reference_ratios[name] * multiplier,
                places=7,
                msg=name,
            )

    def test_shoulder_center_to_nose_length_is_exact_for_tilted_head(self):
        reference = skeleton()
        driving = skeleton()
        driving[sr.NOSE] = (
            sr.shoulder_center(driving) + np.array([0.16, -0.08, -0.25]))

        output, details = sr.retarget_mhr70(
            reference, driving, reference_symmetry="off",
            head_scale=1.25, neck_scale=0.80)

        actual = np.linalg.norm(
            output[sr.NOSE] - sr.shoulder_center(output))
        expected = (
            details["reference_proportions"]["shoulder_to_nose"]
            * details["driving_unit"] * 1.25 * 0.80
        )
        self.assertAlmostEqual(actual, expected, places=7)

    def test_average_symmetry_equalizes_mirrored_bones(self):
        reference = skeleton()
        reference[sr.LEFT_ELBOW] = reference[sr.LEFT_SHOULDER] + (0.50, 0.0, 0.0)
        reference[sr.RIGHT_ELBOW] = reference[sr.RIGHT_SHOULDER] + (-0.30, 0.0, 0.0)

        lengths = sr.reference_lengths(reference, "average")

        self.assertAlmostEqual(lengths[sr.LEFT_ELBOW], 0.40, places=7)
        self.assertAlmostEqual(lengths[sr.RIGHT_ELBOW], 0.40, places=7)

    def test_longer_side_symmetry_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown reference symmetry"):
            sr.reference_lengths(skeleton(), "longer_side")

    def test_perspective_projection_matches_sam3d_camera_convention(self):
        points = np.zeros((70, 3), dtype=np.float64)
        points[0] = (1.0, 2.0, 0.0)

        projected, valid, depth = sr.project_mhr70(
            points, np.array([0.0, 0.0, 5.0]),
            np.array([100.0, 200.0]), 400, 600)

        self.assertTrue(valid[0])
        self.assertAlmostEqual(depth[0], 5.0)
        np.testing.assert_allclose(projected[0], (220.0, 380.0))

    def test_openpose_mapping_uses_absolute_pixel_coordinates(self):
        projected = np.array([(i * 2.0, i * 3.0) for i in range(70)])
        valid = np.ones(70, dtype=bool)

        output = sr.to_pose_keypoint(projected, valid, 200, 300)
        person = output[0]["people"][0]
        body = np.asarray(person["pose_keypoints_2d"]).reshape(18, 3)
        left_hand = np.asarray(person["hand_left_keypoints_2d"]).reshape(21, 3)

        np.testing.assert_allclose(body[2], (6 * 2, 6 * 3, 1.0))
        np.testing.assert_allclose(
            left_hand[0], (sr.LEFT_WRIST * 2,
                           sr.LEFT_WRIST * 3, 1.0))
        self.assertEqual(len(person["face_keypoints_2d"]), 210)
        self.assertFalse(any(person["face_keypoints_2d"]))

    def test_shrink_to_fit_does_not_move_an_in_bounds_pose(self):
        points = np.full((70, 2), 50.0)
        points[:, 0] = np.linspace(10.0, 90.0, 70)
        valid = np.ones(70, dtype=bool)

        fitted, scale = sr.fit_projected(
            points, valid, 100, 100, "shrink_to_fit", margin=40)

        self.assertEqual(scale, 1.0)
        np.testing.assert_array_equal(fitted, points)

    def test_node_accepts_sam3d_output_and_returns_pose_keypoint(self):
        node_class = load_package().NODE_CLASS_MAPPINGS["SAM3DBodyPoseRetarget"]
        node = node_class()
        image = np.zeros((1, 512, 384, 3), dtype=np.float32)

        output, driving_output, report = node.run(
            sam_output(), sam_output(), image,
            "torso", "average",
            1.0, 1.0, 1.0, 1.0, 1.0,
            "off", 16)

        self.assertEqual(output[0]["canvas_width"], 384)
        self.assertEqual(output[0]["canvas_height"], 512)
        person = output[0]["people"][0]
        self.assertEqual(len(person["pose_keypoints_2d"]), 18 * 3)
        self.assertEqual(len(person["hand_left_keypoints_2d"]), 21 * 3)
        self.assertEqual(driving_output[0]["canvas_width"], 384)
        self.assertEqual(driving_output[0]["canvas_height"], 512)
        self.assertIn("SAM 3D Body retargeted", report)
        self.assertIn("Ratios reference->generated", report)
        self.assertIn("shoulder_to_nose:", report)

    def test_node_uses_full_mhr_head_top_for_head_to_heel(self):
        node_class = load_package().NODE_CLASS_MAPPINGS["SAM3DBodyPoseRetarget"]
        node = node_class()
        image = np.zeros((1, 512, 384, 3), dtype=np.float32)
        reference = sam_output()
        driving = sam_output()
        reference["keypoints_3d_full"] = full_keypoints()
        driving["keypoints_3d_full"] = full_keypoints()

        _, _, report = node.run(
            reference, driving, image,
            "head_to_heel", "off",
            1.0, 1.0, 1.0, 1.0, 1.0,
            "off", 16)

        self.assertIn("normalized by head_to_heel", report)
        self.assertIn("reference=184, driving=184", report)

    def test_driving_output_is_direct_projection_without_fit_or_retarget(self):
        node_class = load_package().NODE_CLASS_MAPPINGS["SAM3DBodyPoseRetarget"]
        node = node_class()
        image = np.zeros((1, 512, 384, 3), dtype=np.float32)
        driving_points = skeleton()
        driving_points[sr.NOSE] = (0.45, -0.70, 0.25)
        driving_sam3d = sam_output(driving_points)

        _, driving_output, _ = node.run(
            sam_output(), driving_sam3d, image,
            "torso", "off",
            1.4, 0.8, 1.3, 0.7, 1.2,
            "fit_exactly", 80)

        expected, valid, _ = sr.project_mhr70(
            driving_points,
            driving_sam3d["camera"],
            np.array((800.0, 800.0)),
            384,
            512,
        )
        person = driving_output[0]["people"][0]
        body = np.asarray(person["pose_keypoints_2d"]).reshape(18, 3)
        np.testing.assert_allclose(body[0, :2], expected[sr.NOSE])
        self.assertEqual(body[0, 2], float(valid[sr.NOSE]))

    def test_node_schema_matches_existing_comfyui_sam3dbody_output_type(self):
        package = load_package()
        self.assertEqual(
            set(package.NODE_CLASS_MAPPINGS), {"SAM3DBodyPoseRetarget"})
        inputs = package.NODE_CLASS_MAPPINGS[
            "SAM3DBodyPoseRetarget"].INPUT_TYPES()["required"]

        self.assertEqual(inputs["reference_sam3d"], ("SAM3D_OUTPUT",))
        self.assertEqual(inputs["driving_sam3d"], ("SAM3D_OUTPUT",))
        self.assertEqual(inputs["driving_image"], ("IMAGE",))
        self.assertEqual(inputs["reference_symmetry"], (["average", "off"],))
        self.assertIn("head_to_heel", inputs["size_reference"][0])
        node_class = package.NODE_CLASS_MAPPINGS["SAM3DBodyPoseRetarget"]
        self.assertEqual(
            node_class.RETURN_NAMES,
            ("pose_keypoint", "driving_pose_keypoint", "report"),
        )
        for name in (
                "torso_scale", "shoulder_width_scale", "hip_width_scale",
                "neck_scale", "upper_arm_scale", "forearm_scale",
                "thigh_scale", "shin_scale"):
            self.assertEqual(inputs[name][1]["default"], 1.0)

    def test_invalid_joint_shape_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, r"shape \(70, 3\)"):
            sr.extract_mhr70({"joints": np.zeros((17, 3))})


if __name__ == "__main__":
    unittest.main()
