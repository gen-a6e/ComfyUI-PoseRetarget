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

    def test_average_symmetry_equalizes_mirrored_bones(self):
        reference = skeleton()
        reference[sr.LEFT_ELBOW] = reference[sr.LEFT_SHOULDER] + (0.50, 0.0, 0.0)
        reference[sr.RIGHT_ELBOW] = reference[sr.RIGHT_SHOULDER] + (-0.30, 0.0, 0.0)

        lengths = sr.reference_lengths(reference, "average")

        self.assertAlmostEqual(lengths[sr.LEFT_ELBOW], 0.40, places=7)
        self.assertAlmostEqual(lengths[sr.RIGHT_ELBOW], 0.40, places=7)

    def test_perspective_projection_matches_sam3d_camera_convention(self):
        points = np.zeros((70, 3), dtype=np.float64)
        points[0] = (1.0, 2.0, 0.0)

        projected, valid, depth = sr.project_mhr70(
            points, np.array([0.0, 0.0, 5.0]),
            np.array([100.0, 200.0]), 400, 600)

        self.assertTrue(valid[0])
        self.assertAlmostEqual(depth[0], 5.0)
        np.testing.assert_allclose(projected[0], (220.0, 380.0))

    def test_openpose_mapping_uses_normalized_body_and_hand_coordinates(self):
        projected = np.array([(i * 2.0, i * 3.0) for i in range(70)])
        valid = np.ones(70, dtype=bool)

        output = sr.to_pose_keypoint(projected, valid, 200, 300)
        person = output[0]["people"][0]
        body = np.asarray(person["pose_keypoints_2d"]).reshape(18, 3)
        left_hand = np.asarray(person["hand_left_keypoints_2d"]).reshape(21, 3)

        np.testing.assert_allclose(body[2], (6 * 2 / 200, 6 * 3 / 300, 1.0))
        np.testing.assert_allclose(
            left_hand[0], (sr.LEFT_WRIST * 2 / 200,
                           sr.LEFT_WRIST * 3 / 300, 1.0))
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

        output, report = node.run(
            sam_output(), sam_output(), image,
            "torso", "average",
            1.0, 1.0, 1.0, 1.0, 1.0,
            "off", 16)

        self.assertEqual(output[0]["canvas_width"], 384)
        self.assertEqual(output[0]["canvas_height"], 512)
        person = output[0]["people"][0]
        self.assertEqual(len(person["pose_keypoints_2d"]), 18 * 3)
        self.assertEqual(len(person["hand_left_keypoints_2d"]), 21 * 3)
        self.assertIn("SAM 3D Body retargeted", report)

    def test_node_schema_matches_existing_comfyui_sam3dbody_output_type(self):
        inputs = load_package().NODE_CLASS_MAPPINGS[
            "SAM3DBodyPoseRetarget"].INPUT_TYPES()["required"]

        self.assertEqual(inputs["reference_sam3d"], ("SAM3D_OUTPUT",))
        self.assertEqual(inputs["driving_sam3d"], ("SAM3D_OUTPUT",))
        self.assertEqual(inputs["driving_image"], ("IMAGE",))

    def test_invalid_joint_shape_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, r"shape \(70, 3\)"):
            sr.extract_mhr70({"joints": np.zeros((17, 3))})


if __name__ == "__main__":
    unittest.main()
