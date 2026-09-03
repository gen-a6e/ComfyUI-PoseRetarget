import sys
import types
import unittest
import importlib.util
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pose_retarget as pr  # noqa: E402


def load_node_class():
    root = Path(__file__).resolve().parents[1]
    package_name = "comfyui_pose_retarget_test_package"
    if "torch" not in sys.modules:
        sys.modules["torch"] = types.SimpleNamespace(from_numpy=lambda value: value)
    spec = importlib.util.spec_from_file_location(
        package_name, root / "__init__.py",
        submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    return package.NODE_CLASS_MAPPINGS["PoseRetargetProportions"]


def body_with_unit(unit, face_span=20.0):
    body = np.zeros((18, 3), dtype=np.float32)
    body[:, 2] = 1.0
    body[pr.ROOT, :2] = (0.0, 0.0)
    body[pr.R_HIP, :2] = (-10.0, unit)
    body[pr.L_HIP, :2] = (10.0, unit)
    body[pr.NOSE, :2] = (0.0, -face_span)
    body[pr.R_EYE, :2] = (-face_span / 4.0, -face_span * 1.25)
    body[pr.L_EYE, :2] = (face_span / 4.0, -face_span * 1.25)
    body[pr.R_EAR, :2] = (-face_span / 2.0, -face_span)
    body[pr.L_EAR, :2] = (face_span / 2.0, -face_span)
    return body


def dense_face(width, height):
    xs = np.array([-0.5, -0.25, 0.0, 0.25, 0.5], dtype=np.float32) * width
    ys = np.array([-0.5, 0.0, 0.5], dtype=np.float32) * height
    points = np.array([(x, y, 1.0) for y in ys for x in xs],
                      dtype=np.float32)
    return points


class HeadSizeTests(unittest.TestCase):
    def test_dense_face_ratio_is_transferred_to_current_body(self):
        source_body = body_with_unit(100.0)
        current_body = body_with_unit(200.0)
        source_face = dense_face(40.0, 60.0)
        current_face = dense_face(40.0, 60.0)

        factor, method, details = pr.head_correction_factor(
            source_body, source_face, current_body, current_face, "torso")

        self.assertEqual(method, "face_landmarks")
        self.assertAlmostEqual(factor, 2.0, places=5)
        self.assertAlmostEqual(
            details["target_size"] / 2.0, details["current_size"], places=5)

    def test_manual_head_scale_multiplies_transferred_ratio(self):
        source_body = body_with_unit(100.0)
        current_body = body_with_unit(100.0)

        factor, method, _ = pr.head_correction_factor(
            source_body, dense_face(40.0, 60.0),
            current_body, dense_face(40.0, 60.0),
            "torso", manual_scale=1.25)

        self.assertEqual(method, "face_landmarks")
        self.assertAlmostEqual(factor, 1.25, places=5)

    def test_ear_span_is_used_when_dense_landmarks_are_missing(self):
        source_body = body_with_unit(100.0, face_span=40.0)
        current_body = body_with_unit(100.0, face_span=20.0)

        factor, method, _ = pr.head_correction_factor(
            source_body, None, current_body, None, "torso")

        self.assertEqual(method, "ear_span")
        self.assertAlmostEqual(factor, 2.0, places=5)

    def test_face_scaling_keeps_neck_and_nose_fixed(self):
        body = body_with_unit(100.0, face_span=20.0)
        neck_before = body[pr.ROOT, :2].copy()
        nose_before = body[pr.NOSE, :2].copy()
        eye_offset_before = np.linalg.norm(
            body[pr.R_EYE, :2] - body[pr.NOSE, :2])

        scaled = pr.scale_head_keypoints(body, 1.5)

        np.testing.assert_array_equal(scaled[pr.ROOT, :2], neck_before)
        np.testing.assert_array_equal(scaled[pr.NOSE, :2], nose_before)
        eye_offset_after = np.linalg.norm(
            scaled[pr.R_EYE, :2] - scaled[pr.NOSE, :2])
        self.assertAlmostEqual(
            eye_offset_after, eye_offset_before * 1.5, places=5)

    def test_neck_to_nose_fallback_scales_the_whole_head_chain(self):
        body = body_with_unit(100.0, face_span=20.0)
        neck_before = body[pr.ROOT, :2].copy()
        nose_distance_before = np.linalg.norm(
            body[pr.NOSE, :2] - body[pr.ROOT, :2])

        scaled = pr.scale_head_keypoints(
            body, 1.5, include_neck_to_nose=True)

        np.testing.assert_array_equal(scaled[pr.ROOT, :2], neck_before)
        nose_distance_after = np.linalg.norm(
            scaled[pr.NOSE, :2] - scaled[pr.ROOT, :2])
        self.assertAlmostEqual(
            nose_distance_after, nose_distance_before * 1.5, places=5)

    def test_foreshortening_does_not_shrink_head_joints(self):
        body = body_with_unit(100.0, face_span=2.0)
        lengths = pr._bone_lengths(body)

        factors = pr.foreshortening(
            body, lengths, 100.0, "symmetry_and_canonical", 0.15, 0.75)

        for joint in pr.HEAD_JOINTS:
            self.assertNotIn(joint, factors)

    def test_node_transfers_reference_face_to_body_ratio(self):
        reference_body = body_with_unit(100.0, face_span=20.0)
        driving_body = body_with_unit(200.0, face_span=20.0)
        reference_face = dense_face(80.0, 100.0)
        driving_face = dense_face(20.0, 30.0)
        reference_face[:, :2] += reference_body[pr.NOSE, :2]
        driving_face[:, :2] += driving_body[pr.NOSE, :2]

        def frame(body, face):
            return [{
                "canvas_width": 512,
                "canvas_height": 512,
                "people": [{
                    "pose_keypoints_2d": body.reshape(-1).tolist(),
                    "face_keypoints_2d": face.reshape(-1).tolist(),
                }],
            }]

        node = load_node_class()()
        output, report = node.run(
            frame(reference_body, reference_face),
            frame(driving_body, driving_face),
            "torso", "off", "reference", "neck",
            1.0, 1.0, 1.0, 1.0,
            "off", 0.15, 0.75,
            "off", 16, 0, -1)

        person = output[0]["people"][0]
        output_body = np.asarray(
            person["pose_keypoints_2d"], dtype=np.float32).reshape(-1, 3)
        output_face = np.asarray(
            person["face_keypoints_2d"], dtype=np.float32).reshape(-1, 3)
        reference_ratio = (
            pr._face_extent(reference_face)
            / pr._body_unit(reference_body, "torso"))
        output_ratio = (
            pr._face_extent(output_face)
            / pr._body_unit(output_body, "torso"))
        output_neck_to_nose = np.linalg.norm(
            output_body[pr.NOSE, :2] - output_body[pr.ROOT, :2])

        self.assertAlmostEqual(output_ratio, reference_ratio, places=5)
        self.assertAlmostEqual(output_neck_to_nose, 40.0, places=5)
        self.assertIn("metric=face_landmarks", report)


if __name__ == "__main__":
    unittest.main()
