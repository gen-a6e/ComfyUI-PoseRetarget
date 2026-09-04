import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rtmw3d_inspector as inspector  # noqa: E402


def synthetic_outputs():
    keypoints = np.zeros((1, inspector.KEYPOINT_COUNT, 3), dtype=np.float32)
    keypoints_2d = np.zeros(
        (1, inspector.KEYPOINT_COUNT, 2), dtype=np.float32)
    scores = np.ones((1, inspector.KEYPOINT_COUNT), dtype=np.float32)
    keypoints_2d[0, :, 0] = np.linspace(100.0, 300.0,
                                               inspector.KEYPOINT_COUNT)
    keypoints_2d[0, :, 1] = np.linspace(50.0, 450.0,
                                               inspector.KEYPOINT_COUNT)
    return keypoints, scores, keypoints_2d


class RTMW3DInspectorTests(unittest.TestCase):
    def test_coco_wholebody_schema_has_133_named_points(self):
        self.assertEqual(len(inspector.KEYPOINT_NAMES), 133)
        self.assertEqual(inspector.KEYPOINT_NAMES[0], "nose")
        self.assertEqual(inspector.KEYPOINT_NAMES[90], "face-67")
        self.assertEqual(inspector.KEYPOINT_NAMES[91], "left_hand_root")
        self.assertEqual(inspector.KEYPOINT_NAMES[112], "right_hand_root")
        self.assertTrue(all(
            0 <= a < 133 and 0 <= b < 133
            for a, b in inspector.ALL_EDGES))

    def test_depth_is_centered_on_hip_midpoint(self):
        z = np.zeros(133, dtype=np.float32)
        z[11] = 2.0
        z[12] = 4.0
        z[9] = 1.0
        centered, root, method = inspector.center_relative_depth(
            z, np.ones(133, dtype=np.float32), 0.3)

        self.assertEqual(method, "hip_midpoint")
        self.assertAlmostEqual(root, 3.0)
        self.assertAlmostEqual(float(centered[11]), -1.0)
        self.assertAlmostEqual(float(centered[12]), 1.0)
        self.assertAlmostEqual(float(centered[9]), -2.0)

    def test_low_confidence_hips_use_visible_body_median(self):
        z = np.arange(133, dtype=np.float32)
        scores = np.zeros(133, dtype=np.float32)
        scores[[0, 1, 2]] = 1.0
        centered, root, method = inspector.center_relative_depth(
            z, scores, 0.3)

        self.assertEqual(method, "visible_body_median")
        self.assertAlmostEqual(root, 1.0)
        self.assertAlmostEqual(float(centered[0]), -1.0)

    def test_payload_marks_negative_z_as_nearer(self):
        keypoints, scores, keypoints_2d = synthetic_outputs()
        keypoints[0, 11, 2] = 0.0
        keypoints[0, 12, 2] = 0.0
        keypoints[0, 9, 2] = -0.5
        keypoints[0, 10, 2] = 0.5

        payload = inspector.build_payload(
            Path("pose.png"), 400, 500,
            keypoints, scores, keypoints_2d)

        self.assertEqual(payload["keypoints"][9]["depth"], "nearer")
        self.assertEqual(payload["keypoints"][10]["depth"], "farther")
        self.assertEqual(payload["keypoints"][11]["depth"], "root_plane")
        self.assertEqual(payload["summary"]["nearest"][0]["index"], 9)
        self.assertEqual(payload["summary"]["farthest"][0]["index"], 10)
        self.assertEqual(len(payload["keypoints"]), 133)

    def test_relative_depth_converts_to_native_model_pixels(self):
        z = np.array([
            -inspector.MODEL_Z_RANGE_M, 0.0, inspector.MODEL_Z_RANGE_M,
        ], dtype=np.float32)
        pixels = inspector.relative_depth_to_model_pixels(z)

        np.testing.assert_allclose(
            pixels, [-inspector.MODEL_DEPTH_SIZE / 2, 0.0,
                     inspector.MODEL_DEPTH_SIZE / 2], atol=1e-5)

    def test_inference_shapes_are_validated(self):
        keypoints, scores, keypoints_2d = synthetic_outputs()
        points_3d, confidences, points_2d = (
            inspector.normalize_inference_arrays(
                keypoints, scores[..., None], keypoints_2d))
        self.assertEqual(points_3d.shape, (1, 133, 3))
        self.assertEqual(confidences.shape, (1, 133))
        self.assertEqual(points_2d.shape, (1, 133, 2))

        with self.assertRaisesRegex(ValueError, "expected at least 133"):
            inspector.normalize_inference_arrays(
                keypoints[:, :17], scores[:, :17], keypoints_2d[:, :17])

    def test_html_payload_escapes_script_terminator(self):
        encoded = inspector._json_for_html({"path": "</script><b>bad</b>"})
        self.assertNotIn("</script>", encoded)
        self.assertEqual(
            json.loads(encoded.replace("<\\/", "</"))["path"],
            "</script><b>bad</b>")

    def test_output_directory_sanitizes_filename(self):
        output = inspector.output_directory(
            Path("reports"), Path("my pose (front).png"))
        self.assertEqual(output, Path("reports/my_pose__front_"))

    def test_viewer_uses_fixed_native_model_grid_projections(self):
        viewer = (Path(__file__).resolve().parents[1]
                  / "tools" / "rtmw3d_viewer.html").read_text(
                      encoding="utf-8")
        self.assertIn("available / (frameRadius * 2)", viewer)
        self.assertIn("point.model_x_px - center.model_x_px", viewer)
        self.assertIn("center.model_y_px - point.model_y_px", viewer)
        self.assertIn("point.z_model_px - center.z_model_px", viewer)
        self.assertIn("if (viewMode === 'side') return {x:z, y}", viewer)
        self.assertIn("if (viewMode === 'top') return {x, y:z}", viewer)
        self.assertNotIn("function rotate", viewer)

    def test_viewer_rotates_around_torso_instead_of_visible_bounds(self):
        viewer = (Path(__file__).resolve().parents[1]
                  / "tools" / "rtmw3d_viewer.html").read_text(
                      encoding="utf-8")
        self.assertIn("averagePoint([averagePoint(hips), averagePoint(shoulders)])", viewer)
        self.assertNotIn("camera_x_m", viewer)

    def test_viewer_keeps_original_depth_order_without_drag_rotation(self):
        viewer = (Path(__file__).resolve().parents[1]
                  / "tools" / "rtmw3d_viewer.html").read_text(
                      encoding="utf-8")
        self.assertIn("b.point.z_relative_m - a.point.z_relative_m", viewer)
        self.assertNotIn("'pointermove'", viewer)

    def test_payload_exports_native_model_grid_without_fake_camera_xyz(self):
        keypoints, scores, keypoints_2d = synthetic_outputs()
        keypoints[0, :, 0] = 123.0
        keypoints[0, :, 1] = 234.0
        keypoints[0, 9, 2] = 1.0
        payload = inspector.build_payload(
            Path("pose.png"), 400, 500,
            keypoints, scores, keypoints_2d)

        point = payload["keypoints"][9]
        self.assertEqual(payload["schema"], "rtmw3d-inspector-v2")
        self.assertEqual(point["model_x_px"], 123.0)
        self.assertEqual(point["model_y_px"], 234.0)
        self.assertAlmostEqual(
            point["z_model_px"], inspector.MODEL_DEPTH_PIXELS_PER_M,
            places=5)
        self.assertNotIn("camera_x_m", point)

    def test_viewer_tracks_post_layout_canvas_size(self):
        viewer = (Path(__file__).resolve().parents[1]
                  / "tools" / "rtmw3d_viewer.html").read_text(
                      encoding="utf-8")
        self.assertIn("new ResizeObserver(resizeCanvas).observe(canvas)", viewer)
        self.assertIn("window.addEventListener('load', resizeCanvas)", viewer)


if __name__ == "__main__":
    unittest.main()
