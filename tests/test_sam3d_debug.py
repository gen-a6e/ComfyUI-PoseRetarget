"""描画の整合性を、投影位置・遮蔽・入力不変性で検証する。"""
import importlib
import unittest

import numpy as np

from test_sam3d_retarget import load_package, sam_output

package = load_package()
debug = importlib.import_module(package.__name__ + ".sam3d_debug")


def options_off():
    return dict(show_body=False, show_face=False, show_left_hand=False,
                show_right_hand=False, show_height=False, show_connections=False,
                labels="off")


def mesh_sample():
    data = sam_output()
    # z=5、f=800、384x512なので、原点は(192,256)に投影される。
    data["vertices"] = np.array([[-.2, -.2, 0], [.2, -.2, 0], [0, .2, 0]])
    data["faces"] = np.array([[0, 1, 2]])
    return data


class DebugTests(unittest.TestCase):
    def test_mesh_and_joint_overlay_share_pixels_without_transforming_input(self):
        data = mesh_sample()
        data["joints"][69] = (0, 0, 0)
        data["joints"][[9, 10], 1] = .5  # 腰中央のマーカーと首を重ねない。
        original = data["joints"].copy()
        image = np.zeros((1, 512, 384, 3), dtype=np.float32)
        rendered, report = debug.render_debug(data, image, **options_off())
        self.assertGreater(rendered[0, 256, 192].sum(), 0)
        np.testing.assert_array_equal(rendered[0, 10, 10], [0, 0, 0])
        options = options_off()
        options["show_body"] = True
        marked, _ = debug.render_debug(data, image, **options)
        np.testing.assert_allclose(marked[0, 256, 192], np.array([255, 65, 65])/255)
        np.testing.assert_array_equal(data["joints"], original)
        self.assertFalse(image.any())
        self.assertEqual(marked.shape, image.shape)
        self.assertEqual(marked.dtype, np.float32)
        self.assertIn("mesh_pixels=", report)

    def test_mesh_occlusion_is_order_independent_and_opacity_applied_once(self):
        data = mesh_sample()
        rgb = np.zeros((512, 384, 3), np.float32)
        # 奥の面を傾けて陰影も変える。前後を間違えると色が変わる条件にする。
        depths = np.array([6., 8., 7.])
        back = data["vertices"].copy()
        back[:, :2] *= depths[:, None]/5
        back[:, 2] = depths-5
        verts = np.concatenate((data["vertices"], back))
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        camera, focal = debug.sr.extract_camera(data)
        first, _, _ = debug.mesh_overlay(rgb, verts, faces, camera, focal, .5)
        reverse, _, _ = debug.mesh_overlay(rgb, verts, faces[::-1], camera, focal, .5)
        np.testing.assert_array_equal(first, reverse)
        np.testing.assert_allclose(first[256, 192], np.array([.48, .76, .94])*.5)

    def test_virtual_centers_are_projected_from_3d_not_averaged_in_2d(self):
        data = sam_output()
        data["joints"][5] = (1, 0, -1)
        data["joints"][6] = (-1, 0, 1)
        points, _, _, _, _ = debug.debug_geometry(data)
        camera, focal = debug.sr.extract_camera(data)
        xy, _, _ = debug.sr.project_mhr70(points, camera, focal, 384, 512)
        np.testing.assert_allclose(xy[70], [192, 256])
        self.assertGreater(np.linalg.norm((xy[5]+xy[6])/2-xy[70]), 1)

    def test_all_mhr_points_and_derived_points_can_be_selected(self):
        data = sam_output()
        points, names, colors, _, _ = debug.debug_geometry(data, show_auxiliary=True)
        self.assertTrue(set(range(72)).issubset(colors))
        self.assertEqual(names[69], "69: neck")
        self.assertEqual(names[24], "24: right_thumb_third_joint")
        self.assertEqual(names[45], "45: left_thumb_third_joint")
        self.assertIn("184: selected_head_top", names[72])
        self.assertEqual(len(points), 73)

    def test_group_toggles_and_missing_optional_data(self):
        data = sam_output()
        data.pop("keypoints_3d_full")
        opts = options_off()
        opts.update(show_left_hand=True, show_height=True)
        image, report = debug.render_debug(data, np.zeros((1, 512, 384, 3)), **opts)
        self.assertTrue(image.any())
        self.assertIn("height/head candidates unavailable", report)
        self.assertIn("mesh unavailable", report)
        self.assertNotIn("21: right_thumb_tip |", report)
        self.assertIn("42: left_thumb_tip |", report)
        data["vertices"] = np.zeros((3, 3))
        data["faces"] = np.array([[0, -1, 2]])
        _, report = debug.render_debug(data, np.zeros((1, 512, 384, 3)), **opts)
        self.assertIn("valid integer vertex indices", report)

    def test_behind_camera_mesh_is_skipped_and_batch_is_rejected(self):
        data = mesh_sample()
        data["vertices"][:, 2] = -6
        image, report = debug.render_debug(data, np.zeros((1, 512, 384, 3)), **options_off())
        self.assertFalse(image.any())
        self.assertIn("faces_skipped=1", report)
        with self.assertRaisesRegex(ValueError, "one IMAGE"):
            debug.render_debug(data, np.zeros((2, 512, 384, 3)))

    def test_node_registration_labels_candidates_and_torch_image(self):
        import torch
        node = package.NODE_CLASS_MAPPINGS["SAM3DBodySkeletonDebug"]()
        result, report = node.run(mesh_sample(), torch.zeros((1, 512, 384, 3)),
                                  show_head_candidates=True, show_auxiliary=True)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(tuple(result.shape), (1, 512, 384, 3))
        self.assertIn("307: head_candidate", report)
        self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
