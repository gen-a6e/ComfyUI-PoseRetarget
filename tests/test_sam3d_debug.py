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
    def test_rig_scope_uses_internal_indices_and_parent_hierarchy(self):
        data = sam_output()
        data["joint_coords"] = np.arange(381).reshape(127, 3) / 100.
        points, names, colors, edges, _ = debug.debug_geometry(data, show_rig=True)
        selected = {names[i] for i in colors if names[i].startswith("R")}
        self.assertEqual(len(selected), 17)
        self.assertIn("R126: c_head_null", selected)
        self.assertNotIn("R37: c_spine3", selected)
        head, tip = names.index("R113: c_head"), names.index("R126: c_head_null")
        self.assertIn((tip, head), edges)
        np.testing.assert_array_equal(points[tip], data["joint_coords"][126])
        np.testing.assert_array_equal(points[:70], data["joints"])
        # 全身では親が範囲外だった首→上部脊椎も結び、rootは自分と結ばない。
        _, names, colors, edges, _ = debug.debug_geometry(data, show_rig=True, rig_scope="all", show_height=False)
        rig_edges = [(a, b) for a, b in edges if names[a].startswith("R")]
        self.assertEqual(len(rig_edges), 126)
        self.assertEqual(len([i for i in colors if names[i].startswith("R")]), 127)
        self.assertIn((names.index("R110: c_neck"), names.index("R37: c_spine3")), edges)
        self.assertTrue(all(a != b for a, b in rig_edges))

    def test_rig_projection_uses_same_camera_without_second_axis_flip_or_scaling(self):
        data = sam_output()
        # 表示対象以外をカメラ後方へ。検査点は手計算で確認できる座標にする。
        rig = np.tile([0., 0., -10.], (127, 1))
        rig[113] = [-.2, -.1, 0.]
        rig[126] = [.3, -.6, 1.]
        data["joint_coords"] = rig.copy()
        image = np.zeros((1, 512, 384, 3), dtype=np.float32)
        result, report = debug.render_debug(data, image, show_mesh=False, show_rig=True,
                                            **options_off())
        # (384/2 + 800*.3/6, 512/2 - 800*.6/6) = (232, 176)
        np.testing.assert_allclose(result[0, 176, 232], np.array([240, 255, 255])/255.)
        np.testing.assert_allclose(result[0, 240, 160], np.array([255, 120, 60])/255.)
        self.assertIn("R126: c_head_null | 232.00 176.00 | 6.0000 | visible", report)
        self.assertIn("R110: c_neck | 0.00 0.00 | -5.0000 | not_projectable", report)
        np.testing.assert_array_equal(data["joint_coords"], rig)
        self.assertFalse(image.any())

    def test_rig_connections_toggle_and_raw_singleton_fallback(self):
        data = sam_output()
        rig = np.tile([0., 0., -10.], (127, 1))
        rig[113], rig[126] = (0., -.2, 0.), (0., -1., 0.)
        data.pop("joint_coords")
        data["raw_output"]["pred_joint_coords"] = rig[None]
        opts = options_off()
        source = np.zeros((1, 512, 384, 3), np.float32)
        off, _ = debug.render_debug(data, source, show_mesh=False, show_rig=True, **opts)
        opts["show_connections"] = True
        on, _ = debug.render_debug(data, source, show_mesh=False, show_rig=True, **opts)
        self.assertFalse(off[0, 160, 192].any())
        self.assertTrue(on[0, 160, 192].any())

    def test_missing_or_unusable_rig_does_not_change_existing_overlays(self):
        data = mesh_sample()
        source = np.zeros((1, 512, 384, 3), np.float32)
        expected, _ = debug.render_debug(data, source, show_height=False)
        for value in (None, np.zeros((126, 3)), np.zeros((128, 3)),
                      np.zeros((2, 127, 3)), np.full((127, 3), np.nan), "invalid"):
            with self.subTest(shape=np.shape(value)):
                data["joint_coords"] = value
                result, report = debug.render_debug(data, source, show_rig=True, show_height=False)
                np.testing.assert_array_equal(result, expected)
                self.assertIn("WARNING: internal rig unavailable", report)
                disabled, report = debug.render_debug(data, source, show_rig=False, show_height=False)
                np.testing.assert_array_equal(disabled, expected)
                self.assertNotIn("internal rig unavailable", report)

    def test_rig_and_dense_landmark_indices_do_not_collide(self):
        data = sam_output()
        data["joint_coords"] = np.zeros((127, 3))
        points, names, colors, edges, _ = debug.debug_geometry(
            data, show_rig=True, rig_scope="all", show_head_candidates=True)
        landmark = names.index("126: dense_head_landmark")
        rig_tip = names.index("R126: c_head_null")
        self.assertNotEqual(landmark, rig_tip)
        self.assertIn(landmark, colors)
        self.assertIn(rig_tip, colors)
        self.assertIn((rig_tip, names.index("R113: c_head")), edges)
        np.testing.assert_array_equal(points[landmark], data["keypoints_3d_full"][126])

    def test_height_only_displays_exact_measurement_path_and_no_duplicate_r126(self):
        data = sam_output()
        del data["keypoints_3d_full"]
        _, names, colors, edges, notes = debug.debug_geometry(
            data, show_body=False, show_face=False, show_left_hand=False,
            show_right_hand=False, show_height=True)
        top = names.index("R126: c_head_null")
        self.assertEqual(set(colors), {top, 69, 71, 9, 11, 13, 17, 10, 12, 14, 20})
        self.assertEqual({frozenset(edge) for edge in edges}, {
            frozenset(edge) for edge in ((top,69), (69,71), (9,11), (11,13), (13,17),
                                        (10,12), (12,14), (14,20))})
        self.assertTrue(any("head_top_source=R126" in note for note in notes))
        _, names, colors, _, _ = debug.debug_geometry(data, show_rig=True, show_height=True)
        self.assertEqual(sum(names[i] == "R126: c_head_null" for i in colors), 1)

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
        self.assertEqual("R126: c_head_null", names[72])
        self.assertEqual(len(points), 73)

    def test_group_toggles_and_missing_optional_data(self):
        data = sam_output()
        data.pop("joint_coords")
        opts = options_off()
        opts.update(show_left_hand=True, show_height=True)
        image, report = debug.render_debug(data, np.zeros((1, 512, 384, 3)), **opts)
        self.assertTrue(image.any())
        self.assertIn("height unavailable", report)
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
        self.assertIn("307: dense_head_landmark", report)
        self.assertTrue(torch.isfinite(result).all())
        inputs = list(node.INPUT_TYPES()["required"])
        self.assertEqual(inputs[-2:], ["show_rig", "rig_scope"])


if __name__ == "__main__":
    unittest.main()
