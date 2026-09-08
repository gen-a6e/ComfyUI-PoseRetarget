"""肩の前出しで幅が変わることと、4区間の骨長・方向・旧出力互換を検証。"""
import unittest
import numpy as np

from test_sam3d_retarget import skeleton, sam_output, load_package, sr


def fixture(protracted=False, left_length=.2):
    body = skeleton()
    rig = np.zeros((127, 3), dtype=float)
    rig[37] = (0., -.5, 0.)
    rig[38], rig[74] = (-.03, -.5, -.1), (.03, -.5, -.1)
    # 側方へ.20伸ばす肩と、同じ長さで前へ回した肩（.12/.16/.20の直角三角形）。
    right = np.array([-.12, 0., -.16]) if protracted else np.array([-.2, 0., 0.])
    left = np.array([.6, 0., -.8]) if protracted else np.array([1., 0., 0.])
    rig[39] = rig[38] + right
    rig[75] = rig[74] + left * left_length
    body[sr.RIGHT_SHOULDER], body[sr.LEFT_SHOULDER] = rig[39], rig[75]
    rig[126] = (0., -1.02, 0.)
    return body, rig


def run_node(reference, driving, ref_rig, drv_rig):
    node = load_package().NODE_CLASS_MAPPINGS['SAM3DBodyPoseRetarget']()
    ref, drv = sam_output(reference), sam_output(driving)
    ref['joint_coords'], drv['joint_coords'] = ref_rig, drv_rig
    return node.run(ref, drv, np.zeros((1,512,384,3)), 'off',
                    1., 1., 1., 1., 1., 'off', 16)


class ShoulderRigTests(unittest.TestCase):
    def test_identical_input_reproduces_all_points_without_mutating_inputs(self):
        body, rig = fixture(left_length=.25)
        before_body, before_rig = body.copy(), rig.copy()
        out, details = sr.retarget_mhr70(body, body, reference_symmetry='off',
                                        reference_rig=rig, driving_rig=rig)
        self.assertEqual(details['shoulder_mode'], 'rig_chain')
        np.testing.assert_allclose(out, body, atol=1e-12)
        for i in sr.SHOULDER_RIG_POINTS:
            np.testing.assert_allclose(details['shoulder_rig_points'][i], rig[i], atol=1e-12)
        np.testing.assert_array_equal(body, before_body)
        np.testing.assert_array_equal(rig, before_rig)

    def test_protraction_changes_span_without_changing_segment_lengths(self):
        ref, rr = fixture()
        drv, dr = fixture(protracted=True)
        out, details = sr.retarget_mhr70(ref, drv, reference_symmetry='off',
                                        reference_rig=rr, driving_rig=dr)
        # 長さ.20の左右鎖骨を前へ向けると、肩幅は.46から.30へ変わる。
        self.assertAlmostEqual(np.linalg.norm(out[5]-out[6]), .30)
        self.assertAlmostEqual(np.linalg.norm(ref[5]-ref[6]), .46)
        points = details['shoulder_rig_points']
        np.testing.assert_allclose(points[39]-points[37], (-.15,0.,-.26), atol=1e-12)
        np.testing.assert_allclose(points[75]-points[37], (.15,0.,-.26), atol=1e-12)
        for parent, child in sr.SHOULDER_RIG_EDGES:
            a, b = points[child]-points[parent], dr[child]-dr[parent]
            self.assertAlmostEqual(np.linalg.norm(a), np.linalg.norm(rr[child]-rr[parent]))
            np.testing.assert_allclose(a/np.linalg.norm(a),b/np.linalg.norm(b),atol=1e-12)

    def test_uniform_width_and_torso_scales_have_separate_domains(self):
        ref, rr = fixture()
        drv, dr = fixture(protracted=True)
        out, details = sr.retarget_mhr70(ref, drv, reference_symmetry='off',
            reference_rig=rr, driving_rig=dr, uniform_scale=1.3,
            shoulder_width_scale=.8, torso_scale=1.2)
        p=details['shoulder_rig_points']
        self.assertAlmostEqual(np.linalg.norm(p[37]-out[69]), .1*1.3*1.2)
        self.assertAlmostEqual(np.linalg.norm(out[5]-out[6]), .30*1.3*.8)
        for a,b in sr.SHOULDER_RIG_EDGES:
            self.assertAlmostEqual(np.linalg.norm(p[b]-p[a]),np.linalg.norm(rr[b]-rr[a])*1.3*.8)
        self.assertAlmostEqual(np.linalg.norm(out[7]-out[5]),np.linalg.norm(ref[7]-ref[5])*1.3)
        self.assertAlmostEqual(np.linalg.norm(out[1]-out[2]),np.linalg.norm(ref[1]-ref[2])*1.3)

    def test_symmetry_averages_corresponding_clavicle_segments(self):
        ref, rr = fixture(left_length=.3)
        drv, dr = fixture(protracted=True)
        _, details = sr.retarget_mhr70(ref,drv,reference_rig=rr,driving_rig=dr)
        p=details['shoulder_rig_points']
        self.assertAlmostEqual(np.linalg.norm(p[39]-p[38]),.25)
        self.assertAlmostEqual(np.linalg.norm(p[75]-p[74]),.25)
        _, details = sr.retarget_mhr70(ref,drv,reference_rig=rr,driving_rig=dr,reference_symmetry='off')
        p=details['shoulder_rig_points']
        self.assertAlmostEqual(np.linalg.norm(p[39]-p[38]),.20)
        self.assertAlmostEqual(np.linalg.norm(p[75]-p[74]),.30)

    def test_reference_lengths_ignore_driving_size_and_follow_rotation(self):
        ref, rr=fixture()
        drv, dr=fixture(protracted=True)
        rotation=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
        drv=drv@rotation.T*3.4+(3.,1.,2.)
        dr=dr@rotation.T*3.4+(3.,1.,2.)
        out, details=sr.retarget_mhr70(ref,drv,reference_rig=rr,driving_rig=dr,reference_symmetry='off')
        self.assertAlmostEqual(np.linalg.norm(out[5]-out[6]),.30)
        np.testing.assert_allclose(out[5]-out[6],np.array([.30,0.,0.])@rotation.T,atol=1e-12)
        np.testing.assert_allclose(out[0,[0,2]],drv[0,[0,2]],atol=1e-12)
        self.assertAlmostEqual(out[list(sr.ALIGNMENT_POINTS),1].max(),drv[list(sr.ALIGNMENT_POINTS),1].max())

    def test_bad_or_missing_rig_explicitly_falls_back_to_legacy_output(self):
        ref, rr=fixture();drv, dr=fixture(protracted=True)
        expected,_=sr.retarget_mhr70(ref,drv)
        nan=rr.copy();nan[38,0]=np.nan
        degenerate=rr.copy();degenerate[38]=degenerate[37]
        mismatch=rr.copy();mismatch[39,0]+=.05
        anchor=rr.copy();anchor[37]=ref[69]
        for bad in (None,np.zeros((126,3)),nan,degenerate,mismatch,anchor):
            with self.subTest(kind=str(type(bad))):
                out, details=sr.retarget_mhr70(ref,drv,reference_rig=bad,driving_rig=dr)
                self.assertEqual(details['shoulder_mode'],'legacy_width_fallback')
                self.assertTrue(details['shoulder_warning'])
                np.testing.assert_allclose(out,expected,atol=1e-12)
        out,details=sr.retarget_mhr70(ref,drv,reference_rig=rr,driving_rig=None)
        np.testing.assert_allclose(out,expected,atol=1e-12)
        self.assertIn('driving',details['shoulder_warning'])

    def test_unrelated_nan_r126_does_not_disable_shoulder_transfer(self):
        ref,rr=fixture();drv,dr=fixture(protracted=True)
        rr[126]=np.nan
        out,details=sr.retarget_mhr70(ref,drv,reference_rig=rr,driving_rig=dr)
        self.assertEqual(details['shoulder_mode'],'rig_chain')
        self.assertTrue(np.isfinite(out).all())
        _,report=run_node(ref,drv,rr,dr)
        self.assertIn('reference_height=unavailable',report)
        self.assertIn('shoulder_mode=rig_chain',report)

    def test_extracts_batched_raw_rig_as_an_independent_copy(self):
        _,rr=fixture()
        source={'raw_output':{'pred_joint_coords':rr[None]}}
        got=sr.extract_shoulder_rig(source)
        np.testing.assert_array_equal(got,rr)
        got[38]=999
        self.assertFalse(np.all(rr[38]==999))

    def test_node_uses_rig_and_keeps_internal_diagnostics(self):
        ref,rr=fixture();drv,dr=fixture(protracted=True)
        result=run_node(ref,drv,rr,dr)
        legacy=run_node(ref,drv,None,None)
        self.assertIn('shoulder_mode=rig_chain',result[1])
        self.assertIn('R37->R38:',result[1])
        self.assertIn('R74->R75:',result[1])
        self.assertIn('shoulder_mode=legacy_width_fallback',legacy[1])
        self.assertIn('WARNING: shoulder rig unavailable',legacy[1])
        self.assertNotEqual(result[0],legacy[0])


if __name__=='__main__':
    unittest.main()
