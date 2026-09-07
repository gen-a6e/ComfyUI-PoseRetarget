"""中心リグの骨長と曲率、R113基準の顔、倍率の分離を検証する。"""
import unittest
import numpy as np
from test_sam3d_retarget import skeleton, sam_output, load_package, sr


def fixture(bent=False):
    body = skeleton()
    rig = np.zeros((127, 3))
    rig[1] = (0., -.02, .03)  # 腰中央Hとは意図的に異なる位置。
    vectors = [(0., -.1, 0.)] * 4
    if bent:
        vectors = [(0., -.1, 0.), (0., -.06, -.08),
                   (0., -.06, -.08), (0., -.1, 0.)]
    vectors += [(0., -.1, 0.), (0., -.12, 0.), (0., -.2, 0.)]
    for (parent, child), vector in zip(sr.CENTER_RIG_EDGES, vectors):
        rig[child] = rig[parent] + vector
    rig[38] = rig[37] + (-.035, 0., -.08)
    rig[74] = rig[37] + (.035, 0., -.08)
    rig[39] = rig[38] + (-.20, .02, -.02)
    rig[75] = rig[74] + (.20, .02, -.02)
    body[6], body[5] = rig[39], rig[75]
    body[69] = rig[110] + (0., -.04, -.005)
    face = np.array([(0., -.03, -.1), (.04, -.06, -.105),
                     (-.04, -.06, -.105), (.09, -.04, -.02), (-.09, -.04, -.02)])
    body[:5] = rig[113] + face
    return body, rig


def retarget(ref, drv, rr, dr, **kw):
    return sr.retarget_mhr70(ref, drv, reference_rig=rr, driving_rig=dr,
                            reference_symmetry='off', **kw)


def node_result(ref, drv, rr, dr):
    a, b = sam_output(ref), sam_output(drv)
    a['joint_coords'], b['joint_coords'] = rr, dr
    node = load_package().NODE_CLASS_MAPPINGS['SAM3DBodyPoseRetarget']()
    return node.run(a, b, np.zeros((1, 512, 384, 3)), 'off',
                    1., 1., 1., 1., 1., 'off', 16)


class CenterRigTests(unittest.TestCase):
    def test_self_reproduces_mhr70_and_center_and_shoulder_rigs(self):
        ref, rr = fixture(bent=True)
        before, before_rig = ref.copy(), rr.copy()
        out, d = retarget(ref, ref, rr, rr)
        self.assertEqual(d['center_mode'], 'rig_chain')
        self.assertEqual(d['face_anchor'], 'R113')
        np.testing.assert_allclose(out, ref, atol=1e-12)
        for points in (d['center_rig_points'], d['shoulder_rig_points']):
            for i, point in points.items():
                np.testing.assert_allclose(point, rr[i], atol=1e-12)
        np.testing.assert_array_equal(ref, before)
        np.testing.assert_array_equal(rr, before_rig)

    def test_bent_spine_keeps_each_length_not_endpoint_chord(self):
        ref, rr = fixture()
        drv, dr = fixture(bent=True)
        out, d = retarget(ref, drv, rr, dr)
        p = d['center_rig_points']
        np.testing.assert_allclose(p[37]-p[1], (0., -.32, -.16), atol=1e-12)
        self.assertAlmostEqual(sum(np.linalg.norm(p[b]-p[a]) for a,b in sr.CENTER_RIG_EDGES[:4]), .4)
        self.assertLess(np.linalg.norm(p[37]-p[1]), .4)
        for a,b in sr.CENTER_RIG_EDGES:
            self.assertAlmostEqual(np.linalg.norm(p[b]-p[a]), np.linalg.norm(rr[b]-rr[a]))
            actual = (p[b]-p[a])/np.linalg.norm(p[b]-p[a])
            expected = (dr[b]-dr[a])/np.linalg.norm(dr[b]-dr[a])
            np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_allclose(d['shoulder_rig_points'][37], p[37], atol=1e-12)
        self.assertIsNone(d['shoulder_anchor_length'])
        # 腰・四肢の長さは中心リグの曲がり方で変えない。
        for key in ('hip_width','upper_arm','forearm','thigh','shin'):
            self.assertAlmostEqual(d['reference_measurements'][key], d['generated_measurements'][key])

    def test_scale_domains_and_virtual_offsets(self):
        ref, rr = fixture(); drv, dr = fixture(bent=True)
        out, d = retarget(ref,drv,rr,dr,uniform_scale=1.3,torso_scale=1.2,
                          neck_scale=.8,head_scale=1.1,shoulder_width_scale=.9)
        p = d['center_rig_points']
        for a,b in sr.CENTER_RIG_EDGES:
            part = 1.2 if b in (34,35,36,37) else .8 if b in (110,113) else 1.1
            self.assertAlmostEqual(np.linalg.norm(p[b]-p[a]),np.linalg.norm(rr[b]-rr[a])*1.3*part)
        self.assertAlmostEqual(np.linalg.norm(p[1]-sr.hip_center(out)),np.linalg.norm(rr[1]-sr.hip_center(ref))*1.3)
        self.assertAlmostEqual(np.linalg.norm(out[69]-p[110]),np.linalg.norm(ref[69]-rr[110])*1.3*.8)
        for a,b in sr.SHOULDER_RIG_EDGES:
            q=d['shoulder_rig_points']
            self.assertAlmostEqual(np.linalg.norm(q[b]-q[a]),np.linalg.norm(rr[b]-rr[a])*1.3*.9)
        for i in sr.FACE_POINTS:
            self.assertAlmostEqual(np.linalg.norm(out[i]-p[113]),np.linalg.norm(ref[i]-rr[113])*1.3*1.1)
        self.assertAlmostEqual(np.linalg.norm(out[1]-out[2]),.08*1.3*1.1)

    def test_faces_use_one_rotation_about_head_despite_driving_size(self):
        ref, rr = fixture(); drv, dr = fixture(bent=True)
        rotation=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
        drv=drv@rotation.T*2.4+(1.,2.,3.)
        dr=dr@rotation.T*2.4+(1.,2.,3.)
        out,d=retarget(ref,drv,rr,dr)
        p=d['center_rig_points']
        np.testing.assert_allclose(out[:5]-p[113],(ref[:5]-rr[113])@rotation.T,atol=1e-12)
        for a,b in sr.CENTER_RIG_EDGES:
            self.assertAlmostEqual(np.linalg.norm(p[b]-p[a]),np.linalg.norm(rr[b]-rr[a]))
        np.testing.assert_allclose(out[0,[0,2]],drv[0,[0,2]],atol=1e-12)
        self.assertAlmostEqual(out[list(sr.ALIGNMENT_POINTS),1].max(),drv[list(sr.ALIGNMENT_POINTS),1].max())

    def test_shoulder_width_keeps_face_and_center_when_feet_are_lowest(self):
        ref,rr=fixture();drv,dr=fixture(bent=True)
        a,ad=retarget(ref,drv,rr,dr,shoulder_width_scale=.7)
        b,bd=retarget(ref,drv,rr,dr,shoulder_width_scale=1.3)
        np.testing.assert_allclose(a[:5],b[:5],atol=1e-12)
        for i in sr.CENTER_RIG_CHAIN:
            np.testing.assert_allclose(ad['center_rig_points'][i],bd['center_rig_points'][i],atol=1e-12)
        self.assertGreater(np.linalg.norm(b[5]-b[6]),np.linalg.norm(a[5]-a[6]))

    def test_incomplete_center_falls_back_without_disabling_usable_shoulders(self):
        ref,rr=fixture();drv,dr=fixture(bent=True)
        baseline=rr.copy();baseline[126]=np.nan
        expected,details=retarget(ref,drv,baseline,dr)
        self.assertEqual(details['center_mode'],'legacy_torso_fallback')
        self.assertEqual(details['shoulder_mode'],'rig_chain')
        self.assertEqual(details['face_anchor'],'shoulder_center_fallback')
        for index in sr.CENTER_RIG_CHAIN:
            bad=rr.copy();bad[index]=np.nan
            out,d=retarget(ref,drv,bad,dr)
            self.assertEqual(d['center_mode'],'legacy_torso_fallback')
            self.assertTrue(d['center_warning'])
            self.assertTrue(np.isfinite(out).all())
            if index!=37:
                np.testing.assert_allclose(out,expected,atol=1e-12)
        bad=dr.copy();bad[35]=bad[34]
        out,d=retarget(ref,drv,rr,bad)
        self.assertIn('R34->R35',d['center_warning'])
        np.testing.assert_allclose(out,expected,atol=1e-12)

    def test_zero_virtual_reference_offsets_are_supported(self):
        ref,rr=fixture();drv,dr=fixture(bent=True)
        # 点定義が一致する特殊入力も、必要のない方向を強制せず処理する。
        shift=sr.hip_center(ref)-rr[1]
        rr[list(sr.CENTER_RIG_CHAIN)] += shift
        ref[69]=rr[110]
        out,d=retarget(ref,drv,rr,dr)
        self.assertEqual(d['center_mode'],'rig_chain')
        np.testing.assert_allclose(d['center_rig_points'][1],sr.hip_center(out),atol=1e-12)
        np.testing.assert_allclose(out[69],d['center_rig_points'][110],atol=1e-12)

    def test_degenerate_face_stays_rigid_with_explicit_rotation_warning(self):
        ref,rr=fixture();drv,dr=fixture(bent=True)
        drv[:5]=drv[0]
        out,d=retarget(ref,drv,rr,dr)
        self.assertEqual(d['center_mode'],'rig_chain')
        self.assertEqual(d['face_rotation_source'],'identity_fallback_degenerate_face')
        np.testing.assert_allclose(out[:5]-d['center_rig_points'][113],ref[:5]-rr[113],atol=1e-12)

    def test_node_report_and_unmodified_driving_diagnostics(self):
        ref,rr=fixture();drv,dr=fixture(bent=True)
        current=node_result(ref,drv,rr,dr)
        bad=rr.copy();bad[126]=np.nan
        fallback=node_result(ref,drv,bad,dr)
        for text in ('center_mode=rig_chain','face_anchor=R113','R1->R34:',
                     'R113->R126:','Shoulder origin=generated_R37.'):
            self.assertIn(text,current[2])
        self.assertIn('center_mode=legacy_torso_fallback',fallback[2])
        self.assertIn('WARNING: center rig unavailable',fallback[2])
        self.assertEqual(current[1],fallback[1])
        self.assertEqual(current[3],fallback[3])
        self.assertNotEqual(current[0],fallback[0])

if __name__=='__main__':
    unittest.main()
