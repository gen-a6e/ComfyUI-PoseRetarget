"""Rig/landmark separation: lengths, articulation, roll, scale and atomic fallback."""
import unittest
import numpy as np
from test_center_rig import fixture as center_fixture, node_result
from test_sam3d_retarget import sr
import rig_landmarks as rl


def fixture():
    body, rig = center_fixture()
    for side, sign, base, shoulder, elbow, twist, wrist in (
            ('left', 1, 2, 75, 76, 77, 78), ('right', -1, 18, 39, 40, 41, 42)):
        rig[base] = rig[1] + (sign * .15, .02, -.03)
        rig[base+1] = rig[base] + (0, .42, .04)
        rig[base+2] = rig[base+1] + (0, .39, -.02)
        for i in range(base+3, base+7):
            rig[i] = rig[i-1] + (0, .01, -.035)
        rig[elbow] = rig[shoulder] + (sign * .23, .12, .02)
        rig[twist] = rig[elbow] + (sign * .21, .04, -.05)
        rig[wrist] = rig[twist] + (sign * .015, 0, 0)
        for child in range(wrist+1, wrist+23):
            parent = rl.MHR127_RIG[child][1]
            if parent == wrist:
                # Distinct bases give a well-defined transverse palm axis.
                lateral = {1: -.04, 6: -.02, 10: 0., 14: .025, 18: .045}[child-wrist]
                delta = (sign * .05, 0, lateral)
            else:
                delta = (sign * .025, -.008, 0)
            rig[child] = rig[parent] + delta
    rig[111] = 2 * body[69] - rig[110]
    for m, r in rl.DIRECT.items():
        body[m] = rig[r]
    for m, r in {7:76,8:40,13:4,14:20,15:8,16:8,17:4,18:24,19:24,20:20,
                 63:76,64:40,65:76,66:40,67:75,68:39}.items():
        body[m] = rig[r] + (.004 * ((m % 3)-1), .018, -.012)
    return body, rig


def run(ref, drv, rr, dr, **kw):
    return sr.retarget_mhr70(ref, drv, reference_rig=rr, driving_rig=dr,
                            reference_symmetry=kw.pop('reference_symmetry', 'off'), **kw)


class FullRigTests(unittest.TestCase):
    def test_identity_all_70_points_and_generated_rig(self):
        body, rig = fixture()
        before, rb = body.copy(), rig.copy()
        out, d = run(body, body, rig, rig)
        self.assertEqual(d['skeleton_mode'], 'full_rig', d['skeleton_warning'])
        np.testing.assert_allclose(out, body, atol=1e-12)
        for i, p in d['rig_points'].items():
            np.testing.assert_allclose(p, rig[i], atol=1e-12)
        np.testing.assert_array_equal(body, before)
        np.testing.assert_array_equal(rig, rb)

    def test_global_rotation_and_translation_of_whole_character(self):
        body, rig = fixture()
        rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        shift = np.array([2., -3., 4.])
        drv, dr = body @ rotation + shift, rig @ rotation + shift
        out, d = run(body, drv, rig, dr)
        self.assertEqual(d['skeleton_mode'], 'full_rig')
        np.testing.assert_allclose(out, drv, atol=1e-12)

    def test_articulated_pose_preserves_rig_lengths_not_elbow_landmark_lengths(self):
        ref, rr = fixture(); drv, dr = fixture()
        # Bend right arm and right knee, including attached child subtrees.
        q = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
        for anchor, children in ((40, range(41, 65)), (19, range(20, 25))):
            ids = list(children)
            dr[ids] = dr[anchor] + (dr[ids] - dr[anchor]) @ q
        for m, r in rl.DIRECT.items():
            drv[m] = dr[r]
        out, d = run(ref, drv, rr, dr)
        self.assertEqual(d['skeleton_mode'], 'full_rig')
        p = d['rig_points']
        for (a,b), length in d['rig_reference_lengths'].items():
            self.assertAlmostEqual(np.linalg.norm(p[b]-p[a]), length)
            if length > 1e-8:
                np.testing.assert_allclose((p[b]-p[a])/length,
                    (dr[b]-dr[a])/np.linalg.norm(dr[b]-dr[a]), atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(out[8]-p[40]), np.linalg.norm(ref[8]-rr[40]))
        # The landmark chord is not constrained as a bone anymore.
        self.assertNotAlmostEqual(np.linalg.norm(out[8]-out[6]), np.linalg.norm(ref[8]-ref[6]), places=5)

    def test_surface_point_changes_cannot_change_rig_bones(self):
        ref, rr = fixture(); drv, dr = fixture()
        a, da = run(ref,drv,rr,dr)
        ref[8] += (.06, -.02, .01)
        b, db = run(ref,drv,rr,dr)
        np.testing.assert_allclose(a[41]-a[6], b[41]-b[6], atol=1e-12)
        for edge, length in da['rig_reference_lengths'].items():
            self.assertEqual(length, db['rig_reference_lengths'][edge])
        self.assertGreater(np.linalg.norm((b[8]-b[6])-(a[8]-a[6])), .05)

    def test_all_scale_domains(self):
        body, rig = fixture()
        options = dict(uniform_scale=1.2, arm_scale=1.1, leg_scale=.9,
                       hand_scale=.8, upper_arm_scale=1.3, forearm_scale=.7,
                       thigh_scale=.8, shin_scale=1.1, hip_width_scale=1.2,
                       torso_scale=1.05, neck_scale=.9, head_scale=.95, shoulder_width_scale=.85)
        out,d=run(body,body,rig,rig,**options)
        self.assertEqual(d['skeleton_mode'],'full_rig')
        p=d['rig_points']
        for a,b in rl.LIMB_EDGES:
            expected=np.linalg.norm(rig[b]-rig[a])*1.2*rl.part_scale(b,options)
            self.assertAlmostEqual(np.linalg.norm(p[b]-p[a]),expected)
        self.assertAlmostEqual(np.linalg.norm(out[8]-p[40]),np.linalg.norm(body[8]-rig[40])*1.2*1.1*.7)
        np.testing.assert_allclose(out[69],(p[110]+p[111])*.5,atol=1e-12)
        np.testing.assert_allclose(out[0,[0,2]],body[0,[0,2]],atol=1e-12)
        self.assertAlmostEqual(out[list(sr.ALIGNMENT_POINTS),1].max(),body[list(sr.ALIGNMENT_POINTS),1].max())

    def test_symmetry_averages_rig_segments_and_preserves_surface_offsets(self):
        ref, rr = fixture(); drv, dr = fixture()
        # Move complete right forearm/hand to lengthen upper arm only.
        rr[40:65] += (.08, .03, -.01)
        for m,r in rl.DIRECT.items(): ref[m]=rr[r]
        out,d=run(ref,drv,rr,dr,reference_symmetry='average')
        self.assertEqual(d['skeleton_mode'],'full_rig')
        p=d['rig_points']
        expected=(np.linalg.norm(rr[40]-rr[39])+np.linalg.norm(rr[76]-rr[75]))*.5
        for a,b in ((39,40),(75,76)):
            self.assertAlmostEqual(np.linalg.norm(p[b]-p[a]),expected)
        self.assertAlmostEqual(np.linalg.norm(out[8]-p[40]),np.linalg.norm(ref[8]-rr[40]))

    def test_zero_length_intermediate_is_supported(self):
        body, rig=fixture()
        rig[5]=rig[4]
        out,d=run(body,body,rig,rig)
        self.assertEqual(d['skeleton_mode'],'full_rig')
        np.testing.assert_allclose(out,body,atol=1e-12)

    def test_neck_rounding_residual_is_preserved(self):
        body,rig=fixture()
        body[69] += (.001, -.00390625, .002)
        out,d=run(body,body,rig,rig)
        self.assertEqual(d['skeleton_mode'],'full_rig')
        np.testing.assert_allclose(out,body,atol=1e-12)

    def test_forearm_roll_moves_surface_offset_with_the_limb(self):
        ref,rr=fixture();drv,dr=fixture()
        axis=rr[78]-rr[76];axis/=np.linalg.norm(axis)
        x,y,z=axis
        cross=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
        # Independent Rodrigues rotation: 90 degrees around forearm axis.
        rotation=np.outer(axis,axis)+cross
        ids=list(range(77,101))
        dr[ids]=rr[76]+(rr[ids]-rr[76])@rotation.T
        for m,r in rl.DIRECT.items():drv[m]=dr[r]
        out,d=run(ref,drv,rr,dr)
        self.assertEqual(d['skeleton_mode'],'full_rig')
        np.testing.assert_allclose(out[7]-d['rig_points'][76],
                                   (ref[7]-rr[76])@rotation.T,atol=1e-12)

    def test_driving_size_does_not_change_reference_rig_or_surface_shape(self):
        ref,rr=fixture()
        drv,dr=ref*2.7+3.,rr*2.7+3.
        out,d=run(ref,drv,rr,dr)
        self.assertEqual(d['skeleton_mode'],'full_rig')
        np.testing.assert_allclose(out-out[0],ref-ref[0],atol=1e-12)

    def test_non_finite_mhr_and_scales_remain_errors(self):
        body,rig=fixture()
        for scale in (float('nan'),float('inf'),0.,-1.):
            with self.assertRaisesRegex(ValueError,'scales'):
                run(body,body,rig,rig,uniform_scale=scale)
        body[7]=np.nan
        with self.assertRaisesRegex(ValueError,'non-finite'):
            run(body,body,rig,rig)

    def test_invalid_full_rig_falls_back_atomically(self):
        ref,rr=fixture();drv,dr=fixture()
        for kind in ('nan', 'mismatch', 'collapsed'):
            bad=dr.copy()
            if kind=='nan': bad[41]=np.nan
            elif kind=='mismatch': bad[42]+=.1
            else: bad[41]=bad[40]
            out,d=run(ref,drv,rr,bad)
            self.assertEqual(d['skeleton_mode'],'legacy_mixed_fallback')
            self.assertTrue(d['skeleton_warning'])
            self.assertEqual(d['rig_points'],{})
            self.assertTrue(np.isfinite(out).all())
            np.testing.assert_allclose(out,drv,atol=1e-12)

    def test_every_limb_edge_is_official_parent_and_all_mhr_points_are_covered(self):
        self.assertEqual(len(rl.DIRECT),48)
        for a,b in rl.LIMB_EDGES: self.assertEqual(rl.MHR127_RIG[b][1],a)
        self.assertIn((42,60),rl.LIMB_EDGES) # thumb0
        self.assertIn((42,43),rl.LIMB_EDGES) # pinky0
        self.assertIn((41,42),rl.LIMB_EDGES) # wrist
        self.assertIn((7,8),rl.LIMB_EDGES) # foot chain

    def test_node_reports_approximation_without_changing_driving_outputs(self):
        ref,rr=fixture();drv,dr=fixture()
        result=node_result(ref,drv,rr,dr)
        bad=rr.copy();bad[41]=np.nan
        old=node_result(ref,drv,bad,dr)
        self.assertIn('skeleton_mode=full_rig',result[2])
        self.assertIn('not regenerated from a deformed mesh',result[2])
        self.assertIn('R40->R41:',result[2])
        self.assertIn('legacy_mixed_fallback',old[2])
        self.assertEqual(result[1],old[1]); self.assertEqual(result[3],old[3])


if __name__=='__main__': unittest.main()
