"""Internal-rig limb reconstruction and approximate surface-landmark attachment.

No skinning is performed. Frames use final positional data, not SAM's potentially
pre-hand-fusion global rotations. Row vectors are used throughout.
"""
import numpy as np

try:
    from .mhr127_rig import MHR127_RIG
except ImportError:  # standalone numerical tests
    from mhr127_rig import MHR127_RIG


# Verified against the distributed keypoint_mapping and official rig hierarchy.
DIRECT = {5: 75, 6: 39, 9: 2, 10: 18, 11: 3, 12: 19,
          41: 42, 62: 78}
for start, joints in (
        (21, (64, 63, 62, 61, 59, 58, 57, 56, 55, 54, 53, 52,
              51, 50, 49, 48, 47, 46, 45, 44)),
        (42, (100, 99, 98, 97, 95, 94, 93, 92, 91, 90, 89, 88,
              87, 86, 85, 84, 83, 82, 81, 80))):
    DIRECT.update({start + i: joint for i, joint in enumerate(joints)})

# Only required parent paths, including wrist, thumb0/pinky0 and foot intermediates.
LIMB_POINTS = tuple(range(2, 9)) + tuple(range(18, 25)) + tuple(range(40, 65)) + tuple(range(76, 101)) + (111,)
LIMB_EDGES = tuple((MHR127_RIG[c][1], c) for c in LIMB_POINTS)
_names = {name: i for i, (name, _) in enumerate(MHR127_RIG)}
MIRROR = {i: _names['r_' + name[2:]] for i, (name, _) in enumerate(MHR127_RIG)
          if name.startswith('l_') and 'r_' + name[2:] in _names}
MIRROR.update({b: a for a, b in tuple(MIRROR.items())})


def part_scale(child, s):
    if child in (2, 18):
        return s['hip_width_scale']
    if child in (3, 19):
        return s['leg_scale'] * s['thigh_scale']
    if child in (4, 20):
        return s['leg_scale'] * s['shin_scale']
    if child in tuple(range(5, 9)) + tuple(range(21, 25)):
        return s['leg_scale']
    if child in (40, 76):
        return s['arm_scale'] * s['upper_arm_scale']
    if child in (41, 42, 77, 78):
        return s['arm_scale'] * s['forearm_scale']
    if child == 111:
        return s['neck_scale']
    return s['arm_scale'] * s['hand_scale']


def _frame(primary, candidates):
    length = np.linalg.norm(primary)
    if length <= 1e-8:
        raise ValueError('landmark attachment axis is degenerate')
    x = primary / length
    for candidate in candidates:
        y = candidate - x * np.dot(x, candidate)
        n = np.linalg.norm(y)
        if n > max(1e-8, np.linalg.norm(candidate) * 1e-5):
            y /= n
            return np.stack((x, y, np.cross(x, y)))
    raise ValueError('landmark attachment has no usable transverse axis')


def attachment_frame(rig, region, side):
    """Local position-derived frame. Secondary axis must be shared by both inputs."""
    if region == 'neck':
        return rig[113] - rig[110], [rig[75] - rig[39], rig[37] - rig[1]]
    shoulder, elbow, wrist, index, ring = ((75, 76, 78, 92, 84) if side == 'left'
                                         else (39, 40, 42, 56, 48))
    hip, knee, ankle, ball = ((2, 3, 4, 8) if side == 'left' else (18, 19, 20, 24))
    if region == 'elbow':
        # Palm transverse direction captures roll even with a straight elbow.
        return rig[wrist] - rig[elbow], [rig[index] - rig[ring],
                                        rig[shoulder] - rig[elbow], rig[75] - rig[39]]
    if region == 'shoulder':
        return rig[elbow] - rig[shoulder], [rig[110] - rig[37],
                                           rig[index] - rig[ring], rig[75] - rig[39]]
    return rig[ball] - rig[ankle], [rig[knee] - rig[ankle],
                                   rig[2] - rig[18], rig[37] - rig[1]]


def frame_rotation(reference, driving, region, side):
    a, aa = attachment_frame(reference, region, side)
    b, bb = attachment_frame(driving, region, side)
    # Do not silently pick different roll references in the two poses.
    for x, y in zip(aa, bb):
        try:
            return _frame(a, [x]).T @ _frame(b, [y])
        except ValueError:
            pass
    raise ValueError(f'{side} {region} frame is degenerate')


def place_landmarks(reference, driving, ref, drv, generated, symmetry, scales):
    """Complete one rig and all non-face MHR70 points, atomically on copies."""
    ref, drv = np.asarray(ref), np.asarray(drv)
    needed = sorted(set(LIMB_POINTS) | set(generated) | {a for a, _ in LIMB_EDGES})
    for side, rig, body in (('reference', ref, reference), ('driving', drv, driving)):
        if rig.shape != (127, 3) or not np.isfinite(rig[needed]).all():
            raise ValueError(f'{side} full rig has missing/non-finite required points')
        for point, joint in DIRECT.items():
            if not np.allclose(body[point], rig[joint], rtol=1e-5, atol=1e-6):
                raise ValueError(f'{side} MHR{point} does not match R{joint}')

    points = {i: v.copy() for i, v in generated.items()}
    lengths = {c: float(np.linalg.norm(ref[c] - ref[p])) for p, c in LIMB_EDGES}
    if symmetry == 'average':
        for c, opposite in MIRROR.items():
            if c < opposite and c in lengths and opposite in lengths:
                lengths[c] = lengths[opposite] = (lengths[c] + lengths[opposite]) * .5
    effective = {}
    for parent, child in LIMB_EDGES:
        delta = drv[child] - drv[parent]
        norm = np.linalg.norm(delta)
        length = lengths[child]
        if length > 1e-8 and norm <= 1e-8:
            raise ValueError(f'driving R{parent}->R{child} is degenerate')
        offset = np.zeros(3) if length <= 1e-8 else delta / norm * length
        points[child] = points[parent] + offset * scales['uniform_scale'] * part_scale(child, scales)
        effective[(parent, child)] = length

    output = reference.copy()
    for mhr, joint in DIRECT.items():
        output[mhr] = points[joint]
    # SAM reduced-precision output can round MHR69 independently of the two
    # constituent joints. Preserve its reference residual as an attachment,
    # rather than rejecting real data or losing exact self-reconstruction.
    neck_rotation = frame_rotation(ref, drv, 'neck', 'center')
    neck_residual = reference[69] - (ref[110] + ref[111]) * .5
    output[69] = ((points[110] + points[111]) * .5 + neck_residual @ neck_rotation
                  * scales['uniform_scale'] * scales['neck_scale'])

    # Surface offset is reference-specific (not left/right averaged or a bone).
    # Feet use ball for toe landmarks, ankle for heel/ankle landmarks.
    attachments = (
        ('left', 'elbow', 76, (7, 63, 65), scales['arm_scale'] * scales['forearm_scale']),
        ('right', 'elbow', 40, (8, 64, 66), scales['arm_scale'] * scales['forearm_scale']),
        ('left', 'shoulder', 75, (67,), scales['arm_scale'] * scales['upper_arm_scale']),
        ('right', 'shoulder', 39, (68,), scales['arm_scale'] * scales['upper_arm_scale']),
        ('left', 'foot', 4, (13, 17), scales['leg_scale']),
        ('right', 'foot', 20, (14, 20), scales['leg_scale']),
        ('left', 'foot', 8, (15, 16), scales['leg_scale']),
        ('right', 'foot', 24, (18, 19), scales['leg_scale']),
    )
    for side, region, joint, landmarks, scale in attachments:
        rotation = frame_rotation(ref, drv, region, side)
        indices = list(landmarks)
        output[indices] = points[joint] + (reference[indices] - ref[joint]) @ rotation * scales['uniform_scale'] * scale
    return output, points, effective
