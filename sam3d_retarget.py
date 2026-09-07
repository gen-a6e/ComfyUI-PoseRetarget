"""SAM 3D BodyのMHR70を使って、体型とポーズを3Dで合成する計算モジュール。

処理の流れ:
1. ``SAM3D_OUTPUT``から出力用のMHR70と、骨長転送用のMHR127内部リグを取得する。
2. referenceの3D骨長を取得し、必要なら左右の推定誤差を平均化する。
3. 中心・肩・四肢の各区間はdrivingの方向を使い、顔はR113基準で一体回転する。
4. driving側のカメラで3D座標を2Dピクセル座標へ投影する。
5. BODY18＋左右HAND21の``POSE_KEYPOINT``形式へ変換する。

SAM 3D Body本体やtorchには依存せず、辞書とnumpy配列だけを受け取る。これにより、
ComfyUIとの接続部分を薄く保ち、3D計算を単体テストできるようにしている。
"""

from __future__ import annotations

import numpy as np

try:
    from .rig_landmarks import place_landmarks
except ImportError:  # standalone numerical use
    from rig_landmarks import place_landmarks


EPS = 1e-8
MHR70_COUNT = 70
MHR_KEYPOINT_COUNT = 308

# SAM 3D Bodyが返すMHR70配列内の主要な関節番号。
NOSE = 0
LEFT_EYE, RIGHT_EYE = 1, 2
LEFT_EAR, RIGHT_EAR = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_HIP, RIGHT_HIP = 9, 10
LEFT_KNEE, RIGHT_KNEE = 11, 12
LEFT_ANKLE, RIGHT_ANKLE = 13, 14
LEFT_HEEL, RIGHT_HEEL = 17, 20
RIGHT_WRIST, LEFT_WRIST = 41, 62
NECK = 69

# MHR127内部リグ。MHR70の番号と混同しない。R39=右肩6、R75=左肩5。
RIG_SPINE3 = 37
SHOULDER_RIG_EDGES = ((37, 38), (38, 39), (37, 74), (74, 75))
SHOULDER_RIG_POINTS = (37, 38, 39, 74, 75)
CENTER_RIG_CHAIN = (1, 34, 35, 36, 37, 110, 113, 126)
CENTER_RIG_EDGES = tuple(zip(CENTER_RIG_CHAIN, CENTER_RIG_CHAIN[1:]))
FACE_POINTS = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)
# 実身体点＋首。63〜68の補助点は最下点判定に使わない（足先・手指は含める）。
ALIGNMENT_POINTS = tuple(range(63)) + (NECK,)


# MHR70には単独の腰中心がないため、左右の股関節の中点を仮想rootとして使う。
# 各要素は「子、親、部位」。親から子の順に配置できるよう、体幹側から並べる。
BODY_EDGES = (
    (NECK, None, "torso"),
    (LEFT_HIP, None, "body"),
    (RIGHT_HIP, None, "body"),
    (LEFT_SHOULDER, NECK, "body"),
    (RIGHT_SHOULDER, NECK, "body"),
    (LEFT_ELBOW, LEFT_SHOULDER, "arm"),
    (LEFT_WRIST, LEFT_ELBOW, "arm"),
    (RIGHT_ELBOW, RIGHT_SHOULDER, "arm"),
    (RIGHT_WRIST, RIGHT_ELBOW, "arm"),
    (LEFT_KNEE, LEFT_HIP, "leg"),
    (LEFT_ANKLE, LEFT_KNEE, "leg"),
    (RIGHT_KNEE, RIGHT_HIP, "leg"),
    (RIGHT_ANKLE, RIGHT_KNEE, "leg"),
    (15, LEFT_ANKLE, "foot"),
    (16, LEFT_ANKLE, "foot"),
    (17, LEFT_ANKLE, "foot"),
    (18, RIGHT_ANKLE, "foot"),
    (19, RIGHT_ANKLE, "foot"),
    (20, RIGHT_ANKLE, "foot"),
    (NOSE, NECK, "head"),
    (LEFT_EYE, NOSE, "head"),
    (RIGHT_EYE, NOSE, "head"),
    (LEFT_EAR, LEFT_EYE, "head"),
    (RIGHT_EAR, RIGHT_EYE, "head"),
)


def _finger_edges(wrist, chains):
    """手首から指先へ向かう各指の接続を、共通の骨形式へ変換する。"""
    edges = []
    for chain in chains:
        parent = wrist
        for child in chain:
            edges.append((child, parent, "hand"))
            parent = child
    return tuple(edges)


# MHR70の手は指先から並んでいるため、生成時に使いやすい「手首→指先」順へ並べ直す。
RIGHT_HAND_CHAINS = (
    (24, 23, 22, 21),
    (28, 27, 26, 25),
    (32, 31, 30, 29),
    (36, 35, 34, 33),
    (40, 39, 38, 37),
)
LEFT_HAND_CHAINS = (
    (45, 44, 43, 42),
    (49, 48, 47, 46),
    (53, 52, 51, 50),
    (57, 56, 55, 54),
    (61, 60, 59, 58),
)
HAND_EDGES = (
    _finger_edges(RIGHT_WRIST, RIGHT_HAND_CHAINS)
    + _finger_edges(LEFT_WRIST, LEFT_HAND_CHAINS)
)
RETARGET_EDGES = BODY_EDGES + HAND_EDGES


# 左右対称化で対応させる関節番号。子の番号だけで、その関節へ入る骨を識別できる。
MIRROR_CHILD = {
    LEFT_HIP: RIGHT_HIP,
    LEFT_SHOULDER: RIGHT_SHOULDER,
    LEFT_ELBOW: RIGHT_ELBOW,
    LEFT_WRIST: RIGHT_WRIST,
    LEFT_KNEE: RIGHT_KNEE,
    LEFT_ANKLE: RIGHT_ANKLE,
    15: 18,
    16: 19,
    17: 20,
    LEFT_EYE: RIGHT_EYE,
    LEFT_EAR: RIGHT_EAR,
}
for left_chain, right_chain in zip(LEFT_HAND_CHAINS, RIGHT_HAND_CHAINS):
    MIRROR_CHILD.update(zip(left_chain, right_chain))
MIRROR_CHILD.update({right: left for left, right in tuple(MIRROR_CHILD.items())})


# SAPIENS/SDPoseが期待するCOCO BODY18順。Noneは投影後の左右肩中点。
# OpenPoseの首スロットとMHR69のneckは別定義。内部3Dの69番は変更しない。
COCO18_FROM_MHR70 = (
    NOSE,
    None,
    RIGHT_SHOULDER,
    RIGHT_ELBOW,
    RIGHT_WRIST,
    LEFT_SHOULDER,
    LEFT_ELBOW,
    LEFT_WRIST,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_ANKLE,
    RIGHT_EYE,
    LEFT_EYE,
    RIGHT_EAR,
    LEFT_EAR,
)

# OpenPoseのHAND21順は「手首＋各指を親指から根元→指先」。
RIGHT_HAND_FROM_MHR70 = (
    RIGHT_WRIST,
    24, 23, 22, 21,
    28, 27, 26, 25,
    32, 31, 30, 29,
    36, 35, 34, 33,
    40, 39, 38, 37,
)
LEFT_HAND_FROM_MHR70 = (
    LEFT_WRIST,
    45, 44, 43, 42,
    49, 48, 47, 46,
    53, 52, 51, 50,
    57, 56, 55, 54,
    61, 60, 59, 58,
)

# fitの範囲は、実際にOpenPoseへ出力する点だけで決める。
# 足先・かかと・補助点・MHR69は描かれないため、fitへ影響させない。
# 仮想首点は両肩の間に収まるので、両肩が範囲に含まれれば十分。
OPENPOSE_FROM_MHR70 = tuple(sorted(set(
    COCO18_FROM_MHR70 + LEFT_HAND_FROM_MHR70 + RIGHT_HAND_FROM_MHR70
) - {None}))


def as_numpy(value, name):
    """torchをimportせず、numpy配列やTensor風オブジェクトをnumpyへ変換する。"""
    if value is None:
        raise ValueError(f"SAM3D output is missing {name}")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SAM3D {name} is not a numeric array") from exc


def extract_mhr70(output):
    """SAM3D_OUTPUTから、カメラ座標系のMHR70を安全に取り出す。"""
    if not isinstance(output, dict):
        raise ValueError("SAM3D input must be a SAM3D_OUTPUT dictionary")
    value = output.get("joints")
    if value is None:
        raw = output.get("raw_output") or {}
        value = raw.get("pred_keypoints_3d")
    points = as_numpy(value, "joints")
    while points.ndim > 2 and points.shape[0] == 1:
        points = points[0]
    if points.ndim != 2 or points.shape[0] < MHR70_COUNT or points.shape[1] < 3:
        raise ValueError(
            f"SAM3D joints must have shape (70, 3); received {points.shape}")
    points = points[:MHR70_COUNT, :3].copy()
    if not np.all(np.isfinite(points)):
        raise ValueError("SAM3D joints contain NaN or infinite values")
    return points


def extract_mhr70_2d(output):
    """SAM内部で投影済みのMHR70 XY座標と、有効点maskを取り出す。"""
    if not isinstance(output, dict):
        raise ValueError("SAM3D input must be a SAM3D_OUTPUT dictionary")
    value = output.get("keypoints_2d")
    if value is None:
        raw = output.get("raw_output") or {}
        value = raw.get("pred_keypoints_2d")
    if value is None:
        raise ValueError(
            "SAM3D output is missing raw_output.pred_keypoints_2d; "
            "rerun SAM 3D Body: Process Image"
        )

    points = as_numpy(value, "pred_keypoints_2d")
    while points.ndim > 2 and points.shape[0] == 1:
        points = points[0]
    if points.ndim != 2 or points.shape[0] < MHR70_COUNT or points.shape[1] < 2:
        raise ValueError(
            "SAM3D pred_keypoints_2d must have shape (70, 2); "
            f"received {points.shape}"
        )
    points = points[:MHR70_COUNT, :2].copy()
    valid = np.all(np.isfinite(points), axis=1)
    points[~valid] = 0.0
    return points, valid


def extract_head_top(output):
    """内部リグMHR127のR126（c_head_null）を、身長計測の頭頂点として取り出す。"""
    if not isinstance(output, dict):
        raise ValueError("SAM3D input must be a SAM3D_OUTPUT dictionary")
    value = output.get("joint_coords")
    if value is None:
        value = (output.get("raw_output") or {}).get("pred_joint_coords")
    if value is None:
        raise ValueError(
            "height reporting requires MHR127 joint_coords (R126 c_head_null); "
            "rerun SAM 3D Body: Process Image"
        )
    points = as_numpy(value, "joint_coords")
    while points.ndim > 2 and points.shape[0] == 1:
        points = points[0]
    if points.shape != (127, 3):
        raise ValueError(f"SAM3D joint_coords must have shape (127, 3); received {points.shape}")
    # 単位・軸の変換はProcess Image側で済んでいる。全308点へはフォールバックしない。
    head = points[126]
    if not np.isfinite(head).all():
        raise ValueError("SAM3D R126 c_head_null contains non-finite coordinates")
    return head.copy(), 126


def extract_mhr127(output):
    """中心・肩の再構成用にMHR127を取り出す。使用点の検証は各処理で行う。"""
    if not isinstance(output, dict):
        raise ValueError("SAM3D input must be a SAM3D_OUTPUT dictionary")
    value = output.get("joint_coords")
    if value is None:
        raw = output.get("raw_output")
        value = raw.get("pred_joint_coords") if isinstance(raw, dict) else None
    points = as_numpy(value, "MHR127 joint_coords")
    while points.ndim > 2 and points.shape[0] == 1:
        points = points[0]
    if points.shape != (127, 3):
        raise ValueError(f"rig requires MHR127 (127, 3); received {points.shape}")
    return points.copy()


def extract_shoulder_rig(output):
    """肩リグ追加時の呼び出し名を互換用に維持する。"""
    return extract_mhr127(output)


def _validate_shoulder_rig(rig, body, side, require_neck_anchor=True):
    """別人物・別座標系のリグや潰れた区間を、MHR70へ混ぜないための検証。"""
    points = as_numpy(rig, f"{side} shoulder rig")
    if points.shape != (127, 3):
        raise ValueError(f"{side} shoulder rig must have shape (127, 3)")
    if not np.isfinite(points[list(SHOULDER_RIG_POINTS)]).all():
        raise ValueError(f"{side} shoulder rig contains non-finite shoulder coordinates")
    # 現行SAMのMHR70肩点は、それぞれこの内部関節そのもの。
    for index, shoulder in ((39, RIGHT_SHOULDER), (75, LEFT_SHOULDER)):
        if not np.allclose(points[index], body[shoulder], rtol=1e-5, atol=1e-6):
            raise ValueError(f"{side} R{index} does not match MHR70 shoulder {shoulder}")
    for parent, child in SHOULDER_RIG_EDGES:
        if np.linalg.norm(points[child] - points[parent]) <= EPS:
            raise ValueError(f"{side} shoulder segment R{parent}->R{child} is degenerate")
    # R37は首69を基準に配置するので、その向きも必要。
    if require_neck_anchor and np.linalg.norm(points[RIG_SPINE3] - body[NECK]) <= EPS:
        raise ValueError(f"{side} shoulder anchor MHR69->R37 is degenerate")
    return points


def _place_rig_shoulders(output, reference, driving, reference_rig, driving_rig,
                         symmetry, uniform, torso_scale, width_scale, spine_origin=None):
    """生成R37から肩を構成。中心リグがない旧データのみ首69からR37を逆算。"""
    ref = _validate_shoulder_rig(reference_rig, reference, "reference", spine_origin is None)
    drv = _validate_shoulder_rig(driving_rig, driving, "driving", spine_origin is None)
    lengths = {edge: float(np.linalg.norm(ref[edge[1]] - ref[edge[0]]))
               for edge in SHOULDER_RIG_EDGES}
    if symmetry == "average":
        for right, left in (((37, 38), (37, 74)), ((38, 39), (74, 75))):
            lengths[right] = lengths[left] = (lengths[right] + lengths[left]) * .5

    # 中心リグが使用不能なときだけ、旧方式の首69→R37をtorso_scaleで転送する。
    anchor_length = None
    if spine_origin is None:
        anchor_vector = drv[RIG_SPINE3] - driving[NECK]
        anchor_length = float(np.linalg.norm(ref[RIG_SPINE3] - reference[NECK]))
        spine_origin = (output[NECK] + anchor_vector / np.linalg.norm(anchor_vector)
                        * anchor_length * uniform * float(torso_scale))
    generated = {RIG_SPINE3: spine_origin.copy()}
    for parent, child in SHOULDER_RIG_EDGES:
        direction = drv[child] - drv[parent]
        generated[child] = (generated[parent] + direction / np.linalg.norm(direction)
                            * lengths[(parent, child)] * uniform * float(width_scale))
    output[RIGHT_SHOULDER] = generated[39]
    output[LEFT_SHOULDER] = generated[75]
    return generated, lengths, anchor_length


def _transfer_offset(reference_vector, driving_vector, scale, name):
    """別定義の点を同一視しないための仮想区間。長さはreference、向きはdriving。"""
    if not (np.isfinite(reference_vector).all() and np.isfinite(driving_vector).all()):
        raise ValueError(f"{name} contains non-finite coordinates")
    length = float(np.linalg.norm(reference_vector))
    if length <= EPS:
        return np.zeros(3)
    driving_length = float(np.linalg.norm(driving_vector))
    if driving_length <= EPS:
        raise ValueError(f"{name} driving direction is degenerate")
    return driving_vector / driving_length * length * scale


def _place_center_rig(reference, driving, reference_rig, driving_rig, root,
                      uniform, torso_scale, neck_scale, head_scale, include_neck_landmark=True):
    """骨盤から頭頂まで7区間を転送し、首69の位置も返す。入力は変更しない。"""
    # 肩点との一致も検証し、別座標系・別人物のリグを胴体へ混ぜない。
    ref = _validate_shoulder_rig(reference_rig, reference, "reference", False)
    drv = _validate_shoulder_rig(driving_rig, driving, "driving", False)
    for side, points in (("reference", ref), ("driving", drv)):
        if not np.isfinite(points[list(CENTER_RIG_CHAIN)]).all():
            raise ValueError(f"{side} center rig contains non-finite coordinates")
        for parent, child in CENTER_RIG_EDGES:
            if np.linalg.norm(points[child] - points[parent]) <= EPS:
                raise ValueError(f"{side} center segment R{parent}->R{child} is degenerate")

    # Hは左右股関節の中点、R1は内部ルート。骨盤内のずれには全体倍率だけを適用。
    generated = {1: root + _transfer_offset(
        ref[1] - hip_center(reference), drv[1] - hip_center(driving), uniform, "H->R1")}
    lengths = {}
    for parent, child in CENTER_RIG_EDGES:
        length = float(np.linalg.norm(ref[child] - ref[parent]))
        lengths[(parent, child)] = length
        part_scale = torso_scale if child in (34, 35, 36, 37) else (
            neck_scale if child in (110, 113) else head_scale)
        direction = drv[child] - drv[parent]
        generated[child] = (generated[parent] + direction / np.linalg.norm(direction)
                            * length * uniform * float(part_scale))
    # MHR69はR110そのものではない。既存の首点へのオフセットを別に転送する。
    neck = None
    if include_neck_landmark:
        neck = generated[110] + _transfer_offset(
            reference[NECK] - ref[110], driving[NECK] - drv[110],
            uniform * float(neck_scale), "R110->MHR69")
    return generated, lengths, neck


def hip_center(points):
    """左右の股関節の中点を返す。生成骨格のrootとして使用する。"""
    return (points[LEFT_HIP] + points[RIGHT_HIP]) * 0.5


def shoulder_center(points):
    """左右の肩の中点を返す。肩全体の上下移動や頭の基準に使用する。"""
    return (points[LEFT_SHOULDER] + points[RIGHT_SHOULDER]) * 0.5


def _edge_delta(points, child, parent):
    """親から子へ向かう3Dベクトルを返す。親がNoneなら腰中心を使う。"""
    origin = hip_center(points) if parent is None else points[parent]
    return points[child] - origin


def estimated_height(points, head_top):
    """R126→首→腰中央と、左右平均の脚→かかとの3D距離による概算身長。"""
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (MHR70_COUNT, 3):
        raise ValueError("height measurement requires MHR70 points with shape (70, 3)")
    head_top = np.asarray(head_top, dtype=np.float64).reshape(-1)
    if head_top.size != 3 or not np.all(np.isfinite(head_top)):
        raise ValueError("head_top must contain three finite coordinates")
    # 鼻・肩中央は経由しない。腰中央→股関節の横方向の距離も身長には足さない。
    head_neck = np.linalg.norm(head_top - points[NECK])
    torso = np.linalg.norm(points[NECK] - hip_center(points))
    leg_lengths = []
    for hip, knee, ankle, heel in (
            (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, LEFT_HEEL),
            (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_HEEL)):
        leg_lengths.append(
            np.linalg.norm(points[knee] - points[hip])
            + np.linalg.norm(points[ankle] - points[knee])
            + np.linalg.norm(points[heel] - points[ankle])
        )
    height = head_neck + torso + 0.5 * sum(leg_lengths)
    if not np.isfinite(height) or height <= EPS:
        raise ValueError("SAM3D skeleton has no usable estimated height")
    return float(height)


def reference_lengths(points, symmetry="average"):
    """referenceの各骨長を取得し、必要なら左右の推定誤差を平均化する。"""
    lengths = {
        child: float(np.linalg.norm(_edge_delta(points, child, parent)))
        for child, parent, _ in RETARGET_EDGES
    }
    if symmetry == "off":
        # offでは左右差を体型情報としてそのまま残す。
        return lengths
    if symmetry != "average":
        raise ValueError(f"unknown reference symmetry mode: {symmetry}")

    visited = set()
    # 同じ左右ペアを二重処理しないよう、番号を並べ替えた組を記録する。
    for child, mirror in MIRROR_CHILD.items():
        pair = tuple(sorted((child, mirror)))
        if pair in visited or child not in lengths or mirror not in lengths:
            continue
        visited.add(pair)
        left, right = lengths[child], lengths[mirror]
        value = (left + right) * 0.5
        lengths[child] = value
        lengths[mirror] = value
    return lengths


def body_measurements(points):
    """reportへ表示する主要部位の3D長を返す。左右部位は平均値にする。"""
    return {
        "torso": float(np.linalg.norm(points[NECK] - hip_center(points))),
        "shoulder_width": float(np.linalg.norm(
            points[LEFT_SHOULDER] - points[RIGHT_SHOULDER])),
        "hip_width": float(np.linalg.norm(
            points[LEFT_HIP] - points[RIGHT_HIP])),
        "shoulder_to_nose": float(np.linalg.norm(
            points[NOSE] - shoulder_center(points))),
        "upper_arm": 0.5 * (
            float(np.linalg.norm(points[LEFT_ELBOW] - points[LEFT_SHOULDER]))
            + float(np.linalg.norm(
                points[RIGHT_ELBOW] - points[RIGHT_SHOULDER]))),
        "forearm": 0.5 * (
            float(np.linalg.norm(points[LEFT_WRIST] - points[LEFT_ELBOW]))
            + float(np.linalg.norm(
                points[RIGHT_WRIST] - points[RIGHT_ELBOW]))),
        "thigh": 0.5 * (
            float(np.linalg.norm(points[LEFT_KNEE] - points[LEFT_HIP]))
            + float(np.linalg.norm(points[RIGHT_KNEE] - points[RIGHT_HIP]))),
        "shin": 0.5 * (
            float(np.linalg.norm(points[LEFT_ANKLE] - points[LEFT_KNEE]))
            + float(np.linalg.norm(
                points[RIGHT_ANKLE] - points[RIGHT_KNEE]))),
    }


def _unit_direction(primary, fallback):
    """drivingを優先し、長さを1にした安定なボーン方向を返す。"""
    # driving側の骨が潰れている場合だけreference方向へ退避する。
    for value in (primary, fallback):
        norm = float(np.linalg.norm(value))
        if np.isfinite(norm) and norm > EPS:
            return value / norm
    return np.zeros(3, dtype=np.float64)


def _place_edge(output, driving, reference, child, parent, target_length):
    """drivingの方向と指定した骨長を使い、親から子の3D位置を決める。"""
    direction = _unit_direction(
        driving[child] - driving[parent],
        reference[child] - reference[parent],
    )
    output[child] = output[parent] + direction * target_length


def _face_rotation(reference, driving):
    """顔5点の対応から、reference→drivingの共通回転をKabsch法で求める。

    重心移動・全体サイズを除去して向きだけを推定する。戻り値は行ベクトル用。
    顔立ちの差は回転推定に影響し得るが、出力の顔形状を変形することはない。
    """
    clouds = []
    for points in (reference, driving):
        face = points[list(FACE_POINTS)]
        centered = face - face.mean(axis=0)
        size = float(np.linalg.norm(centered))
        if not np.isfinite(size) or size <= EPS:
            return np.eye(3), "identity_fallback_degenerate_face"
        clouds.append(centered / size)
    u, singular, vt = np.linalg.svd(clouds[0].T @ clouds[1])
    # 平面上の顔でも回転は求まるが、点・直線に潰れると回転軸が定まらない。
    if singular[1] <= singular[0] * 1e-6:
        return np.eye(3), "identity_fallback_degenerate_face"
    correction = np.eye(3)
    # 左右反転を許可しない。常にdet=+1の回転として顔形状を保持する。
    correction[2, 2] = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    return u @ correction @ vt, "kabsch_face5"


def _align_to_driving(points, driving):
    """形状を保った平行移動で、鼻X/Zと最下点Yをdrivingに合わせる。"""
    indices = np.asarray(ALIGNMENT_POINTS)
    generated_bottom = int(indices[np.argmax(points[indices, 1])])
    driving_bottom = int(indices[np.argmax(driving[indices, 1])])
    # SAMの投影用座標はY正方向が下。カメラ並進を加えてもこの差は変わらない。
    translation = driving[NOSE] - points[NOSE]
    translation[1] = driving[driving_bottom, 1] - points[generated_bottom, 1]
    return points + translation, translation, generated_bottom, driving_bottom


def _retarget_full_rig(reference, driving, reference_rig, driving_rig, symmetry, scales):
    """Rig lengths first, then MHR70 output landmarks; no MHR limb lengths used."""
    reference_rig = as_numpy(reference_rig, 'reference MHR127')
    driving_rig = as_numpy(driving_rig, 'driving MHR127')
    u = scales['uniform_scale']
    center, center_lengths, _ = _place_center_rig(
        reference, driving, reference_rig, driving_rig, hip_center(driving),
        u, scales['torso_scale'], scales['neck_scale'], scales['head_scale'],
        include_neck_landmark=False)
    temporary = driving.copy()
    shoulders, shoulder_lengths, _ = _place_rig_shoulders(
        temporary, reference, driving, reference_rig, driving_rig,
        symmetry, u, scales['torso_scale'], scales['shoulder_width_scale'], center[37])
    output, rig, lengths = place_landmarks(
        reference, driving, reference_rig, driving_rig, {**center, **shoulders},
        symmetry, scales)
    rotation, source = _face_rotation(reference, driving)
    output[:5] = rig[113] + (reference[:5] - reference_rig[113]) @ rotation * u * scales['head_scale']
    output, shift, bottom, driving_bottom = _align_to_driving(output, driving)
    rig = {i: v + shift for i, v in rig.items()}
    lengths.update(center_lengths)
    lengths.update(shoulder_lengths)
    return output, {
        'base_scale': u, 'size_source': 'reference',
        'skeleton_mode': 'full_rig', 'skeleton_warning': None,
        'landmark_mode': 'position_frame_offset_approximation',
        'rig_points': rig, 'rig_reference_lengths': lengths,
        'face_rotation_source': source, 'face_anchor': 'R113',
        'center_mode': 'rig_chain', 'center_warning': None,
        'center_rig_points': {i: rig[i] for i in center},
        'center_reference_lengths': center_lengths,
        'shoulder_mode': 'rig_chain', 'shoulder_warning': None,
        'shoulder_pose_scale': None, 'shoulder_anchor_length': None,
        'shoulder_rig_points': {i: rig[i] for i in shoulders},
        'shoulder_reference_lengths': shoulder_lengths,
        'alignment_translation': shift, 'generated_bottom_index': bottom,
        'driving_bottom_index': driving_bottom,
        'reference_measurements': body_measurements(reference),
        'generated_measurements': body_measurements(output),
    }


def retarget_mhr70(reference, driving, reference_symmetry="average",
                   uniform_scale=1.0,
                   leg_scale=1.0, arm_scale=1.0, head_scale=1.0,
                   hand_scale=1.0, torso_scale=1.0,
                   shoulder_width_scale=1.0, hip_width_scale=1.0,
                   neck_scale=1.0, upper_arm_scale=1.0,
                   forearm_scale=1.0, thigh_scale=1.0,
                   shin_scale=1.0, reference_rig=None, driving_rig=None):
    """referenceの3D骨長とdrivingの3D方向・ポーズを合成する。"""
    # 入力をfloat64へ統一し、全関節が期待どおり70点あることを先に保証する。
    reference = np.asarray(reference, dtype=np.float64)
    driving = np.asarray(driving, dtype=np.float64)
    if reference.shape != (MHR70_COUNT, 3) or driving.shape != (MHR70_COUNT, 3):
        raise ValueError("reference and driving joints must both have shape (70, 3)")
    if not (np.isfinite(reference).all() and np.isfinite(driving).all()):
        raise ValueError('MHR70 contains non-finite coordinates')
    if reference_symmetry not in ('off', 'average'):
        raise ValueError(f'unknown reference symmetry mode: {reference_symmetry}')
    scales = dict(uniform_scale=uniform_scale, leg_scale=leg_scale, arm_scale=arm_scale,
                  head_scale=head_scale, hand_scale=hand_scale, torso_scale=torso_scale,
                  shoulder_width_scale=shoulder_width_scale, hip_width_scale=hip_width_scale,
                  neck_scale=neck_scale, upper_arm_scale=upper_arm_scale,
                  forearm_scale=forearm_scale, thigh_scale=thigh_scale, shin_scale=shin_scale)
    scales = {key: float(value) for key, value in scales.items()}
    if any(not np.isfinite(value) or value <= 0 for value in scales.values()):
        raise ValueError('scales must be finite and positive')
    try:
        return _retarget_full_rig(reference, driving, reference_rig, driving_rig,
                                  reference_symmetry, scales)
    except ValueError as exc:
        # Atomic fallback: no newly reconstructed limb or attachment leaks into
        # the previous mixed implementation. Never label that path as full_rig.
        skeleton_warning = str(exc)

    # SAM 3D Bodyが推定したreferenceの3D骨長を、正規化せず直接使用する。
    # 出力骨長は「reference骨長 × uniform_scale × 部位別scale」。
    uniform = float(uniform_scale)
    bone_lengths = reference_lengths(reference, reference_symmetry)
    reference_measurements = body_measurements(reference)
    driving_measurements = body_measurements(driving)

    # drivingを土台にし、補助点は再構成しない。最後の全体平行移動は補助点にも適用する。
    output = driving.copy()
    # 再構成中はdrivingの腰中心を仮の起点にし、最後に鼻・最下点で位置を合わせる。
    root = hip_center(driving)

    # 腰: 左右を別々の骨として伸ばすと中心がずれるため、腰中心から対称に配置する。
    # 向きはdriving、幅はreferenceの実際の3D距離を使用する。
    hip_axis = _unit_direction(
        driving[LEFT_HIP] - driving[RIGHT_HIP],
        reference[LEFT_HIP] - reference[RIGHT_HIP],
    )
    hip_width = (
        reference_measurements["hip_width"] * uniform
        * float(hip_width_scale)
    )
    output[LEFT_HIP] = root + hip_axis * hip_width * 0.5
    output[RIGHT_HIP] = root - hip_axis * hip_width * 0.5

    # 胴体: 腰中心→首の向きはdriving、長さはreferenceから移す。
    torso_direction = _unit_direction(
        driving[NECK] - hip_center(driving),
        reference[NECK] - hip_center(reference),
    )
    torso_length = (
        reference_measurements["torso"] * uniform
        * float(torso_scale)
    )
    output[NECK] = root + torso_direction * torso_length

    # 新しい中心構造をすべて検証してから採用する。欠損時は直前版の配置を維持。
    center_mode, center_warning = "rig_chain", None
    center_rig_points, center_lengths = {}, {}
    try:
        center_rig_points, center_lengths, neck = _place_center_rig(
            reference, driving, reference_rig, driving_rig, root,
            uniform, torso_scale, neck_scale, head_scale)
    except ValueError as exc:
        center_mode, center_warning = "legacy_torso_fallback", str(exc)
    else:
        output[NECK] = neck

    # 肩: 端から端の幅を固定せず、背骨→鎖骨起点→肩の4区間を転送する。
    # リグのない旧データは従来方式へ戻し、reportで明示する。
    shoulder_mode = "rig_chain"
    shoulder_warning = None
    shoulder_pose_scale = None
    shoulder_rig_points, shoulder_lengths, shoulder_anchor_length = {}, {}, None
    try:
        shoulder_rig_points, shoulder_lengths, shoulder_anchor_length = _place_rig_shoulders(
            output, reference, driving, reference_rig, driving_rig,
            reference_symmetry, uniform, torso_scale, shoulder_width_scale,
            spine_origin=center_rig_points.get(RIG_SPINE3))
    except ValueError as exc:
        shoulder_mode = "legacy_width_fallback"
        shoulder_warning = str(exc)
        shoulder_axis = _unit_direction(
            driving[LEFT_SHOULDER] - driving[RIGHT_SHOULDER],
            reference[LEFT_SHOULDER] - reference[RIGHT_SHOULDER])
        shoulder_width = (reference_measurements["shoulder_width"] * uniform
                          * float(shoulder_width_scale))
        driving_shoulder_offset = shoulder_center(driving) - driving[NECK]
        driving_torso = driving_measurements["torso"]
        shoulder_pose_scale = uniform
        if np.isfinite(driving_torso) and driving_torso > EPS:
            shoulder_pose_scale *= reference_measurements["torso"] / driving_torso
        output_shoulder_center = output[NECK] + driving_shoulder_offset * shoulder_pose_scale
        output[LEFT_SHOULDER] = output_shoulder_center + shoulder_axis * shoulder_width * .5
        output[RIGHT_SHOULDER] = output_shoulder_center - shoulder_axis * shoulder_width * .5

    # 腕と脚: 各ボーンの向きはdriving、長さはreferenceの実骨長を使う。
    # 大分類のarm/leg倍率と、上腕・前腕・腿・脛の詳細倍率は乗算する。
    edge_scales = (
        (LEFT_ELBOW, LEFT_SHOULDER,
         float(arm_scale) * float(upper_arm_scale)),
        (RIGHT_ELBOW, RIGHT_SHOULDER,
         float(arm_scale) * float(upper_arm_scale)),
        (LEFT_WRIST, LEFT_ELBOW,
         float(arm_scale) * float(forearm_scale)),
        (RIGHT_WRIST, RIGHT_ELBOW,
         float(arm_scale) * float(forearm_scale)),
        (LEFT_KNEE, LEFT_HIP,
         float(leg_scale) * float(thigh_scale)),
        (RIGHT_KNEE, RIGHT_HIP,
         float(leg_scale) * float(thigh_scale)),
        (LEFT_ANKLE, LEFT_KNEE,
         float(leg_scale) * float(shin_scale)),
        (RIGHT_ANKLE, RIGHT_KNEE,
         float(leg_scale) * float(shin_scale)),
    )
    for child, parent, part_scale in edge_scales:
        _place_edge(
            output, driving, reference, child, parent,
            bone_lengths[child] * uniform * part_scale,
        )

    # 足: 足首を親に持つつま先・かかとのみを、driving方向へ再配置する。
    for child, parent, _ in BODY_EDGES:
        if parent not in (LEFT_ANKLE, RIGHT_ANKLE):
            continue
        _place_edge(
            output, driving, reference, child, parent,
            bone_lengths[child] * uniform * float(leg_scale),
        )

    # 顔5点は一体で回転させる。頭の軸1本だけでは顔の左右向きは決まらない。
    face_rotation, face_rotation_source = _face_rotation(reference, driving)
    if center_mode == "rig_chain":
        face_anchor = "R113"
        face_indices = list(FACE_POINTS)
        output[face_indices] = center_rig_points[113] + (
            (reference[face_indices] - as_numpy(reference_rig, "reference rig")[113]) @ face_rotation
        ) * uniform * float(head_scale)
    else:
        # リグが不完全な場合だけ直前版の肩中央→鼻配置へ戻す。
        face_anchor = "shoulder_center_fallback"
        nose_direction = _unit_direction(
            driving[NOSE] - shoulder_center(driving),
            reference[NOSE] - shoulder_center(reference))
        neck_length = (reference_measurements["shoulder_to_nose"] * uniform
                       * float(head_scale) * float(neck_scale))
        output[NOSE] = shoulder_center(output) + nose_direction * neck_length
        face_indices = list(FACE_POINTS[1:])
        output[face_indices] = output[NOSE] + (
            (reference[face_indices] - reference[NOSE]) @ face_rotation
        ) * uniform * float(head_scale)

    # 手: 手首を起点に各指を根元から指先へ順番に配置する。
    for child, parent, _ in HAND_EDGES:
        _place_edge(
            output, driving, reference, child, parent,
            bone_lengths[child] * uniform
            * float(arm_scale) * float(hand_scale),
        )

    # 形状が完成してから全点を同量だけ移す。足固定や関節方向の補正は行わない。
    output, translation, generated_bottom, driving_bottom = _align_to_driving(output, driving)
    # 診断用の内部点も、返すMHR70と同じ最終座標系へ移す。
    shoulder_rig_points = {index: point + translation
                           for index, point in shoulder_rig_points.items()}
    center_rig_points = {index: point + translation
                        for index, point in center_rig_points.items()}

    # 呼び出し側でreferenceと生成後の実骨長を比較できるようにする。
    details = {
        "skeleton_mode": "legacy_mixed_fallback",
        "skeleton_warning": skeleton_warning,
        "landmark_mode": "legacy_mhr_edges",
        "rig_points": {},
        "rig_reference_lengths": {},
        "base_scale": uniform,
        "size_source": "reference",
        "face_rotation_source": face_rotation_source,
        "face_anchor": face_anchor,
        "center_mode": center_mode,
        "center_warning": center_warning,
        "center_rig_points": center_rig_points,
        "center_reference_lengths": center_lengths,
        "alignment_translation": translation,
        "generated_bottom_index": generated_bottom,
        "driving_bottom_index": driving_bottom,
        "shoulder_pose_scale": shoulder_pose_scale,
        "shoulder_mode": shoulder_mode,
        "shoulder_warning": shoulder_warning,
        "shoulder_rig_points": shoulder_rig_points,
        "shoulder_reference_lengths": shoulder_lengths,
        "shoulder_anchor_length": shoulder_anchor_length,
        "reference_measurements": reference_measurements,
        "generated_measurements": body_measurements(output),
    }
    return output, details


def extract_camera(output):
    """SAM3D_OUTPUTから透視投影用のカメラ移動量と焦点距離を取り出す。"""
    if not isinstance(output, dict):
        raise ValueError("driving SAM3D input must be a dictionary")
    raw = output.get("raw_output") or {}
    camera = output.get("camera")
    if camera is None:
        camera = raw.get("pred_cam_t")
    focal = output.get("focal_length")
    if focal is None:
        focal = raw.get("focal_length")

    camera = as_numpy(camera, "camera translation").reshape(-1)
    focal = as_numpy(focal, "focal length").reshape(-1)
    if camera.size < 3 or focal.size < 1:
        raise ValueError("SAM3D camera data has an invalid shape")
    if not np.all(np.isfinite(camera[:3])) or not np.all(np.isfinite(focal)):
        raise ValueError("SAM3D camera data contains invalid values")
    fx = float(focal[0])
    fy = float(focal[1] if focal.size > 1 else focal[0])
    if fx <= EPS or fy <= EPS:
        raise ValueError("SAM3D focal length must be positive")
    return camera[:3], np.array([fx, fy], dtype=np.float64)


def project_mhr70(points, camera, focal_xy, width, height):
    """SAM 3D Bodyと同じ透視投影式で、MHR70を絶対ピクセル座標へ変換する。"""
    # 関節は人物ローカル座標なので、まず推定されたカメラ移動量を加える。
    camera_points = np.asarray(points, dtype=np.float64) + camera[None, :]
    depth = camera_points[:, 2]
    valid = np.isfinite(camera_points).all(axis=1) & (depth > EPS)
    projected = np.zeros((len(points), 2), dtype=np.float64)
    # 主点は画像中央。depthが0以下の点はカメラの後ろなので投影しない。
    projected[valid, 0] = (
        focal_xy[0] * camera_points[valid, 0] / depth[valid] + width * 0.5)
    projected[valid, 1] = (
        focal_xy[1] * camera_points[valid, 1] / depth[valid] + height * 0.5)
    return projected, valid, depth


def projected_difference(first, first_valid, second, second_valid, indices):
    """指定した関節群における2組のXY座標差をピクセル単位で集計する。"""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_valid = np.asarray(first_valid, dtype=bool)
    second_valid = np.asarray(second_valid, dtype=bool)
    indices = np.asarray(indices, dtype=np.int64)
    valid = first_valid[indices] & second_valid[indices]
    if not np.any(valid):
        return {"count": 0, "rms": None, "max": None}
    delta = first[indices[valid]] - second[indices[valid]]
    distances = np.linalg.norm(delta, axis=1)
    return {
        "count": int(len(distances)),
        "rms": float(np.sqrt(np.mean(distances ** 2))),
        "max": float(np.max(distances)),
    }


def fit_projected(points, valid, width, height, mode="shrink_to_fit", margin=16):
    """出力する体・手の有効点を基準に、縦横比を保ってcanvas内へ収める。"""
    points = np.asarray(points, dtype=np.float64).copy()
    if mode == "off" or not np.any(valid):
        return points, 1.0
    if mode not in {"shrink_to_fit", "fit_exactly"}:
        raise ValueError(f"unknown fit mode: {mode}")

    fit_indices = np.asarray(OPENPOSE_FROM_MHR70, dtype=np.int64)
    fit_indices = fit_indices[np.asarray(valid, dtype=bool)[fit_indices]]
    if not fit_indices.size:
        # 描画できる点がない場合、補助点だけで拡大・移動しない。
        return points, 1.0
    subset = points[fit_indices]
    lo = subset.min(axis=0)
    hi = subset.max(axis=0)
    if mode == "shrink_to_fit" and (
            lo[0] >= 0 and lo[1] >= 0 and hi[0] <= width and hi[1] <= height):
        # 出力する有効点が収まっていれば、位置やサイズを一切変えない。
        return points, 1.0

    max_margin = max(0.0, min(width, height) * 0.5 - 1.0)
    margin = min(max(float(margin), 0.0), max_margin)
    available = np.array([width - 2 * margin, height - 2 * margin], dtype=np.float64)
    span = np.maximum(hi - lo, EPS)
    # 幅・高さのうち厳しい側に合わせ、同じ倍率をXY両方へ適用する。
    scale = float(np.min(available / span))
    if mode == "shrink_to_fit":
        scale = min(1.0, scale)
    center = (lo + hi) * 0.5
    target_center = np.array([width * 0.5, height * 0.5], dtype=np.float64)
    # 範囲計算から除外した点にも同じ変換を適用し、座標系は揃えておく。
    points[valid] = (points[valid] - center) * scale + target_center
    return points, scale


def _openpose_field(projected, valid, indices):
    """指定点を並べ替え、Noneのスロットは2D左右肩中点として作る。"""
    out = np.zeros((len(indices), 3), dtype=np.float64)
    for output_index, mhr_index in enumerate(indices):
        if mhr_index is None:
            # 3D中点の投影ではなく2D中点。左右肩の奥行きが違っても線の中央になる。
            # 片肩が無効なら首点も無効とし、69番での代用はしない。
            if valid[LEFT_SHOULDER] and valid[RIGHT_SHOULDER]:
                out[output_index, :2] = (
                    projected[LEFT_SHOULDER] + projected[RIGHT_SHOULDER]
                ) * 0.5
                out[output_index, 2] = 1.0
            continue
        if valid[mhr_index]:
            out[output_index, 0] = projected[mhr_index, 0]
            out[output_index, 1] = projected[mhr_index, 1]
            out[output_index, 2] = 1.0
    return [round(float(value), 6) for value in out.reshape(-1)]


def to_pose_keypoint(projected, valid, width, height):
    """MHR70の体と手を、絶対ピクセル座標のPOSE_KEYPOINTへ変換する。"""
    # MHR70からOpenPose互換の顔70点は復元できないため、固定長のconfidence 0で返す。
    person = {
        "pose_keypoints_2d": _openpose_field(
            projected, valid, COCO18_FROM_MHR70),
        "face_keypoints_2d": [0.0] * (70 * 3),
        "hand_left_keypoints_2d": _openpose_field(
            projected, valid, LEFT_HAND_FROM_MHR70),
        "hand_right_keypoints_2d": _openpose_field(
            projected, valid, RIGHT_HAND_FROM_MHR70),
    }
    return [{
        "canvas_width": int(width),
        "canvas_height": int(height),
        "people": [person],
    }]


def image_size(image):
    """ComfyUIのIMAGEテンソルから、canvasに使う幅と高さを取得する。"""
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) < 3:
        raise ValueError("driving_image must be a ComfyUI IMAGE tensor")
    height, width = int(shape[-3]), int(shape[-2])
    if width <= 0 or height <= 0:
        raise ValueError("driving_image has an invalid size")
    return width, height
