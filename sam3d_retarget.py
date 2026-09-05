"""SAM 3D BodyのMHR70を使って、体型とポーズを3Dで合成する計算モジュール。

処理の流れ:
1. ``SAM3D_OUTPUT``からreferenceとdrivingのMHR70座標を取得する。
2. referenceの3D骨長を取得し、必要なら左右の推定誤差を平均化する。
3. drivingのボーン方向・肩の上下・人物位置へreferenceの実骨長を適用する。
4. driving側のカメラで3D座標を2Dピクセル座標へ投影する。
5. BODY18＋左右HAND21の``POSE_KEYPOINT``形式へ変換する。
6. カメラ深度の奥から手前へ骨線を重ねた``IMAGE``も生成する。

SAM 3D Body本体やtorchには依存せず、辞書とnumpy配列だけを受け取る。これにより、
ComfyUIとの接続部分を薄く保ち、3D計算を単体テストできるようにしている。
"""

from __future__ import annotations

import numpy as np


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


# OpenPose BODY18と同じ接続。各色はRGBで、黒背景上でも部位を追いやすい配色にする。
# 3D深度で並べ替えるため、ここでは描画順を固定しない。
BODY_DRAW_EDGES = (
    (NECK, RIGHT_SHOULDER, (255, 85, 0)),
    (RIGHT_SHOULDER, RIGHT_ELBOW, (255, 170, 0)),
    (RIGHT_ELBOW, RIGHT_WRIST, (255, 255, 0)),
    (NECK, LEFT_SHOULDER, (85, 255, 0)),
    (LEFT_SHOULDER, LEFT_ELBOW, (0, 255, 0)),
    (LEFT_ELBOW, LEFT_WRIST, (0, 255, 85)),
    (NECK, RIGHT_HIP, (0, 255, 170)),
    (RIGHT_HIP, RIGHT_KNEE, (0, 255, 255)),
    (RIGHT_KNEE, RIGHT_ANKLE, (0, 170, 255)),
    (NECK, LEFT_HIP, (0, 85, 255)),
    (LEFT_HIP, LEFT_KNEE, (0, 0, 255)),
    (LEFT_KNEE, LEFT_ANKLE, (85, 0, 255)),
    (NECK, NOSE, (255, 0, 0)),
    (NOSE, RIGHT_EYE, (170, 0, 255)),
    (RIGHT_EYE, RIGHT_EAR, (255, 0, 255)),
    (NOSE, LEFT_EYE, (255, 0, 170)),
    (LEFT_EYE, LEFT_EAR, (255, 0, 85)),
)

# 指ごとに色を変える。左右の手で同じ色体系を使い、OpenPoseらしい表示にする。
HAND_DRAW_COLORS = (
    (255, 80, 80),
    (255, 190, 80),
    (80, 255, 120),
    (80, 190, 255),
    (210, 80, 255),
)


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


# SAPIENS/SDPoseが期待するCOCO BODY18順に、MHR70の関節番号を対応させる。
COCO18_FROM_MHR70 = (
    NOSE,
    NECK,
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


def extract_head_top(output, mhr70=None):
    """MHR全308点から頭頂点を選び、その3D座標と元の番号を返す。"""
    if not isinstance(output, dict):
        raise ValueError("SAM3D input must be a SAM3D_OUTPUT dictionary")
    value = output.get("keypoints_3d_full")
    if value is None:
        raw = output.get("raw_output") or {}
        value = raw.get("pred_keypoints_3d_full")
    if value is None:
        raise ValueError(
            "height reporting requires full MHR keypoints from the updated "
            "ComfyUI-SAM3DBody; rerun SAM 3D Body: Process Image"
        )

    points = as_numpy(value, "full MHR keypoints")
    while points.ndim > 2 and points.shape[0] == 1:
        points = points[0]
    if (
        points.ndim != 2
        or points.shape[0] < MHR_KEYPOINT_COUNT
        or points.shape[1] < 3
    ):
        raise ValueError(
            "SAM3D full MHR keypoints must have shape (308, 3); "
            f"received {points.shape}"
        )
    points = points[:MHR_KEYPOINT_COUNT, :3]

    body = extract_mhr70(output) if mhr70 is None else np.asarray(
        mhr70, dtype=np.float64
    )
    if body.shape != (MHR70_COUNT, 3):
        raise ValueError("mhr70 must have shape (70, 3)")

    # 70〜307番は密な頭部・顔ランドマーク。首から顔中心へ向かう軸上で最も遠い点を
    # 頭頂とする。腕の点を候補に含めないため、手を上げても頭頂と誤認しない。
    # 頭そのものの軸を使うので、首を傾けた姿勢にも追従する。
    head_center = np.mean(
        body[[NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR]], axis=0
    )
    head_axis = head_center - body[NECK]
    axis_length = np.linalg.norm(head_axis)
    if not np.isfinite(axis_length) or axis_length <= EPS:
        raise ValueError("SAM3D skeleton has no usable neck-to-head direction")
    head_axis /= axis_length

    dense = points[MHR70_COUNT:]
    valid = np.all(np.isfinite(dense), axis=1)
    dense = dense[valid]
    dense_indices = np.arange(MHR70_COUNT, MHR_KEYPOINT_COUNT)[valid]
    if not dense.size or np.max(np.ptp(dense, axis=0)) <= EPS:
        raise ValueError("SAM3D full MHR head keypoints are empty")

    selected = int(np.argmax((dense - body[NECK]) @ head_axis))
    return dense[selected].copy(), int(dense_indices[selected])


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
    """頭頂からかかとまでの、姿勢変化に強い概算3D身長を返す。"""
    # 直立時の上下差ではなく骨の長さを加算するため、屈伸や前屈でも縮みにくい。
    torso = np.linalg.norm(points[NECK] - hip_center(points))
    body_height = (
        torso
        + np.linalg.norm(points[NOSE] - shoulder_center(points))
        + 0.5 * (
            np.linalg.norm(points[LEFT_KNEE] - points[LEFT_HIP])
            + np.linalg.norm(points[RIGHT_KNEE] - points[RIGHT_HIP]))
        + 0.5 * (
            np.linalg.norm(points[LEFT_ANKLE] - points[LEFT_KNEE])
            + np.linalg.norm(points[RIGHT_ANKLE] - points[RIGHT_KNEE]))
    )
    # body_heightに頭頂→鼻と、左右平均の足首→かかとを加えて全身長にする。
    head_top = np.asarray(head_top, dtype=np.float64).reshape(-1)
    if head_top.size < 3 or not np.all(np.isfinite(head_top[:3])):
        raise ValueError("head_top must contain three finite coordinates")
    height = (
        body_height
        + np.linalg.norm(head_top[:3] - points[NOSE])
        + 0.5 * (
            np.linalg.norm(points[LEFT_HEEL] - points[LEFT_ANKLE])
            + np.linalg.norm(points[RIGHT_HEEL] - points[RIGHT_ANKLE])
        )
    )
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


def retarget_mhr70(reference, driving, reference_symmetry="average",
                   uniform_scale=1.0,
                   leg_scale=1.0, arm_scale=1.0, head_scale=1.0,
                   hand_scale=1.0, torso_scale=1.0,
                   shoulder_width_scale=1.0, hip_width_scale=1.0,
                   neck_scale=1.0, upper_arm_scale=1.0,
                   forearm_scale=1.0, thigh_scale=1.0,
                   shin_scale=1.0):
    """referenceの3D骨長とdrivingの3D方向・ポーズを合成する。"""
    # 入力をfloat64へ統一し、全関節が期待どおり70点あることを先に保証する。
    reference = np.asarray(reference, dtype=np.float64)
    driving = np.asarray(driving, dtype=np.float64)
    if reference.shape != (MHR70_COUNT, 3) or driving.shape != (MHR70_COUNT, 3):
        raise ValueError("reference and driving joints must both have shape (70, 3)")

    # SAM 3D Bodyが推定したreferenceの3D骨長を、正規化せず直接使用する。
    # 出力骨長は「reference骨長 × uniform_scale × 部位別scale」。
    uniform = float(uniform_scale)
    bone_lengths = reference_lengths(reference, reference_symmetry)
    reference_measurements = body_measurements(reference)
    driving_measurements = body_measurements(driving)

    # drivingを土台にすると、明示的に再配置しない補助点も元の位置を維持できる。
    output = driving.copy()
    # 人物の配置はdriving基準なので、腰中心そのものは移動させない。
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

    # 肩: 左右を結ぶ軸の傾きはdriving、肩幅はreferenceから移す。
    shoulder_axis = _unit_direction(
        driving[LEFT_SHOULDER] - driving[RIGHT_SHOULDER],
        reference[LEFT_SHOULDER] - reference[RIGHT_SHOULDER],
    )
    shoulder_width = (
        reference_measurements["shoulder_width"] * uniform
        * float(shoulder_width_scale)
    )
    # 両肩の中点を首へ固定すると「肩を落とす／すくめる」ポーズが消える。
    # そこで肩中央の首に対する上下・奥行きを、胴体長比でreference体格へ換算する。
    driving_shoulder_offset = shoulder_center(driving) - driving[NECK]
    driving_torso = driving_measurements["torso"]
    shoulder_pose_scale = uniform
    if np.isfinite(driving_torso) and driving_torso > EPS:
        shoulder_pose_scale *= reference_measurements["torso"] / driving_torso
    output_shoulder_center = (
        output[NECK]
        + driving_shoulder_offset * shoulder_pose_scale
    )
    output[LEFT_SHOULDER] = (
        output_shoulder_center + shoulder_axis * shoulder_width * 0.5)
    output[RIGHT_SHOULDER] = (
        output_shoulder_center - shoulder_axis * shoulder_width * 0.5)

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

    # 頭: 肩中央→鼻をreferenceの実際の3D距離として保証する。
    # 首や肩の複数ボーンを順に足す方式にしないことで、推定誤差の累積を避ける。
    driving_shoulders = shoulder_center(driving)
    reference_shoulders = shoulder_center(reference)
    nose_direction = _unit_direction(
        driving[NOSE] - driving_shoulders,
        reference[NOSE] - reference_shoulders,
    )
    neck_length = (
        reference_measurements["shoulder_to_nose"] * uniform
        * float(head_scale) * float(neck_scale)
    )
    output[NOSE] = shoulder_center(output) + nose_direction * neck_length

    # 目と耳: 鼻から外側へ、drivingの顔向きとreferenceの骨長で配置する。
    for child, parent in (
            (LEFT_EYE, NOSE), (RIGHT_EYE, NOSE),
            (LEFT_EAR, LEFT_EYE), (RIGHT_EAR, RIGHT_EYE)):
        _place_edge(
            output, driving, reference, child, parent,
            bone_lengths[child] * uniform * float(head_scale),
        )

    # 手: 手首を起点に各指を根元から指先へ順番に配置する。
    for child, parent, _ in HAND_EDGES:
        _place_edge(
            output, driving, reference, child, parent,
            bone_lengths[child] * uniform
            * float(arm_scale) * float(hand_scale),
        )

    # 呼び出し側でreferenceと生成後の実骨長を比較できるようにする。
    details = {
        "base_scale": uniform,
        "size_source": "reference",
        "shoulder_pose_scale": shoulder_pose_scale,
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


def fit_projected(points, valid, width, height, mode="shrink_to_fit", margin=16):
    """縦横比を維持したまま、有効な2D点をcanvas内へ収める。"""
    points = np.asarray(points, dtype=np.float64).copy()
    if mode == "off" or not np.any(valid):
        return points, 1.0
    if mode not in {"shrink_to_fit", "fit_exactly"}:
        raise ValueError(f"unknown fit mode: {mode}")

    subset = points[valid]
    lo = subset.min(axis=0)
    hi = subset.max(axis=0)
    if mode == "shrink_to_fit" and (
            lo[0] >= 0 and lo[1] >= 0 and hi[0] <= width and hi[1] <= height):
        # すでに全点が収まっている場合は、位置やサイズを一切変えない。
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
    points[valid] = (points[valid] - center) * scale + target_center
    return points, scale


def _openpose_field(projected, valid, indices):
    """MHR70の指定点を、OpenPoseの[x, y, confidence]配列へ並べ替える。"""
    out = np.zeros((len(indices), 3), dtype=np.float64)
    for output_index, mhr_index in enumerate(indices):
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


def _draw_segment(image, start, end, color, thickness):
    """外部描画ライブラリを使わず、丸端の線分をRGB画像へ描く。"""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    radius = max(int(np.ceil(float(thickness) * 0.5)), 1)
    height, width = image.shape[:2]
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
        return

    # 画面外の長大な投影線を先にcanvas周辺へ切り詰め、不要なサンプル生成を防ぐ。
    segment = end - start
    lower = np.array((-radius, -radius), dtype=np.float64)
    upper = np.array((width - 1 + radius, height - 1 + radius), dtype=np.float64)
    enter, leave = 0.0, 1.0
    for direction, distance in (
        (-segment[0], start[0] - lower[0]),
        (segment[0], upper[0] - start[0]),
        (-segment[1], start[1] - lower[1]),
        (segment[1], upper[1] - start[1]),
    ):
        if abs(direction) <= EPS:
            if distance < 0.0:
                return
            continue
        boundary = distance / direction
        if direction < 0.0:
            enter = max(enter, boundary)
        else:
            leave = min(leave, boundary)
        if enter > leave:
            return
    original_start = start
    start = original_start + enter * segment
    end = original_start + leave * segment
    segment = end - start
    sample_count = max(int(np.ceil(np.linalg.norm(segment))) + 1, 1)
    amount = np.linspace(0.0, 1.0, sample_count)
    samples = np.rint(start + amount[:, None] * segment).astype(np.int64)
    rgb = np.asarray(color, dtype=np.float32) / 255.0

    # 線分の各サンプルへ小さな円を押す方式なら、4K画像でも巨大なbbox配列を作らない。
    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            if offset_x ** 2 + offset_y ** 2 > radius ** 2:
                continue
            xx = samples[:, 0] + offset_x
            yy = samples[:, 1] + offset_y
            inside = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
            image[yy[inside], xx[inside]] = rgb


def _hand_draw_edges():
    """左右の指骨を、描画色付きの線分一覧にする。"""
    edges = []
    for wrist, chains in (
        (RIGHT_WRIST, RIGHT_HAND_CHAINS),
        (LEFT_WRIST, LEFT_HAND_CHAINS),
    ):
        for color, chain in zip(HAND_DRAW_COLORS, chains):
            parent = wrist
            for child in chain:
                edges.append((parent, child, color))
                parent = child
    return tuple(edges)


HAND_DRAW_EDGES = _hand_draw_edges()
DEPTH_DRAW_EDGES = BODY_DRAW_EDGES + HAND_DRAW_EDGES


def render_depth_pose(projected, depth, valid, width, height, thickness=None):
    """3D深度の奥から手前へ骨線を重ねたComfyUI IMAGEを返す。

    通常のPOSE_KEYPOINTはZを保持できない。ここでは各線分の両端の平均depthを使う
    painter's algorithmにより、胴体より後ろの手へ胴体線が重なるように描画する。
    """
    projected = np.asarray(projected, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if projected.shape != (MHR70_COUNT, 2):
        raise ValueError("projected MHR70 must have shape (70, 2)")
    if depth.shape != (MHR70_COUNT,) or valid.shape != (MHR70_COUNT,):
        raise ValueError("depth and valid must each have shape (70,)")
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("pose image dimensions must be positive")
    if thickness is None:
        thickness = max(2, int(round(min(width, height) / 128.0)))

    primitives = []
    for start_index, end_index, color in DEPTH_DRAW_EDGES:
        if not valid[start_index] or not valid[end_index]:
            continue
        mean_depth = 0.5 * (depth[start_index] + depth[end_index])
        if not np.isfinite(mean_depth):
            continue
        primitives.append((float(mean_depth), start_index, end_index, color))

    # カメラ座標Zが大きいほど遠い。遠い線を先に、近い線を後から描画する。
    primitives.sort(key=lambda item: item[0], reverse=True)
    image = np.zeros((height, width, 3), dtype=np.float32)
    for _, start_index, end_index, color in primitives:
        _draw_segment(
            image,
            projected[start_index],
            projected[end_index],
            color,
            thickness,
        )
    return image[None, ...]


def image_size(image):
    """ComfyUIのIMAGEテンソルから、canvasに使う幅と高さを取得する。"""
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) < 3:
        raise ValueError("driving_image must be a ComfyUI IMAGE tensor")
    height, width = int(shape[-3]), int(shape[-2])
    if width <= 0 or height <= 0:
        raise ValueError("driving_image has an invalid size")
    return width, height
