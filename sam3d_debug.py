"""入力SAM骨格の確認用描画。マージ・fitは行わず、元画像の座標系を維持する。

NumPyでメッシュのZバッファを作り、Pillowで関節とラベルを上描きする。
内部関節も見える透視図なので、点・線をメッシュの深度で隠すことはしない。
"""

import numpy as np

from . import sam3d_retarget as sr


# SAM公式 mhr70.py の original_keypoint_info と同じ名前・番号。
JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_big_toe_tip", "left_small_toe_tip",
    "left_heel", "right_big_toe_tip", "right_small_toe_tip", "right_heel",
]
for _side in ("right", "left"):
    for _finger in ("thumb", "index", "middle", "ring", "pinky"):
        JOINT_NAMES.extend(f"{_side}_{_finger}_{part}" for part in
                           ("tip", "first_joint", "second_joint", "third_joint"))
    JOINT_NAMES.append(f"{_side}_wrist")
JOINT_NAMES.extend(("left_olecranon", "right_olecranon", "left_cubital_fossa",
                    "right_cubital_fossa", "left_acromion", "right_acromion", "neck"))

GROUPS = {
    "body": tuple(range(5, 21)) + (41, 62, 69),
    "face": tuple(range(5)),
    "left_hand": sr.LEFT_HAND_FROM_MHR70,
    "right_hand": sr.RIGHT_HAND_FROM_MHR70,
    "auxiliary": tuple(range(63, 69)),
}
COLORS = {"body": (70, 195, 255), "face": (70, 240, 140),
          "left_hand": (255, 150, 55), "right_hand": (235, 100, 245),
          "auxiliary": (180, 180, 180), "height": (255, 230, 70)}


def _array3(value, name):
    """単一人物の(N,3)データのみ許可し、バッチを黙って捨てない。"""
    array = sr.as_numpy(value, name)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1] != 3 or not len(array):
        raise ValueError(f"{name} must have shape (N, 3)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def mesh_overlay(rgb, vertices, faces, camera, focal, opacity):
    """三角形の最前面だけを半透明合成する。重なる面で不透明度を累積させない。"""
    vertices = _array3(vertices, "vertices")
    faces = _array3(faces, "faces")
    if (faces != np.floor(faces)).any() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("faces must contain valid integer vertex indices")
    faces = faces.astype(np.int64)
    h, w = rgb.shape[:2]
    xy, valid, depth = sr.project_mhr70(vertices, camera, focal, w, h)
    # 非常に近い面を無理に描かない。関節投影や既存のマージ処理には影響しない。
    usable = valid & (depth > 1e-5) & np.isfinite(xy).all(axis=1)
    zbuffer = np.full((h, w), np.inf, dtype=np.float32)
    shade = np.zeros((h, w), dtype=np.float32)
    skipped = 0
    for face in faces:
        if not usable[face].all():
            skipped += 1
            continue
        triangle = xy[face]
        lo = np.maximum(np.floor(triangle.min(axis=0)), 0)
        hi = np.minimum(np.ceil(triangle.max(axis=0)), (w - 1, h - 1))
        if (lo > hi).any():
            continue
        x0, y0 = lo.astype(int)
        x1, y1 = hi.astype(int)
        a, b, c = triangle
        denominator = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
        if abs(denominator) < 1e-10:
            continue
        normal = np.cross(vertices[face[1]]-vertices[face[0]],
                          vertices[face[2]]-vertices[face[0]])
        light = 0.4 + 0.6 * abs(normal[2]) / max(np.linalg.norm(normal), 1e-12)
        # 行を分割して、大きな三角形でも一時配列が画像全体の何倍にもならないようにする。
        for row in range(y0, y1 + 1, 64):
            end = min(row + 64, y1 + 1)
            xx = np.arange(x0, x1 + 1)[None, :] + 0.5
            yy = np.arange(row, end)[:, None] + 0.5
            u = ((b[1]-c[1])*(xx-c[0]) + (c[0]-b[0])*(yy-c[1])) / denominator
            v = ((c[1]-a[1])*(xx-c[0]) + (a[0]-c[0])*(yy-c[1])) / denominator
            t = 1 - u - v
            inside = (u >= -1e-8) & (v >= -1e-8) & (t >= -1e-8)
            # 透視投影下では逆深度を補間する。
            inv_z = u/depth[face[0]] + v/depth[face[1]] + t/depth[face[2]]
            z = np.full_like(inv_z, np.inf)
            np.divide(1, inv_z, out=z, where=inside & (inv_z > 0))
            region = zbuffer[row:end, x0:x1+1]
            nearer = inside & (z < region)
            region[nearer] = z[nearer]
            shade[row:end, x0:x1+1][nearer] = light
    mask = np.isfinite(zbuffer)
    result = rgb.copy()
    mesh_color = np.array([0.48, 0.76, 0.94], dtype=np.float32)
    result[mask] = rgb[mask]*(1-opacity) + shade[mask, None]*mesh_color*opacity
    return result, int(mask.sum()), skipped


def debug_geometry(mesh_data, show_body=True, show_face=True, show_left_hand=True,
                   show_right_hand=True, show_height=True, show_head_candidates=False,
                   show_auxiliary=False):
    """描画点・線を組み立てる。中心点は3Dで計算してから投影する。"""
    body = sr.extract_mhr70(mesh_data)
    points = list(body) + [sr.shoulder_center(body), sr.hip_center(body)]
    names = [f"{i}: {name}" for i, name in enumerate(JOINT_NAMES)]
    names += ["S: shoulder_center", "H: hip_center"]
    enabled = dict(body=show_body, face=show_face, left_hand=show_left_hand,
                   right_hand=show_right_hand, auxiliary=show_auxiliary)
    colors = {}
    for group, on in enabled.items():
        if on:
            colors.update({i: COLORS[group] for i in GROUPS[group]})
    for index, color in ((69, (255, 65, 65)), (5, (55, 120, 255)),
                         (6, (55, 120, 255)), (0, (65, 240, 100))):
        if index in colors:
            colors[index] = color
    if show_body or show_height:
        colors.update({70: (255, 230, 70), 71: (180, 130, 255)})
    edges = [(child, 71 if parent is None else parent)
             for child, parent, _ in sr.RETARGET_EDGES]
    # 幅と、実際にマージで使う肩中央→鼻・首→肩中央も明示する。
    edges += [(5, 6), (9, 10), (69, 70), (70, 0)]
    notes = []
    if show_height or show_head_candidates:
        try:
            head, index = sr.extract_head_top(mesh_data, body)
            selected = len(points)
            points.append(head)
            names.append(f"{index}: selected_head_top (estimated)")
            colors[selected] = COLORS["height"]
            if show_height:
                height_ids = (0, 9, 10, 11, 12, 13, 14, 17, 20, 69)
                for i in height_ids:
                    colors.setdefault(i, COLORS["height"])
                edges.append((0, selected))
                notes.append(f"estimated_height={sr.estimated_height(body, head):.3f} m; "
                             f"selected_head_top_full_index={index}")
            if show_head_candidates:
                full = mesh_data.get("keypoints_3d_full")
                if full is None:
                    full = (mesh_data.get("raw_output") or {}).get("pred_keypoints_3d_full")
                full = sr.as_numpy(full, "full keypoints")
                while full.ndim > 2 and full.shape[0] == 1:
                    full = full[0]
                full = full[:308, :3]
                for i in range(70, 308):
                    if i != index and np.isfinite(full[i]).all():
                        colors[len(points)] = (160, 170, 110)
                        points.append(full[i])
                        names.append(f"{i}: head_candidate")
        except ValueError as exc:
            notes.append(f"WARNING: height/head candidates unavailable: {exc}")
    edges = [(a, b) for a, b in edges if a in colors and b in colors]
    return np.asarray(points), names, colors, edges, notes


def _clip_line(a, b, width, height):
    """画面外へ伸びる線も端まで描くが、巨大座標はPillowへ渡さない。"""
    delta = b - a
    low, high = 0.0, 1.0
    for origin, direction, limit in zip(a, delta, (width-1, height-1)):
        if abs(direction) < 1e-12:
            if origin < 0 or origin > limit:
                return None
        else:
            near, far = sorted((-origin/direction, (limit-origin)/direction))
            low, high = max(low, near), min(high, far)
            if low > high:
                return None
    return tuple(a + low*delta), tuple(a + high*delta)


def render_debug(mesh_data, image, mesh_opacity=0.35, show_mesh=True,
                 show_body=True, show_face=True, show_left_hand=True,
                 show_right_hand=True, show_height=True, show_head_candidates=False,
                 show_auxiliary=False, show_connections=True,
                 labels="index_and_name", point_radius=4, font_size=12):
    """1画像・1人物を描画し、IMAGE用のfloat32配列と診断reportを返す。"""
    from PIL import Image, ImageDraw, ImageFont

    rgb = sr.as_numpy(image, "image")
    if rgb.ndim != 4 or rgb.shape[0] != 1 or rgb.shape[-1] != 3:
        raise ValueError("Debug node requires one IMAGE with shape (1, H, W, 3)")
    if not np.isfinite(rgb).all() or min(rgb.shape[1:3]) < 1:
        raise ValueError("image contains invalid pixels or dimensions")
    if not 0 <= mesh_opacity <= 1:
        raise ValueError("mesh_opacity must be between 0 and 1")
    if labels not in ("off", "index", "index_and_name"):
        raise ValueError("unknown labels mode")
    if not 1 <= point_radius <= 20 or not 8 <= font_size <= 32:
        raise ValueError("invalid point_radius or font_size")
    rgb = np.clip(rgb[0], 0, 1).astype(np.float32)
    h, w = rgb.shape[:2]
    camera, focal = sr.extract_camera(mesh_data)
    points, names, colors, edges, notes = debug_geometry(
        mesh_data, show_body, show_face, show_left_hand, show_right_hand,
        show_height, show_head_candidates, show_auxiliary)
    if show_mesh and mesh_opacity > 0:
        try:
            vertices = mesh_data.get("vertices")
            if vertices is None:
                vertices = (mesh_data.get("raw_output") or {}).get("pred_vertices")
            rgb, pixels, skipped = mesh_overlay(
                rgb, vertices, mesh_data.get("faces"), camera, focal, mesh_opacity)
            notes.append(f"mesh_pixels={pixels}; near/behind_camera_faces_skipped={skipped}")
        except ValueError as exc:
            notes.append(f"WARNING: mesh unavailable: {exc}")
    xy, valid, depth = sr.project_mhr70(points, camera, focal, w, h)
    valid &= np.isfinite(xy).all(axis=1)
    canvas = Image.fromarray(np.rint(rgb*255).astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:  # 古いPillowは既定フォントのサイズ指定に未対応。
            font = ImageFont.load_default()
            notes.append("WARNING: fallback bitmap font uses a fixed size on this Pillow version")
    if show_connections:
        for a, b in edges:
            if valid[a] and valid[b]:
                segment = _clip_line(xy[a], xy[b], w, h)
                if segment:
                    draw.line(segment, fill=colors[a], width=2)
    visible = []
    for i in colors:
        x, y = xy[i]
        if valid[i] and 0 <= x < w and 0 <= y < h:
            visible.append(i)
    # ラベルは簡単な衝突回避を行う。密な手指はindex表示や部位切替でも確認できる。
    occupied = []
    label_layout = []
    if labels != "off":
        for i in visible:
            x, y = xy[i]
            label = names[i] if labels == "index_and_name" else names[i].split(":")[0]
            box = draw.textbbox((0, 0), label, font=font)
            tw, th = box[2]-box[0]+4, box[3]-box[1]+4
            chosen = None
            offsets = [0]
            for step in range(1, int(h / (th+3)) + 1):
                offsets.extend((-step*(th+3), step*(th+3)))
            found = False
            for offset in offsets:
                for side_x in (x+point_radius+3, x-point_radius-3-tw):
                    tx = max(0, min(side_x, w-tw))
                    ty = max(0, min(y+offset, h-th))
                    chosen = (tx, ty, tx+tw, ty+th)
                    if not any(tx < b[2] and tx+tw > b[0] and ty < b[3] and ty+th > b[1]
                               for b in occupied):
                        found = True
                        break
                if found:
                    break
            occupied.append(chosen)
            tx, ty = chosen[:2]
            label_layout.append((i, chosen, box, label))
        # 全引き出し線を先に描き、別のラベルの文字を線で横切らないようにする。
        for i, chosen, _, _ in label_layout:
            draw.line((tuple(xy[i]), (chosen[0], (chosen[1]+chosen[3])/2)),
                      fill=colors[i], width=1)
        for i, chosen, box, label in label_layout:
            draw.rectangle(chosen, fill=(15, 18, 24))
            draw.text((chosen[0]+2-box[0], chosen[1]+2-box[1]),
                      label, font=font, fill=colors[i])
    # ラベルが関節そのものを塗りつぶさないよう、マーカーを最後に描く。
    for i in visible:
        x, y = xy[i]
        r = point_radius
        bounds = (x-r, y-r, x+r, y+r)
        if i >= 70:
            draw.rectangle(bounds, fill=colors[i], outline="black")
        else:
            draw.ellipse(bounds, fill=colors[i], outline="black")
    notes[:0] = [
        "Input SAM skeleton only; no retarget, no fit. Use the original inference image.",
        "Joints/lines are x-ray overlays (not mesh-occluded). L/R are subject's sides.",
        "Red=69 neck; blue=5/6 shoulders; green=0 nose; yellow=S shoulder center; violet=H hip center.",
        "Head top is a selected estimate, not a verified anatomical landmark.",
        f"visible_points={len(visible)}/{len(colors)}; canvas={w}x{h}",
        "point | pixel_x pixel_y | estimated_camera_z | visibility",
    ]
    for i in colors:
        state = "visible" if i in visible else ("off_canvas" if valid[i] else "not_projectable")
        notes.append(f"{names[i]} | {xy[i,0]:.2f} {xy[i,1]:.2f} | {depth[i]:.4f} | {state}")
    return np.asarray(canvas, dtype=np.float32)[None] / 255.0, "\n".join(notes)


class SAM3DBodySkeletonDebug:
    """入力画像・メッシュ・MHR関節を重ねて確認する独立ノード。"""

    @classmethod
    def INPUT_TYPES(cls):
        required = {"mesh_data": ("SAM3D_OUTPUT",), "image": ("IMAGE",),
                    "mesh_opacity": ("FLOAT", {"default": 0.35, "min": 0.0,
                                               "max": 1.0, "step": 0.05})}
        for name, default in (("show_mesh", True), ("show_body", True),
                              ("show_face", True), ("show_left_hand", True),
                              ("show_right_hand", True), ("show_height", True),
                              ("show_head_candidates", False), ("show_auxiliary", False),
                              ("show_connections", True)):
            required[name] = ("BOOLEAN", {"default": default})
        required.update({"labels": (["index_and_name", "index", "off"],),
                         "point_radius": ("INT", {"default": 4, "min": 1, "max": 20}),
                         "font_size": ("INT", {"default": 12, "min": 8, "max": 32})})
        return {"required": required}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("debug_image", "report")
    FUNCTION = "run"
    CATEGORY = "pose-retarget/debug"

    def run(self, mesh_data, image, **options):
        result, report = render_debug(mesh_data, image, **options)
        # ComfyUI IMAGEはCPU float32 Tensor。単体テストではnumpyも受け付ける。
        if hasattr(image, "detach"):
            import torch
            result = torch.from_numpy(result)
        return result, report
