# ComfyUI-PoseRetarget

SAM 3D Bodyで推定した2つの3D骨格を組み合わせ、
「reference画像の体型でdriving画像のポーズ」を取るOpenPose骨格を作るComfyUIノードです。

referenceのMHR70骨格から3D骨長比を測り、drivingの3Dボーン方向へ適用したあと、
driving側のカメラと焦点距離で2Dへ透視投影します。横向きや手足をカメラへ向けた
ポーズでも、2Dの見かけの長さから奥行きを推測する必要がありません。

## インストール

```bash
cd ComfyUI/custom_nodes
git clone git@github.com:gen-a6e/ComfyUI-PoseRetarget.git
```

このリポジトリ自体に追加の依存パッケージはありません（numpyはComfyUIに同梱）。

3D推定には既存の
[`ComfyUI-SAM3DBody`](https://github.com/PozzettiAndrea/ComfyUI-SAM3DBody)
を使います。ComfyUI Managerで`SAM3DBody`を検索してインストールし、
両方の拡張機能を読み込むためにComfyUIを再起動してください。

モデルのロードとVRAM管理は`ComfyUI-SAM3DBody`が担当します。
このリポジトリではSAM 3D Body本体やモデルを複製しません。

## SAM 3D Body Pose Retarget

`(Down)Load SAM 3D Body Model`の出力を、reference用とdriving用の
`SAM 3D Body: Process Image`へ接続し、両方とも`inference_type=full`で実行します。

`SAM 3D Body: Process Image`からは、一番上の`mesh_data`出力を使います。
`mesh_data`の型が、このノードの入力する`SAM3D_OUTPUT`です。
`skeleton`出力（`SKELETON`型）ではありません。

| 入力 | 説明 |
|---|---|
| `reference_sam3d` | reference側の`mesh_data` |
| `driving_sam3d` | driving側の`mesh_data` |
| `driving_image` | drivingに使った元画像。出力キャンバスの幅・高さを取得するために必要 |

| パラメータ | 既定 | 説明 |
|---|---|---|
| `size_reference` | torso | referenceの骨長比率と、drivingの肩ポーズ比率を測る3D基準。torso / shoulder_width / body_height / head_to_heel |
| `reference_symmetry` | average | averageは左右の推定誤差を平均化。offは左右それぞれの3D骨長をそのまま使用 |
| `uniform_scale` | 1.0 | 腰中央を基準にした全身サイズ |
| `leg_scale` | 1.0 | 脚と足の追加倍率 |
| `arm_scale` | 1.0 | 腕と手の追加倍率 |
| `head_scale` | 1.0 | 首から上の追加倍率 |
| `hand_scale` | 1.0 | 手指の追加倍率 |
| `fit_to_canvas` | shrink_to_fit | はみ出した場合のキャンバス調整 |
| `canvas_margin` | 16 | fit_exactly、または縮小が必要な場合の余白 |
| `torso_scale` | 1.0 | 腰中央から首までの胴長の追加倍率 |
| `shoulder_width_scale` | 1.0 | 肩幅の追加倍率 |
| `hip_width_scale` | 1.0 | 腰幅の追加倍率 |
| `neck_scale` | 1.0 | 肩中央から鼻までの長さの追加倍率 |
| `upper_arm_scale` | 1.0 | 上腕の追加倍率（`arm_scale`との積） |
| `forearm_scale` | 1.0 | 前腕の追加倍率（`arm_scale`との積） |
| `thigh_scale` | 1.0 | 太腿の追加倍率（`leg_scale`との積） |
| `shin_scale` | 1.0 | 脛の追加倍率（`leg_scale`との積） |

```text
(Down)Load SAM 3D Body Model ─┬→ Process Image ← reference画像 ─┐
                              └→ Process Image ← driving画像  ─┤
driving画像 ───────────────────────────────────────────────────┤
                                                               ↓
                                                  SAM 3D Body Pose Retarget
                                                               ↓
                                                  SDPose Draw → ControlNet
```

出力する`POSE_KEYPOINT`は、ComfyUI標準の`SDPose Draw Keypoints`に合わせた
絶対ピクセル座標です。

出力は、reference体型とdrivingポーズを合成した`pose_keypoint`、drivingの
MHR70を変形・fitなしで直接投影した`driving_pose_keypoint`、処理内容を示す
`report`の順です。

referenceの各骨長は`size_reference`で正規化し、reference自身の基準サイズへ戻して転送します。
そのため基本式は`出力骨長 = reference骨長比率 × reference_unit × uniform_scale × 部位別scale`で、各scaleが1.0ならreferenceの実際の3D骨長を維持します。drivingからは3D方向・ポーズを使用し、drivingの全体サイズは骨長の展開に使用しません。
肩幅、腰幅、肩中央から鼻までの長さは最終骨格上で直接保証されます。
肩中央の首に対する上下・奥行きと左右の肩線の傾きはdrivingから維持します。
`report`にはreferenceと生成後の主要な正規化比率を`reference->generated`形式で表示します。

`head_to_heel`は、MHRの全308キーポイントから選んだ頭頂点に、頭・胴体・左右平均の脚・かかとの3D骨格長を加えた、姿勢に影響されにくい全身長を基準にします。この選択肢を使う場合は、全キーポイント出力に対応した`ComfyUI-SAM3DBody`でreferenceとdrivingを再実行してください。

MHR70には鼻・目・耳はありますが、輪郭や口を含む密な顔ランドマークはありません。
そのためBODY18の顔点は出力し、`face_keypoints_2d`の70点はゼロconfidenceにします。
左右の手はそれぞれ21点を出力します。

現在は1画像・1人用です。`SAM 3D Body: Process Image`が選んだ先頭の人物を使います。

## ライセンス

MIT
