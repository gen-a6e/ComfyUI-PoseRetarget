# ComfyUI-PoseRetarget

SAM 3D Bodyで推定した2つの3D骨格を組み合わせ、
「reference画像の体型でdriving画像のポーズ」を取るOpenPose骨格を作るComfyUIノードです。

referenceのMHR70骨格から3D骨長を直接取得し、drivingの3Dボーン方向へ適用したあと、
driving側のカメラと焦点距離で2Dへ透視投影します。横向きや手足をカメラへ向けた
ポーズでも、2Dの見かけの長さから奥行きを推測する必要がありません。

## インストール

```bash
cd ComfyUI/custom_nodes
git clone git@github.com:gen-a6e/ComfyUI-PoseRetarget.git
```

追加インストールは不要です（numpyと、デバッグ描画用のPillowはComfyUIに同梱）。

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
`report`、SAM内部で投影済みの2D点をそのまま変換した
`sam_raw_driving_pose_keypoint`の順です。

`driving_pose_keypoint`と`sam_raw_driving_pose_keypoint`は、SAMの3D手座標を
こちらで再投影した結果と、SAM内部の2D投影結果を比較する診断用です。`report`には
左右の手について`raw2d_vs_reprojected`のRMS差・最大差をピクセル単位で表示します。
raw側だけが正しい場合は再投影処理、両方が同じ場合はSAMのMHR70手座標より前段を
調査できます。

referenceの各骨長は正規化せず、SAM 3D Bodyが推定した3D距離を直接転送します。
基本式は`出力骨長 = reference骨長 × uniform_scale × 部位別scale`です。各scaleが1.0ならreferenceの3D骨長を維持し、drivingからは3D方向・ポーズを使用します。
肩幅、腰幅、肩中央から鼻までの長さは最終骨格上で直接保証されます。
肩中央の首に対する上下・奥行きはreference胴体長÷driving胴体長で体格換算し、左右の肩線の傾きはdrivingから維持します。
canvas fitの範囲計算は、実際に出力するBODY18と左右HAND21の有効点だけを対象にします。描画しないつま先・かかと・補助点は縮小率や配置に影響しません。画面外にある手・肘などの出力点は、引き続きfit対象です。
`report`にはreferenceとdrivingの概算身長、およびreferenceと生成後の主要な実骨長を`reference->generated`形式で表示します。

概算身長は、MHRの全308キーポイントから選んだ頭頂点に、頭・胴体・左右平均の脚・かかとの3D骨格長を加えて計測します。単眼画像からの推定値なので実測身長ではありません。身長を表示するには、全キーポイント出力に対応した`ComfyUI-SAM3DBody`でreferenceとdrivingを実行してください。

身長とraw 2D診断は補足情報です。全308点が欠ける・使用不能な場合は該当側の身長を`unavailable`とし、通常のポーズ生成は続行します。SAM内部2Dが欠ける・使用不能な場合もマージ結果とdriving再投影は返し、`sam_raw_driving_pose_keypoint`はキャンバス情報と空の`people`リストを返します。再投影座標をraw座標の代わりには使いません。省略理由は`report`の`WARNING`に表示します。MHR70やカメラなど、ポーズ生成に必須の情報が不正な場合は引き続きエラーになります。

MHR70には鼻・目・耳はありますが、輪郭や口を含む密な顔ランドマークはありません。
そのためBODY18の顔点は出力し、`face_keypoints_2d`の70点はゼロconfidenceにします。
左右の手はそれぞれ21点を出力します。

現在は1画像・1人用です。`SAM 3D Body: Process Image`が選んだ先頭の人物を使います。

## SAM 3D Body Skeleton Debug

入力SAM骨格の位置を、元画像と半透明メッシュに重ねて確認する独立ノードです。
マージ後の骨格は扱わず、既存のPose Retargetノードの計算・入出力も変更しません。

```text
Process Image の mesh_data → SAM 3D Body Skeleton Debug → debug_image → Preview Image
推論に使った元画像 ──────────↗                           → report
```

画像はSAM推論時と同じサイズ・内容を接続してください。自動fit・リサイズ・左右反転はせず、
メッシュと関節に同じSAMカメラ・焦点距離・画像中心を使います。別サイズ画像との誤接続は
入力データに元画像サイズがないため自動検出できません。1画像・1人物専用です。

| 設定 | 内容 |
|---|---|
| `show_mesh` / `mesh_opacity` | メッシュ表示と不透明度。CPUのZバッファで最前面を半透明合成 |
| `show_body` | 肩・腰・腕・脚・足先・かかと・首 |
| `show_face` | 鼻・目・耳 |
| `show_left_hand` / `show_right_hand` | 左右HAND21。左右は被写体本人の左右 |
| `show_height` | 現在の身長計測に使う点と、選択された頭頂候補。身長計算自体は変更しない |
| `show_head_candidates` | 全308点のうち頭頂選択の対象になる70〜307番も表示（既定OFF） |
| `show_auxiliary` | マージで再配置しない63〜68番の補助点（既定OFF） |
| `show_connections` | 関節接続、肩幅・腰幅・中心点との線 |
| `labels` | 番号＋名前／番号のみ／OFF |
| `point_radius` / `font_size` | 点の半径とラベル文字サイズ |

69番neckは赤、5・6番肩は青、0番鼻は緑。3Dで計算した肩中央（S）は黄、腰中央（H）は紫の
四角マーカーです。関節・線はメッシュの**内部も見える上描き**で、遮蔽判定は行いません。
点の重なりが密なときは部位を絞り、ラベルを番号のみにしてください。reportにも表示点の
名前・ピクセル座標・推定カメラ奥行き・画面内外を一覧します。

頭頂は現在の選択ロジックによる推定候補で、解剖学的な頭頂を保証しません。
全308点がない場合は頭頂・身長表示を省略し、理由をreportへ出します。メッシュが欠損・不正な
場合も警告付きで関節描画を続行します。MHR70とカメラの異常はエラーになります。
頭頂候補を表示する場合は全308点を出す
[`gen-a6e/ComfyUI-SAM3DBody`](https://github.com/gen-a6e/ComfyUI-SAM3DBody/tree/fix/pytorch-device-compat)
の`fix/pytorch-device-compat`ブランチが必要です。

## ライセンス

MIT
