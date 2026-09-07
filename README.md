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
| `reference_symmetry` | average | 身体・手の左右骨長を平均化。offは左右差を維持。顔は設定によらずreference形状を維持 |
| `uniform_scale` | 1.0 | 全身サイズ。再構成後に鼻X/Z・最下点Yで位置合わせ |
| `leg_scale` | 1.0 | 脚と足の追加倍率 |
| `arm_scale` | 1.0 | 腕と手の追加倍率 |
| `head_scale` | 1.0 | 首から上の追加倍率 |
| `hand_scale` | 1.0 | 手指の追加倍率 |
| `torso_scale` | 1.0 | 腰中央から首までの胴長の追加倍率 |
| `shoulder_width_scale` | 1.0 | 肩リグ4区間の追加倍率。肩幅と肩周囲の前後・上下オフセットにも作用 |
| `hip_width_scale` | 1.0 | 腰幅の追加倍率 |
| `neck_scale` | 1.0 | 肩中央から鼻までの長さの追加倍率 |
| `upper_arm_scale` | 1.0 | 上腕の追加倍率（`arm_scale`との積） |
| `forearm_scale` | 1.0 | 前腕の追加倍率（`arm_scale`との積） |
| `thigh_scale` | 1.0 | 太腿の追加倍率（`leg_scale`との積） |
| `shin_scale` | 1.0 | 脛の追加倍率（`leg_scale`との積） |
| `fit_to_canvas` | off | 既定ではキャンバス調整なし。必要な場合にfitを選択 |
| `canvas_margin` | 16 | fit_exactly、または縮小が必要な場合の余白 |

`fit_to_canvas`と`canvas_margin`は入力設定の末尾にあります。旧版から更新する場合は、保存済みwidget値の順序が変わるため、既存ノードの設定を控えてからノードを追加し直してください。

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
腰幅、肩中央から鼻までの長さは最終骨格上で直接保証されます。肩幅は左右肩間の距離を固定せず、下記の肩リグを組み立てた結果として決まります。
顔の目・耳は個別の骨長転送ではなく、referenceの鼻からの相対位置をひとまとまりで回転・拡縮します。鼻・両目・両耳の5点にKabsch法を適用してreferenceからdrivingへの回転を推定し、`uniform_scale × head_scale`を適用して生成鼻へ配置します。referenceの目幅・耳幅・左右差を維持し、`reference_symmetry=average`でも顔は平均化しません。鼻の配置式は従来どおりで、`neck_scale`は顔内部のサイズを変えません。
回転は顔点の対応からの推定で、内部リグの頭回転を直接使用する方式ではありません。顔が直線・一点に潰れて向きを決められない場合はreferenceの顔向きを維持し、`report`に警告と`face_rotation=identity_fallback_degenerate_face`を表示します。通常は`face_rotation=kabsch_face5`です。driving再投影・raw出力・デバッグの元座標は変更しません。
肩は内部リグMHR127を使い、以下の4区間の**長さをreference、3D方向をdriving**から取ります。

```text
R37 c_spine3
├─ R38 r_clavicle → R39 r_uparm（MHR70右肩6）
└─ R74 l_clavicle → R75 l_uparm（MHR70左肩5）
```

R37は生成した首MHR69を基準に配置します。首69→R37の方向はdriving、長さはreferenceに`uniform_scale × torso_scale`を掛けた値です。肩の4区間には`uniform_scale × shoulder_width_scale`を適用します。幅だけを横方向に動かす方式ではないため、肩の前後・上下位置にも影響します。`reference_symmetry=average`なら左右の対応区間をそれぞれ平均し、`off`なら左右別の長さを保持します。

肩を前に出す・開くといった方向の変化に応じて、最終的な左右肩間距離も変わります。端から端の肩幅がreferenceと同じであることは保証しません。R37から鎖骨起点への区間はリグ上の接続であり、解剖学的な一本の骨の長さとは限りません。

`report`には通常`shoulder_mode=rig_chain`と、左右平均後のreference区間長→生成区間長、首69→R37の長さを表示します。内部リグは既存の`mesh_data["joint_coords"]`または`raw_output["pred_joint_coords"]`から取得し、入力・widget・出力順の変更はありません。

リグが欠ける・対象区間が潰れる・肩点R39/R75がMHR70肩点と一致しない場合は、**警告付きで従来の肩幅固定方式**へ戻します（`shoulder_mode=legacy_width_fallback`）。この場合のみ、肩中央の首に対するオフセットをreference胴体長÷driving胴体長で換算します。未使用のR126などの異常は肩リグ転送を止めません。旧データでも生成できますが、実機検証では`rig_chain`になっていることを確認してください。
配置は、3D骨格を組み立てた後に全点を同量だけ平行移動し、**鼻のX・Zと、最下点のYをdrivingへ合わせます**。SAMの投影用座標はY正方向が下なので、最下点はY最大の点です。身体・顔・手指・足先・かかと・首を対象にし、補助点63〜68は除外します。生成側とdriving側で最下点の関節が異なっても、それぞれの最下点の高さを揃えます。手が最下点なら手が基準であり、接地を保証するIKではありません。骨長・各点の相対位置は変えず、腰中央は固定しません。`report`には移動量と両方の最下点インデックスを表示します。
これは3D座標での整列なので、鼻のYや画像上の足元まで一致するとは限りません。`fit_to_canvas=off`ならこの投影配置を維持し、fitを有効にすると2Dでさらに移動・拡縮されます。driving比較用2出力やデバッグの元骨格にはこの整列を適用しません。
OpenPoseの首スロット（BODY18の1番）は、DWPoseと同様に**投影後の左右肩の2D中点**を出力します。マージ結果・driving再投影・SAM raw 2Dの3出力に共通です。片肩が無効なら首点のconfidenceも0にします。内部3DのMHR69は変更せず、骨格マージ・身長計測・デバッグでは本来の首点を使います。デバッグのSは3D肩中央を投影した点なので、透視投影ではOpenPoseの2D中点と一致しない場合があります。
canvas fitの範囲計算は、実際に出力するBODY18と左右HAND21の有効点だけを対象にします。描画しないつま先・かかと・補助点・MHR69は縮小率や配置に影響しません。画面外にある手・肘などの出力点は、引き続きfit対象です。
`report`にはreferenceとdrivingの概算身長、およびreferenceと生成後の主要な実骨長を`reference->generated`形式で表示します。

概算身長は、内部リグの`R126: c_head_null`を頭頂として直接使い、以下の3D直線距離を合計します。

```text
概算身長 = 距離(R126, 69) + 距離(69, H)
         + (左脚長 + 右脚長) / 2
左脚長 = 距離(9, 11) + 距離(11, 13) + 距離(13, 17)
右脚長 = 距離(10, 12) + 距離(12, 14) + 距離(14, 20)
H = 左右股関節(9, 10)の3D中点
```

鼻や肩中央は経由せず、Hから股関節への腰幅方向の距離も加算しません。全308点からの
頭頂候補選択は廃止しました。身長計測にはMHR70と`joint_coords`（内部リグ127点）が必要で、
全308点は不要です。`report`の頭頂出典も`reference=R126, driving=R126`と表示します。
単眼推定と区間の直線距離に基づく概算なので、実測の直立身長ではありません。

身長とraw 2D診断は補足情報です。内部リグのR126が欠ける・使用不能な場合は該当側の身長を`unavailable`とし、通常のポーズ生成は続行します。SAM内部2Dが欠ける・使用不能な場合もマージ結果とdriving再投影は返し、`sam_raw_driving_pose_keypoint`はキャンバス情報と空の`people`リストを返します。再投影座標をraw座標の代わりには使いません。省略理由は`report`の`WARNING`に表示します。MHR70やカメラなど、ポーズ生成に必須の情報が不正な場合は引き続きエラーになります。

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
| `show_height` | R126→69→Hと左右の脚→かかとの計測点・経路 |
| `show_head_candidates` | 全308点の70〜307番を参考表示（既定OFF）。互換性のため旧widget名を維持。身長計算には使わない |
| `show_auxiliary` | マージで再配置しない63〜68番の補助点（既定OFF） |
| `show_connections` | 関節接続、肩幅・腰幅・中心点との線 |
| `labels` | 番号＋名前／番号のみ／OFF |
| `point_radius` / `font_size` | 点の半径とラベル文字サイズ |
| `show_rig` | 内部リグMHR127の表示（既定OFF）。MHR70とは別の関節体系 |
| `rig_scope` | `head_neck`は頭・首のR110〜R126、`all`は全127関節 |

69番neckは赤、5・6番肩は青、0番鼻は緑。3Dで計算した肩中央（S）は黄、腰中央（H）は紫の
四角マーカーです。関節・線はメッシュの**内部も見える上描き**で、遮蔽判定は行いません。
点の重なりが密なときは部位を絞り、ラベルを番号のみにしてください。reportにも表示点の
名前・ピクセル座標・推定カメラ奥行き・画面内外を一覧します。

頭頂はR126を使用します。R126が取得できない場合は頭頂・身長表示だけを省略し、理由をreportへ出します。メッシュが欠損・不正な
場合も警告付きで関節描画を続行します。MHR70とカメラの異常はエラーになります。
密な頭部ランドマークを参考表示する場合のみ、全308点を出す
[`gen-a6e/ComfyUI-SAM3DBody`](https://github.com/gen-a6e/ComfyUI-SAM3DBody/tree/fix/pytorch-device-compat)
の`fix/pytorch-device-compat`ブランチが必要です。

### 内部リグで頭頂付近を確認する

`show_rig=ON`、`rig_scope=head_neck`にすると、`mesh_data["joint_coords"]`の
頭・首の内部関節を表示します。追加の入力接続やSAM側のコード更新は不要です。
MHRの番号と混同しないよう、`R126: c_head_null`のように`R`を付けたラベルと菱形を使います。
`R113: c_head`はオレンジ、`R126: c_head_null`は白の大きなマーカー、ほかはシアンです。
親子接続は`show_connections`で切り替えられ、頭・首表示では範囲外の親へは線を引きません。

まずメッシュとリグだけを見るなら、`show_body / show_face / show_left_hand / show_right_hand /
show_height`をOFF、`show_mesh / show_rig`をONにしてください。身長の計測経路も確認する場合は
`show_height`もONにします。R126のマーカーは共有し、重複表示しません。内部リグの親子接続と
身長計測用の接続は別であり、身長はR113を経由せずR126→MHR69の直線距離を使います。

内部リグの座標はProcess Image側でメートル換算・Y/Z反転済みのため、そのままメッシュと
同じカメラで投影します。名前と親子階層は、mhr_model.ptから抽出された
[MHR127対応表（固定版）](https://github.com/AmmarkoV/SAM3DBody-cpp/blob/db3fd03dd6e556aaf1774bbcb02f0a9c040b862b/src/mhr_joint_table.h)
を使用します。127点以外の構成は対応外です。内部リグが欠損・不正な場合は、その表示だけを
省略してreportに理由を出し、既存のメッシュ・MHR点の描画は継続します。
設定は既存widgetの末尾へ追加しています。更新後に設定が見えない場合はデバッグノードを
追加し直してください。Pose Retargetノードの入出力・計算に変更はありません。

## ライセンス

MIT
