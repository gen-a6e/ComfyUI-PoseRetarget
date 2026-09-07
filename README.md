# ComfyUI-PoseRetarget

SAM 3D Bodyで推定した2つの3D骨格を組み合わせ、
「reference画像の体型でdriving画像のポーズ」を取るOpenPose骨格を作るComfyUIノードです。

referenceのMHR127内部リグから3D区間長を取得し、drivingのリグ方向へ適用します。
再構成したリグからMHR70の出力点を配置したあと、
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
| `head_scale` | 1.0 | R113→R126と、R113基準の顔5点の追加倍率 |
| `hand_scale` | 1.0 | 手指の追加倍率 |
| `torso_scale` | 1.0 | R1→R34→R35→R36→R37の背骨4区間の追加倍率 |
| `shoulder_width_scale` | 1.0 | 肩リグ4区間の追加倍率。肩幅と肩周囲の前後・上下オフセットにも作用 |
| `hip_width_scale` | 1.0 | R1→左右股関節の区間倍率。左右幅だけでなく上下・前後成分にも作用 |
| `neck_scale` | 1.0 | R37→R110→R113、R110→R111と首69の微小位置差の追加倍率 |
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

referenceの各区間長は正規化せず、SAM 3D Bodyが推定した3D距離を直接転送します。
基本式は`出力区間長 = reference区間長 × uniform_scale × 部位別scale`です。

### 背骨・首・頭の中心リグ

```text
R1 root → R34 → R35 → R36 → R37（背骨）
                              ├─ R38 → R39（右肩6）
                              ├─ R74 → R75（左肩5）
                              └─ R110（首）→ R113（頭）→ R126（頭頂）
```

中心の7区間は**長さをreference、3D方向をdriving**から取得します。腰中央H→首69の直線距離を固定する方式ではなく、途中の曲がりも区間ごとに反映します。したがって、H→69の直線距離はreferenceと同じになる保証はありません。各接続はリグ上の区間であり、解剖学的な一本の骨と同一ではありません。

| 対象 | 適用倍率 |
|---|---|
| R1→R34→R35→R36→R37 | `uniform_scale × torso_scale` |
| R37→R110→R113 | `uniform_scale × neck_scale` |
| R113→R126 | `uniform_scale × head_scale` |
| R113から鼻・目・耳への相対位置 | `uniform_scale × head_scale` |
| 肩の4区間 | `uniform_scale × shoulder_width_scale` |
| 腰中央H→R1の補助オフセット | `uniform_scale` |
| R110→R111とMHR69の中点からの微小位置差 | `uniform_scale × neck_scale` |

Hは左右股関節9・10の中点、R1は内部ルートで別の位置です。drivingのHを仮の起点とし、H→R1もreferenceの距離・drivingの方向で配置します。最終的な股関節はR1から生成するため、Hの位置自体は固定しません。首69は生成R110/R111の中点を基準とし、SAMの丸めなどによるreferenceの中点からの残差も回転・転送して保持します。H→R1とこの残差は骨長ではなく配置用です。

### 顔はR113を基準に一体で配置

鼻0・左右目1/2・左右耳3/4の5点へ、共通の回転・倍率を適用します。

```text
生成顔点 = 生成R113 + 回転(reference顔点 − reference R113) × uniform_scale × head_scale
```

回転は顔5点からKabsch法で推定します。referenceの目幅・耳幅・鼻との位置関係・左右差と、R113から各顔点までの距離を保持します。顔は`reference_symmetry=average`でも平均化しません。目を先に作って鼻を個別に接続する処理や、内部リグの目をMHR70の目へ置き換える処理は行いません。

通常は`face_anchor=R113`。肩中央→鼻の距離は固定せず、顔の局所配置は肩幅に依存しません。ただし最後の全体整列で、最下点の変化に応じて顔を含む全身の位置が変わることはあります。`neck_scale`は首の区間を変えますが、顔内部のサイズには作用しません。

回転は内部リグの回転行列ではなく、顔点の対応からの推定です。顔が直線・一点に潰れて向きを決められない場合はreferenceの方向を維持し、`face_rotation=identity_fallback_degenerate_face`と警告を表示します。通常は`face_rotation=kabsch_face5`です。R113→R126の方向は中心リグ転送によるもので、顔のKabsch回転とは別に取得します。

OpenPose BODY18には背骨途中・内部頭点・頭頂のスロットがありません。中心リグは肩や顔の配置計算と区間長のreportに使い、R126そのものを画像へ描く出力は追加していません。既存のSkeleton Debugは引き続き入力SAM専用です。

### 全身リグとMHR70の役割分離

肩は生成したR37から、R37→R38→R39 / R37→R74→R75の4区間を再構成します。首69からR37を逆算しません。`reference_symmetry=average`なら左右の対応区間をそれぞれ平均し、`off`なら左右別の長さを保持します。肩の倍率は前後・上下オフセットにも作用し、左右肩間の距離は区間を組み立てた結果で決まります。

通常は**骨長・方向を内部リグから取得し、MHR70は出力用の目印として配置**します。MHR70の肘や足首の点間距離を腕・脛の骨長には使用しません。

| リグ区間（右／左） | 適用倍率（すべてにuniform_scaleを乗算） |
|---|---|
| R1→R18／R1→R2 | `hip_width_scale` |
| R18→R19／R2→R3 | `leg_scale × thigh_scale` |
| R19→R20／R3→R4 | `leg_scale × shin_scale` |
| R20→R21→R22→R23→R24／R4→R5→R6→R7→R8 | `leg_scale` |
| R39→R40／R75→R76 | `arm_scale × upper_arm_scale` |
| R40→R41→R42／R76→R77→R78 | `arm_scale × forearm_scale` |
| 手首から各指先（thumb0・pinky0を含む公式の親子経路） | `arm_scale × hand_scale` |

`reference_symmetry=average`は左右の対応する**リグ区間長**を平均します。顔と表面点の位置差は平均化せず、referenceの左右差を保持します。全127点を骨長として足すわけではなく、必要な経路を使用します。零長の区間はreference側も零なら許容し、非零のreference区間に対してdrivingの方向が潰れていれば旧方式へ戻します。

MHR70のうち内部リグと一致する48点は、生成リグの対応座標を直接使います。メッシュ由来の肘・足首・足先・かかと・補助点は、次の**近似**で配置します。

```text
生成MHR点 = 生成した基準リグ点
          + 部位の回転(reference MHR点 − reference基準リグ点) × 倍率
```

- 肘7/8・肘補助点：R76/R40を基準に前腕の倍率。
- 足首13/14・かかと17/20：R4/R20を基準に`leg_scale`。
- つま先15/16・18/19：R8/R24を基準に`leg_scale`。
- 肩補助点67/68：R75/R39を基準に上腕の倍率。fit・最下点判定には引き続き含めません。
- 鼻・目・耳：前述のR113＋共通の顔回転を維持します。

部位の回転は最終リグ座標から求めます。前腕は前腕軸と手の横方向、足部は足の長軸と脛方向などから座標枠を作り、reference/drivingで共通の軸を使用します。両方で使用可能な軸がなければフォールバックします。**リグ座標だけでは全てのねじりを一意に決められず、関節曲げによる皮膚変形も再現しません。** SAMの回転配列はhand fusion前の値が残る版があるため、この方式では使用しません。

メッシュは変形・再生成しません。MHR70も変形後メッシュからの厳密な再計算ではありません。Depth出力は今後の検討事項です。

通常のreportで`skeleton_mode=full_rig`と`landmark_mode=position_frame_offset_approximation`を確認してください。リグ区間長は`Rig lengths effective_reference->generated`、MHR70の点間距離は`MHR landmark distances`として分けて表示します。後者はポーズとオフセットで変わり、骨長の一致を保証する指標ではありません。

### 旧データとの互換動作

全身リグや座標枠が使用不能なら、新しい四肢処理を部分的に混ぜず、`skeleton_mode=legacy_mixed_fallback`と理由を表示して直前版の混合方式へ戻します。

内部リグは既存の`mesh_data["joint_coords"]`または`raw_output["pred_joint_coords"]`から取得します。リグの欠損・不正値・中心区間の退化などで中心構造を使えない場合は、警告と`center_mode=legacy_torso_fallback`を出し、直前版のH→69／肩中央→鼻の配置へ戻します。肩のリグだけ使用可能なら4区間転送は維持し、肩も使用不能なら`shoulder_mode=legacy_width_fallback`として肩幅固定方式へ戻します。たとえばR126が欠ける場合、身長はunavailable、中心は旧方式、肩は4区間方式になり得ます。

中心リグまで使えない旧方式へ戻った場合の倍率も直前版どおりです。`torso_scale`はH→69、`head_scale × neck_scale`は肩中央→鼻へ作用します。新方式の比較では必ず`skeleton_mode=full_rig`を確認してください。入力・widget順・出力順は変更していないため、今回の更新によるノード再追加は不要です。ただし**各倍率の適用先が変わるので、以前の補正値は一旦1.0を基準に再確認してください**。

配置は、3D骨格を組み立てた後に全点を同量だけ平行移動し、**鼻のX・Zと、最下点のYをdrivingへ合わせます**。SAMの投影用座標はY正方向が下なので、最下点はY最大の点です。身体・顔・手指・足先・かかと・首を対象にし、補助点63〜68は除外します。生成側とdriving側で最下点の関節が異なっても、それぞれの最下点の高さを揃えます。手が最下点なら手が基準であり、接地を保証するIKではありません。骨長・各点の相対位置は変えず、腰中央は固定しません。`report`には移動量と両方の最下点インデックスを表示します。
これは3D座標での整列なので、鼻のYや画像上の足元まで一致するとは限りません。`fit_to_canvas=off`ならこの投影配置を維持し、fitを有効にすると2Dでさらに移動・拡縮されます。driving比較用2出力やデバッグの元骨格にはこの整列を適用しません。
OpenPoseの首スロット（BODY18の1番）は、DWPoseと同様に**投影後の左右肩の2D中点**を出力します。マージ結果・driving再投影・SAM raw 2Dの3出力に共通です。片肩が無効なら首点のconfidenceも0にします。これは出力形式への変換だけで、内部MHR69を肩中点に置き換える処理ではありません。デバッグのSは3D肩中央を投影した点なので、透視投影ではOpenPoseの2D中点と一致しない場合があります。
canvas fitの範囲計算は、実際に出力するBODY18と左右HAND21の有効点だけを対象にします。描画しないつま先・かかと・補助点・MHR69は縮小率や配置に影響しません。画面外にある手・肘などの出力点は、引き続きfit対象です。
`report`にはreferenceとdrivingの概算身長、およびリグ区間長とMHR70点間距離を分けて表示します。

補足表示の概算身長の算出式は今回変更していません（中心リグ7区間の合計とは別の指標）。内部リグの`R126: c_head_null`を頭頂として直接使い、以下の3D直線距離を合計します。

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
| `show_auxiliary` | 63〜68番の補助点（既定OFF）。このデバッグノードは入力SAMのみを表示 |
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
