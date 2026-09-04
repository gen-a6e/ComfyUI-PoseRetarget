# comfyui-pose-retarget

参照画像の体型と、別画像のポーズを組み合わせてOpenPose骨格を作るComfyUIノードです。

- `SAM 3D Body Pose Retarget`: SAM 3D Bodyの3D骨格を使う高精度版
- `Pose Retarget (keep body proportions)`: DWPose/OpenPoseだけで動く軽量な2D版

既存のリターゲット系ノードは全身を一律に拡大縮小するものが多く、頭身などの比率は変わりません。
これは部位ごとの骨の長さを個別に差し替えます。

## インストール

```
cd ComfyUI/custom_nodes
git clone git@github.com:gen-a6e/comfyui-pose-retarget.git
```

このリポジトリ自体に追加の依存パッケージはありません（numpyはComfyUIに同梱）。
ComfyUI を再起動してください。

フォルダ名は何でも構いません。ブラウザ側の JS を使っていないので、
Manager がフォルダ名を変えても壊れません。

## SAM 3D Body版

3D推定には既存の
[`ComfyUI-SAM3DBody`](https://github.com/PozzettiAndrea/ComfyUI-SAM3DBody)
を利用します。ComfyUI Managerで`SAM3DBody`を検索してインストールしてください。
モデルのロードとVRAM管理は同拡張機能が担当し、このリポジトリではモデルを複製しません。
`(Down)Load SAM 3D Body Model`の出力を、reference用とdriving用の
`SAM 3D Body: Process Image`へ接続し、両方とも`inference_type=full`で実行します。

### SAM 3D Body Pose Retarget

`SAM 3D Body: Process Image`が出力する2つの`SAM3D_OUTPUT`を受け取ります。
referenceのMHR70骨格から3D骨長比を測り、drivingの3Dボーン方向へ適用したあと、
driving側のカメラと焦点距離で2Dへ透視投影します。横向きや手足をカメラへ向けた
ポーズでも、2Dの見かけの長さから奥行きを推測する必要がありません。

| 入力 | 説明 |
|---|---|
| `reference_sam3d` | reference画像を`SAM 3D Body: Process Image`へ通した`SAM3D_OUTPUT` |
| `driving_sam3d` | driving画像を同ノードへ通した`SAM3D_OUTPUT` |
| `driving_image` | drivingに使った元画像。出力キャンバスの幅・高さを取得するために必要 |

| パラメータ | 既定 | 説明 |
|---|---|---|
| `size_reference` | torso | 画面内の人物サイズを合わせるための3D基準。torso / shoulder_width / body_height |
| `reference_symmetry` | average | referenceの左右で推定誤差が出たときの骨長補正 |
| `uniform_scale` | 1.0 | 腰中央を基準にした全身サイズ |
| `leg_scale` | 1.0 | 脚と足の追加倍率 |
| `arm_scale` | 1.0 | 腕と手の追加倍率 |
| `head_scale` | 1.0 | 首から上の追加倍率 |
| `hand_scale` | 1.0 | 手指の追加倍率 |
| `fit_to_canvas` | shrink_to_fit | はみ出した場合のキャンバス調整 |
| `canvas_margin` | 16 | fit_exactly、または縮小が必要な場合の余白 |

MHR70には鼻・目・耳はありますが、輪郭や口を含む密な顔ランドマークはありません。
そのためBODY18の顔点は出力し、`face_keypoints_2d`の70点はゼロconfidenceにします。
左右の手はそれぞれ21点を出力します。

```text
(Down)Load SAM 3D Body Model ─┬→ Process Image ← reference画像 ─┐
                              └→ Process Image ← driving画像  ─┤
driving画像 ───────────────────────────────────────────────────┤
                                                               ↓
                                                  SAM 3D Body Pose Retarget
                                                               ↓
                                                  SDPose Draw → ControlNet
```

現在は1画像・1人用です。複数人の場合、`SAM 3D Body: Process Image`が選んだ先頭の人物を使います。

## 2Dノード

### Pose Retarget (keep body proportions)

| 入力 | 説明 |
|---|---|
| `reference_pose` | POSE_KEYPOINT。**体型**の出どころ。自分のキャラの全身画像を DWPose に通したもの |
| `driving_pose` | POSE_KEYPOINT。**ポーズ**の出どころ |

| パラメータ | 既定 | 説明 |
|---|---|---|
| `size_reference` | torso | 画面上のサイズを決める基準。torso が最も安定。肩や脚が隠れる構図なら shoulder_width / head_size |
| `reference_symmetry` | longer_side | reference側の測定誤差を補正する。曲がりや奥行きで片側だけ短く測られた場合、左右の骨を同じ長さにする |
| `anchor` | hips | 位置を合わせる基準点。全身なら hips、上半身中心なら neck |
| `uniform_scale` | 1.0 | 全体の大きさ |
| `leg_scale` | 1.0 | 脚だけ追加で伸縮。頭身の微調整用 |
| `arm_scale` | 1.0 | 腕だけ追加で伸縮 |
| `head_scale` | 1.0 | 頭部だけ追加で伸縮 |
| `foreshorten_mode` | symmetry | driving側の奥行き表現を保持する。カメラ方向を向いて短く見える手足を、出力でも同じ比率だけ短くする |
| `foreshorten_floor` | 0.15 | 奥行き補正でボーンを短くする場合の下限倍率 |
| `canonical_trigger` | 0.75 | symmetry_and_canonical のときだけ有効。標準比率のこの割合を下回ったら奥行きと判定 |
| `fit_to_canvas` | shrink_to_fit | shrink_to_fit=はみ出したときだけ縮める（余白は保証されない）。fit_exactly=常に余白の枠に合わせる。off=何もしない |
| `canvas_margin` | 16 | 余白（px）。fit_exactly のときは必ずこの余白が空く |

出力は `POSE_KEYPOINT` と、確認用の `report` 文字列です。
このノードは1人専用です。複数人が入力された場合は先頭の人物だけを使い、
追加の人物を無視したことを `report` に表示します。
顔や手が検出されなかった場合は、標準の `SDPose Draw Keypoints` が読み込めるように
顔70点・左右の手21点のゼロconfidenceデータを補完します。

## つなぎ方

```
キャラの全身画像 ─→ DWPose Estimator ─┐
                                      ├→ Pose Retarget ─→ SDPose Draw Keypoints ─→ ControlNet
ポーズ画像       ─→ DWPose Estimator ─┘
```

DWPose Estimator は `comfyui_controlnet_aux` のものです。
`POSE_KEYPOINT` 出力のほうを繋いでください（画像出力ではありません）。

## reference_symmetry と foreshorten_mode の違い

どちらも左右の骨の長さを比較しますが、**処理する入力と目的が逆**です。

- `reference_symmetry`はreference側に使います。参照画像で片腕だけ曲がっていたり、
  奥行き方向を向いて短く検出されたりした場合、その短さを体型として記憶しないための補正です。
  - `longer_side`（既定）: 左右とも長い側の長さに揃える
  - `average`: 左右の平均値に揃える
  - `off`: 左右を揃えず、検出された長さをそのまま使う
- `foreshorten_mode`はdriving側に使います。手足がカメラ方向を向いて短く見える場合、
  その見かけの短さをポーズの一部として出力へ残す処理です。

処理順は次のようになります。

```text
出力の骨の長さ
  = 左右補正済みのreference骨長
  × referenceとdrivingの全体サイズ比
  × drivingの奥行き短縮率
```

例えば、全体サイズ比を `1.0` とし、referenceの左右の腕が `80 / 60` と検出されたとします。
`reference_symmetry=longer_side`なら、まず本来の骨長を `80 / 80` に補正します。
次に、drivingで片腕だけカメラ方向を向き `80 / 40` に見えている場合、
`foreshorten_mode=symmetry`が短い側へ `0.5` を掛けます。最終的な出力は `80 / 40` となり、
referenceの体型を使いながらdrivingの奥行き感も維持できます。

通常の推奨設定は `reference_symmetry=longer_side` と
`foreshorten_mode=symmetry` の組み合わせです。
正面かつ左右対称で正確に検出されたreferenceを使う場合や、意図的に左右の骨長が異なる
キャラクターの場合だけ `reference_symmetry=off` を検討してください。
drivingの奥行きによる短縮を無視して、常にreferenceの骨長をそのまま使いたい場合は
`foreshorten_mode=off` にします。

## 奥行き方向（foreshorten_mode）

腕をカメラ方向に突き出すと、画像上では腕が短く写ります。単純に参照画像の長さを
当てはめると、この「短く見える」情報が消えて腕が画面内で伸び切ってしまいます。

そこで、見かけの短さを比率として保存します。ポーズ側で腕が本来の 40% の長さに
見えているなら、出力でも参照画像の腕の 40% にします。

「本来の長さ」の推定方法が3つのモードです。

- **symmetry**（既定）— 左右対称性を使います。右腕が短く写っていて左腕が伸びていれば、
  左腕の長さが本来の長さです。正面向きのポーズには一切影響しません。安全なので既定にしています。
- **symmetry_and_canonical** — 上に加えて、標準的な人体比率をフォールバックに使います。
  両腕とも手前に突き出しているような、左右対称性では判定できないポーズ向けです。
  ただし「腕が短いキャラクター」と「腕を突き出したポーズ」は2Dでは原理的に区別できません。
  誤って手足が縮む場合は `canonical_trigger` を下げるか symmetry に戻してください。
- **off** — 奥行きを無視します。

## 仕組み

1. 参照骨格から、親子関係にある各ボーンの長さを測る
2. 胴（首→腰）の長さで割って、キャンバス非依存の比率にする
3. ポーズ骨格の胴の長さを掛け直す。これで画面上のサイズはポーズ画像側に合う
4. 首を起点に、首→肩→肘→手首 のように木構造をたどる。
   各ボーンの**向きはポーズ骨格から**、**長さは参照骨格から**取って座標を積み上げる

顔と手のキーポイントは、鼻・手首の移動量に追従して平行移動し、
顔は参照画像の顔と身体の比率、手は肘→手首の長さの変化率でスケールします。
枠内に収める処理は顔と手の点も含めて判定するので、顔が枠外で切れることはありません。

首の長さはCOCO-18の肩中央点（neck、index 1）から鼻（index 0）までの距離として扱います。
身体を組み直した後の実際の出力サイズを再計測し、referenceの
`肩中央→鼻 / size_reference` 比率を基準に頭部全体を平行移動します。
横向きでは鼻の前方成分が2D画像上に現れるため、drivingの肩中央→鼻を
身体の上方向（腰中央→肩中央）へ射影し、三角比の`1 / cos`に相当する倍率を加えます。
頭を大きく傾けた場合の過剰な伸長を避けるため、この投影倍率は最大1.5倍です。
この補正は顔を拡縮しないため、drivingの顔サイズ・向き・表情を維持します。

### 顔サイズの算出

参照画像から顔サイズと身体サイズの比率を測り、その比率を
出力骨格へ移植します。画像の解像度や人物の写る大きさが違っても、頭身を維持できます。

顔サイズは、reference側と出力側の両方で利用できる最も精密な方法を次の順に選びます。

1. DWPoseの詳細な顔ランドマーク領域
2. 左右の耳の間隔
3. 左右の目の間隔
4. 首から鼻までの距離

異なる種類の測定値を直接比較すると倍率が狂うため、必ず両側で同じ方法を使います。
詳細な顔ランドマークがある場合は外れ値の影響を減らした顔領域の対角線を使います。
顔の向きと表情はdriving側のランドマークを維持し、大きさだけをreference側の比率へ合わせます。
通常は鼻の位置を固定して、その周囲の顔ランドマークと目・耳だけを拡大縮小するため、
顔サイズの補正によって首から鼻までの距離が伸びることはありません。
詳細顔点・耳・目がすべて取得できず、首→鼻へフォールバックした場合だけは、
この距離自体を顔サイズの代替値として補正します。
`head_scale`は、その結果に対する手動の追加倍率です。

## 参照画像の選び方

このノードは参照画像から骨の長さを**実測**します。したがって参照画像は
「測れる姿勢」である必要があります。

- 全身が写っていること（特に足首まで）
- 両脚をまっすぐ下ろしていること
- 腕を体から離していること（背中に回していると腕が測れません）
- カメラに対して正面。極端なアオリやフカンは避ける

手持ちの画像がこの条件を満たさない場合は、Qwen-Image-Edit などで一度
直立姿勢に編集してから使うと確実です。見栄えは不要で、幾何学的に
正しければ十分です。参照画像は一度作れば使い回せます。

`reference_symmetry`（既定 longer_side）は、片側だけ曲がっている程度なら
自動で補正します。左右差が大きいときは report に WARNING を出すので、
参照画像を作り直す判断に使ってください。

## 制限

- COCO-18（DWPose / OpenPose の標準18点）専用です。BODY-25 には未対応
- 1人専用です。複数人を検出した場合は、参照画像・ポーズ画像とも先頭の人物だけを使います
- 2D 情報しかないため、奥行きは見かけの長さからの推定です。左右どちらかが正面を向いていれば
  正確ですが、両手足を同時に手前へ突き出したポーズは原理的に曖昧さが残ります
- 参照画像は全身が写っているものを使ってください。脚が写っていないと脚の長さを測れず、
  その部位はポーズ画像の長さがそのまま使われます
- 首が検出できないフレームはそのまま素通しします
- 詳細な顔ランドマークは髪や頭頂部を含まないため、算出値は厳密な頭部外形ではなくControlNetへ渡す顔領域の大きさです

## ライセンス

MIT
