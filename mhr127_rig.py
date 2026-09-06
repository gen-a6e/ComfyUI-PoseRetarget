"""SAMが使用するMHR127リグの名前と親番号（MHR70の番号とは別体系）。

SAM3DBody-cppがmhr_model.ptから抽出した対応表を、版を固定して収録する。
https://github.com/AmmarkoV/SAM3DBody-cpp/blob/db3fd03dd6e556aaf1774bbcb02f0a9c040b862b/src/mhr_joint_table.h
実行時の外部取得やモデル再ロードは不要。127点以外にはこの対応表を適用しない。
c_head_nullという名前だけで、解剖学的な頭頂点と断定してはいけない。
"""

# (名前, 親のリグ番号)。-1はrootであり、接続線を持たない。
MHR127_RIG = (
    ("body_world", -1),  # R0
    ("root", 0),  # R1
    ("l_upleg", 1),  # R2
    ("l_lowleg", 2),  # R3
    ("l_foot", 3),  # R4
    ("l_talocrural", 4),  # R5
    ("l_subtalar", 5),  # R6
    ("l_transversetarsal", 6),  # R7
    ("l_ball", 7),  # R8
    ("l_lowleg_twist1_proc", 3),  # R9
    ("l_lowleg_twist2_proc", 3),  # R10
    ("l_lowleg_twist3_proc", 3),  # R11
    ("l_lowleg_twist4_proc", 3),  # R12
    ("l_upleg_twist0_proc", 2),  # R13
    ("l_upleg_twist1_proc", 2),  # R14
    ("l_upleg_twist2_proc", 2),  # R15
    ("l_upleg_twist3_proc", 2),  # R16
    ("l_upleg_twist4_proc", 2),  # R17
    ("r_upleg", 1),  # R18
    ("r_lowleg", 18),  # R19
    ("r_foot", 19),  # R20
    ("r_talocrural", 20),  # R21
    ("r_subtalar", 21),  # R22
    ("r_transversetarsal", 22),  # R23
    ("r_ball", 23),  # R24
    ("r_lowleg_twist1_proc", 19),  # R25
    ("r_lowleg_twist2_proc", 19),  # R26
    ("r_lowleg_twist3_proc", 19),  # R27
    ("r_lowleg_twist4_proc", 19),  # R28
    ("r_upleg_twist0_proc", 18),  # R29
    ("r_upleg_twist1_proc", 18),  # R30
    ("r_upleg_twist2_proc", 18),  # R31
    ("r_upleg_twist3_proc", 18),  # R32
    ("r_upleg_twist4_proc", 18),  # R33
    ("c_spine0", 1),  # R34
    ("c_spine1", 34),  # R35
    ("c_spine2", 35),  # R36
    ("c_spine3", 36),  # R37
    ("r_clavicle", 37),  # R38
    ("r_uparm", 38),  # R39
    ("r_lowarm", 39),  # R40
    ("r_wrist_twist", 40),  # R41
    ("r_wrist", 41),  # R42
    ("r_pinky0", 42),  # R43
    ("r_pinky1", 43),  # R44
    ("r_pinky2", 44),  # R45
    ("r_pinky3", 45),  # R46
    ("r_pinky_null", 46),  # R47
    ("r_ring1", 42),  # R48
    ("r_ring2", 48),  # R49
    ("r_ring3", 49),  # R50
    ("r_ring_null", 50),  # R51
    ("r_middle1", 42),  # R52
    ("r_middle2", 52),  # R53
    ("r_middle3", 53),  # R54
    ("r_middle_null", 54),  # R55
    ("r_index1", 42),  # R56
    ("r_index2", 56),  # R57
    ("r_index3", 57),  # R58
    ("r_index_null", 58),  # R59
    ("r_thumb0", 42),  # R60
    ("r_thumb1", 60),  # R61
    ("r_thumb2", 61),  # R62
    ("r_thumb3", 62),  # R63
    ("r_thumb_null", 63),  # R64
    ("r_lowarm_twist1_proc", 40),  # R65
    ("r_lowarm_twist2_proc", 40),  # R66
    ("r_lowarm_twist3_proc", 40),  # R67
    ("r_lowarm_twist4_proc", 40),  # R68
    ("r_uparm_twist0_proc", 39),  # R69
    ("r_uparm_twist1_proc", 39),  # R70
    ("r_uparm_twist2_proc", 39),  # R71
    ("r_uparm_twist3_proc", 39),  # R72
    ("r_uparm_twist4_proc", 39),  # R73
    ("l_clavicle", 37),  # R74
    ("l_uparm", 74),  # R75
    ("l_lowarm", 75),  # R76
    ("l_wrist_twist", 76),  # R77
    ("l_wrist", 77),  # R78
    ("l_pinky0", 78),  # R79
    ("l_pinky1", 79),  # R80
    ("l_pinky2", 80),  # R81
    ("l_pinky3", 81),  # R82
    ("l_pinky_null", 82),  # R83
    ("l_ring1", 78),  # R84
    ("l_ring2", 84),  # R85
    ("l_ring3", 85),  # R86
    ("l_ring_null", 86),  # R87
    ("l_middle1", 78),  # R88
    ("l_middle2", 88),  # R89
    ("l_middle3", 89),  # R90
    ("l_middle_null", 90),  # R91
    ("l_index1", 78),  # R92
    ("l_index2", 92),  # R93
    ("l_index3", 93),  # R94
    ("l_index_null", 94),  # R95
    ("l_thumb0", 78),  # R96
    ("l_thumb1", 96),  # R97
    ("l_thumb2", 97),  # R98
    ("l_thumb3", 98),  # R99
    ("l_thumb_null", 99),  # R100
    ("l_lowarm_twist1_proc", 76),  # R101
    ("l_lowarm_twist2_proc", 76),  # R102
    ("l_lowarm_twist3_proc", 76),  # R103
    ("l_lowarm_twist4_proc", 76),  # R104
    ("l_uparm_twist0_proc", 75),  # R105
    ("l_uparm_twist1_proc", 75),  # R106
    ("l_uparm_twist2_proc", 75),  # R107
    ("l_uparm_twist3_proc", 75),  # R108
    ("l_uparm_twist4_proc", 75),  # R109
    ("c_neck", 37),  # R110
    ("c_neck_twist1_proc", 110),  # R111
    ("c_neck_twist0_proc", 110),  # R112
    ("c_head", 110),  # R113
    ("c_jaw", 113),  # R114
    ("c_teeth", 114),  # R115
    ("c_jaw_null", 114),  # R116
    ("c_tongue0", 114),  # R117
    ("c_tongue1", 117),  # R118
    ("c_tongue2", 118),  # R119
    ("c_tongue3", 119),  # R120
    ("c_tongue4", 120),  # R121
    ("r_eye", 113),  # R122
    ("r_eye_null", 122),  # R123
    ("l_eye", 113),  # R124
    ("l_eye_null", 124),  # R125
    ("c_head_null", 113),  # R126
)

RIG_HEAD = 113
RIG_HEAD_NULL = 126
RIG_HEAD_NECK = tuple(range(110, 127))

