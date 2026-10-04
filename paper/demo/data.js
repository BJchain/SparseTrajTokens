window.DEMO_DATA = {
 "generated": "2026-09-26",
 "models": {
  "rgb": "Qwen3-VL-8B · frames only (same SFT data)",
  "ours": "Ours · frames + trajectory tokens"
 },
 "rainbow": "violet = start, red = end",
 "clips": [
  {
   "key": "complex_interaction",
   "name": "Complex interaction",
   "clip": "sbcUC3xaXfE_475542209",
   "fig": "fig4",
   "split": "heldout_test",
   "T": 34,
   "fps": 5.0,
   "duration_s": 6.6,
   "paper_frames": [
    0,
    16,
    33
   ],
   "n_masklets_rerun": 16,
   "objects": [
    {
     "iid": 3,
     "label": "person",
     "kind": "nonrigid",
     "n_keypoints": 6,
     "mask_matched": true,
     "mask_overlap": 0.96,
     "color": "#1F76D6"
    },
    {
     "iid": 6,
     "label": "seal",
     "kind": "nonrigid",
     "n_keypoints": 4,
     "mask_matched": true,
     "mask_overlap": 0.991,
     "color": "#FF7F0E"
    }
   ],
   "captions": [
    "Moving in the world frame: the person, the seal. Staying in place: 2 platforms.",
    "The person moves about 6.4 m in the world frame at an average speed of 1.43 m/s; it accelerates over the clip; it reverses direction around frame 24; it moves away from the seal.",
    "The seal moves about 3.5 m in the world frame at an average speed of 1.41 m/s; it reverses direction around frame 1; it moves away from the person."
   ],
   "question": {
    "id": "scene_sbcUC3xaXfE_475542209_disp_grid_5df62b6a",
    "text": "In the world frame (ignoring camera motion), approximately how far (in meters) did the seal move between frame 1 and frame 27 of 34?",
    "options": {
     "A": "0.58",
     "B": "1.1",
     "C": "0.78",
     "D": "1.4"
    },
    "answer": "D",
    "subtype": "disp_grid",
    "gt_value": 1.5609,
    "gt_unit": "m",
    "i": 1,
    "j": 27,
    "keyline": "displacement_m:",
    "obj": "the seal",
    "obj_iid": 6,
    "obj_color": "#FF7F0E"
   },
   "outputs": {
    "rgb": {
     "head": "Qwen3-VL-8B · frames only (same SFT data)",
     "lines": [
      "<evidence>",
      "object: the seal",
      "position_frame_1_m: (1.6, 1.3, 6.8)",
      "position_frame_27_m: (0.5, 1.4, 7.1)",
      "displacement_m: 1.1",
      "</evidence>",
      "<answer>B</answer>"
     ],
     "letter": "B",
     "correct": false,
     "value": 1.1
    },
    "ours": {
     "head": "Ours · frames + trajectory tokens",
     "lines": [
      "<evidence>",
      "object: the seal",
      "position_frame_1_m: (1.8, 1.4, 7.6)",
      "position_frame_27_m: (0.8, 1.4, 8.8)",
      "displacement_m: 1.6",
      "</evidence>",
      "<answer>D</answer>"
     ],
     "letter": "D",
     "correct": true,
     "value": 1.6
    },
    "qwen_zs": {
     "head": "Qwen3-VL-8B · zero-shot",
     "letter": "B",
     "correct": false
    }
   },
   "videos": {
    "tiles": {
     "path": "videos/complex_interaction_tiles.mp4",
     "w": 1024,
     "h": 1024,
     "frames": 34,
     "bytes": 3868360
    },
    "both": {
     "path": "videos/complex_interaction_both.mp4",
     "w": 512,
     "h": 512,
     "frames": 34,
     "bytes": 990514
    },
    "traj": {
     "path": "videos/complex_interaction_traj.mp4",
     "w": 512,
     "h": 512,
     "frames": 34,
     "bytes": 976385
    },
    "masks": {
     "path": "videos/complex_interaction_masks.mp4",
     "w": 512,
     "h": 512,
     "frames": 34,
     "bytes": 978729
    },
    "input": {
     "path": "videos/complex_interaction_input.mp4",
     "w": 512,
     "h": 512,
     "frames": 34,
     "bytes": 959624
    },
    "sbs": {
     "path": "videos/complex_interaction_sbs.mp4",
     "w": 1024,
     "h": 512,
     "frames": 34,
     "bytes": 1923115
    }
   },
   "poster": "posters/complex_interaction.jpg"
  },
  {
   "key": "multiple_people",
   "name": "Multiple people",
   "clip": "2ekr1IfOn5c_101835169",
   "fig": "fig4",
   "split": "test",
   "T": 34,
   "fps": 5.0,
   "duration_s": 6.6,
   "paper_frames": [
    0,
    16,
    33
   ],
   "n_masklets_rerun": 14,
   "objects": [
    {
     "iid": 1,
     "label": "man in striped shirt",
     "kind": "nonrigid",
     "n_keypoints": 6,
     "mask_matched": true,
     "mask_overlap": 0.946,
     "color": "#1F76D6"
    },
    {
     "iid": 3,
     "label": "man in white shirt",
     "kind": "nonrigid",
     "n_keypoints": 10,
     "mask_matched": true,
     "mask_overlap": 0.942,
     "color": "#FF7F0E"
    },
    {
     "iid": 5,
     "label": "woman in white dress",
     "kind": "nonrigid",
     "n_keypoints": 6,
     "mask_matched": true,
     "mask_overlap": 0.793,
     "color": "#2CA02C"
    },
    {
     "iid": 6,
     "label": "shopping bag",
     "kind": "nonrigid",
     "n_keypoints": 4,
     "mask_matched": true,
     "mask_overlap": 0.981,
     "color": "#D62728"
    }
   ],
   "captions": [
    "Moving in the world frame: the man in striped shirt, the man in white shirt, the woman in white dress, the shopping bag. Staying in place: the tent canopy, 2 potted plants, the boat on trailer (green), the boat on trailer (brown), 2 sign stands.",
    "The man in striped shirt moves about 1.5 m in the world frame at an average speed of 0.45 m/s; it reverses direction around frame 4; it moves away from the woman in white dress; it moves faster than the woman in white dress.",
    "The man in white shirt moves about 1.8 m in the world frame at an average speed of 0.42 m/s; it reverses direction around frame 12; it moves faster than the woman in white dress.",
    "The woman in white dress moves about 0.9 m in the world frame at an average speed of 0.26 m/s; it accelerates over the clip; it reverses direction around frame 16; it moves away from the man in striped shirt.",
    "The shopping bag moves about 1.6 m in the world frame at an average speed of 0.3 m/s; it accelerates over the clip; it moves away from the woman in white dress."
   ],
   "question": {
    "id": "scene_2ekr1IfOn5c_101835169_speed_grid_75a03fb6",
    "text": "In the world frame (ignoring camera motion), what was the average speed (in meters per second) of the man in white shirt between frame 4 and frame 18 of 34?",
    "options": {
     "A": "0.29",
     "B": "0.39",
     "C": "0.52",
     "D": "0.21"
    },
    "answer": "A",
    "subtype": "speed_grid",
    "gt_value": 0.2622,
    "gt_unit": "m/s",
    "i": 4,
    "j": 18,
    "keyline": "speed_mps:",
    "obj": "the man in white shirt",
    "obj_iid": 3,
    "obj_color": "#FF7F0E"
   },
   "outputs": {
    "rgb": {
     "head": "Qwen3-VL-8B · frames only (same SFT data)",
     "lines": [
      "<evidence>",
      "object: the man in white shirt",
      "position_frame_4_m: (0.1, 0.1, 4.1)",
      "position_frame_18_m: (-0.4, 0.1, 3.3)",
      "displacement_m: 0.9",
      "time_s: 2.80",
      "speed_mps: 0.32",
      "</evidence>",
      "<answer>A</answer>"
     ],
     "letter": "A",
     "correct": true,
     "value": 0.32
    },
    "ours": {
     "head": "Ours · frames + trajectory tokens",
     "lines": [
      "<evidence>",
      "object: the man in white shirt",
      "position_frame_4_m: (0.1, -0.2, 3.5)",
      "position_frame_18_m: (-0.4, 0.1, 3.0)",
      "displacement_m: 0.7",
      "time_s: 2.80",
      "speed_mps: 0.26",
      "</evidence>",
      "<answer>A</answer>"
     ],
     "letter": "A",
     "correct": true,
     "value": 0.26
    },
    "qwen_zs": {
     "head": "Qwen3-VL-8B · zero-shot",
     "letter": "B",
     "correct": false
    }
   },
   "videos": {
    "tiles": {
     "path": "videos/multiple_people_tiles.mp4",
     "w": 1024,
     "h": 1024,
     "frames": 34,
     "bytes": 3781373
    },
    "both": {
     "path": "videos/multiple_people_both.mp4",
     "w": 512,
     "h": 512,
     "frames": 34,
     "bytes": 996369
    },
    "traj": {
     "path": "videos/multiple_people_traj.mp4",
     "w": 512,
     "h": 512,
     "frames": 34,
     "bytes": 945260
    },
    "masks": {
     "path": "videos/multiple_people_masks.mp4",
     "w": 512,
     "h": 512,
     "frames": 34,
     "bytes": 943289
    },
    "input": {
     "path": "videos/multiple_people_input.mp4",
     "w": 512,
     "h": 512,
     "frames": 34,
     "bytes": 884605
    },
    "sbs": {
     "path": "videos/multiple_people_sbs.mp4",
     "w": 1024,
     "h": 512,
     "frames": 34,
     "bytes": 1878957
    }
   },
   "poster": "posters/multiple_people.jpg"
  },
  {
   "key": "swapping_positions",
   "name": "Objects swapping positions",
   "clip": "Kh42LoZOS2Y_102502503",
   "fig": "fig4",
   "split": "test",
   "T": 34,
   "fps": 5.0,
   "duration_s": 6.6,
   "paper_frames": [
    0,
    20,
    33
   ],
   "n_masklets_rerun": 8,
   "objects": [
    {
     "iid": 2,
     "label": "man",
     "kind": "nonrigid",
     "n_keypoints": 5,
     "mask_matched": true,
     "mask_overlap": 0.926,
     "color": "#1F76D6"
    },
    {
     "iid": 3,
     "label": "woman",
     "kind": "nonrigid",
     "n_keypoints": 5,
     "mask_matched": true,
     "mask_overlap": 0.926,
     "color": "#FF7F0E"
    }
   ],
   "captions": [
    "Moving in the world frame: the man, the woman. Staying in place: the tank, the tree, the fountain, 2 chain barriers, the lamp post.",
    "The man moves about 0.5 m in the world frame at an average speed of 0.23 m/s; it reverses direction around frame 15; it approaches the woman.",
    "The woman moves about 0.3 m in the world frame at an average speed of 0.26 m/s; it accelerates over the clip; it reverses direction around frame 21; it approaches the man."
   ],
   "question": {
    "id": "scene_Kh42LoZOS2Y_102502503_speed_grid_dda38c97",
    "text": "In the world frame (ignoring camera motion), what was the average speed (in meters per second) of the woman between frame 25 and frame 32 of 34?",
    "options": {
     "A": "0.21",
     "B": "0.39",
     "C": "0.52",
     "D": "0.29"
    },
    "answer": "B",
    "subtype": "speed_grid",
    "gt_value": 0.4307,
    "gt_unit": "m/s",
    "i": 25,
    "j": 32,
    "keyline": "speed_mps:",
    "obj": "the woman",
    "obj_iid": 3,
    "obj_color": "#FF7F0E"
   },
   "outputs": {
    "rgb": {
     "head": "Qwen3-VL-8B · frames only (same SFT data)",
     "lines": [
      "<evidence>",
      "object: the woman",
      "position_frame_25_m: (-0.8, 0.4, 3.0)",
      "position_frame_32_m: (-0.4, 0.5, 3.1)",
      "displacement_m: 0.4",
      "time_s: 1.40",
      "speed_mps: 0.29",
      "</evidence>",
      "<answer>D</answer>"
     ],
     "letter": "D",
     "correct": false,
     "value": 0.29
    },
    "ours": {
     "head": "Ours · frames + trajectory tokens",
     "lines": [
      "<evidence>",
      "object: the woman",
      "position_frame_25_m: (-0.9, 0.2, 2.4)",
      "position_frame_32_m: (-0.4, 0.3, 2.7)",
      "displacement_m: 0.6",
      "time_s: 1.40",
      "speed_mps: 0.40",
      "</evidence>",
      "<answer>B</answer>"
     ],
     "letter": "B",
     "correct": true,
     "value": 0.4
    },
    "qwen_zs": {
     "head": "Qwen3-VL-8B · zero-shot",
     "letter": "D",
     "correct": false
    }
   },
   "videos": {
    "tiles": {
     "path": "videos/swapping_positions_tiles.mp4",
     "w": 1024,
     "h": 1024,
     "frames": 34,
     "bytes": 2374621
    },
    "both": {
     "path": "videos/swapping_positions_both.mp4",
     "w": 512,
     "h": 512,
     "frames": 34,
     "bytes": 617691
    },
    "traj": {
     "path": "videos/swapping_positions_traj.mp4",
     "w": 512,
     "h": 512,
     "frames": 34,
     "bytes": 589471
    },
    "masks": {
     "path": "videos/swapping_positions_masks.mp4",
     "w": 512,
     "h": 512,
     "frames": 34,
     "bytes": 600603
    },
    "input": {
     "path": "videos/swapping_positions_input.mp4",
     "w": 512,
     "h": 512,
     "frames": 34,
     "bytes": 571745
    },
    "sbs": {
     "path": "videos/swapping_positions_sbs.mp4",
     "w": 1024,
     "h": 512,
     "frames": 34,
     "bytes": 1185740
    }
   },
   "poster": "posters/swapping_positions.jpg"
  },
  {
   "key": "complex_video",
   "name": "Complex video",
   "clip": "yRsUHBgOt8k_61795128",
   "fig": "fig4",
   "split": "test",
   "T": 34,
   "fps": 5.0,
   "duration_s": 6.6,
   "paper_frames": [
    0,
    16,
    33
   ],
   "n_masklets_rerun": 29,
   "objects": [
    {
     "iid": 0,
     "label": "person",
     "kind": "nonrigid",
     "n_keypoints": 7,
     "mask_matched": true,
     "mask_overlap": 0.996,
     "color": "#1F76D6"
    },
    {
     "iid": 1,
     "label": "person",
     "kind": "nonrigid",
     "n_keypoints": 7,
     "mask_matched": true,
     "mask_overlap": 0.978,
     "color": "#FF7F0E"
    },
    {
     "iid": 3,
     "label": "person",
     "kind": "nonrigid",
     "n_keypoints": 7,
     "mask_matched": true,
     "mask_overlap": 0.97,
     "color": "#2CA02C"
    },
    {
     "iid": 4,
     "label": "person",
     "kind": "nonrigid",
     "n_keypoints": 7,
     "mask_matched": true,
     "mask_overlap": 0.816,
     "color": "#D62728"
    },
    {
     "iid": 6,
     "label": "person",
     "kind": "nonrigid",
     "n_keypoints": 8,
     "mask_matched": true,
     "mask_overlap": 0.885,
     "color": "#9467BD"
    },
    {
     "iid": 7,
     "label": "person",
     "kind": "nonrigid",
     "n_keypoints": 6,
     "mask_matched": true,
     "mask_overlap": 0.792,
     "color": "#8C564B"
    },
    {
     "iid": 8,
     "label": "stroller",
     "kind": "rigid",
     "n_keypoints": 3,
     "mask_matched": true,
     "mask_overlap": 0.891,
     "color": "#E377C2"
    },
    {
     "iid": 9,
     "label": "person",
     "kind": "nonrigid",
     "n_keypoints": 5,
     "mask_matched": true,
     "mask_overlap": 0.876,
     "color": "#BCBD22"
    },
    {
     "iid": 10,
     "label": "person",
     "kind": "nonrigid",
     "n_keypoints": 7,
     "mask_matched": true,
     "mask_overlap": 0.927,
     "color": "#17BECF"
    },
    {
     "iid": 11,
     "label": "person",
     "kind": "nonrigid",
     "n_keypoints": 5,
     "mask_matched": true,
     "mask_overlap": 0.711,
     "color": "#7F7F7F"
    },
    {
     "iid": 12,
     "label": "person",
     "kind": "nonrigid",
     "n_keypoints": 5,
     "mask_matched": false,
     "mask_overlap": null,
     "color": "#1F77B4"
    },
    {
     "iid": 13,
     "label": "person",
     "kind": "nonrigid",
     "n_keypoints": 4,
     "mask_matched": true,
     "mask_overlap": 0.81,
     "color": "#FFBB78"
    },
    {
     "iid": 15,
     "label": "person",
     "kind": "nonrigid",
     "n_keypoints": 5,
     "mask_matched": true,
     "mask_overlap": 0.832,
     "color": "#1F76D6"
    },
    {
     "iid": 16,
     "label": "person",
     "kind": "nonrigid",
     "n_keypoints": 5,
     "mask_matched": true,
     "mask_overlap": 0.975,
     "color": "#FF7F0E"
    },
    {
     "iid": 18,
     "label": "person",
     "kind": "nonrigid",
     "n_keypoints": 4,
     "mask_matched": false,
     "mask_overlap": null,
     "color": "#2CA02C"
    },
    {
     "iid": 21,
     "label": "person",
     "kind": "nonrigid",
     "n_keypoints": 4,
     "mask_matched": true,
     "mask_overlap": 0.901,
     "color": "#D62728"
    },
    {
     "iid": 23,
     "label": "person",
     "kind": "nonrigid",
     "n_keypoints": 6,
     "mask_matched": false,
     "mask_overlap": null,
     "color": "#9467BD"
    }
   ],
   "captions": [
    "Moving in the world frame: 16 persons, the stroller. Staying in place: 5 signs, 2 persons.",
    "The stroller moves about 3.8 m in the world frame at an average speed of 1.04 m/s; it reverses direction around frame 4."
   ],
   "question": {
    "id": "scene_yRsUHBgOt8k_61795128_speed_grid_7dfb88c4",
    "text": "In the world frame (ignoring camera motion), what was the average speed (in meters per second) of the stroller between frame 5 and frame 16 of 34?",
    "options": {
     "A": "0.70",
     "B": "1.7",
     "C": "0.95",
     "D": "1.3"
    },
    "answer": "A",
    "subtype": "speed_grid",
    "gt_value": 0.6414,
    "gt_unit": "m/s",
    "i": 5,
    "j": 16,
    "keyline": "speed_mps:",
    "obj": "the stroller",
    "obj_iid": 8,
    "obj_color": "#E377C2"
   },
   "outputs": {
    "rgb": {
     "head": "Qwen3-VL-8B · frames only (same SFT data)",
     "lines": [
      "<evidence>",
      "object: the stroller",
      "position_frame_5_m: (-0.7, 1.3, 8.3)",
      "position_frame_16_m: (-0.6, 1.1, 6.2)",
      "displacement_m: 2.1",
      "time_s: 2.20",
      "speed_mps: 0.95",
      "</evidence>",
      "<answer>C</answer>"
     ],
     "letter": "C",
     "correct": false,
     "value": 0.95
    },
    "ours": {
     "head": "Ours · frames + trajectory tokens",
     "lines": [
      "<evidence>",
      "object: the stroller",
      "position_frame_5_m: (-0.6, 1.5, 8.7)",
      "position_frame_16_m: (-0.6, 1.3, 6.9)",
      "displacement_m: 1.8",
      "time_s: 2.20",
      "speed_mps: 0.80",
      "</evidence>",
      "<answer>A</answer>"
     ],
     "letter": "A",
     "correct": true,
     "value": 0.8
    },
    "qwen_zs": {
     "head": "Qwen3-VL-8B · zero-shot",
     "letter": "C",
     "correct": false
    }
   },
   "videos": {
    "tiles": {
     "path": "videos/complex_video_tiles.mp4",
     "w": 1024,
     "h": 1024,
     "frames": 34,
     "bytes": 6986808
    },
    "both": {
     "path": "videos/complex_video_both.mp4",
     "w": 512,
     "h": 512,
     "frames": 34,
     "bytes": 1888428
    },
    "traj": {
     "path": "videos/complex_video_traj.mp4",
     "w": 512,
     "h": 512,
     "frames": 34,
     "bytes": 1806798
    },
    "masks": {
     "path": "videos/complex_video_masks.mp4",
     "w": 512,
     "h": 512,
     "frames": 34,
     "bytes": 1698696
    },
    "input": {
     "path": "videos/complex_video_input.mp4",
     "w": 512,
     "h": 512,
     "frames": 34,
     "bytes": 1585620
    },
    "sbs": {
     "path": "videos/complex_video_sbs.mp4",
     "w": 1024,
     "h": 512,
     "frames": 34,
     "bytes": 3475272
    }
   },
   "poster": "posters/complex_video.jpg"
  },
  {
   "key": "distant_animal",
   "name": "Distant animal",
   "clip": "N6EH3Lk0eRo_255522189",
   "fig": "fig4b",
   "split": "test",
   "T": 34,
   "fps": 5.0,
   "duration_s": 6.6,
   "paper_frames": [
    0,
    16,
    32
   ],
   "n_masklets_rerun": 11,
   "objects": [
    {
     "iid": 8,
     "label": "horse",
     "kind": "nonrigid",
     "n_keypoints": 4,
     "mask_matched": true,
     "mask_overlap": 0.721,
     "color": "#1F76D6"
    }
   ],
   "captions": [
    "Moving in the world frame: the horse. Staying in place: 9 trees.",
    "The horse moves about 1.4 m in the world frame at an average speed of 0.23 m/s; it accelerates over the clip."
   ],
   "question": {
    "id": "scene_N6EH3Lk0eRo_255522189_speed_grid_f27c768c",
    "text": "In the world frame (ignoring camera motion), what was the average speed (in meters per second) of the horse between frame 0 and frame 32 of 34?",
    "options": {
     "A": "0.29",
     "B": "0.39",
     "C": "0.21",
     "D": "0.16"
    },
    "answer": "C",
    "subtype": "speed_grid",
    "gt_value": 0.2175,
    "gt_unit": "m/s",
    "i": 0,
    "j": 32,
    "keyline": "speed_mps:",
    "obj": "the horse",
    "obj_iid": 8,
    "obj_color": "#1F76D6"
   },
   "outputs": {
    "rgb": {
     "head": "Qwen3-VL-8B · frames only (same SFT data)",
     "lines": [
      "<evidence>",
      "object: the horse",
      "position_frame_0_m: (0.1, -0.0, 13.0)",
      "position_frame_32_m: (2.6, 0.0, 10.4)",
      "displacement_m: 3.6",
      "time_s: 6.41",
      "speed_mps: 0.56",
      "</evidence>",
      "<answer>B</answer>"
     ],
     "letter": "B",
     "correct": false,
     "value": 0.56
    },
    "ours": {
     "head": "Ours · frames + trajectory tokens",
     "lines": [
      "<evidence>",
      "object: the horse",
      "position_frame_0_m: (0.1, 0.2, 5.1)",
      "position_frame_32_m: (1.4, 0.1, 4.7)",
      "displacement_m: 1.4",
      "time_s: 6.41",
      "speed_mps: 0.22",
      "</evidence>",
      "<answer>C</answer>"
     ],
     "letter": "C",
     "correct": true,
     "value": 0.22
    },
    "qwen_zs": {
     "head": "Qwen3-VL-8B · zero-shot",
     "letter": "B",
     "correct": false
    }
   },
   "videos": {
    "tiles": {
     "path": "videos/distant_animal_tiles.mp4",
     "w": 1024,
     "h": 1024,
     "frames": 34,
     "bytes": 1514448
    },
    "both": {
     "path": "videos/distant_animal_both.mp4",
     "w": 512,
     "h": 512,
     "frames": 34,
     "bytes": 386314
    },
    "traj": {
     "path": "videos/distant_animal_traj.mp4",
     "w": 512,
     "h": 512,
     "frames": 34,
     "bytes": 379514
    },
    "masks": {
     "path": "videos/distant_animal_masks.mp4",
     "w": 512,
     "h": 512,
     "frames": 34,
     "bytes": 384615
    },
    "input": {
     "path": "videos/distant_animal_input.mp4",
     "w": 512,
     "h": 512,
     "frames": 34,
     "bytes": 376432
    },
    "sbs": {
     "path": "videos/distant_animal_sbs.mp4",
     "w": 1024,
     "h": 512,
     "frames": 34,
     "bytes": 756230
    }
   },
   "poster": "posters/distant_animal.jpg"
  },
  {
   "key": "crossing_truck",
   "name": "Crossing truck",
   "clip": "WWdorMyLg3M_426693360",
   "fig": "fig4b",
   "split": "test",
   "T": 34,
   "fps": 5.0,
   "duration_s": 6.6,
   "paper_frames": [
    7,
    10,
    14
   ],
   "n_masklets_rerun": 19,
   "objects": [
    {
     "iid": 0,
     "label": "white box truck",
     "kind": "rigid",
     "n_keypoints": 4,
     "mask_matched": true,
     "mask_overlap": 0.995,
     "color": "#1F76D6"
    }
   ],
   "captions": [
    "Moving in the world frame: the white box truck. Staying in place: 2 yellow tactile pavings, the traffic light pole, 2 green railings.",
    "The white box truck moves about 16.4 m in the world frame at an average speed of 6.31 m/s; it accelerates over the clip; it reverses direction around frame 19."
   ],
   "question": {
    "id": "scene_WWdorMyLg3M_426693360_disp_grid_1b56523e",
    "text": "In the world frame (ignoring camera motion), approximately how far (in meters) did the white box truck move between frame 7 and frame 14 of 34?",
    "options": {
     "A": "12",
     "B": "6.4",
     "C": "4.7",
     "D": "8.7"
    },
    "answer": "D",
    "subtype": "disp_grid",
    "gt_value": 8.1401,
    "gt_unit": "m",
    "i": 7,
    "j": 14,
    "keyline": "displacement_m:",
    "obj": "the white box truck",
    "obj_iid": 0,
    "obj_color": "#1F76D6"
   },
   "outputs": {
    "rgb": {
     "head": "Qwen3-VL-8B · frames only (same SFT data)",
     "lines": [
      "<evidence>",
      "object: the white box truck",
      "position_frame_7_m: (4.1, 1.0, 9.3)",
      "position_frame_14_m: (-0.2, -0.2, 10.0)",
      "displacement_m: 4.5",
      "</evidence>",
      "<answer>C</answer>"
     ],
     "letter": "C",
     "correct": false,
     "value": 4.5
    },
    "ours": {
     "head": "Ours · frames + trajectory tokens",
     "lines": [
      "<evidence>",
      "object: the white box truck",
      "position_frame_7_m: (5.6, 2.6, 13.6)",
      "position_frame_14_m: (-2.7, 2.3, 14.8)",
      "displacement_m: 8.4",
      "</evidence>",
      "<answer>D</answer>"
     ],
     "letter": "D",
     "correct": true,
     "value": 8.4
    },
    "qwen_zs": {
     "head": "Qwen3-VL-8B · zero-shot",
     "letter": "B",
     "correct": false
    }
   },
   "videos": {
    "tiles": {
     "path": "videos/crossing_truck_tiles.mp4",
     "w": 1024,
     "h": 1024,
     "frames": 34,
     "bytes": 3500624
    },
    "both": {
     "path": "videos/crossing_truck_both.mp4",
     "w": 512,
     "h": 512,
     "frames": 34,
     "bytes": 894664
    },
    "traj": {
     "path": "videos/crossing_truck_traj.mp4",
     "w": 512,
     "h": 512,
     "frames": 34,
     "bytes": 881154
    },
    "masks": {
     "path": "videos/crossing_truck_masks.mp4",
     "w": 512,
     "h": 512,
     "frames": 34,
     "bytes": 866187
    },
    "input": {
     "path": "videos/crossing_truck_input.mp4",
     "w": 512,
     "h": 512,
     "frames": 34,
     "bytes": 861651
    },
    "sbs": {
     "path": "videos/crossing_truck_sbs.mp4",
     "w": 1024,
     "h": 512,
     "frames": 34,
     "bytes": 1745632
    }
   },
   "poster": "posters/crossing_truck.jpg"
  },
  {
   "key": "camera_motion",
   "name": "Camera motion",
   "clip": "7uTgv66Q29U_36169503",
   "fig": "fig4b",
   "split": "test",
   "T": 28,
   "fps": 5.0,
   "duration_s": 5.4,
   "paper_frames": [
    4,
    11,
    19
   ],
   "n_masklets_rerun": 25,
   "objects": [
    {
     "iid": 0,
     "label": "person",
     "kind": "nonrigid",
     "n_keypoints": 6,
     "mask_matched": true,
     "mask_overlap": 1.0,
     "color": "#1F76D6"
    },
    {
     "iid": 8,
     "label": "person",
     "kind": "nonrigid",
     "n_keypoints": 4,
     "mask_matched": true,
     "mask_overlap": 1.0,
     "color": "#FF7F0E"
    },
    {
     "iid": 12,
     "label": "person",
     "kind": "nonrigid",
     "n_keypoints": 6,
     "mask_matched": true,
     "mask_overlap": 0.617,
     "color": "#2CA02C"
    },
    {
     "iid": 15,
     "label": "person",
     "kind": "nonrigid",
     "n_keypoints": 7,
     "mask_matched": true,
     "mask_overlap": 0.826,
     "color": "#D62728"
    }
   ],
   "captions": [
    "Moving in the world frame: 4 persons. Staying in place: the door, 12 paintings, 2 persons."
   ],
   "question": {
    "id": "scene_7uTgv66Q29U_36169503_cam_disp_grid_84fb6571",
    "text": "Approximately how far (in meters) did the camera move between frame 4 and frame 19 of 28?",
    "options": {
     "A": "1.1",
     "B": "0.78",
     "C": "0.58",
     "D": "1.4"
    },
    "answer": "C",
    "subtype": "cam_disp_grid",
    "gt_value": 0.6351,
    "gt_unit": "m",
    "i": 4,
    "j": 19,
    "keyline": "camera_displacement_m:",
    "obj": null,
    "obj_iid": null,
    "obj_color": null
   },
   "outputs": {
    "rgb": {
     "head": "Qwen3-VL-8B · frames only (same SFT data)",
     "lines": [
      "<evidence>",
      "camera_frame_4_m: (-0.1, -0.0, 0.4)",
      "camera_frame_19_m: (-0.7, -0.0, 1.5)",
      "camera_displacement_m: 1.3",
      "</evidence>",
      "<answer>D</answer>"
     ],
     "letter": "D",
     "correct": false,
     "value": 1.3
    },
    "ours": {
     "head": "Ours · frames + trajectory tokens",
     "lines": [
      "<evidence>",
      "camera_frame_4_m: (-0.1, 0.0, 0.1)",
      "camera_frame_19_m: (-0.6, 0.0, 0.5)",
      "camera_displacement_m: 0.6",
      "</evidence>",
      "<answer>C</answer>"
     ],
     "letter": "C",
     "correct": true,
     "value": 0.6
    },
    "qwen_zs": {
     "head": "Qwen3-VL-8B · zero-shot",
     "letter": "D",
     "correct": false
    }
   },
   "videos": {
    "tiles": {
     "path": "videos/camera_motion_tiles.mp4",
     "w": 1024,
     "h": 1024,
     "frames": 28,
     "bytes": 2445172
    },
    "both": {
     "path": "videos/camera_motion_both.mp4",
     "w": 512,
     "h": 512,
     "frames": 28,
     "bytes": 642734
    },
    "traj": {
     "path": "videos/camera_motion_traj.mp4",
     "w": 512,
     "h": 512,
     "frames": 28,
     "bytes": 613296
    },
    "masks": {
     "path": "videos/camera_motion_masks.mp4",
     "w": 512,
     "h": 512,
     "frames": 28,
     "bytes": 626100
    },
    "input": {
     "path": "videos/camera_motion_input.mp4",
     "w": 512,
     "h": 512,
     "frames": 28,
     "bytes": 589353
    },
    "sbs": {
     "path": "videos/camera_motion_sbs.mp4",
     "w": 1024,
     "h": 512,
     "frames": 28,
     "bytes": 1211170
    }
   },
   "poster": "posters/camera_motion.jpg"
  },
  {
   "key": "carried_object",
   "name": "Carried object",
   "clip": "uDWH1bRUvYQ_217984651",
   "fig": "fig4b",
   "split": "heldout_test",
   "T": 24,
   "fps": 15.0,
   "duration_s": 1.533,
   "paper_frames": [
    9,
    16,
    23
   ],
   "n_masklets_rerun": 17,
   "objects": [
    {
     "iid": 2,
     "label": "person",
     "kind": "nonrigid",
     "n_keypoints": 6,
     "mask_matched": true,
     "mask_overlap": 0.89,
     "color": "#1F76D6"
    },
    {
     "iid": 3,
     "label": "person",
     "kind": "nonrigid",
     "n_keypoints": 7,
     "mask_matched": true,
     "mask_overlap": 0.958,
     "color": "#FF7F0E"
    },
    {
     "iid": 4,
     "label": "person",
     "kind": "nonrigid",
     "n_keypoints": 7,
     "mask_matched": true,
     "mask_overlap": 0.874,
     "color": "#2CA02C"
    },
    {
     "iid": 8,
     "label": "person",
     "kind": "nonrigid",
     "n_keypoints": 8,
     "mask_matched": true,
     "mask_overlap": 0.817,
     "color": "#D62728"
    },
    {
     "iid": 14,
     "label": "handbag",
     "kind": "nonrigid",
     "n_keypoints": 6,
     "mask_matched": true,
     "mask_overlap": 0.727,
     "color": "#9467BD"
    }
   ],
   "captions": [
    "Moving in the world frame: 4 persons, the handbag. Staying in place: 6 paintings, the window, the chair.",
    "The handbag moves about 0.9 m in the world frame at an average speed of 0.59 m/s."
   ],
   "question": {
    "id": "scene_uDWH1bRUvYQ_217984651_disp_grid_fddedbbd",
    "text": "In the world frame (ignoring camera motion), approximately how far (in meters) did the handbag move between frame 9 and frame 23 of 24?",
    "options": {
     "A": "0.58",
     "B": "1.4",
     "C": "1.1",
     "D": "0.78"
    },
    "answer": "A",
    "subtype": "disp_grid",
    "gt_value": 0.6215,
    "gt_unit": "m",
    "i": 9,
    "j": 23,
    "keyline": "displacement_m:",
    "obj": "the handbag",
    "obj_iid": 14,
    "obj_color": "#9467BD"
   },
   "outputs": {
    "rgb": {
     "head": "Qwen3-VL-8B · frames only (same SFT data)",
     "lines": [
      "<evidence>",
      "object: the handbag",
      "position_frame_9_m: (0.8, 1.5, 5.6)",
      "position_frame_23_m: (1.0, 1.6, 6.9)",
      "displacement_m: 1.3",
      "</evidence>",
      "<answer>B</answer>"
     ],
     "letter": "B",
     "correct": false,
     "value": 1.3
    },
    "ours": {
     "head": "Ours · frames + trajectory tokens",
     "lines": [
      "<evidence>",
      "object: the handbag",
      "position_frame_9_m: (1.0, 1.5, 6.5)",
      "position_frame_23_m: (1.2, 1.5, 7.1)",
      "displacement_m: 0.6",
      "</evidence>",
      "<answer>A</answer>"
     ],
     "letter": "A",
     "correct": true,
     "value": 0.6
    },
    "qwen_zs": {
     "head": "Qwen3-VL-8B · zero-shot",
     "letter": "B",
     "correct": false
    }
   },
   "videos": {
    "tiles": {
     "path": "videos/carried_object_tiles.mp4",
     "w": 1024,
     "h": 1024,
     "frames": 24,
     "bytes": 1309931
    },
    "both": {
     "path": "videos/carried_object_both.mp4",
     "w": 512,
     "h": 512,
     "frames": 24,
     "bytes": 374585
    },
    "traj": {
     "path": "videos/carried_object_traj.mp4",
     "w": 512,
     "h": 512,
     "frames": 24,
     "bytes": 326696
    },
    "masks": {
     "path": "videos/carried_object_masks.mp4",
     "w": 512,
     "h": 512,
     "frames": 24,
     "bytes": 327976
    },
    "input": {
     "path": "videos/carried_object_input.mp4",
     "w": 512,
     "h": 512,
     "frames": 24,
     "bytes": 280114
    },
    "sbs": {
     "path": "videos/carried_object_sbs.mp4",
     "w": 1024,
     "h": 512,
     "frames": 24,
     "bytes": 653485
    }
   },
   "poster": "posters/carried_object.jpg"
  },
  {
   "key": "approaching_person",
   "name": "Approaching person",
   "clip": "q-B2xu8B3Js_29963297",
   "fig": "fig4b",
   "split": "heldout_test",
   "T": 26,
   "fps": 5.0,
   "duration_s": 5.0,
   "paper_frames": [
    0,
    3,
    6
   ],
   "n_masklets_rerun": 13,
   "objects": [
    {
     "iid": 5,
     "label": "person",
     "kind": "nonrigid",
     "n_keypoints": 7,
     "mask_matched": true,
     "mask_overlap": 0.973,
     "color": "#1F76D6"
    }
   ],
   "captions": [
    "Moving in the world frame: the person. Staying in place: the railing, the tree, the sign, 2 doors.",
    "The person moves about 1.1 m in the world frame at an average speed of 0.68 m/s."
   ],
   "question": {
    "id": "scene_q-B2xu8B3Js_29963297_speed_grid_c8681538",
    "text": "In the world frame (ignoring camera motion), what was the average speed (in meters per second) of the person between frame 0 and frame 6 of 26?",
    "options": {
     "A": "0.52",
     "B": "0.95",
     "C": "0.70",
     "D": "1.3"
    },
    "answer": "A",
    "subtype": "speed_grid",
    "gt_value": 0.5938,
    "gt_unit": "m/s",
    "i": 0,
    "j": 6,
    "keyline": "speed_mps:",
    "obj": "the person",
    "obj_iid": 5,
    "obj_color": "#1F76D6"
   },
   "outputs": {
    "rgb": {
     "head": "Qwen3-VL-8B · frames only (same SFT data)",
     "lines": [
      "<evidence>",
      "object: the person",
      "position_frame_0_m: (0.8, 1.6, 6.5)",
      "position_frame_6_m: (0.4, 1.2, 4.8)",
      "displacement_m: 1.8",
      "time_s: 1.20",
      "speed_mps: 1.53",
      "</evidence>",
      "<answer>D</answer>"
     ],
     "letter": "D",
     "correct": false,
     "value": 1.53
    },
    "ours": {
     "head": "Ours · frames + trajectory tokens",
     "lines": [
      "<evidence>",
      "object: the person",
      "position_frame_0_m: (0.8, 0.5, 4.8)",
      "position_frame_6_m: (0.3, 0.6, 4.3)",
      "displacement_m: 0.7",
      "time_s: 1.20",
      "speed_mps: 0.58",
      "</evidence>",
      "<answer>A</answer>"
     ],
     "letter": "A",
     "correct": true,
     "value": 0.58
    },
    "qwen_zs": {
     "head": "Qwen3-VL-8B · zero-shot",
     "letter": "C",
     "correct": false
    }
   },
   "videos": {
    "tiles": {
     "path": "videos/approaching_person_tiles.mp4",
     "w": 1024,
     "h": 1024,
     "frames": 26,
     "bytes": 3203445
    },
    "both": {
     "path": "videos/approaching_person_both.mp4",
     "w": 512,
     "h": 512,
     "frames": 26,
     "bytes": 828528
    },
    "traj": {
     "path": "videos/approaching_person_traj.mp4",
     "w": 512,
     "h": 512,
     "frames": 26,
     "bytes": 821224
    },
    "masks": {
     "path": "videos/approaching_person_masks.mp4",
     "w": 512,
     "h": 512,
     "frames": 26,
     "bytes": 763622
    },
    "input": {
     "path": "videos/approaching_person_input.mp4",
     "w": 512,
     "h": 512,
     "frames": 26,
     "bytes": 754521
    },
    "sbs": {
     "path": "videos/approaching_person_sbs.mp4",
     "w": 1024,
     "h": 512,
     "frames": 26,
     "bytes": 1586596
    }
   },
   "poster": "posters/approaching_person.jpg"
  }
 ]
};
