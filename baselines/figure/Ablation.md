| Arm (inputs at inference) | Model | Frames | Traj tok. | **Numeric-3** | Heldout | Test | Cam.Disp | Obj.Disp | Speed | Non-num. |
|---|---|---|---|---|---|---|---|---|---|---|
| Question only | final (joint) | - | - | **26.8** | 27.5 | 26.2 | 27.2 | 27.0 | 25.9 | 49.5 |
| RGB frames only | RGB-only FT | Y | - | **48.3** | 47.8 | 48.8 | 53.7 | 44.5 | 43.9 | 80.8 |
| RGB + donor tokens (control) | final (joint) | Y | donor | **29.6** | 30.6 | 28.8 | 30.7 | 28.8 | 28.6 | 69.0 |
| RGB + trajectory tokens (full) | final (joint) | Y | Y | **82.2** | 81.7 | 82.5 | 89.4 | 77.6 | 75.2 | 90.3 |
| *(not in figure) Trajectory tokens only (no frames)* | final (joint) | - | Y | *82.2* | *81.5* | *82.8* | *89.4* | *77.7* | *75.5* | *90.8* |
