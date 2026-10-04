# Demo 网页:图 4 / 图 4b 的 9 个 clip,数据管线结果逐帧画进视频(2026-09-26)

- `index.html` 本地直接双击打开即可(所有资源都是相对路径,不需要服务器);要在别的机器看:`python3 -m http.server 8765` 后 `ssh -L 8765:localhost:8765 ...`。
- 线上版本(私有链接,claude.ai Artifact)见会话记录 / memory。
- `videos/<key>_tiles.mp4`:2×2 拼图(左上 输入 | 右上 掩码 | 左下 轨迹 | 右下 掩码+轨迹),网页用它做图层切换,一个文件天然同步。
- `videos/<key>_{input,masks,traj,both}.mp4`:单图层 512×512;`<key>_sbs.mp4`:输入 | 掩码+轨迹 左右并排 1024×512 —— 给剪辑软件用。
- `posters/<key>.jpg`:末帧(完整轨迹)。`data.js`:网页数据(物体清单 / 配色 / T8 caption / TrajQA-B 题目 / 两个模型原样输出)。
- 帧率 = 各 clip 的 fps_effective(8 个 5 fps、carried_object 15 fps),即真实时间轴。
- 画法与论文图 4 帧图同一套代码(复用 `paper/figure_scripts/render_figs.py`),`render_demo_videos.py` 运行时把图里选的帧逐像素核对,不一致会报错。
- 重新出图:`python paper/demo/render_demo_videos.py [--keys k1,k2]`(临时帧目录用 `DEMO_TMP` 指定)。
- 9 个 clip = 图 4 的 4 例(complex_interaction / multiple_people / swapping_positions / complex_video)+ 图 4b 的 5 例(distant_animal / crossing_truck / camera_motion / carried_object / approaching_person)。手–物体那个 clip(QE_pA0S3ttc_161895229)没进任何图,也没做。
