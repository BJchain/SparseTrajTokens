"""SAM 3D 推理后端选择:官方 `sam-3d-objects` / `Fast-SAM3D`(ICML26,training-free 加速 fork)。

**为什么需要这一层**:两个仓库的顶层包同名(`sam3d_objects`)、notebook 模块也同名
(`inference`),只能靠 sys.path 顺序二选一 —— 而且必须在【任何】`import sam3d_objects`
之前决定。官方那份是 `pip install -e` 装的(site-packages 里一个 .pth 指过去),所以把
Fast-SAM3D 的根目录 `sys.path.insert(0, ...)` 就能整棵树盖掉。

本模块同时把 Fast-SAM3D 的调用契约抹平成官方那套,让 `src/stage1/schema.py` 的
`run_sam3d_single()` 一行都不用改。2026-08-09 逐行读 fork 后确认的差异:

1. `Inference(config: DictConfig, compile, args)` 收的是【已经 load 好的 config】而不是
   路径,并且要调用方自己设 `config.workspace_dir`(官方是 `Inference(path)` 自己 load)。
2. `Inference.__call__` 【不接受】`with_layout_postprocess` 形参(内部写死 True)。
3. 三个加速模块的超参靠 `get_params(argparse.Namespace)` 注入;2D 频谱复杂度 HFER 靠
   `get_hfer(float)` 注入。**HFER 是逐实例的**(从该实例的 RGBA 抠图算 FFT),官方 demo
   只跑单物体所以没暴露这点 —— 批量跑必须每次调用前重算重注入,否则所有实例共用第一个
   物体的复杂度,mesh 聚合档位就全错了。
4. **fork 基于更旧的 SAM3D 快照**:`InferencePipelinePointMap.__init__` 里
   `layout_post_optimization_method` 默认是 `None`(官方默认是那个函数),并且没有官方后来
   加的 GS 版后优化 `layout_post_optimization_method_GS`。照搬 fork = 【完全不做 layout
   后优化】,位姿必然退化(项目 2026-06-30 已实证这一步是 Path 1 朝向正确的关键)。
   本模块因此默认把 fork 自带的 mesh 版 `layout_post_optimization` 通过配置接回去,
   `layout_opt="off"` 可关掉做 A/B。
5. `sample_sparse_structure` 里写死 `process_and_visualize(..., draw_spatial=True)`,
   每个实例都会往 CWD 的 ./visualization 写一个 plotly HTML。这里 monkeypatch 掉。

用法(必须在 import 任何 sam3d 相关模块之前调用):
    from src.stage1.sam3d_engine import build_inference
    inference = build_inference("fastsam3d", pipeline_yaml=..., compile=False)
    out = inference(rgb, mask, seed=42, with_layout_postprocess=True)   # 官方签名
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = _ROOT / "third_party" / "sam-3d-objects"
FAST_ROOT = _ROOT / "third_party" / "Fast-SAM3D"
OFFICIAL_CKPT = OFFICIAL_ROOT / "checkpoints" / "hf"          # 权重只有这一份(fork 里只有 yaml)

# Fast-SAM3D 三个加速模块的超参。取自仓库 Quick Start 的 infer.sh(论文配置);注意
# notebook/infer.py 的 argparse 默认值和 infer.sh 不一致(slat_* 三个都不同),以 infer.sh 为准。
FAST_ACCEL_DEFAULTS = dict(
    ss_faster_stride=3, ss_warmup=2, ss_order=1, ss_momentum_beta=0.5,      # 阶段1 步缓存
    slat_thresh=1.5, slat_warmup=3, slat_token_ratio=0.1,                   # 阶段2 token 雕刻
    mesh_spectral_threshold_low=0.5, mesh_spectral_threshold_high=0.7,      # 阶段3 谱感知聚合
)


# --------------------------------------------------------------------------- HFER
def hfer_from_rgb_mask(rgb: np.ndarray, mask: np.ndarray,
                       target_size: int = 256, radius_ratio: float = 0.15) -> float:
    """把 Fast-SAM3D 的 `fft.fft2d.calculate_hfer_robust(<RGBA png 路径>)` 改写成吃数组。

    上游那份只接受【路径】,而且读的是 RGBA 抠图 png(RGB 是原图、alpha 是 mask)。我们
    手上是 (rgb, mask) 两个数组,写临时文件再读回去纯属浪费,所以逐步复刻:
      alpha bbox(pad 2)裁剪 -> 双边 resize 到 256(INTER_AREA)-> mask 外填物体灰度均值
      -> 2D Hanning 窗 -> FFT -> 半径 15% 以外的幅度能量占比。
    等价性已用仓库自带 demo 图逐位比对(atol 1e-12 通过,见 verify_fastsam3d_hfer.py)。
    """
    import cv2

    gray = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2GRAY)
    alpha = (np.asarray(mask).astype(bool).astype(np.uint8)) * 255
    coords = cv2.findNonZero(alpha)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = 2
        x, y = max(0, x - pad), max(0, y - pad)
        w = min(gray.shape[1] - x, w + 2 * pad)
        h = min(gray.shape[0] - y, h + 2 * pad)
        crop_gray, crop_alpha = gray[y:y + h, x:x + w], alpha[y:y + h, x:x + w]
    else:
        crop_gray, crop_alpha = gray, np.full_like(gray, 255)

    rg = cv2.resize(crop_gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
    ra = cv2.resize(crop_alpha, (target_size, target_size), interpolation=cv2.INTER_AREA)
    obj = rg[ra > 0]
    mean_val = float(np.mean(obj)) if obj.size > 0 else 128.0
    filled = np.full_like(rg, mean_val, dtype=np.float32)
    np.copyto(filled, rg.astype(np.float32), where=(ra > 0))
    win = np.outer(np.hanning(target_size), np.hanning(target_size))
    img = filled * win

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mag = np.abs(np.fft.fftshift(np.fft.fft2(img)))
    mag[crow, ccol] = 0
    total = mag.sum() + 1e-8
    r = int(min(rows, cols) * radius_ratio)
    yy, xx = np.ogrid[:rows, :cols]
    low = ((xx - ccol) ** 2 + (yy - crow) ** 2) <= r * r
    return float(mag[~low].sum() / total)


# --------------------------------------------------------------------------- 适配器
class _FastSAM3DInference:
    """把 Fast-SAM3D 的 `Inference` 包成官方签名,并逐实例注入 HFER。"""

    def __init__(self, inner, accel: dict):
        self._inner = inner
        self._accel = accel

    def __call__(self, image, mask, seed=None, with_layout_postprocess=True, pointmap=None):
        # fork 的 __call__ 没有 with_layout_postprocess 形参(内部写死 True)。我们上游一直
        # 传 True,所以语义一致;传 False 会被静默忽略,这里直接拦下,免得埋一个假开关。
        if not with_layout_postprocess:
            raise ValueError("Fast-SAM3D 的 pipeline 写死 with_layout_postprocess=True,"
                             "不支持关掉;要对照请用 layout_opt='off' 构建。")
        self._inner.get_hfer(hfer_from_rgb_mask(np.asarray(image), np.asarray(mask)))
        return self._inner(image, mask, seed=seed, pointmap=pointmap)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _silence_fft3d_html():
    """fork 在 `sample_sparse_structure` 里写死 draw_spatial=True,每个实例往 CWD 的
    ./visualization 落一个 plotly HTML。批量跑没有意义且污染工作目录 —— 换成同名包装,
    强制 draw_spatial=False(返回值不变)。"""
    from sam3d_objects.pipeline import inference_pipeline as ip

    orig = ip.process_and_visualize

    def quiet(coords_value, output_dir="./visualization", filter_radius=8,
              draw_spatial=True, draw_freq=True):
        return orig(coords_value, output_dir=output_dir, filter_radius=filter_radius,
                    draw_spatial=False, draw_freq=False)

    ip.process_and_visualize = quiet


def _patch_missing_solver():
    """Fast-SAM3D 发布版的一个真 bug:`flow_matching/model.py` 从 solver 里 import
    `Euler_easy_ss` 并注册进 solver 表,但 `solver.py` 根本没定义它 —— 于是整棵模型树
    import 就炸。那是 EasyCache 对照臂用的(`--enable_easy`),我们永远不开。

    这里在【导入模型树之前】把名字补上,但补的是一个"用到就报错"的占位类,绝不悄悄换成
    别的 solver 跑出一份看起来正常、其实算法不对的结果。(所有中间 __init__.py 都是空的,
    所以单独 import solver 不会提前触发 model.py。)"""
    import importlib

    mod = importlib.import_module(
        "sam3d_objects.model.backbone.generator.flow_matching.solver")
    if hasattr(mod, "Euler_easy_ss"):
        return

    class Euler_easy_ss:                                    # noqa: N801  (跟上游命名一致)
        def __init__(self, *a, **k):
            raise NotImplementedError(
                "Fast-SAM3D 发布版没有 Euler_easy_ss(EasyCache 的 SS 求解器),"
                "本项目也不用它;走到这里说明有人开了 --enable_easy。")

    mod.Euler_easy_ss = Euler_easy_ss
    print("[stage1] 已补 Fast-SAM3D 缺失的 solver 名 Euler_easy_ss(占位,用到即报错)", flush=True)


def _patch_dino_source():
    """fork 把 `Dino.__init__` 的默认 `repo_or_dir` 写死成了作者机器上的路径
    (`/data3/wmq/.../torch-cache/hub/facebookresearch_dinov2_main`, source='local'),
    在别人机器上必然 FileNotFoundError。两个 *_faster.yaml 里的 embedder 又没写这两个参数,
    所以只能改类的默认值。

    改回官方那份的取值(`facebookresearch/dinov2` + source='github'),而不是指向本机 hub
    缓存目录 —— 这样两条臂(官方 / Fast)加载 DINO 的路径【完全一致】,A/B 里少一个变量。
    离线可行:torch.hub 会命中 ~/.cache/torch/hub 的缓存(调用方已 monkeypatch trust_repo)。"""
    import sam3d_objects.model.backbone.dit.embedder.dino as dino_mod

    orig = dino_mod.Dino.__init__

    def patched(self, *a, **k):
        k.setdefault("repo_or_dir", "facebookresearch/dinov2")
        k.setdefault("source", "github")
        return orig(self, *a, **k)

    dino_mod.Dino.__init__ = patched


def _patch_pointmap_intrinsics():
    """fork 的第二个真 bug:`compute_pointmap` 在【深度模型自己给了内参】的分支里漏了
    `point_map_tensor["intrinsics"] = intrinsics`(官方有这个 else)。MoGe 恰好总是给内参,
    于是 `run_post_optimization` 里 KeyError: 'intrinsics' —— 而 fork 用 try/except 把它
    吞成一行 logger.error,**layout 后优化就这么被静默跳过了**,位姿退回未优化状态。
    实测:补之前日志里每个实例都是 "Error during layout post optimization: 'intrinsics'"。

    补法:让 MoGe 每次推理把内参挂在类上,compute_pointmap 返回时缺了就补进去 —— 用的是
    深度模型【自己】的内参,和官方那条 else 分支等价,不引入估计误差。"""
    from sam3d_objects.pipeline.depth_models.moge import MoGe
    from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap as P

    if getattr(P, "_s04_intr_patched", False):
        return
    moge_call = MoGe.__call__

    def moge_patched(self, image):
        out = moge_call(self, image)
        MoGe._s04_last_intrinsics = out.get("intrinsics")
        return out

    MoGe.__call__ = moge_patched
    orig = P.compute_pointmap

    def patched(self, image, pointmap=None):
        MoGe._s04_last_intrinsics = None
        d = orig(self, image, pointmap)
        if "intrinsics" not in d and getattr(MoGe, "_s04_last_intrinsics", None) is not None:
            d["intrinsics"] = MoGe._s04_last_intrinsics
        return d

    P.compute_pointmap = patched
    P._s04_intr_patched = True


def _quiet_loguru():
    """fork 每个 diffusion 步都打 INFO,批量跑日志会被淹掉;只留 WARNING 及以上
    (layout 后优化失败是 ERROR,仍然看得见)。"""
    import sys as _sys

    from loguru import logger

    logger.remove()
    logger.add(_sys.stderr, level="WARNING")


def _build_fast(pipeline_yaml, compile: bool, layout_opt: str, accel: dict):
    from omegaconf import OmegaConf

    # 结果顺序 [FAST_ROOT, FAST_ROOT/notebook, ...]:`sam3d_objects`/`fft` 命中 fork 的包,
    # `inference` 在 FAST_ROOT 下找不到再落到 notebook/ —— 都拿到 fork 那份。
    for p in (str(FAST_ROOT / "notebook"), str(FAST_ROOT)):
        sys.path.insert(0, p)
    # 保险:官方仓库是 editable 装的(.pth 直接把路径塞进 sys.path),确认没抢在前面
    for p in list(sys.path):
        if p and Path(p).resolve() == OFFICIAL_ROOT.resolve():
            sys.path.remove(p)
            sys.path.append(p)

    _patch_missing_solver()
    _patch_dino_source()
    from inference import Inference               # noqa: E402  (fork 的 notebook/inference.py)
    import sam3d_objects                          # noqa: F401,E402  确认解析到 fork
    got = Path(sam3d_objects.__file__).resolve()
    if FAST_ROOT.resolve() not in got.parents:
        raise RuntimeError(f"sam3d_objects 解析到了 {got},不是 Fast-SAM3D —— sys.path 顺序被抢了")

    cfg = OmegaConf.load(str(pipeline_yaml))
    # 权重、moge、其余 yaml 都用官方那份(fork 的 checkpoints/hf 只有 yaml 且路径写死作者机器);
    # 只把两个 generator 配置换成 fork 的 *_faster.yaml(绝对路径 -> os.path.join 会忽略
    # workspace_dir,不用往官方目录里塞文件)。权重仍是官方 ckpt:加速是 training-free 的。
    cfg.workspace_dir = str(OFFICIAL_CKPT)
    cfg.ss_generator_config_path = str(FAST_ROOT / "checkpoints" / "hf" / "ss_generator_faster.yaml")
    cfg.slat_generator_config_path = str(FAST_ROOT / "checkpoints" / "hf" / "slat_generator_faster.yaml")
    if layout_opt == "mesh":
        # 见模块 docstring 第 4 条:fork 的默认是 None = 不做 layout 后优化。
        cfg.layout_post_optimization_method = OmegaConf.create(
            {"_target_": "sam3d_objects.pipeline.inference_utils.layout_post_optimization",
             "_partial_": True})
    elif layout_opt != "off":
        raise ValueError(f"layout_opt 只支持 'mesh' / 'off',收到 {layout_opt!r}")

    ns = argparse.Namespace(**accel, enable_ss_faster=True, enable_slat_token=True,
                            enable_mesh_aggregation=True, enable_acceleration=True,
                            enable_taylor=False, enable_easy=False)
    inference = Inference(cfg, compile=compile, args=ns)
    inference.get_params(ns)
    inference.get_hfer(0.0)                       # 占位;每次 __call__ 前会按实例重算
    _silence_fft3d_html()
    if layout_opt == "mesh":
        _patch_pointmap_intrinsics()
    _quiet_loguru()
    print(f"[stage1] Fast-SAM3D 就绪 (layout_opt={layout_opt}, accel={accel})", flush=True)
    return _FastSAM3DInference(inference, accel)


def _build_official(pipeline_yaml, compile: bool):
    sys.path.insert(0, str(OFFICIAL_ROOT / "notebook"))
    from inference import Inference               # noqa: E402
    print(f"[stage1] 官方 SAM 3D 就绪 ({pipeline_yaml})", flush=True)
    return Inference(str(pipeline_yaml), compile=compile)


def build_inference(engine: str = "sam3d", pipeline_yaml=None, compile: bool = False,
                    layout_opt: str = "mesh", accel: dict | None = None):
    """构建推理后端。`engine`: 'sam3d'(官方)| 'fastsam3d'(Fast-SAM3D 加速 fork)。
    两者的返回对象都支持 `obj(rgb, mask, seed=..., with_layout_postprocess=True) -> dict`。"""
    if pipeline_yaml is None:
        pipeline_yaml = OFFICIAL_CKPT / "pipeline.yaml"
    if engine == "sam3d":
        return _build_official(pipeline_yaml, compile)
    if engine == "fastsam3d":
        a = dict(FAST_ACCEL_DEFAULTS)
        a.update(accel or {})
        return _build_fast(pipeline_yaml, compile, layout_opt, a)
    raise ValueError(f"未知 engine {engine!r}(支持 'sam3d' / 'fastsam3d')")
