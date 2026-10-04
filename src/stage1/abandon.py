"""提前放弃:在把候选池烧完之前就判出"这个实例达不到 accept_onmask",把它的候选数砍掉。

**两道门,一前一后**(2026-08-19 加了第二道):

| 门 | 何时判 | 输入 | 代价 |
|---|---|---|---|
| ① 预判门 `abandon_frac` | 花第一次 SAM 3D 调用【之前】 | top-1 候选帧的 4 个 mask 特征 | 0 |
| ② on-mask 门 `onmask_gate` | 第 1 次调用【之后】 | 该次实测的 on_mask | 1 次调用 |

② 贵一次调用,但信号强得多 —— mask 几何只是"这一帧好不好看",而 on_mask 是早停判据本身。

## 为什么必须按引擎分开(2026-08-19)

标签是"达不达得到 accept_onmask",而**达标率是引擎的函数**:同一 29 场景 A/B,官方 SAM3D
52.1% vs Fast-SAM3D 25.5%。特征(mask 几何)与引擎无关,失败的**成因**却不同 —— 官方失败
主要来自"这一帧本来就难"(mask 小/被裁/破碎),Fast-SAM3D 的失败还叠了一层加速管线本身的
网格退化,那部分**看 mask 是看不出来的**。实测(142~2337 实例,按 clip 分组 5 折 CV):

| 数据 | 预判门 AUC | 判读 |
|---|---|---|
| 官方 SAM3D,1k 批(2337 实例 / 483 clip) | **0.716 ± 0.012** | 稳,可用 |
| 官方 SAM3D,ab 29 场景(144 实例) | 0.738 ± 0.080 | 同上,小样本对照臂 |
| **Fast-SAM3D,同 29 场景(142 实例)** | **0.546 ± 0.249** | **≈ 掷硬币** |

而且学出来的权重结构不一样。同一口径(k=3 标签、同一份代码重拟合)对比标准化后的权重
`[area, completeness, occlusion, compactness]`:官方 1k 批 `[-1.12, -0.24, -0.27, -0.58]`
—— `area` 主导且为负(mask 越大越不容易失败,符合直觉);Fast-SAM3D `[+0.06, +0.29, -0.31,
+0.00]` —— **`area` 权重 ≈ 0**,主导项换成了 occlusion 且整体量级很小(= 学不出东西)。
(现役官方 npz 是 2026-08-12 用 508 实例训的,权重 `[-1.44, -0.49, -1.02, -4.40]`,
compactness 更重;两者都指向"官方这边 mask 几何确实带信息"。)**拿官方模型去跑 Fast-SAM3D 是把一个校准错的分类器当真值用**,所以
`get_model()` 按引擎取不同的 npz,取不到就返回 None(不退回另一个引擎的模型)。

反过来,on-mask 门在 Fast-SAM3D 上白捡(同一批数据,k=3 回放):

| on-mask 门 | Fast-SAM3D 省调用 / 丢达标 | 官方 SAM3D 省调用 / 丢达标 |
|---|---|---|
| 0.55 | −6.9% / **0** | −9.0% / 1.5% |
| **0.65** | **−15.7% / 0** | −15.7% / **2.8%** |
| 0.75 | −32.4% / 2.8% | −25.1% / 4.6% |

Fast-SAM3D 上 rank-1 on_mask < 0.65 的 33 个实例**没有一个**被后面的候选救回来(AUC 0.789);
官方在同一阈值下要丢 65 个真达标实例。这就是"Fast-SAM3D 适用版本"的全部内容:**预判门退到
辅助位,主力换成 on-mask 门**。代价是那些反正都失败的实例会留下略差的网格
(best-on-mask 均值 −0.0238,与 08-13 用户接受的 −0.0099 同量级)。

## 官方 SAM3D 的预判门(不变,2026-08-12 实测依据 STAGE1_FRAME_SCORER_PLAN.md §8/§9)

  * **72% 的 SAM 3D 调用花在注定失败的实例上**,它们把候选池烧满却拿不到 on-mask ≥ 0.9。
  * 天花板对比:完美排序只能省 13%,完美提前放弃能省 44%。**杠杆完全不在排序上。**
  * cap=2 时达标实例里 79% 本来就在前 2 个候选内命中,所以名义误伤 ≫ 真丢 mesh:
    判死 20% -> 真丢 1.2%;判死 30% -> 2.4%。

⚠️ **数据的自截尾**:采数那一跑本身开着预判门,被砍到 cap 的实例只观测到前 cap 个候选,
"它在第 cap+1 个候选会不会达标"永远未知。所以**预判门无法在开着它采的数据上重新评估**
(能评的只有它没判死的那部分)。重训用 `src/stage1/train_abandon.py`,它会把这类实例剔除
并打印剔除数。想干净地重测预判门,得跑一次 `--abandon-frac 0`。

⚠️ 两个模型都只在 **Stereo4D** 上训过 —— 那也正是 110k 的目标数据集。MOVi/DAVIS 没进训练集。
**换数据集前先重新验证。**

`abandon_frac=0` 且 `onmask_gate=0` 时本模块完全不参与,行为与改动前逐位一致。
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

_DIR = os.path.dirname(__file__)
# 按引擎取模型。**故意不设 fallback**:用错引擎的模型比不用更糟(见上表)。
MODEL_PATHS = {
    "sam3d":     os.path.join(_DIR, "abandon_model.npz"),
    "fastsam3d": os.path.join(_DIR, "abandon_model_fastsam3d.npz"),
}
_MODEL_PATH = MODEL_PATHS["sam3d"]        # 向后兼容:老调用点 get_model() 仍取官方模型


class AbandonModel:
    """逻辑回归 + 分位阈值表。分数越高 = 越可能达不到 accept_onmask。"""

    def __init__(self, mean, std, weights, bias, fracs, thresholds, meta=None):
        self.mean = np.asarray(mean, float)
        self.std = np.asarray(std, float)
        self.weights = np.asarray(weights, float)
        self.bias = float(bias)
        self.fracs = np.asarray(fracs, float)
        self.thresholds = np.asarray(thresholds, float)
        self.meta = meta or {}

    # --- 元信息(老模型的 npz 里没有这些键 -> 给出与改动前等价的默认值)
    @property
    def onmask_gate(self) -> float:
        """rank-1 实测 on_mask 低于此值就停;0 = 不启用(官方模型即为 0)。"""
        return float(self.meta.get("onmask_gate", 0.0) or 0.0)

    @property
    def rec_frac(self) -> float:
        """建议的预判门 abandon_frac。**仅提示不强制** —— 超了只打印警告,不改用户给的值。"""
        return float(self.meta.get("rec_frac", 0.2) or 0.2)

    @property
    def engine(self) -> str:
        return str(self.meta.get("engine", "sam3d"))

    @classmethod
    def load(cls, path: str = _MODEL_PATH) -> "AbandonModel":
        d = np.load(path)
        keys = ("n_train", "base_rate", "auc_cv", "auc_cv_std", "n_clips",
                "engine", "k", "onmask_gate", "rec_frac", "n_censored")
        return cls(d["mean"], d["std"], d["weights"], d["bias"], d["fracs"], d["thresholds"],
                   meta={k: d[k].item() for k in keys if k in d})

    def score(self, feats: np.ndarray) -> float:
        """P(这个实例达不到 accept_onmask)。`feats` = mask_features 的 (4,) 向量。"""
        z = (np.asarray(feats, float) - self.mean) / self.std
        return float(1.0 / (1.0 + np.exp(-(z @ self.weights + self.bias))))

    def threshold_for(self, frac: float) -> float:
        """把"判死比例"换成分数阈值。阈值取自**训练分布**的分位,不依赖当前 batch —— 否则小 batch
        里会把该批最差的那几个无条件判死,哪怕它们其实都不错。"""
        f = float(frac)
        if f <= 0:
            return float("inf")               # 永不放弃
        if f >= 1:
            return float("-inf")              # 全部放弃(只用于调试)
        return float(np.interp(f, self.fracs, self.thresholds,
                               left=self.thresholds[0], right=self.thresholds[-1]))

    def describe(self) -> str:
        m = self.meta
        return (f"engine={self.engine} k={m.get('k', '?')} n={m.get('n_train', '?')} "
                f"CV_AUC={m.get('auc_cv', float('nan')):.3f}±{m.get('auc_cv_std', float('nan')):.3f} "
                f"注定失败率={m.get('base_rate', float('nan')):.3f} on-mask门={self.onmask_gate:g}")


_CACHE: "dict[str, Optional[AbandonModel]]" = {}


def get_model(engine: str = "sam3d", path: "Optional[str]" = None) -> Optional[AbandonModel]:
    """按引擎带缓存地加载。模型文件缺失/引擎不认识就返回 None(调用方退回"不放弃"),
    **绝不退回另一个引擎的模型** —— 校准错的分类器比没有分类器更糟。"""
    key = path or str(engine)
    if key in _CACHE:
        return _CACHE[key]
    p = path or MODEL_PATHS.get(str(engine))
    if p is None:
        print(f"[abandon] 引擎 {engine!r} 没有对应模型 — 本次不启用提前放弃", flush=True)
        _CACHE[key] = None
        return None
    try:
        m = AbandonModel.load(p)
        print(f"[abandon] 载入 {os.path.basename(p)}: {m.describe()}", flush=True)
        if m.meta.get("auc_cv", 1.0) < 0.6:
            print(f"[abandon] ⚠️ 该模型预判门 CV AUC 只有 {m.meta['auc_cv']:.3f} —— 接近掷硬币,"
                  f"建议 --abandon-frac <= {m.rec_frac:g},主要靠 on-mask 门 {m.onmask_gate:g}",
                  flush=True)
        _CACHE[key] = m
    except Exception as e:
        print(f"[abandon] 模型加载失败 ({type(e).__name__}: {e}) — 本次不启用提前放弃",
              flush=True)
        _CACHE[key] = None
    return _CACHE[key]
