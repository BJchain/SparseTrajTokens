"""
Stage 0-pre helper: Qwen-VL (Alibaba DashScope / 千问百炼) client.

Used by run_stage0_pre.py to (1) ENUMERATE every distinct instance that appears
anywhere in a video and (2) classify each instance rigid / non-rigid. Its output
drives SAM 3 (concept text + optional box prompt) and rides downstream as the
authoritative object_type — replacing Stage 2b's tracker-dependent residual gate.

Why a hand-rolled urllib client (no openai SDK)
------------------------------------------------
  * Zero extra deps — the `qwen` env only needs stdlib + Pillow (already a pipeline dep).
  * The box / connectivity facts were ALL validated 2026-06-24 (see memory
    project_qwen_dashscope_setup):
      - base_url (OpenAI-compatible):  https://dashscope.aliyuncs.com/compatible-mode/v1
      - key:        env DASHSCOPE_API_KEY  (exported in ~/.bashrc)
      - PROXY:      this box runs a Clash proxy on 127.0.0.1:17890 that is hostile to
                    Aliyun. Direct is fine AND faster -> we FORCE proxy-bypass with
                    urllib ProxyHandler({}). Do not rely on Clash auto-routing.
      - model:      qwen3-vl-plus  (account-accessible; -latest aliases & open qwen2.5-vl
                    are 403 access_denied).
      - images:     local file:// is unsupported on the OpenAI-compatible path -> send
                    base64 data URLs. Multi-frame "video" understanding uses a
                    {"type":"video","video":[url,...]} content block.
      - 🔑 COORDS:  grounding returns [{"bbox":[x1,y1,x2,y2],"label":..}] in a 0..1000
                    NORMALISED frame (x rel. width, y rel. height), resolution-independent.
                    We divide by 1000 -> normalised [0,1] xyxy. (Earlier confusion: y2=807
                    on a 480px-tall image is NOT a pixel overflow, it's 80.7% of height.)
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Optional, Sequence

import numpy as np
from PIL import Image

# --- validated constants ----------------------------------------------------
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3-vl-plus"
_COORD_SCALE = 1000.0  # Qwen-VL grounding coordinate frame

# A single opener that NEVER uses a proxy (ProxyHandler({}) wins over env http(s)_proxy).
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class QwenError(RuntimeError):
    pass


# =============================================================================
# low-level transport
# =============================================================================
def _api_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise QwenError(
            "DASHSCOPE_API_KEY not set. It is exported in ~/.bashrc; a non-login shell "
            "won't auto-source it, so export it before running this stage."
        )
    return key


def _chat(messages: list, model: str, *, max_tokens: int = 2048,
          temperature: float = 0.0, timeout: int = 90, retries: int = 4) -> str:
    """POST a chat-completions request (proxy-bypassed) and return the message content."""
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    url = f"{BASE_URL}/chat/completions"

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with _NO_PROXY_OPENER.open(req, timeout=timeout) as resp:
                out = json.load(resp)
            return out["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            # 4xx (bad request / access denied) won't fix on retry — fail fast.
            if 400 <= e.code < 500 and e.code != 429:
                raise QwenError(f"HTTP {e.code}: {detail}") from e
            last_err = QwenError(f"HTTP {e.code}: {detail}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as e:
            last_err = e
        time.sleep(1.5 * (attempt + 1))
    raise QwenError(f"chat request failed after {retries} attempts: {last_err}")


# =============================================================================
# image encoding
# =============================================================================
def _to_data_url(frame, *, quality: int = 90) -> str:
    """np.ndarray (H,W,3) uint8 RGB | path | PIL.Image  ->  base64 JPEG data URL."""
    if isinstance(frame, np.ndarray):
        img = Image.fromarray(frame.astype(np.uint8), "RGB")
    elif isinstance(frame, Image.Image):
        img = frame.convert("RGB")
    else:  # path-like
        img = Image.open(frame).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


# =============================================================================
# JSON / coordinate parsing
# =============================================================================
def _salvage_objects(t: str) -> list:
    """Scan for top-level {...} objects and parse each independently. Recovers as many
    complete items as possible from a TRUNCATED array (e.g. max_tokens cut mid-list)."""
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(t):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    out.append(json.loads(t[start:i + 1]))
                except json.JSONDecodeError:
                    pass
                start = None
    return out


def _extract_json(text: str):
    """Pull a JSON array/object out of an LLM reply (tolerates ```json fences, prose, and
    truncation)."""
    t = text.strip()
    if "```" in t:  # strip code fences
        m = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
        if m:
            t = m.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # whole/outermost parse failed (often truncation) -> salvage individual objects
    salvaged = _salvage_objects(t)
    if salvaged:
        return salvaged
    # last resort: a single outermost {...}
    i, j = t.find("{"), t.rfind("}")
    if 0 <= i < j:
        try:
            return [json.loads(t[i:j + 1])]
        except json.JSONDecodeError:
            pass
    raise QwenError(f"could not parse JSON from model reply: {text[:300]!r}")


def _norm_bbox(raw) -> Optional[list]:
    """[x1,y1,x2,y2] in 0..1000 -> normalised [0,1] xyxy, ordered & clamped. None if invalid."""
    if not (isinstance(raw, (list, tuple)) and len(raw) == 4):
        return None
    try:
        x1, y1, x2, y2 = (float(v) / _COORD_SCALE for v in raw)
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    clamp = lambda v: max(0.0, min(1.0, v))
    box = [clamp(x1), clamp(y1), clamp(x2), clamp(y2)]
    if box[2] - box[0] < 1e-4 or box[3] - box[1] < 1e-4:
        return None  # degenerate
    return box


def xyxy_to_xywh(box: Sequence[float]) -> list:
    """Normalised [x1,y1,x2,y2] -> SAM 3's [x_min, y_min, w, h] (all in [0,1])."""
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


# =============================================================================
# high-level VLM tasks
# =============================================================================
# Shared SCOPE: countable discrete objects ("things"), NOT amorphous background ("stuff").
# Includes movable agents/items AND discrete static objects (tree/pole/sign); excludes
# boundary-less background. (Scope decision by user, 2026-06-24.)
_SCOPE = (
    "List every DISTINCT, individually-segmentable WHOLE object INSTANCE that has a CLEAR CLOSED "
    "boundary: people, animals, vehicles, movable items, AND discrete bounded standing objects "
    "(tree, pole, sign, traffic light, bench). EXCLUDE: "
    "(a) amorphous background 'stuff' with no single-object boundary — sky, ground, road, "
    "pavement, crosswalk, manhole, water, puddle, wall, building, grass, foliage / pile / heap; "
    "(b) PARTS or attachments of a larger object — e.g. a person's helmet / boot / vest / glove, a "
    "horse's saddle, a car's wheel or light (segment the whole person / horse / car instead); "
    "(c) massed or repeated background scenery — do NOT enumerate a row or cluster of similar trees "
    "or poles as many instances; report only the few prominent, clearly-separated ones. "
    "Use SIMPLE noun phrases (a category word, optionally one visual adjective) — no relations, "
    "comparatives, actions, or numbers."
)

_DETECT_PROMPT = (
    "Detect objects in this image. " + _SCOPE + " Return ONLY a JSON list of "
    "{\"bbox\":[x1,y1,x2,y2],\"label\":\"<noun>\"}; one entry per instance "
    "(repeat the label if there are several)."
)

_ENUMERATE_PROMPT = (
    "These frames are sampled in order from one video. List every object that appears at "
    "ANY point in the video, including objects that enter, leave, or are briefly occluded. "
    + _SCOPE + " Return ONLY a JSON list of {\"label\":\"<noun>\",\"count\":<int>} with the "
    "max number of simultaneously-visible instances of that label."
)

_MOTION_PROMPT = (
    "These frames are sampled IN TEMPORAL ORDER from one video clip. For each listed object, "
    "judge whether the OBJECT ITSELF moves during this clip: DYNAMIC = the object translates, "
    "rotates, or deforms/articulates in the WORLD at any point of the clip (a driving car, a "
    "walking person, a swaying branch); STATIC = the object stays physically still for the whole "
    "clip (a parked car, a tree, a pole, a traffic sign, a building fixture). CRITICAL: the "
    "camera may be moving, so apparent image motion caused by CAMERA movement does NOT count — "
    "judge only the object's own displacement/deformation relative to the static world (ground, "
    "buildings). If a label covers several instances, answer DYNAMIC if ANY of them moves. "
    "Return ONLY a JSON list of {\"label\":\"<the same label>\",\"motion\":\"dynamic\"|\"static\","
    "\"confidence\":<0..1>}. Labels given: "
)

# COMBINED single-call prompt (2026-07-31 orchestration speedup #3): one video-block call
# returns enumerate + seed box + rigid/non-rigid + dynamic/static for every instance,
# replacing the 7-call legacy path (1 enumerate + 4 detect + 1 type + 1 motion). The
# per-task wording is copied from the validated prompts above (scope / camera-motion
# discount / rigid examples) so the semantics stay aligned with the legacy path.
_COMBINED_PROMPT = (
    "These {n} frames are sampled IN TEMPORAL ORDER from one video clip; they are numbered "
    "0 to {last} in the order given. "
    + _SCOPE +
    " Include objects that appear at ANY point of the clip (entering, leaving, or briefly "
    "occluded). For EACH such object label output ONE JSON entry with ALL of these fields: "
    "\"label\": the noun phrase; "
    "\"count\": the max number of simultaneously-visible instances of that label; "
    "\"frame\": the frame NUMBER (0..{last}, among the given frames) where the object is "
    "largest and clearest; "
    "\"bbox\": [x1,y1,x2,y2] of one representative instance ON that frame, coordinates in a "
    "0-1000 normalised frame (x relative to width, y relative to height), or null if it "
    "cannot be localised; "
    "\"object_type\": \"rigid\" if it is physically rigid (keeps a fixed shape while it "
    "moves — e.g. car, box, ball, bottle) or \"non-rigid\" if it deforms / articulates / is "
    "a living creature (e.g. person, animal, cloth, rope); "
    "\"type_confidence\": 0..1; "
    "\"motion\": \"dynamic\" if the OBJECT ITSELF translates, rotates, or deforms/"
    "articulates in the WORLD at any point of the clip (a driving car, a walking person), "
    "\"static\" if it stays physically still for the whole clip (a parked car, a tree, a "
    "pole, a building fixture). CRITICAL: the camera may be moving, so apparent image "
    "motion caused by CAMERA movement does NOT count — judge only the object's own "
    "displacement/deformation relative to the static world. If a label covers several "
    "instances, answer dynamic if ANY of them moves; "
    "\"motion_confidence\": 0..1. "
    "Return ONLY a raw JSON array of these entries — no markdown, no code fences, no "
    "commentary, no trailing text."
)


def analyze_video(frames: Sequence, *, model: str = DEFAULT_MODEL) -> list:
    """SINGLE-call combined analysis: temporally-ordered sampled frames (video block) ->
    [{label, count, frame_pos, bbox, object_type, type_confidence, motion,
      motion_confidence}] where bbox is normalised [0,1] xyxy (None if not localised) and
    frame_pos indexes into the GIVEN frame list (caller maps it back to video frame idx).
    Replaces enumerate_video + detect_instances + classify_types + classify_motion with
    one round trip; fields the model omits get the same conservative defaults the legacy
    path used (non-rigid / dynamic / conf 0.0 — applied by the caller)."""
    n = len(frames)
    content = [
        {"type": "video", "video": [_to_data_url(f) for f in frames]},
        {"type": "text", "text": _COMBINED_PROMPT.format(n=n, last=n - 1)},
    ]
    reply = _chat([{"role": "user", "content": content}], model, max_tokens=3072)
    out = []
    for item in _extract_json(reply):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip().lower()
        if not label:
            continue
        try:
            count = max(1, int(item.get("count", 1)))
        except (TypeError, ValueError):
            count = 1
        try:
            frame_pos = int(item.get("frame", n // 2))
        except (TypeError, ValueError):
            frame_pos = n // 2
        frame_pos = max(0, min(n - 1, frame_pos))
        otype = str(item.get("object_type", "")).strip().lower()
        if otype not in ("rigid", "non-rigid"):
            otype = None
        motion = str(item.get("motion", "")).strip().lower()
        if motion not in ("dynamic", "static"):
            motion = None

        def _conf(key):
            try:
                return float(item.get(key, 1.0))
            except (TypeError, ValueError):
                return 1.0

        out.append({
            "label": label, "count": count, "frame_pos": frame_pos,
            "bbox": _norm_bbox(item.get("bbox") or item.get("bbox_2d")),
            "object_type": otype, "type_confidence": _conf("type_confidence"),
            "motion": motion, "motion_confidence": _conf("motion_confidence"),
        })
    return out


_TYPE_PROMPT = (
    "For each listed object, decide whether it is physically RIGID (keeps a fixed shape "
    "while it moves — e.g. car, box, ball, bottle) or NON-RIGID (deforms / articulates / "
    "is a living creature — e.g. person, animal, cloth, rope). Return ONLY a JSON list of "
    "{\"label\":\"<the same label>\",\"object_type\":\"rigid\"|\"non-rigid\","
    "\"confidence\":<0..1>}. Labels given: "
)


def detect_instances(frame, *, model: str = DEFAULT_MODEL) -> list:
    """Single frame -> [{label, bbox}] where bbox is normalised [0,1] xyxy (None if undetected)."""
    content = [
        {"type": "image_url", "image_url": {"url": _to_data_url(frame)}},
        {"type": "text", "text": _DETECT_PROMPT},
    ]
    # Cluttered scenes can make the model emit dozens of boxes (slow). Cap generation:
    # prominent objects are listed first, _extract_json salvages complete items from a
    # truncated list, and we only need ONE seed box per concept downstream.
    reply = _chat([{"role": "user", "content": content}], model, max_tokens=512)
    out = []
    for item in _extract_json(reply):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip().lower()
        if not label:
            continue
        out.append({"label": label, "bbox": _norm_bbox(item.get("bbox") or item.get("bbox_2d"))})
    return out


def enumerate_video(frames: Sequence, *, model: str = DEFAULT_MODEL) -> list:
    """Sampled frames as a video block -> [{label, count}] union over the whole clip."""
    video_urls = [_to_data_url(f) for f in frames]
    content = [
        {"type": "video", "video": video_urls},
        {"type": "text", "text": _ENUMERATE_PROMPT},
    ]
    reply = _chat([{"role": "user", "content": content}], model)
    out = []
    for item in _extract_json(reply):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip().lower()
        if not label:
            continue
        try:
            count = max(1, int(item.get("count", 1)))
        except (TypeError, ValueError):
            count = 1
        out.append({"label": label, "count": count})
    return out


def classify_motion(frames: Sequence, labels: Sequence[str], *, model: str = DEFAULT_MODEL) -> dict:
    """Temporally-ordered frames (video block) + label list ->
    {label: {'motion': 'dynamic'|'static', 'confidence': float}}.
    Judges the object's OWN world motion over the clip (camera motion excluded, per prompt).
    Same mechanism/format as classify_types; labels missing from the reply are simply absent
    (callers should default them to 'dynamic' — the conservative full-pipeline route)."""
    labels = [str(l).strip().lower() for l in labels if str(l).strip()]
    if not labels:
        return {}
    content = [
        {"type": "video", "video": [_to_data_url(f) for f in frames]},
        {"type": "text", "text": _MOTION_PROMPT + json.dumps(labels)},
    ]
    reply = _chat([{"role": "user", "content": content}], model)
    out = {}
    for item in _extract_json(reply):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip().lower()
        motion = str(item.get("motion", "")).strip().lower()
        if label not in labels or motion not in ("dynamic", "static"):
            continue
        try:
            conf = float(item.get("confidence", 1.0))
        except (TypeError, ValueError):
            conf = 1.0
        out[label] = {"motion": motion, "confidence": conf}
    return out


def classify_types(frame, labels: Sequence[str], *, model: str = DEFAULT_MODEL) -> dict:
    """Frame + label list -> {label: {'object_type': 'rigid'|'non-rigid', 'confidence': float}}."""
    labels = [str(l).strip().lower() for l in labels if str(l).strip()]
    if not labels:
        return {}
    content = [
        {"type": "image_url", "image_url": {"url": _to_data_url(frame)}},
        {"type": "text", "text": _TYPE_PROMPT + json.dumps(labels)},
    ]
    reply = _chat([{"role": "user", "content": content}], model)
    out = {}
    for item in _extract_json(reply):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip().lower()
        otype = str(item.get("object_type", "")).strip().lower()
        if label not in labels or otype not in ("rigid", "non-rigid"):
            continue
        try:
            conf = float(item.get("confidence", 1.0))
        except (TypeError, ValueError):
            conf = 1.0
        out[label] = {"object_type": otype, "confidence": conf}
    return out
