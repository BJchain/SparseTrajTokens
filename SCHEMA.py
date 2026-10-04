"""Compatibility shim. The former monolithic SCHEMA.py was split into per-stage modules:
    src/stage0/schema.py  — Masklet, VideoInput (Stage 0 inputs)
    src/stage1/schema.py  — frame selection + SAM 3D recon + compose_scene + stage0_to_stage1
    src/stage3/schema.py  — Keypoint
    src/stage4/schema.py  — TrackerQuery, build_tracker_query, pass2 gap re-query
    src/geometry.py       — shared geometry/quaternion/mesh-extraction utilities

This module re-exports EVERY public + private name from those modules so that:
  - existing `from SCHEMA import X` imports (incl. underscore-prefixed) keep working unchanged;
  - old pickles (video_input.pkl / pass1_keypoints.pkl) that reference SCHEMA.VideoInput /
    SCHEMA.Keypoint by module path still unpickle — getattr(SCHEMA, name) resolves here.
New code should import from the per-stage modules directly; this shim is for back-compat.

LOADING NOTE — why importlib-by-path instead of `import src.geometry`:
  run_stage4_d4rt.py prepends third_party/D4rt to sys.path, whose OWN top-level `src` package
  (with __init__.py, a regular package) then SHADOWS our namespace `src`. A plain
  `import src.geometry` there resolves to D4RT's `src` and fails. So we load our 5 modules from
  EXPLICIT file paths and register them in sys.modules under their canonical `src.*` names. Their
  internal `from src.geometry import ...` then hits the sys.modules cache (no path resolution), so
  our `src.*.schema` and D4RT's `src.core`/`src.model` coexist in one process.
"""
import os as _os
import sys as _sys
import importlib.util as _ilu

_HERE = _os.path.dirname(_os.path.abspath(__file__))

# (canonical module name, file path) — dependency order: deps before dependents.
_MODULES = [
    ("src.geometry", "src/geometry.py"),
    ("src.stage0.schema", "src/stage0/schema.py"),
    ("src.stage3.schema", "src/stage3/schema.py"),
    ("src.stage1.schema", "src/stage1/schema.py"),
    ("src.stage4.schema", "src/stage4/schema.py"),
]


def _load(modname, relpath):
    if modname in _sys.modules:
        return _sys.modules[modname]
    spec = _ilu.spec_from_file_location(modname, _os.path.join(_HERE, relpath))
    mod = _ilu.module_from_spec(spec)
    _sys.modules[modname] = mod          # register BEFORE exec so intra-imports hit the cache
    spec.loader.exec_module(mod)
    return mod


_loaded = [_load(_n, _p) for _n, _p in _MODULES]
for _mod in _loaded:
    for _name in dir(_mod):
        if not _name.startswith("__"):
            globals()[_name] = getattr(_mod, _name)
del _mod, _name, _loaded
