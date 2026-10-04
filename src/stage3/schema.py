"""Stage 3 keypoint dataclass — split out of the former monolithic SCHEMA.py.
`Keypoint` is PICKLED into pass1_keypoints.pkl; the SCHEMA.py shim re-exports it."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Keypoint:
    """A Stage 3 keypoint, defined on the STATIC reference-frame geometry,
    expressed in the SHARED scene coordinate frame (same frame as InstanceMesh.verts).

    v2 pass ordering: keypoints are sampled in pass1 from Stage 2a SEMANTIC parts (the
    motion-corrected parts don't exist until Stage 2b, which runs AFTER the tracker).
    So `part_id` here is the SEMANTIC part id. The motion-part assignment is attached
    later by pass2 as a SEPARATE keypoint_id -> motion_part_id map (Stage 2b output) and
    does NOT mutate this field or the tracker query index."""
    keypoint_id: int
    instance_id: int
    position: np.ndarray              # (3,) in scene frame
    part_id: Optional[int] = None     # SEMANTIC part id (Stage 2a) at pass1; see docstring
