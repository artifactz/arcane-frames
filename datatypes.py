from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class Video:
    _id: int
    filename: str
    num_frames: Optional[int] = None
    crop: Optional[str] = None

@dataclass
class FrameFeatures:
    frame_index: int
    frame_type: Optional[str] = None
    num_gftt: Optional[int] = None
    num_gftt_halfres: Optional[int] = None
    laplace_mean: Optional[float] = None

@dataclass
class Frame:
    rgb: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    u: Optional[np.ndarray] = None
    v: Optional[np.ndarray] = None
