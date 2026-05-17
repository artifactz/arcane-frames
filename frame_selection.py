from typing import Iterable, Callable, Sequence, Iterator
import numpy as np
from datatypes import FrameFeatures
import db, resnet_util
from quality_estimation import inference


def select_frame_between_i_frames(
    filename: str,
    score_func: Callable[[Iterable[FrameFeatures]], Sequence[float]],
    min_cut_distance=24
) -> set[int]:
    """
    Chooses best frame per I frame interval wrt. score function.

    Args:
        filename: Video filename.
        score_func: Function that takes an iterable of FrameFeatures and returns a sequence of scores (one per frame).
        min_cut_distance: Minimum distance between selected frames. If two I frames are closer than this, only one will
                          be selected.

    Returns:
        Set of selected frame indices.
    """
    frame_features = db.get_frame_features(filename)
    i_indices = np.array([i for i, f in enumerate(frame_features) if f.frame_type == "I"])

    # Remove indices too close to their predecessor
    mask = np.empty_like(i_indices, dtype=bool)
    mask[0] = True
    mask[1:] = np.diff(i_indices) >= min_cut_distance
    i_indices = i_indices[mask]

    # Best frame per I frame interval
    frame_scores = score_func(frame_features)
    return {max(range(i1, i2), key=lambda i: frame_scores[i]) for i1, i2 in zip(i_indices, i_indices[1:])}

def score_mixed_01(frame_features: Iterable[FrameFeatures]) -> np.ndarray:
    frame_types_score = {"B": 1.0, "P": 2.0, "I": 3.0, None: 0.0}
    frame_types = np.array([frame_types_score[f.frame_type] for f in frame_features], dtype=float)
    nums_gftt = np.array([f.num_gftt for f in frame_features], dtype=float)
    nums_gftt_halfres = np.array([f.num_gftt_halfres for f in frame_features], dtype=float)
    laplace_means = np.array([f.laplace_mean for f in frame_features], dtype=float)

    frame_types = (frame_types - frame_types.mean()) / frame_types.std()
    nums_gftt = (nums_gftt - nums_gftt.mean()) / nums_gftt.std()
    nums_gftt_halfres = (nums_gftt_halfres - nums_gftt_halfres.mean()) / nums_gftt_halfres.std()
    laplace_means = (laplace_means - laplace_means.mean()) / laplace_means.std()

    scores = 0.1 * frame_types + 0.2 * nums_gftt + 0.4 * nums_gftt_halfres + 0.3 * laplace_means
    return scores

def score_nums_gftt_halfres(frame_features: Iterable[FrameFeatures]) -> np.ndarray:
    nums_gftt_halfres = np.array([f.num_gftt_halfres for f in frame_features], dtype=float)
    scores = (nums_gftt_halfres - nums_gftt_halfres.mean()) / nums_gftt_halfres.std()
    return scores


def filter_frames_quality(filename: str, indices: list[int], quality_threshold=0.35) -> Iterator[int]:
    """Filters given frame indices by quality estimation."""
    for i, (embed, prob) in enumerate(resnet_util.iter_frames_resnet(filename, indices)):
        quality = inference.from_resnet(embed, prob)
        if quality > quality_threshold:
            yield indices[i]
