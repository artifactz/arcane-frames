import numpy as np
import db


def select_frame_between_i_frames(
    filename: str,
    min_quality=0.35,
    min_distance=24
) -> set[int]:
    """
    Chooses best frame per I frame interval wrt. score function.

    Args:
        filename: Video filename.
        min_quality: Minimum quality for a frame to be selected.
        min_distance: Minimum distance between selected frames. If two I frames are closer than this, only one will be selected.

    Returns:
        Set of selected frame indices.
    """
    video = db.get_video(filename)
    indexed_frame_types = db.get_frame_types(video._id)
    i_indices = np.array([i for i, frame_type in indexed_frame_types if frame_type == "I"])

    # Remove indices too close to their predecessor
    mask = np.empty_like(i_indices, dtype=bool)
    mask[0] = True
    mask[1:] = np.diff(i_indices) >= min_distance
    i_indices = i_indices[mask]

    # Best frame per I frame interval
    indexed_qualities = db.get_frame_qualities(video._id)
    max_quality_index = indexed_qualities[-1][0]
    i_indices = [i for i in i_indices if i <= max_quality_index]

    selected_indices = [max(range(i1, i2), key=lambda i: indexed_qualities[i][1]) for i1, i2 in zip(i_indices, i_indices[1:])]
    return {i for i in selected_indices if indexed_qualities[i][1] > min_quality}
