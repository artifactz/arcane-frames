from typing import Iterator
from PIL import Image
import numpy as np
import db, ffmpeg
from feature_extraction import resnet50_worker


def iter_frames_resnet(filename: str, indices: list[int]) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Retrieves and/or calculates resnet embeddings and probabilities for the given frames
    and makes sure they are stored in db.
    """
    video = db.get_video(filename)

    db.ensure_frame_resnet_table()
    cached_resnet = dict(db.get_frames_resnet(video._id, indices))

    missing_indices = [i for i in indices if not i in cached_resnet]
    if missing_indices:
        frame_iter = ffmpeg.iter_frames(missing_indices, filename, crop=video.crop)
        image_iter = ((i, Image.fromarray(frame.rgb)) for i, frame in frame_iter)
        for resnet_result in resnet50_worker.iter_inference(image_iter):
            frame_index, (embed, prob) = resnet_result
            db.set_frame_resnet(video._id, frame_index, embed, prob)
            cached_resnet[frame_index] = (embed, prob)

    return (cached_resnet[i] for i in indices)
