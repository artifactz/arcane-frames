from typing import Iterator
from itertools import batched
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


def ensure_frame_quality(filename: str, batch_size=64):
    """
    Ensures that all frames of the video have a quality estimation in the database.
    """
    from feature_extraction import resnet50
    from quality_estimation import inference

    video = db.get_video(filename)
    available_indices = list(reversed(sorted([i for i, _ in db.get_frame_qualities(video._id)])))
    if available_indices:
        missing_indices = []
        for i in range(video.num_frames):
            if available_indices and i == available_indices[-1]:
                available_indices.pop()
            else:
                missing_indices.append(i)
    else:
        missing_indices = None

    for indexed_frames in batched(ffmpeg.buffered(ffmpeg.iter_frames(missing_indices, filename, crop=video.crop), batch_size), batch_size):
        batch = [Image.fromarray(frame.rgb) for _, frame in indexed_frames]
        embeds, probs = resnet50.from_images(batch)
        qualities = inference.from_resnets(embeds, probs)
        for i, (frame_index, _) in enumerate(indexed_frames):
            db.set_frame_quality(video._id, frame_index, qualities[i])
