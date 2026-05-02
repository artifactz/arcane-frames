from typing import Iterable
import os, re
from PIL import Image
import numpy as np
from datatypes import FrameFeatures
import ffmpeg, db

from upscaling.upscaler import Upscaler


def extract_indices(filename: str, min_cut_distance=24) -> set[int]:
    """
    Chose frame indices to extract based on frame features.
    """
    frame_features = db.get_frame_features(filename)
    frame_scores = calculate_frame_scores(frame_features)
    i_indices = np.array([i for i, f in enumerate(frame_features) if f.frame_type == "I"])

    # Remove indices too close to their predecessor
    mask = np.empty_like(i_indices)
    mask[0] = True
    mask[1:] = np.diff(i_indices) >= min_cut_distance
    i_indices = i_indices[mask]

    # Best frame per I frame interval
    return {max(range(i1, i2), key=lambda i: frame_scores[i]) for i1, i2 in zip(i_indices, i_indices[1:])}

def extract_images(filename: str, out_folder: str, upscaler: Upscaler, frame_indices: Iterable):
    """
    Extract frames from the given video file and saves them as PNG images in the specified output folder.
    """
    suffix = f"{m[1]}_" if (m := re.match(r".*(S\d+E\d+).*", filename)) else f"{os.path.basename(filename)}_"
    video = db.get_video(filename)
    os.makedirs(out_folder, exist_ok=True)

    for i, frame in ffmpeg.iter_frames(frame_indices, filename, pix_fmt=upscaler.pix_fmt, crop=video.crop):
        out_path = f"{out_folder}/{suffix}{i:06d}.png"
        # print(f"Extracting frame {i} to {out_path}...")
        upscaled = upscaler.upscale(frame)
        upscaled_1080p = _resample_1080p(upscaled)
        upscaled_1080p.save(out_path)

def _resample_1080p(array_or_image: np.ndarray | Image.Image) -> Image.Image:
    img = array_or_image if isinstance(array_or_image, Image.Image) else Image.fromarray(array_or_image)
    factor = 1080 / img.height
    img = img.resize((int(img.width * factor), int(img.height * factor)), resample=Image.LANCZOS)
    return img

def calculate_frame_scores(frame_features: Iterable[FrameFeatures]) -> list[float]:
    frame_types_score = {"B": 1.0, "P": 2.0, "I": 3.0}
    frame_types = np.array([frame_types_score[f.frame_type] for f in frame_features])
    nums_gftt = np.array([float(f.num_gftt) for f in frame_features])
    nums_gftt_halfres = np.array([float(f.num_gftt_halfres) for f in frame_features])
    laplace_means = np.array([f.laplace_mean for f in frame_features])

    frame_types = (frame_types - frame_types.mean()) / frame_types.std()
    nums_gftt = (nums_gftt - nums_gftt.mean()) / nums_gftt.std()
    nums_gftt_halfres = (nums_gftt_halfres - nums_gftt_halfres.mean()) / nums_gftt_halfres.std()
    laplace_means = (laplace_means - laplace_means.mean()) / laplace_means.std()

    # scores = 0.1 * frame_types + 0.5 * nums_gftt + 0.4 * laplace_means
    # scores = nums_gftt.astype(float)
    scores = nums_gftt_halfres.astype(float)  # seems best so far TODO
    return scores.tolist()


if __name__ == "__main__":
    from upscaling.diffusion import DiffusionUpscaler
    from upscaling.onnx_model import OnnxModelUpscaler, OnnxModelYuvUpscaler
    from upscaling.super_image_model import SuperImageModelUpscaler, SuperImageModelYuvUpscaler

    # upscaler = DiffusionUpscaler()
    # upscaler = SuperImageModelYuvUpscaler(4)
    upscaler = OnnxModelYuvUpscaler(4)

    for video in db.get_videos():
        indices = extract_indices(video.filename)
        extract_images(video.filename, "export/images", upscaler, indices)
