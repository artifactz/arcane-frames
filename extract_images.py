from typing import Iterable
import os, re
from PIL import Image
import numpy as np
import ffmpeg, db

from upscaling.upscaler import Upscaler


def extract_images(filename: str, out_folder: str, upscaler: Upscaler, frame_indices: Iterable):
    """
    Extracts frames from the given video file and saves them as PNG images at 1080p (height) in the specified output
    folder.
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


if __name__ == "__main__":
    from frame_selection import select_frame_between_i_frames#, score_nums_gftt_halfres, filter_frames_quality
    # from upscaling.diffusion import DiffusionUpscaler as Upscaler
    # from upscaling.onnx_model import OnnxModelUpscaler as Upscaler
    from upscaling.onnx_model import OnnxModelYuvUpscaler as Upscaler
    # from upscaling.super_image_model import SuperImageModelUpscaler as Upscaler
    # from upscaling.super_image_model import SuperImageModelYuvUpscaler as Upscaler

    upscaler = Upscaler(4)

    for video in db.get_videos():
        indices = select_frame_between_i_frames(video.filename)
        extract_images(video.filename, "export/images", upscaler, indices)
