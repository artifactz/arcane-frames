from typing import Optional
import multiprocessing
import numpy as np
import cv2
from datatypes import FrameFeatures
import ffmpeg


def get_video_features(filename: str, crop: Optional[str] = None) -> list[FrameFeatures]:
    with ffmpeg.FfmpegVideoReader(filename, crop=crop) as reader:
        gray_iter = (cv2.cvtColor(frame.rgb, cv2.COLOR_BGR2GRAY) for frame in reader)

        pool = multiprocessing.Pool()
        features_iter = pool.imap(
            calculate_frame_features,
            enumerate(gray_iter)
        )

        return list(features_iter)

def calculate_frame_features(args) -> FrameFeatures:
    frame_index, gray = args
    num_gftt = get_num_gftt(gray)
    num_gftt_halfres = get_num_gftt_halfres(gray)
    laplace_mean = get_laplace_mean(gray)
    return FrameFeatures(frame_index=frame_index, num_gftt=num_gftt, num_gftt_halfres=num_gftt_halfres, laplace_mean=laplace_mean)

def get_num_gftt(gray: np.ndarray) -> int:
    """
    Returns the number of Shi-Tomasi corners detected in the given image array.
    """
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=0, qualityLevel=0.01, minDistance=10)
    num_features = 0 if corners is None else len(corners)
    return num_features

def get_num_gftt_halfres(gray: np.ndarray) -> int:
    """
    Returns the number of Shi-Tomasi corners detected in the given image array, after downscaling it to half
    resolution.
    """
    halfres = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2))
    return get_num_gftt(halfres)

def get_laplace_mean(gray: np.ndarray) -> float:
    """
    Returns the mean of the absolute values of the Laplacian of the given image array, which is a measure of its
    sharpness.
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    laplace = cv2.Laplacian(blurred, cv2.CV_64F)
    laplace_mean = float(np.mean(np.abs(laplace)))
    return laplace_mean
