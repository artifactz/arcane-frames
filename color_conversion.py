from typing import Optional
import numpy as np


def yuv_to_rgb(color_space: Optional[str], y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    if color_space == "bt709":
        return yuv420_full_bt709_to_rgb(y, u, v)
    elif color_space == "bt601" or color_space is None:
        return yuv420_full_bt601_to_rgb(y, u, v)
    else:
        raise ValueError(f"Unsupported color space: {color_space}")

def yuv420_full_bt709_to_rgb(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Convert full-range BT.709 YUV (Y in [0,255], U/V centered at 128) to RGB uint8.
    Args:
        Y: H x W uint8
        U: H x W uint8 (U upscaled to full resolution)
        V: H x W uint8 (V upscaled to full resolution)
    Returns:
        rgb: H x W x 3 uint8 (RGB)
    """
    Yf = Y.astype(np.float32)
    Uf = U.astype(np.float32) - 128.0
    Vf = V.astype(np.float32) - 128.0

    R = Yf + 1.5748 * Vf
    G = Yf - 0.187324 * Uf - 0.468124 * Vf
    B = Yf + 1.8556 * Uf

    rgb = np.stack((R, G, B), axis=-1)
    np.clip(rgb, 0, 255, out=rgb)
    return rgb.astype(np.uint8)

def yuv420_full_bt601_to_rgb(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Convert full-range BT.601 YUV (Y in [0,255], U/V centered at 128) to RGB uint8.
    Args:
        Y: H x W uint8
        U: H x W uint8 (U upscaled to full resolution)
        V: H x W uint8 (V upscaled to full resolution)
    Returns:
        rgb: H x W x 3 uint8 (RGB)
    """
    Yf = Y.astype(np.float32)
    Uf = U.astype(np.float32) - 128.0
    Vf = V.astype(np.float32) - 128.0

    R = Yf + 1.402 * Vf
    G = Yf - 0.344136 * Uf - 0.714136 * Vf
    B = Yf + 1.772 * Uf

    rgb = np.stack((R, G, B), axis=-1)
    np.clip(rgb, 0, 255, out=rgb)
    return rgb.astype(np.uint8)
