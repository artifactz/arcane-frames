from typing import Optional
import os, requests
from PIL import Image
import numpy as np

# pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-13/pypi/simple/
import onnxruntime

from datatypes import Frame
from .upscaler import RgbUpscaler, YuvUpscaler
from color_conversion import yuv_to_rgb


class OnnxModelUpscaler(RgbUpscaler):
    def __init__(self, scale: int):
        assert(scale == 2 or scale == 4)
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        model_filename = ensure_model(scale)
        self.ort_session = onnxruntime.InferenceSession(model_filename, options)

    def upscale(self, frame: Frame) -> np.ndarray:
        return _upscale_array(self.ort_session, frame.rgb)


class OnnxModelYuvUpscaler(YuvUpscaler):
    def __init__(self, scale: int, resample_before_2nd_upscale: Optional[float] = None):
        """
        Args:
            scale: 2 or 4
            resample_before_2nd_upscale: If set, resample the RGB image by this factor before upscaling it.
        """
        assert(scale == 2 or scale == 4)
        self.resample_before_2nd_upscale = resample_before_2nd_upscale

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1

        model_filename = ensure_model(2)
        self.ort_session_x2 = onnxruntime.InferenceSession(model_filename, options)
        self.ort_session = (
            self.ort_session_x2
            if scale == 2 else
            onnxruntime.InferenceSession(ensure_model(4), options)
        )

    def upscale(self, frame: Frame) -> np.ndarray:
        # Upscale U and V, which are at half resolution
        upscaled_u = _upscale_array_1channel(self.ort_session_x2, frame.u)
        upscaled_v = _upscale_array_1channel(self.ort_session_x2, frame.v)
        rgb_array = yuv_to_rgb(frame.color_space, frame.y, upscaled_u, upscaled_v)

        if self.resample_before_2nd_upscale:
            rgb_array = _resample(rgb_array, self.resample_before_2nd_upscale)

        return _upscale_array(self.ort_session, rgb_array)


def _resample(array: np.ndarray, factor=0.5) -> np.ndarray:
    img = Image.fromarray(array)
    img = img.resize((int(img.width * factor), int(img.height * factor)), resample=Image.LANCZOS)
    return np.array(img)

def _upscale_array_1channel(session: onnxruntime.InferenceSession, array: np.ndarray) -> np.ndarray:
    array = np.concatenate([array[:, :, np.newaxis]] * 3, axis=2)
    array = _upscale_array(session, array)
    return array[:, :, 0]

def _upscale_array(session: onnxruntime.InferenceSession, array: np.ndarray) -> np.ndarray:
    x = pre_process(array)
    ort_inputs = {session.get_inputs()[0].name: x}
    ort_outs = session.run(None, ort_inputs)
    return post_process(ort_outs[0])

def pre_process(array: np.ndarray) -> np.ndarray:
    assert(array.shape[2] == 3)
    # H, W, C -> C, H, W
    array = np.transpose(array[:, :, ::-1], (2, 0, 1))
    # C, H, W -> 1, C, H, W
    return np.expand_dims(array, axis=0).astype(np.float32)

def post_process(array: np.ndarray) -> np.ndarray:
    # 1, C, H, W -> C, H, W
    array = np.squeeze(array)
    # C, H, W -> H, W, C
    return np.transpose(array, (1, 2, 0))[:, :, ::-1].astype(np.uint8)

def ensure_model(scale: int, model_folder="models") -> str:
    """
    Ensure the ONNX model for the given scale exists locally, downloading it if necessary.
    """
    basename = f"modelx{scale}.ort"
    os.makedirs(model_folder, exist_ok=True)
    model_path = f"{model_folder}/{basename}"

    if not os.path.isfile(model_path):
        # print(f"Downloading model for x{scale} upscale...")
        url = f"https://huggingface.co/spaces/bookbot/Image-Upscaling-Playground/resolve/9eab909/models/{basename}"
        response = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(response.content)

    return model_path


# # This is worse
# def upscale(img_array: np.ndarray) -> np.ndarray:
#     a = downscale_nearest(img_array, 2)
#     b = post_process(inference(pre_process(a), 4))
#     c = upscale_nearest(img_array, 2)
#     laplace = b.astype(int)
#     laplace -= c
#     # d = upscale_bilinear(img_array, 2)
#     d = post_process(inference(pre_process(img_array), 2))
#     laplace += d
#     result = np.clip(laplace, 0, 255).astype(np.uint8)
#     return result
