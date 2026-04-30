from typing import Optional
import numpy as np
import cv2
from super_image import MsrnModel, EdsrModel, DrlnModel, ImageLoader, PreTrainedModel
from PIL import Image
from .upscaler import RgbUpscaler, YuvUpscaler
from datatypes import Frame


class SuperImageModelUpscaler(RgbUpscaler):
    def __init__(self, scale: int):
        assert(scale == 2 or scale == 4)
        self.model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=scale)

        # Much slower than EDSR and no notable quality improvement
        # model_x2 = MsrnModel.from_pretrained('eugenesiow/msrn-bam', scale=2)
        # model_x4 = MsrnModel.from_pretrained('eugenesiow/msrn-bam', scale=4)

        # Runs on CPU for whatever reason, slow
        # model_x2 = DrlnModel.from_pretrained('eugenesiow/drln', scale=2)
        # model_x4 = DrlnModel.from_pretrained('eugenesiow/drln', scale=4)

    def upscale(self, frame: Frame):
        upscaled = _upscale_array(self.model, frame.rgb)
        return upscaled


class SuperImageModelYuvUpscaler(YuvUpscaler):
    def __init__(self, scale: int, resample_before_2nd_upscale: Optional[float] = None):
        """
        Args:
            scale: 2 or 4
            resample_before_2nd_upscale: If set, resample the RGB image by this factor before upscaling it.
        """
        assert(scale == 2 or scale == 4)
        self.resample_before_2nd_upscale = resample_before_2nd_upscale
        self.model_x2 = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
        self.model = (
            self.model_x2
            if scale == 2 else
            EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4)
        )

    def upscale(self, frame: Frame):
        # Upscale U and V, which are at half resolution
        upscaled_u = _upscale_array(self.model_x2, frame.u, channels=1)
        upscaled_v = _upscale_array(self.model_x2, frame.v, channels=1)

        # Combine to RGB
        yuv_array = cv2.merge((frame.y, upscaled_u, upscaled_v))
        rgb_array = cv2.cvtColor(yuv_array, cv2.COLOR_YUV2RGB)

        # Model works better at a smaller scale
        if self.resample_before_2nd_upscale:
            rgb_image = _resample(rgb_array, self.resample_before_2nd_upscale)
        else:
            rgb_image = Image.fromarray(rgb_array)

        # Upscale RGB
        upscaled_rgb = _upscale_array(self.model, rgb_image)
        return upscaled_rgb


def _upscale_array(model: PreTrainedModel, array_or_image: np.ndarray | Image.Image, channels: int = 3) -> np.ndarray:
    image = array_or_image if isinstance(array_or_image, Image.Image) else Image.fromarray(array_or_image)
    inputs = ImageLoader.load_image(image)
    preds = model(inputs)
    upscaled = preds.data.cpu().numpy()
    if channels == 1:
        upscaled = upscaled[0, 0]
    elif channels == 3:
        upscaled = upscaled[0].transpose((1, 2, 0))
    upscaled = np.clip(upscaled * 255.0, 0, 255).astype(np.uint8)
    return upscaled

def _resample(array: np.ndarray, factor=0.5) -> Image.Image:
    img = Image.fromarray(array)
    return img.resize((int(img.width * factor), int(img.height * factor)), resample=Image.LANCZOS)
