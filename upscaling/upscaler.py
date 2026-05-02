from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
from datatypes import Frame


class Upscaler(ABC):
    pix_fmt: str

    @abstractmethod
    def upscale(self, frame: Frame) -> np.ndarray | Image.Image:
        raise NotImplementedError

class RgbUpscaler(Upscaler):
    pix_fmt = "rgb24"

class YuvUpscaler(Upscaler):
    pix_fmt = "yuvj420p"
