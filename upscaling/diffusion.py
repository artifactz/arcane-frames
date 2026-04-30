from PIL import Image

# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
import torch

from diffusers import StableDiffusionUpscalePipeline
from datatypes import Frame
from .upscaler import RgbUpscaler


class DiffusionUpscaler(RgbUpscaler):
    def __init__(
            self,
            model_id="stabilityai/stable-diffusion-x4-upscaler",
            prompt="",#"UHD, 4k, extremely detailed, professional, vibrant, not grainy, smooth",
            negative_prompt=None,
            num_inference_steps=25,
            noise_level=5,
        ):
        pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipeline = pipeline.to("cuda")
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.num_inference_steps = num_inference_steps
        self.noise_level = noise_level

    def upscale(
            self,
            frame: Frame,
        ):
        image = Image.fromarray(frame.rgb)

        # Downsampling using BOX filter leads to much sharper and more detailed diffusion results than LANCZOS
        # Not downsampling at all makes diffusion take unfeasibly long
        image = image.resize((image.width // 2, image.height // 2), resample=Image.BOX)

        upscaled_image = self.pipeline(
            image=image,
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            num_inference_steps=self.num_inference_steps,
            noise_level=self.noise_level,
        ).images[0]
        return upscaled_image
