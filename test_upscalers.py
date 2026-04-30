from upscaling.upscaler import Upscaler
from upscaling.diffusion import DiffusionUpscaler
from upscaling.onnx_model import OnnxModelUpscaler, OnnxModelYuvUpscaler
from upscaling.super_image_model import SuperImageModelUpscaler, SuperImageModelYuvUpscaler
from extract_images import extract_images


def test_upscaler(upscaler: Upscaler):
    suffix = upscaler.__class__.__name__
    test_frames = {2178, 3654, 4439, 6594, 7432, 12181}
    extract_images("episodes/Arcane S01E01.mp4", f"export/images_{suffix}", upscaler, test_frames)


if __name__ == "__main__":
    test_upscaler(OnnxModelUpscaler(4))
    test_upscaler(OnnxModelYuvUpscaler(4))
    test_upscaler(SuperImageModelUpscaler(4))
    test_upscaler(SuperImageModelYuvUpscaler(4))
    # test_upscaler(DiffusionUpscaler())
