from typing import Iterable, Iterator
import multiprocessing
from PIL import Image
import numpy as np


def iter_inference(indexed_images: Iterable[tuple[int, Image.Image]]) -> Iterator[tuple[int, tuple[np.ndarray, np.ndarray]]]:
    pool = multiprocessing.Pool(1)
    features_iter = pool.imap(work, indexed_images)

    return features_iter

def work(indexed_image: tuple[int, Image.Image]) -> tuple[int, tuple[np.ndarray, np.ndarray]]:
    from . import resnet50
    index, image = indexed_image
    return index, resnet50.from_image(image)
