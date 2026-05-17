"""
Module for labeling video frames with quality ratings for quality estimation model training.

See:
label_quality.py
"""

from typing import Iterator
import multiprocessing, threading, os, time
from pathlib import Path
import numpy as np
import cv2
from datatypes import Video
import db, ffmpeg
from .labels import save, load


class Labeler:
    def __init__(self):
        self.labels = load()
        self.label_list = [(filename, int(frame_index)) for filename, frame_labels in self.labels.items() for frame_index in frame_labels]
        self.label_index = 0
        self.max_label_index = 0
        self.label_images = {}

        self.frame_generator_worker_running = True
        self.frame_generator_thread = threading.Thread(target=self._frame_generator_worker)
        self.frame_generator_thread.start()

        self.frame_loader_worker_running = True
        self.frame_loader_thread = threading.Thread(target=self._frame_loader_task)
        self.frame_loader_thread.start()

    def prev(self):
        if self.label_index > 0:
            self.label_index -= 1

    def next(self):
        if self.label_index < len(self.label_list) - 1:
            self.label_index += 1
            if self.label_index > self.max_label_index:
                self.max_label_index = self.label_index

    def draw_image_ui(self, w: int, h: int) -> np.ndarray:
        img = self.get_image()

        # Crop  TODO extend
        assert img.shape[0] == h
        dx = (img.shape[1] - w) // 2
        img = np.array(img[:, dx : w + dx, :], copy=True)

        # Draw rating
        filename, frame_index = self.label_list[self.label_index]
        rating = self.labels[filename][frame_index]
        if rating is not None:
            rating_str = ('O' * rating) + ('-' * (5 - rating))
            cv2.putText(img, rating_str, (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2.0, (255, 100, 0), 3)  # color format is BGR
        return img

    def get_image(self) -> np.ndarray:
        while ((img := self.label_images.get(self.label_list[self.label_index]) if self.label_list else None) is None):
            time.sleep(0.1)

        return np.array(img, copy=True)

    def set_label(self, value: int):
        filename, frame_index = self.label_list[self.label_index]
        self.labels[filename][frame_index] = value
        # print(f"Set label for frame {frame_index} of video {filename} to {value}.")
        save(self.labels)

    def _frame_generator_worker(self, lookahead=32):
        """Retrieves extracted frames from the queue and adds them to the labeling list."""
        p, q = create_frame_generator_process(self.labels)
        while self.frame_generator_worker_running:
            if self.max_label_index < len(self.label_list) - lookahead:
                time.sleep(0.2)
                continue
            filename, frame_index, _ = q.get()
            self.labels.setdefault(filename, {})[frame_index] = None
            self.label_list.append((filename, frame_index))
            # print(f"Added frame {frame_index} of video {filename} to labeling queue.")
        p.terminate()
        p.kill()
        p.join()
        p.close()

    def _frame_loader_task(self, lookbehind=5, lookahead=10):
        """Keeps frames around the current label index loaded in memory."""
        while not self.label_list:
            time.sleep(0.1)

        label_key = self.label_list[self.label_index]
        self.label_images[label_key] = cv2.imread(get_frame_image_path(*label_key))
        min_loaded_index = max_loaded_index = self.label_index
        # print(f"Loaded initial frame {label_key[1]} of video {label_key[0]} into memory.")

        while self.frame_loader_worker_running:
            if min_loaded_index < self.label_index - lookbehind:
                label_key = self.label_list[min_loaded_index]
                self.label_images.pop(label_key, None)
                min_loaded_index += 1
                # print(f"Unloaded frame {label_key[1]} of video {label_key[0]} from memory (head).", min_loaded_index, max_loaded_index)
            elif max_loaded_index > self.label_index + lookahead:
                label_key = self.label_list[max_loaded_index]
                self.label_images.pop(label_key, None)
                max_loaded_index -= 1
                # print(f"Unloaded frame {label_key[1]} of video {label_key[0]} from memory (tail).", min_loaded_index, max_loaded_index)
            elif max_loaded_index < self.label_index + lookahead and max_loaded_index < len(self.label_list) - 1:
                label_key = self.label_list[max_loaded_index + 1]
                self.label_images[label_key] = cv2.imread(get_frame_image_path(*label_key))
                max_loaded_index += 1
                # print(f"Loaded frame {label_key[1]} of video {label_key[0]} into memory.", min_loaded_index, max_loaded_index)
            elif min_loaded_index > self.label_index - lookbehind and min_loaded_index > 0:
                label_key = self.label_list[min_loaded_index - 1]
                self.label_images[label_key] = cv2.imread(get_frame_image_path(*label_key))
                min_loaded_index -= 1
                # print(f"Loaded frame {label_key[1]} of video {label_key[0]} into memory.", min_loaded_index, max_loaded_index)
            else:
                time.sleep(0.1)

    def close(self):
        self.frame_generator_worker_running = False
        self.frame_loader_worker_running = False
        save(self.labels)
        self.frame_generator_thread.join()
        self.frame_loader_thread.join()


def create_frame_generator_process(labels: dict[str, dict[str, int | None]]) -> tuple[multiprocessing.Process, multiprocessing.Queue]:
    num_labels = {k: len(v) for k, v in labels.items()}
    q = multiprocessing.Queue(1)
    p = multiprocessing.Process(target=_frame_generator_task, args=(num_labels, q))
    p.start()
    return p, q


def _frame_generator_task(num_labels: dict[str, int], results_queue: multiprocessing.Queue):
    """
    Chooses and extracts frames for labeling and saves them to disk.

    Args:
        num_labels: A dictionary mapping video filenames to the number of frames already labeled for that video.
        results_queue: A multiprocessing queue used to communicate generated frame information back to the main process.
            Each item is a tuple of (video_filename, frame_index, frame_image_path).
    """
    from upscaling.onnx_model import OnnxModelYuvUpscaler
    from extract_images import _resample_1080p
    upscaler = OnnxModelYuvUpscaler(4)

    def iter_frames(frames_per_video=32) -> Iterator[tuple[Video, list]]:
        videos = list(db.get_videos())
        for video in videos:
            num_labels.setdefault(video.filename, 0)
            if num_labels[video.filename] < frames_per_video:
                # Choose equally distributed frames initially
                step = video.num_frames // frames_per_video
                indices = list(range(step // 2, video.num_frames, step))[num_labels[video.filename]:]
                yield (video, indices)
                num_labels[video.filename] += len(indices)
        while True:
            # Choose video with lowest label density
            video = min(videos, key=lambda v: num_labels[v.filename] / v.num_frames)
            # Choose random frames
            yield (video, sorted(np.random.random_integers(0, video.num_frames, frames_per_video)))
            num_labels[video.filename] += frames_per_video

    os.makedirs("frames", exist_ok=True)

    for video, indices in iter_frames():
        for i, frame in ffmpeg.iter_frames(indices, video.filename, pix_fmt=upscaler.pix_fmt, crop=video.crop):
            upscaled = upscaler.upscale(frame)
            upscaled = _resample_1080p(upscaled)
            filename = get_frame_image_path(video.filename, i)
            upscaled.save(filename)
            results_queue.put((video.filename, i, filename))


def get_frame_image_path(video_filename: str, frame_index: int) -> str:
    return f"frames/{Path(video_filename).stem}_{frame_index:06d}.png"
