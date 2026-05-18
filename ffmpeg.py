from typing import Iterable, Iterator, Optional
from dataclasses import dataclass
import subprocess, re, threading, os
from collections import Counter
import numpy as np
from tqdm import tqdm
from datatypes import Frame


def iter_frames(frame_indices: Iterable[int], *args, **kwargs) -> Iterator[tuple[int, Frame]]:
    frame_indices = set(frame_indices)
    last_frame_index = max(frame_indices)
    with FfmpegVideoReader(*args, **kwargs) as reader:
        for i, frame in enumerate(reader):
            if i in frame_indices:
                yield i, frame
                if i == last_frame_index:
                    break


@dataclass
class StreamSpecs:
    resolution: Optional[tuple[int]] = None
    duration_seconds: Optional[float] = None
    fps: Optional[float] = None
    pix_fmt: Optional[str] = None
    color_space: Optional[str] = None


class FfmpegVideoReader:
    def __init__(self, filename: str, pix_fmt: str = "rgb24", crop: Optional[str] = None):
        args = get_ffmpeg_reader_args(filename, pix_fmt=pix_fmt, crop=crop)
        self.process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        self.input_specs = self.detect_input()
        crop_size = get_crop_size(crop) if crop else None
        self.output_specs = StreamSpecs(
            resolution=crop_size or self.input_specs.resolution,
            duration_seconds=self.input_specs.duration_seconds,
            fps=self.input_specs.fps,
            pix_fmt=pix_fmt,
            color_space=self.input_specs.color_space,
        )

        if self.output_specs.pix_fmt in ["yuv420p", "yuvj420p"] and self.input_specs.pix_fmt not in ["yuv420p", "yuvj420p"]:
            print(
                f"Warning: Output pixel format '{self.output_specs.pix_fmt}' only makes sense with matching input "
                f"pixel format. Detected input pixel format is '{self.input_specs.pix_fmt}'."
            )
        
        self.stderr_bytes = bytes()
        self.stderr_consumer = None
        self.progressbar = None
        self.basename = os.path.basename(filename)

    @property
    def num_frame_bytes(self):
        w, h = self.output_specs.resolution
        if self.output_specs.pix_fmt in ["yuv420p", "yuvj420p"]:
            return w * h + 2 * (w // 2) * (h // 2)
        if self.output_specs.pix_fmt == "rgb24":
            return w * h * 3
        raise ValueError(f"Unknown pixel format: {self.output_specs.pix_fmt}")

    def __iter__(self) -> Iterator[Frame]:
        num_frame_bytes = self.num_frame_bytes

        self.progressbar = tqdm(
            total=int(self.output_specs.duration_seconds * self.output_specs.fps),
            desc=f"Reading {self.basename}",
            unit=" frames",
        )

        self.stderr_consumer = StreamConsumer(self.process.stderr)

        while True:
            raw_frame = self.process.stdout.read(num_frame_bytes)

            if not raw_frame:
                break  # No more frames to read

            if self.output_specs.pix_fmt in ["yuv420p", "yuvj420p"]:
                w, h = self.output_specs.resolution
                y_size = w * h
                u_size = (w // 2) * (h // 2)

                y = np.frombuffer(raw_frame[:y_size], dtype=np.uint8).reshape((h, w))
                u = np.frombuffer(raw_frame[y_size : y_size + u_size], dtype=np.uint8).reshape((h // 2, w // 2))
                v = np.frombuffer(raw_frame[y_size + u_size :], dtype=np.uint8).reshape((h // 2, w // 2))

                frame = Frame(y=y, u=u, v=v, color_space=self.output_specs.color_space)

            elif self.output_specs.pix_fmt == "rgb24":
                w, h = self.output_specs.resolution
                rgb = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3))
                frame = Frame(rgb=rgb, color_space=self.output_specs.color_space)

            else:
                raise ValueError(f"Unknown pixel format: {self.output_specs.pix_fmt}")

            self.progressbar.update(1)
            yield frame

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        if self.progressbar:
            self.progressbar.close()
        self.process.stdout.close()
        self.process.stderr.close()
        if self.stderr_consumer:
            self.stderr_consumer.join()
        self.process.wait()

    def detect_input(self) -> StreamSpecs:
        """Read resolution, duration, fps, and pixel format from stderr."""
        resolution = None
        fps = None
        duration_seconds = None
        pix_fmt = "unknown"
        color_space = None

        while not (line := self.process.stderr.readline().decode("utf-8")).startswith("Output #0"):
            # print(line)

            if (m := re.match(r"\s*Stream\s+#.+\s+(\d\d+x\d\d+)\s+.+\s+(\d+)\s+fps.+", line)):
                resolution = [int(x) for x in m[1].split("x")]
                w, h = resolution
                print(f"Detected resolution: {w}x{h}")

                fps = float(m[2])
                print(f"Detected fps: {fps}")

                if "yuv420p" in line:
                    pix_fmt = "yuv420p"
                if "yuvj420p" in line:
                    pix_fmt = "yuvj420p"
                print(f"Detected pix_fmt: {pix_fmt}")

                if "bt709" in line:
                    color_space = "bt709"
                if "bt601" in line:
                    color_space = "bt601"
                print(f"Detected color space: {color_space}")

            if (m := re.match(r"\s*Duration:\s+(\d\d:\d\d:\d\d\.\d+),.+", line)):
                duration = m[1]
                duration_seconds = sum(
                    float(x) * 60**i for i, x in enumerate(reversed(duration.split(":")))
                )
                print(f"Detected duration: {duration}, which is {duration_seconds} seconds")

        return StreamSpecs(
            resolution=resolution,
            duration_seconds=duration_seconds,
            fps=fps,
            pix_fmt=pix_fmt,
            color_space=color_space,
        )


def get_ffmpeg_reader_args(filename: str, pix_fmt: str = "rgb24", crop: Optional[str] = None):
    args = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        filename,
        "-fps_mode",
        "passthrough",
    ]
    if crop:
        args += ["-vf", f"crop={crop}"]
    args += [
        "-f",
        "image2pipe",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "-",
    ]
    return args

def get_crop_size(crop: str) -> tuple[int]:
    w, h, _, _ = [int(x) for x in crop.split(":")]
    return w, h

def ffmpeg_crop_detect(filename) -> Optional[str]:
    """
    Uses ffmpeg's cropdetect filter to analyze a video and suggest a crop filter string (e.g., "640:480:0:0").
    """
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            filename,
            "-vf",
            "cropdetect",
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _, stderr = process.communicate()

    crop_counts = Counter()
    for line in stderr.splitlines():
        if (m := re.match(r".*crop=(\d+:\d+:\d+:\d+).*", line)):
            crop_counts[m[1]] += 1
    
    if crop_counts:
        most_common_crop, _count = crop_counts.most_common(1)[0]
        return most_common_crop


def ffprobe_frame_types(filename) -> Iterator[str]:
    """
    Reads video frame types from a file using ffprobe and yields them as strings (e.g., "I", "P", "B").
    """
    process = subprocess.Popen(
        [
            "ffprobe",
            "-hide_banner",
            "-threads",
            "auto",
            "-select_streams",
            "v",
            "-show_entries",
            "frame=pict_type",
            filename,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stderr_consumer = StreamConsumer(process.stderr)

    while (line := process.stdout.readline().decode("utf-8")):
        if line.startswith("pict_type="):
            frame_type = line.strip().split("=")[1]
            yield frame_type

    process.stdout.close()
    process.stderr.close()
    stderr_consumer.join()
    process.wait()


class StreamConsumer:
    def __init__(self, stream):
        self.stream = stream
        self.content_bytes = bytes()
        self.thread = threading.Thread(target=self._consume_stream_task)
        self.thread.start()

    def _consume_stream_task(self):
        try:
            while b := self.stream.read(1024):
                self.content_bytes += b
        except Exception:
            pass

    def join(self):
        self.thread.join()
        return self.content_bytes
