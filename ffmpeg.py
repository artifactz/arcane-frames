from typing import Iterator, Optional
from dataclasses import dataclass
import subprocess, re, threading
from collections import Counter
import numpy as np
from tqdm import tqdm
from datatypes import Frame


@dataclass
class StreamSpecs:
    resolution: Optional[tuple[int]] = None
    duration_seconds: Optional[float] = None
    fps: Optional[float] = None
    pix_fmt: Optional[str] = None


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
        )

        if self.output_specs.pix_fmt == "yuv420p" and self.input_specs.pix_fmt != "yuv420p":
            print(
                f"Warning: Output pixel format 'yuv420p' only makes sense with matching input pixel format. "
                f"Detected input pixel format is '{self.input_specs.pix_fmt}'."
            )
        
        self.stderr_thread = None
        self.progressbar = None

    @property
    def num_frame_bytes(self):
        w, h = self.output_specs.resolution
        if self.output_specs.pix_fmt == "yuv420p":
            return w * h + 2 * (w // 2) * (h // 2)
        if self.output_specs.pix_fmt == "rgb24":
            return w * h * 3
        raise ValueError(f"Unknown pixel format: {self.output_specs.pix_fmt}")

    def __iter__(self) -> Iterator[Frame]:
        num_frame_bytes = self.num_frame_bytes

        self.progressbar = tqdm(
            total=int(self.output_specs.duration_seconds * self.output_specs.fps),
            desc="Reading video",
            unit=" frames",
        )

        self.stderr_thread = self.start_stderr_consumer_thread()

        while True:
            raw_frame = self.process.stdout.read(num_frame_bytes)

            if not raw_frame:
                break  # No more frames to read

            if self.output_specs.pix_fmt == "yuv420p":
                w, h = self.output_specs.resolution
                y_size = w * h
                u_size = (w // 2) * (h // 2)

                y = np.frombuffer(raw_frame[:y_size], dtype=np.uint8).reshape((h, w))
                u = np.frombuffer(raw_frame[y_size : y_size + u_size], dtype=np.uint8).reshape((h // 2, w // 2))
                v = np.frombuffer(raw_frame[y_size + u_size :], dtype=np.uint8).reshape((h // 2, w // 2))

                frame = Frame(y=y, u=u, v=v)

            elif self.output_specs.pix_fmt == "rgb24":
                w, h = self.output_specs.resolution
                frame = Frame(rgb=np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3)))

            else:
                raise ValueError(f"Unknown pixel format: {self.output_specs.pix_fmt}")

            self.progressbar.update(1)
            yield frame

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.progressbar.close()
        self.process.stdout.close()
        self.process.stderr.close()
        self.stderr_thread.join()
        self.process.wait()

    def detect_input(self) -> StreamSpecs:
        """Read resolution, duration, fps, and pixel format from stderr."""
        resolution = None
        fps = None
        duration_seconds = None
        pix_fmt = "unknown"

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
                print(f"Detected pix_fmt: {fps}")

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
        )

    def start_stderr_consumer_thread(self):
        def consume_stderr():
            """Reads and discards stderr to prevent block due to filled-up buffer."""
            while self.process.stderr.read(1024):
                pass

        t = threading.Thread(target=consume_stderr)
        t.start()
        return t


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
            "-select_streams",
            "v",
            "-show_entries",
            "frame=pict_type",
            filename,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    while (line := process.stdout.readline().decode("utf-8")):
        if line.startswith("pict_type="):
            frame_type = line.strip().split("=")[1]
            yield frame_type

    process.stdout.close()
    process.stderr.close()
    process.wait()
