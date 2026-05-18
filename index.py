import os, functools, contextlib
import db, ffmpeg
from feature_extraction import cv_features


def scan(folder: str, overwrite=False, verbose=True):
    """
    Scans a folder for video files, extracts their frame types using ffmpeg, and stores the information in the database.
    """
    _scan_with_args = lambda: _scan(folder, overwrite)
    if verbose:
        _scan_with_args()
    else:
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            _scan_with_args()

def _scan(folder: str, overwrite=False):
    print(f"# Scanning {folder}\n")
    filenames = get_video_paths(folder)
    if not filenames:
        print("No video files found.")
        return

    for filename in filenames:
        print(f"## Processing {filename}\n")
        v = db.get_video(filename)

        if not v or overwrite:
            db.ensure_filename(filename)
            print("* Created DB entry.")
        else:
            print("* DB entry exists.")

        if not v or not v.crop or overwrite:
            print("* Detecting crop", end="", flush=True)
            crop = ffmpeg.ffmpeg_crop_detect(filename)
            db.set_crop(filename, crop)
            print(f"={crop}")
        else:
            crop = v.crop
            print(f"* Crop in DB: {v.crop}")

        @functools.cache
        def _frame_types():
            return list(ffmpeg.ffprobe_frame_types(filename))

        frame_features = db.get_frame_features(filename)
        if not frame_features or frame_features[0].frame_type is None or overwrite:
            print("* Detecting frame types", end="", flush=True)
            db.set_video_features(filename, frame_types=_frame_types())
            print(f" ({len(_frame_types())})")
        else:
            print(f"* Frame types in DB ({len(frame_features)}).")

        if not v or not v.num_frames or overwrite:
            print("* Counting num_frames", end="", flush=True)
            n = len(_frame_types())
            db.set_num_frames(filename, n)
            print(f"={n}")
        else:
            print(f"* num_frames in DB: {v.num_frames}")

        if (
            not frame_features or
            frame_features[0].num_gftt is None or
            frame_features[0].num_gftt_halfres is None or
            frame_features[0].laplace_mean is None or
            overwrite
        ):
            print(f"* Extracting frame features")
            video_features = cv_features.get_video_features(filename, crop)
            nums_gftt = [f.num_gftt for f in video_features]
            nums_gftt_halfres = [f.num_gftt_halfres for f in video_features]
            laplace_means = [f.laplace_mean for f in video_features]
            db.set_video_features(filename, nums_gftt=nums_gftt, nums_gftt_halfres=nums_gftt_halfres, laplace_means=laplace_means)
        else:
            print("* Frame features in DB.")

        print()

def get_video_paths(folder: str):
    """
    Scans a folder for video files and returns a list of their paths.
    """
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.flv']
    return [f"{folder}/{f}" for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in video_extensions]


if __name__ == "__main__":
    db.ensure_tables()
    scan("episodes")
    db.dump_videos()
