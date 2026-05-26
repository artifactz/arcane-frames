import os, contextlib
import db, ffmpeg
from resnet_util import ensure_frame_quality


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
        video = db.get_video(filename)

        if not video or overwrite:
            video = db.ensure_filename(filename)
            print("* Created DB entry.")
        else:
            print("* DB entry exists.")

        if not video or not video.crop or overwrite:
            print("* Detecting crop", end="", flush=True)
            crop = ffmpeg.ffmpeg_crop_detect(filename)
            db.set_crop(filename, crop)
        else:
            print("* From DB crop", end="", flush=True)
            crop = video.crop
        print(f"={crop}")

        frame_types = db.get_frame_types(video._id)
        if not frame_types or overwrite:
            print("* Detecting frame types", end="", flush=True)
            frame_types = list(ffmpeg.ffprobe_frame_types(filename))
            db.set_frame_types(video._id, frame_types)
            db.set_num_frames(filename, len(frame_types))
        else:
            print(f"* Frame types from DB", end="", flush=True)
        print(f" ({len(frame_types)})")

        frame_qualities = db.get_frame_qualities(video._id)
        if not frame_qualities or len(frame_qualities) < len(frame_types) or overwrite:
            print("* Estimating frame qualities")
            ensure_frame_quality(filename, batch_size=64)
            print("* Done estimating frame qualities")
        else:
            print(f"* Frame qualities from DB ({len(frame_qualities)}).")

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
