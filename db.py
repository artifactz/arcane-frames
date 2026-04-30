from typing import Optional, Iterable
from itertools import repeat, batched, islice
import sqlite3
from datatypes import Video, FrameFeatures

conn = sqlite3.connect('database.db')

def ensure_videos_table():
    conn.execute("""CREATE TABLE IF NOT EXISTS videos (
                        id INTEGER PRIMARY KEY,
                        filename TEXT NOT NULL,
                        num_frames INTEGER,
                        crop TEXT
                    )""")
    conn.commit()

def get_videos() -> Iterable[Video]:
    return (
        Video(_id=row[0], filename=row[1], num_frames=row[2], crop=row[3])
        for row in conn.execute("SELECT id, filename, num_frames, crop FROM videos")
    )

def get_video(filename: str, raise_not_found=False) -> Video:
    cursor = conn.execute("SELECT id, filename, num_frames, crop FROM videos WHERE filename = ?", (filename,))
    row = cursor.fetchone()
    if row:
        return Video(_id=row[0], filename=row[1], num_frames=row[2], crop=row[3])
    elif raise_not_found:
        raise ValueError(f"Video '{filename}' not found in database.")
    else:
        return None

def ensure_filename(filename: str):
    cursor = conn.execute("SELECT filename FROM videos WHERE filename = ?", (filename,))
    if cursor.fetchone() is None:
        conn.execute("INSERT INTO videos (filename) VALUES (?)", (filename,))
        conn.commit()

def set_num_frames(filename: str, num_frames: int):
    conn.execute("UPDATE videos SET num_frames = ? WHERE filename = ?", (num_frames, filename))
    conn.commit()

def set_crop(filename: str, crop: str):
    conn.execute("UPDATE videos SET crop = ? WHERE filename = ?", (crop, filename))
    conn.commit()

def dump_videos():
    cursor = conn.execute("SELECT * FROM videos")
    _dump_rows(cursor)

def ensure_frame_features_table():
    conn.execute("""CREATE TABLE IF NOT EXISTS frame_features (
                        id INTEGER PRIMARY KEY,
                        video_id INTEGER NOT NULL,
                        frame_index INTEGER NOT NULL,
                        frame_type TEXT CHECK(frame_type IN ('I','P','B') OR frame_type IS NULL),
                        num_gftt INTEGER,
                        num_gftt_halfres INTEGER,
                        laplace_mean REAL,

                        FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE,
                        UNIQUE(video_id, frame_index)
                    )""")
    conn.commit()

def get_frame_features(filename: str) -> list[FrameFeatures]:
    video_id = conn.execute('SELECT id FROM videos WHERE filename = ?', (filename,)).fetchone()[0]

    cursor = conn.execute("""SELECT frame_index, frame_type, num_gftt, num_gftt_halfres, laplace_mean
                             FROM frame_features WHERE video_id = ? ORDER BY frame_index""", (video_id,))
    features = [
        FrameFeatures(
            frame_index=row[0],
            frame_type=row[1],
            num_gftt=row[2],
            num_gftt_halfres=row[3],
            laplace_mean=row[4]
        )
        for row in cursor
    ]
    return features

def set_video_features(
    filename: str,
    frame_types: Optional[Iterable[str]] = None,
    nums_gftt: Optional[Iterable[int]] = None,
    nums_gftt_halfres: Optional[Iterable[int]] = None,
    laplace_means: Optional[Iterable[float]] = None,
    _chunk_size=1000
):
    video = filename if isinstance(filename, Video) else get_video(filename, raise_not_found=True)
    with conn:
        for chunk in batched(_iter_rows(video._id, frame_types, nums_gftt, nums_gftt_halfres, laplace_means), _chunk_size):
            conn.executemany("""INSERT INTO frame_features (video_id, frame_index, frame_type, num_gftt, num_gftt_halfres, laplace_mean)
                                VALUES (?, ?, ?, ?, ?, ?)
                                ON CONFLICT(video_id, frame_index) DO UPDATE SET
                                    frame_type        = COALESCE(excluded.frame_type,       frame_features.frame_type),
                                    num_gftt          = COALESCE(excluded.num_gftt,         frame_features.num_gftt),
                                    num_gftt_halfres  = COALESCE(excluded.num_gftt_halfres, frame_features.num_gftt_halfres),
                                    laplace_mean      = COALESCE(excluded.laplace_mean,     frame_features.laplace_mean);
                                """, chunk)

def _iter_rows(
    video_id: int,
    frame_types: Optional[Iterable[str]],
    nums_gftt: Optional[Iterable[int]],
    nums_gftt_halfres: Optional[Iterable[int]],
    laplace_means: Optional[Iterable[float]]
) -> Iterable[tuple]:
    assert frame_types or nums_gftt or laplace_means

    return (
        (
            video_id, i, frame_type, num_gftt, num_gftt_halfres, laplace_mean
        )
        for i, (frame_type, num_gftt, num_gftt_halfres, laplace_mean) in enumerate(zip(
            frame_types or repeat(None),
            nums_gftt or repeat(None),
            nums_gftt_halfres or repeat(None),
            laplace_means or repeat(None)
        ))
    )

def dump_frame_features():
    cursor = conn.execute("""SELECT videos.filename, frame_index, frame_type, num_gftt, laplace_mean
                             FROM frame_features
                             JOIN videos ON frame_features.video_id = videos.id
                             ORDER BY videos.filename, frame_index""")
    _dump_rows(cursor)

def _dump_rows(rows, _max_item_len=64, _max_rows=16):
    for row in islice(rows, _max_rows):
        row = list(row)
        for i in range(len(row)):
            if row[i] and isinstance(row[i], str) and len(row[i]) > _max_item_len:
                row[i] = row[i][:_max_item_len] + "..."
        print(row)
