from typing import Iterable, Iterator
from itertools import batched, islice
import sqlite3
import numpy as np
from datatypes import Video


conn = sqlite3.connect('database.db')


def ensure_tables():
    ensure_videos_table()
    ensure_frame_type_table()
    ensure_frame_resnet_table()
    ensure_frame_quality_table()

def ensure_videos_table():
    conn.execute("""CREATE TABLE IF NOT EXISTS videos (
                        id INTEGER PRIMARY KEY,
                        filename TEXT NOT NULL,
                        num_frames INTEGER,
                        crop TEXT
                    )""")
    conn.commit()

def get_videos() -> Iterator[Video]:
    return (
        Video(_id=row[0], filename=row[1], num_frames=row[2], crop=row[3])
        for row in conn.execute("SELECT id, filename, num_frames, crop FROM videos")
    )

def get_video(filename: str, raise_not_found=False) -> Video | None:
    cursor = conn.execute("SELECT id, filename, num_frames, crop FROM videos WHERE filename = ?", (filename,))
    row = cursor.fetchone()
    if row:
        return Video(_id=row[0], filename=row[1], num_frames=row[2], crop=row[3])
    elif raise_not_found:
        raise ValueError(f"Video '{filename}' not found in database.")

def ensure_filename(filename: str):
    cursor = conn.execute("SELECT filename FROM videos WHERE filename = ?", (filename,))
    if video := cursor.fetchone() is None:
        conn.execute("INSERT INTO videos (filename) VALUES (?)", (filename,))
        conn.commit()
        video = get_video(filename)
    return video

def set_num_frames(filename: str, num_frames: int):
    conn.execute("UPDATE videos SET num_frames = ? WHERE filename = ?", (num_frames, filename))
    conn.commit()

def set_crop(filename: str, crop: str):
    conn.execute("UPDATE videos SET crop = ? WHERE filename = ?", (crop, filename))
    conn.commit()

def dump_videos():
    cursor = conn.execute("SELECT * FROM videos")
    _dump_rows(cursor)


def ensure_frame_type_table():
    conn.execute("""CREATE TABLE IF NOT EXISTS frame_type (
                        id INTEGER PRIMARY KEY,
                        video_id INTEGER NOT NULL,
                        frame_index INTEGER NOT NULL,
                        frame_type TEXT CHECK(frame_type IN ('I','P','B') OR frame_type IS NULL),

                        FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE,
                        UNIQUE(video_id, frame_index)
                    )""")
    conn.commit()

def get_frame_types(video_id: int) -> list[tuple[int, str]]:
    cursor = conn.execute("""SELECT frame_index, frame_type
                             FROM frame_type WHERE video_id = ? ORDER BY frame_index""", (video_id,))
    return [(frame_index, frame_type) for frame_index, frame_type in cursor]

def set_frame_types(video_id: int, frame_types: Iterable[str], _chunk_size=1000):
    with conn:
        for chunk in batched(enumerate(frame_types), _chunk_size):
            conn.executemany(f"""INSERT INTO frame_type (video_id, frame_index, frame_type)
                                 VALUES ({video_id}, ?, ?)
                                 ON CONFLICT(video_id, frame_index) DO UPDATE SET
                                    frame_type = excluded.frame_type
                                 """, chunk)


def ensure_frame_resnet_table():
    conn.execute("""CREATE TABLE IF NOT EXISTS frame_resnet (
                        id INTEGER PRIMARY KEY,
                        video_id INTEGER NOT NULL,
                        frame_index INTEGER NOT NULL,
                        embed BLOB NOT NULL,
                        prob BLOB NOT NULL,

                        FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE,
                        UNIQUE(video_id, frame_index)
                    )""")
    conn.commit()

def get_frames_resnet(video_id: int, frame_indices: Iterable[int] | None = None) -> Iterator[tuple[int, tuple[np.ndarray | None, np.ndarray | None]]]:
    if frame_indices:
        placeholders = ','.join('?' for _ in frame_indices)
        query = f"""SELECT frame_index, embed, prob FROM frame_resnet
                    WHERE video_id = ? AND frame_index IN ({placeholders})"""
        cursor = conn.execute(query, (video_id, *frame_indices))
    else:
        query = "SELECT frame_index, embed, prob FROM frame_resnet WHERE video_id = ?"
        cursor = conn.execute(query, (video_id,))
    for row in cursor:
        frame_index = row[0]
        embed = np.frombuffer(row[1], dtype=np.float32) if row[1] else None
        prob = np.frombuffer(row[2], dtype=np.float32) if row[2] else None
        yield frame_index, (embed, prob)

def set_frame_resnet(video_id: int, frame_index: int, embed: np.ndarray | None = None, prob: np.ndarray | None = None):
    embed_ = embed.astype(np.float32).tobytes() if embed else None
    prob_ = prob.astype(np.float32).tobytes() if prob else None
    conn.execute("""INSERT INTO frame_resnet (video_id, frame_index, embed, prob)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(video_id, frame_index) DO UPDATE SET
                        embed = excluded.embed,
                        prob  = excluded.prob;
                """, (video_id, frame_index, embed_, prob_))
    conn.commit()


def ensure_frame_quality_table():
    conn.execute("""CREATE TABLE IF NOT EXISTS frame_quality (
                        id INTEGER PRIMARY KEY,
                        video_id INTEGER NOT NULL,
                        frame_index INTEGER NOT NULL,
                        quality REAL NOT NULL,

                        FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE,
                        UNIQUE(video_id, frame_index)
                    )""")
    conn.commit()

def get_frame_qualities(video_id: int) -> list[tuple[int, float]]:
    cursor = conn.execute("""SELECT frame_index, quality
                             FROM frame_quality WHERE video_id = ? ORDER BY frame_index""", (video_id,))
    return [(frame_index, frame_quality) for frame_index, frame_quality in cursor]

def set_frame_quality(video_id: int, frame_index: int, quality: float):
    conn.execute("""INSERT INTO frame_quality (video_id, frame_index, quality)
                    VALUES (?, ?, ?)
                    ON CONFLICT(video_id, frame_index) DO UPDATE SET
                        quality = excluded.quality;
                """, (int(video_id), int(frame_index), float(quality)))
    conn.commit()


def _dump_rows(rows, _max_item_len=64, _max_rows=16):
    for row in islice(rows, _max_rows):
        row = list(row)
        for i in range(len(row)):
            if row[i] and isinstance(row[i], str) and len(row[i]) > _max_item_len:
                row[i] = row[i][:_max_item_len] + "..."
        print(row)
