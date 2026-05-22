import os
import tempfile
from collections.abc import Generator
from typing import Any

import cv2
from core.config import FRAMES_DIR, VIDEO_SAMPLE_FPS


def _open_video(video_bytes: bytes) -> tuple[cv2.VideoCapture, str]:
    """Write video_bytes to a temp file and return (cap, tmp_path).

    Caller is responsible for cap.release() and os.unlink(tmp_path).
    Raises ValueError if the file cannot be opened by OpenCV.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        os.unlink(tmp_path)
        raise ValueError("Could not open video file")
    return cap, tmp_path


def _iter_sampled_frames(
    cap: cv2.VideoCapture,
    sample_fps: float,
) -> Generator[tuple[int, Any], None, None]:
    """Yield (saved_idx, frame) for frames sampled at sample_fps."""
    native_fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, round(native_fps / sample_fps))
    frame_idx      = 0
    saved_idx      = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            yield saved_idx, frame
            saved_idx += 1
        frame_idx += 1


def extract_frames_memory(
    video_bytes: bytes,
    sample_fps: float = VIDEO_SAMPLE_FPS,
) -> list[dict[str, bytes | str]]:
    """Extract frames from video bytes entirely in memory (no disk I/O).

    Returns a list of {"filename": str, "content": bytes} dicts where
    content is each frame PNG-encoded.  Used by combined pipelines so
    intermediate frames are never written to disk.
    """
    cap, tmp_path = _open_video(video_bytes)
    try:
        frames: list[dict[str, bytes | str]] = []
        for saved_idx, frame in _iter_sampled_frames(cap, sample_fps):
            success, buf = cv2.imencode(".png", frame)
            if success:
                frames.append({
                    "filename": f"frame_{saved_idx:06d}.png",
                    "content":  bytes(buf),
                })
        cap.release()
    finally:
        os.unlink(tmp_path)
    return frames


def extract_frames(
    video_bytes: bytes,
    job_id: str,
    sample_fps: float = VIDEO_SAMPLE_FPS,
) -> list[dict[str, str]]:
    """Write video bytes to a temp file, extract frames at sample_fps,
    save each frame as a PNG to FRAMES_DIR/{job_id}/, and return a
    list of {"filename": str, "path": str} dicts.
    """
    out_dir = os.path.join(FRAMES_DIR, job_id)
    os.makedirs(out_dir, exist_ok=True)

    cap, tmp_path = _open_video(video_bytes)
    try:
        frames: list[dict[str, str]] = []
        for saved_idx, frame in _iter_sampled_frames(cap, sample_fps):
            filename = f"frame_{saved_idx:06d}.png"
            path     = os.path.join(out_dir, filename)
            cv2.imwrite(path, frame)
            frames.append({"filename": filename, "path": path})
        cap.release()
    finally:
        os.unlink(tmp_path)
    return frames
