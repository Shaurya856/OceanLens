import os
import tempfile
import cv2
from core.config import FRAMES_DIR, VIDEO_SAMPLE_FPS


def extract_frames_memory(
    video_bytes: bytes,
    sample_fps: float = VIDEO_SAMPLE_FPS,
) -> list[dict[str, bytes | str]]:
    """Extract frames from video bytes entirely in memory (no disk I/O).

    Returns a list of {"filename": str, "content": bytes} dicts where
    content is each frame PNG-encoded.  Used by combined pipelines so
    intermediate frames are never written to disk.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        native_fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, round(native_fps / sample_fps))

        frames:    list[dict[str, bytes | str]] = []
        frame_idx  = 0
        saved_idx  = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                success, buf = cv2.imencode(".png", frame)
                if success:
                    frames.append({
                        "filename": f"frame_{saved_idx:06d}.png",
                        "content":  bytes(buf),
                    })
                    saved_idx += 1
            frame_idx += 1

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

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        native_fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, round(native_fps / sample_fps))

        frames:    list[dict[str, str]] = []
        frame_idx  = 0
        saved_idx  = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                filename = f"frame_{saved_idx:06d}.png"
                path     = os.path.join(out_dir, filename)
                cv2.imwrite(path, frame)
                frames.append({"filename": filename, "path": path})
                saved_idx += 1
            frame_idx += 1

        cap.release()
    finally:
        os.unlink(tmp_path)

    return frames
