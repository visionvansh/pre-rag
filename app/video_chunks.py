from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from PIL import Image


@dataclass
class VideoChunk:
    chunk_index: int
    start_sec: float
    end_sec: float
    frames: list[Image.Image]


def _duration_seconds(container: av.container.InputContainer, stream) -> float:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return float(container.duration / av.time_base)
    raise ValueError("Could not determine video duration")


def probe_video(path: str | Path) -> dict:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return {
            "duration_sec": _duration_seconds(container, stream),
            "width": int(stream.codec_context.width or 0),
            "height": int(stream.codec_context.height or 0),
            "average_rate": float(stream.average_rate) if stream.average_rate else None,
        }


def _frame_time(frame, stream) -> float | None:
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is not None and stream.time_base is not None:
        return float(frame.pts * stream.time_base)
    return None


def sample_interval(container, stream, start_sec: float, end_sec: float, max_frames: int) -> list[Image.Image]:
    seek_pts = int(start_sec / float(stream.time_base)) if stream.time_base else 0
    container.seek(max(0, seek_pts), stream=stream, backward=True, any_frame=False)

    target_count = max(2, max_frames)
    targets = np.linspace(start_sec, end_sec, num=target_count, endpoint=False)
    target_i = 0
    sampled: list[Image.Image] = []
    last_image = None

    for frame in container.decode(stream):
        t = _frame_time(frame, stream)
        if t is None or t < start_sec:
            continue
        if t >= end_sec:
            break
        image = frame.to_image().convert("RGB")
        last_image = image
        while target_i < len(targets) and t >= targets[target_i]:
            sampled.append(image.copy())
            target_i += 1

    if not sampled and last_image is not None:
        sampled = [last_image]
    if len(sampled) == 1:
        sampled.append(sampled[0].copy())
    if len(sampled) % 2:
        sampled = sampled[:-1]
    return sampled[:max_frames]


def iter_video_chunks(path: str | Path, chunk_seconds: float = 10.0, max_frames: int = 32):
    path = Path(path)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration = _duration_seconds(container, stream)
        count = int(math.ceil(duration / chunk_seconds))
        for idx in range(count):
            start = idx * chunk_seconds
            end = min(duration, start + chunk_seconds)
            frames = sample_interval(container, stream, start, end, max_frames)
            if not frames:
                continue
            yield VideoChunk(idx, start, end, frames)
