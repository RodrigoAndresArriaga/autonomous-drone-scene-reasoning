# Video clipping utility: extract time-range segments via ffmpeg.
# Fast path: stream copy (no re-encode). Fallback: re-encode if copy fails.

import shutil
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path


# Verify ffmpeg and ffprobe are available before attempting video operations
def _check_ffmpeg() -> None:
    """Raise clear error if ffmpeg or ffprobe not on PATH."""
    for cmd in ("ffmpeg", "ffprobe"):
        if not shutil.which(cmd):
            raise RuntimeError(
                f"{cmd} not found on PATH. Install ffmpeg (e.g. apt install ffmpeg) to use video clipping."
            )


def get_video_duration(path: str) -> float:
    """Get video duration in seconds via ffprobe."""
    _check_ffmpeg()
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(Path(path).resolve()),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


# Fast path: stream copy without re-encoding (best for keyframe-aligned clips)
def _run_ffmpeg_copy(source: str, start: float, duration: float, out_path: Path) -> bool:
    """Run ffmpeg stream copy. Returns True if success."""
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-t",
            str(duration),
            "-i",
            str(Path(source).resolve()),
            "-c",
            "copy",
            "-an",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and out_path.exists()


# Fallback: re-encode when stream copy fails (non-keyframe starts, format issues)
def _run_ffmpeg_reencode(
    source: str, start: float, duration: float, out_path: Path, fps: int = 4
) -> bool:
    """Run ffmpeg re-encode (fallback when copy fails)."""
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-t",
            str(duration),
            "-i",
            str(Path(source).resolve()),
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-an",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and out_path.exists()


@contextmanager
def extract_clip_ctx(
    source_path: str,
    start_sec: float,
    duration_sec: float,
    *,
    reencode_fps: int | None = None,
):
    """
    Context manager: extract video segment, yield path, cleanup on exit.
    Fast path: stream copy. Fallback: re-encode if copy fails.
    """
    _check_ffmpeg()
    source = Path(source_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source video not found: {source}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="cosmos_clip_"))
    out_path = tmp_dir / f"clip_{uuid.uuid4().hex[:8]}.mp4"

    try:
        # Try fast stream copy first
        if _run_ffmpeg_copy(str(source), start_sec, duration_sec, out_path):
            yield out_path
            return
        # Fall back to re-encode if copy fails (handles non-keyframe starts)
        fps = reencode_fps if reencode_fps is not None else 4
        if _run_ffmpeg_reencode(str(source), start_sec, duration_sec, out_path, fps):
            yield out_path
        else:
            raise RuntimeError(
                f"ffmpeg failed to extract clip from {source} "
                f"[{start_sec:.1f}s, +{duration_sec:.1f}s]. "
                "Check ffmpeg logs or source file."
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
