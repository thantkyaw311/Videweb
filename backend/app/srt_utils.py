from pathlib import Path

def srt_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def write_srt_from_segments(segments, out_path: Path):
    with out_path.open("w", encoding="utf-8") as fh:
        for idx, seg in enumerate(segments, start=1):
            start = srt_timestamp(seg["start"])
            end = srt_timestamp(seg["end"])
            text = seg["text"].strip()
            fh.write(f"{idx}\n{start} --> {end}\n{text}\n\n")
