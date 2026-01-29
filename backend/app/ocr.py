import os
from pathlib import Path
import cv2
from paddleocr import PaddleOCR
from typing import List, Dict
from .srt_utils import write_srt_from_segments

# Initialize PaddleOCR with GPU if available
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)

def extract_frames(video_path: Path, interval_sec: float):
    vidcap = cv2.VideoCapture(str(video_path))
    fps = vidcap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps else 0
    frames = []
    t = 0.0
    while t < duration:
        frame_no = int(round(t * fps))
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        success, image = vidcap.read()
        if not success:
            break
        frames.append((t, image.copy()))
        t += interval_sec
    vidcap.release()
    return frames

def detect_text_in_frame(image):
    try:
        results = ocr_engine.ocr(image, cls=True)
    except Exception:
        results = []
    lines = []
    for line in results:
        if len(line) >= 2 and isinstance(line[1], tuple):
            txt = line[1][0]
        elif len(line) >= 2 and isinstance(line[1], list):
            txt = line[1][0]
        else:
            txt = str(line)
        lines.append(txt)
    return lines

def group_texts_to_segments(detected_items: List[Dict], window=2.0):
    if not detected_items:
        return []
    detected_items = sorted(detected_items, key=lambda x: x['time'])
    segments = []
    cur = {"start": detected_items[0]["time"], "end": detected_items[0]["time"] + window, "text": detected_items[0]["text"]}
    for it in detected_items[1:]:
        if it["time"] <= cur["end"] + 0.5:
            cur["end"] = max(cur["end"], it["time"] + window)
            cur["text"] = cur["text"].strip() + " " + it["text"].strip()
        else:
            segments.append(cur)
            cur = {"start": it["time"], "end": it["time"] + window, "text": it["text"]}
    segments.append(cur)
    return segments

def translate_text_via_openai(text: str, target_language: str = "en"):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        prompt = f"Translate the following text to {target_language} preserving meaning and punctuation. Output only the translation.\n\nText:\n{text}"
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            max_tokens=800,
        )
        return resp.choices[0].message["content"].strip()
    except Exception:
        return text

def ocr_from_video_and_generate_srt(video_path: Path, out_dir: Path, frame_interval: float = 1.0, target_languages: List[str] = None):
    if target_languages is None:
        target_languages = ["en"]

    frames = extract_frames(video_path, interval_sec=frame_interval)
    detected = []
    for (t, image) in frames:
        lines = detect_text_in_frame(image)
        if lines:
            full = " ".join([l for l in lines if l.strip()])
            detected.append({"time": t, "text": full})

    segments = group_texts_to_segments(detected, window=max(1.0, frame_interval))

    ocr_original_srt = out_dir / "ocr_detected_original.srt"
    write_srt_from_segments(segments, ocr_original_srt)

    for lang in target_languages:
        if lang == 'original':
            continue
        translated_segments = []
        for seg in segments:
            ttxt = translate_text_via_openai(seg['text'], target_language=lang)
            translated_segments.append({"start": seg["start"], "end": seg["end"], "text": ttxt})
        outp = out_dir / f"ocr_translated_{lang}.srt"
        write_srt_from_segments(translated_segments, outp)

    return True
