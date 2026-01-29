import os
import subprocess
from pathlib import Path
from typing import Optional, List, Dict
import whisper
import torch
from .srt_utils import write_srt_from_segments

# Extract audio using ffmpeg to wav (16k/16-bit) which Whisper works well with
def extract_audio_to_wav(video_path: Path, out_wav: Path):
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ac", "1", "-ar", "16000", "-vn",
        str(out_wav)
    ]
    subprocess.check_call(cmd)


def transcribe_audio_generate_srts(video_path: Path, out_dir: Path, target_languages: Optional[List[str]] = None, whisper_model_size: str = "small"):
    """
    Uses local Whisper (OpenAI's whisper) running on GPU if available.
    Produces:
      - transcript_original.srt
      - transcript_translated_<lang>.srt for each requested language (if translation available)

    target_languages: list like ['en', 'my']
    """
    if target_languages is None:
        target_languages = ["en"]

    audio_path = out_dir / "audio.wav"
    extract_audio_to_wav(video_path, audio_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(whisper_model_size, device=device)

    # Original-language transcription (no translation)
    result = model.transcribe(str(audio_path), verbose=False)
    segments = []
    for s in result.get("segments", []):
        segments.append({"start": float(s["start"]), "end": float(s["end"], "text": s["text"].strip()})

    original_srt = out_dir / "transcript_original.srt"
    write_srt_from_segments(segments, original_srt)

    # For English: Whisper supports task='translate' which translates to English
    # We'll attempt to produce an English SRT by running the model with task='translate'
    if "en" in target_languages:
        try:
            trans_res_en = model.transcribe(str(audio_path), task="translate", verbose=False)
            en_segments = []
            for s in trans_res_en.get("segments", []):
                en_segments.append({"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()})
            en_srt = out_dir / "transcript_translated_en.srt"
            write_srt_from_segments(en_segments, en_srt)
        except Exception as e:
            # if translation via whisper fails, fall back to copying original
            en_srt = out_dir / "transcript_translated_en.srt"
            write_srt_from_segments(segments, en_srt)

    # For other languages (e.g., Burmese 'my'), use OpenAI API if available as a fallback
    if any(l for l in target_languages if l != "en"):
        try:
            from openai import OpenAI
            openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            for lang in target_languages:
                if lang == "en":
                    continue
                translated_segments = []
                for seg in segments:
                    prompt = f"Translate the following text to {lang}. Keep it concise. Text:\n\n{seg['text']}"
                    chat_resp = openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1000,
                    )
                    try:
                        ttext = chat_resp.choices[0].message["content"].strip()
                    except Exception:
                        ttext = seg["text"]
                    translated_segments.append({"start": seg["start"], "end": seg["end"], "text": ttext})
                outp = out_dir / f"transcript_translated_{lang}.srt"
                write_srt_from_segments(translated_segments, outp)
        except Exception:
            # No OpenAI key or error -> write placeholder files with original text
            for lang in target_languages:
                if lang == "en":
                    continue
                outp = out_dir / f"transcript_translated_{lang}.srt"
                write_srt_from_segments(segments, outp)

    return True
