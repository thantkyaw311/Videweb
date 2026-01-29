# backend/app/transcribe.py
# Local Whisper transcription (GPU-enabled) + SRT generation + local translation support
import os
import subprocess
from pathlib import Path
from typing import Optional, List, Dict

import torch
import whisper
from moviepy.editor import VideoFileClip

from .srt_utils import write_srt_from_segments
from .translation_local import translate_text_local
from .status import update_status

# Extract audio using moviepy for reliability across containers
def extract_audio_to_wav(video_path: Path, out_wav: Path):
    try:
        with VideoFileClip(str(video_path)) as clip:
            if clip.audio is None:
                # No audio stream
                raise ValueError("no-audio")
            # write_audiofile will choose ffmpeg under the hood; set parameters for whisper
            clip.audio.write_audiofile(str(out_wav), fps=16000, nbytes=2, codec='pcm_s16le')
    except Exception as e:
        # re-raise for caller to handle, but preserve type
        raise

def transcribe_audio_generate_srts(video_path: Path, out_dir: Path, target_languages: Optional[List[str]] = None, whisper_model_size: str = "small"):
    """
    - Extracts audio using moviepy, runs local Whisper (on GPU if available) to produce segments.
    - Writes:
        - transcript_original.srt
        - transcript_translated_<lang>.srt for each target language (en, my etc.)
    - Handles missing audio gracefully by writing a small SRT with a notice and updating status.
    """
    if target_languages is None:
        target_languages = ["en", "my"]

    audio_path = out_dir / "audio.wav"

    # Try to extract audio
    try:
        update_status(out_dir, step="extracting_audio", percent=5.0, detail="Extracting audio with moviepy")
        extract_audio_to_wav(video_path, audio_path)
    except Exception as e:
        # Handle no audio gracefully: create a small SRT explaining the issue and skip transcription
        update_status(out_dir, step="no_audio", percent=100.0, detail="No audio track found")
        notice_segments = [{"start": 0.0, "end": 2.0, "text": "[No audio track found in this video]"}]
        write_srt_from_segments(notice_segments, out_dir / "transcript_original.srt")
        for lang in target_languages:
            write_srt_from_segments(notice_segments, out_dir / f"transcript_translated_{lang}.srt")
        return True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    update_status(out_dir, step="loading_model", percent=10.0, detail=f"Loading Whisper model ({whisper_model_size}) on {device}")
    model = whisper.load_model(whisper_model_size, device=device)

    update_status(out_dir, step="transcribing", percent=20.0, detail="Running Whisper transcription")
    try:
        result = model.transcribe(str(audio_path), task="transcribe", verbose=False)
    except Exception as e:
        update_status(out_dir, step="error", percent=0.0, detail=f"Transcription error: {e}")
        raise

    segments = []
    for s in result.get("segments", []):
        segments.append({
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": s["text"].strip()
        })

    # If no segments detected, write a placeholder and return
    if not segments:
        update_status(out_dir, step="no_speech_detected", percent=100.0, detail="No speech detected in audio")
        notice_segments = [{"start": 0.0, "end": 2.0, "text": "[No speech detected in audio]"}]
        write_srt_from_segments(notice_segments, out_dir / "transcript_original.srt")
        for lang in target_languages:
            write_srt_from_segments(notice_segments, out_dir / f"transcript_translated_{lang}.srt")
        return True

    update_status(out_dir, step="writing_original_srt", percent=60.0, detail="Writing original SRT")
    original_srt = out_dir / "transcript_original.srt"
    write_srt_from_segments(segments, original_srt)

    # 2) English SRT: try Whisper translation task if requested
    if "en" in target_languages:
        try:
            update_status(out_dir, step="translating_en", percent=70.0, detail="Generating English translation via Whisper")
            trans_res_en = model.transcribe(str(audio_path), task="translate", verbose=False)
            en_segments = []
            for s in trans_res_en.get("segments", []):
                en_segments.append({"start": float(s["start"], "end": float(s["end"]), "text": s["text"].strip()})
            en_srt = out_dir / "transcript_translated_en.srt"
            write_srt_from_segments(en_segments, en_srt)
        except Exception:
            # fallback: translate each segment via local translation model
            update_status(out_dir, step="translating_en_fallback", percent=75.0, detail="Fallback English translation via local model")
            translated_segments = []
            for seg in segments:
                try:
                    ttext = translate_text_local(seg["text"], tgt_lang="en", src_lang="auto")
                except Exception:
                    ttext = seg["text"]
                translated_segments.append({"start": seg["start"], "end": seg["end"], "text": ttext})
            outp = out_dir / "transcript_translated_en.srt"
            write_srt_from_segments(translated_segments, outp)

    # 3) Other languages via local translation model (e.g., Burmese 'my')
    for lang in target_languages:
        if lang == "en":
            continue
        update_status(out_dir, step=f"translating_{lang}", percent=80.0, detail=f"Translating to {lang}")
        translated_segments = []
        for seg in segments:
            try:
                ttext = translate_text_local(seg["text"], tgt_lang=lang, src_lang="auto")
            except Exception:
                ttext = seg["text"]
            translated_segments.append({"start": seg["start"], "end": seg["end"], "text": ttext})
        outp = out_dir / f"transcript_translated_{lang}.srt"
        write_srt_from_segments(translated_segments, outp)

    update_status(out_dir, step="transcription_done", percent=95.0, detail="Transcription tasks complete")
    return True
