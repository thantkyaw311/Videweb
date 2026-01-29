from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
import shutil
import asyncio
from pathlib import Path
from .transcribe import transcribe_audio_generate_srts
from .ocr import ocr_from_video_and_generate_srt
from typing import Optional
import zipfile

app = FastAPI(title="Video -> Transcripts + OCR SRTs")
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    target_languages: Optional[str] = Form("en,my"),
    frame_interval: Optional[float] = Form(1.0),  # seconds between OCR frames
    whisper_model_size: Optional[str] = Form("small"),
):
    # Accept common video content types
    if file.content_type not in ("video/mp4", "video/avi", "video/quicktime", "video/mov", "video/x-msvideo"):
        return JSONResponse(status_code=400, content={"error": "Unsupported file type"})

    file_id = uuid.uuid4().hex
    saved_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    with saved_path.open("wb") as out_f:
        shutil.copyfileobj(file.file, out_f)

    job_out_dir = OUTPUT_DIR / file_id
    job_out_dir.mkdir(parents=True, exist_ok=True)

    # Normalize target languages list
    langs = [l.strip() for l in target_languages.split(",") if l.strip()]

    # Run transcription + OCR concurrently
    transcription_task = asyncio.create_task(
        transcribe_audio_generate_srts(saved_path, job_out_dir, target_languages=langs, whisper_model_size=whisper_model_size)
    )
    ocr_task = asyncio.create_task(
        ocr_from_video_and_generate_srt(saved_path, job_out_dir, frame_interval=float(frame_interval), target_languages=langs)
    )

    results = await asyncio.gather(transcription_task, ocr_task, return_exceptions=True)

    # Check for errors
    errors = [str(r) for r in results if isinstance(r, Exception)]
    if errors:
        return JSONResponse(status_code=500, content={"error": errors})

    # zip results
    zip_path = OUTPUT_DIR / f"{file_id}_results.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in job_out_dir.iterdir():
            zf.write(f, arcname=f.name)

    return {
        "status": "done",
        "files": [str(p.name) for p in job_out_dir.iterdir()],
        "download": f"/download/{zip_path.name}"
    }

@app.get("/download/{zipname}")
async def download_zip(zipname: str):
    path = OUTPUT_DIR / zipname
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "not found"})
    return FileResponse(path, filename=zipname)