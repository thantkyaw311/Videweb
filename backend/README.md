# Videweb backend

This backend provides:
- Local Whisper transcription (runs on GPU if available)
- PaddleOCR-based on-screen caption detection (GPU-enabled)
- SRT generation for original audio transcript, translated transcripts (English + Burmese by default), and OCR-detected captions

Important setup notes:

1. GPU drivers and CUDA must be installed on the host. Install the matching `paddlepaddle-gpu` package per your CUDA version. See https://www.paddlepaddle.org.cn/install/quick for instructions.

2. Install Python dependencies. It's recommended to use a virtual environment. Example:

```bash
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

Note: `paddlepaddle-gpu` installation may require a specific wheel for your CUDA version. Adjust accordingly.

3. Run the app:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4. Put your video processing load onto a machine with GPU. The Whisper model size determines GPU memory usage (`small` recommended for most GPUs with >=6GB VRAM).

5. For Burmese translation: the code currently uses the OpenAI Chat API if `OPENAI_API_KEY` is set. You can swap this with a local translation model (e.g., Hugging Face transformers) if you prefer a fully local pipeline.


---

End of commit content.
