"""Local web server: pick a piece from the curated library, upload a
recording (audio or video), get back a closeness percentage and
measure-level feedback, computed via chroma+DTW audio-to-score alignment.
"""
import shutil
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from .library import LIBRARY, LIBRARY_BY_ID
from ..audio_perf_score import score_audio_performance, load_model

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_PATH = REPO_ROOT / "pretrained_weights/han_measnote_gru/checkpoint_best.pt"
STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "virtuosonet_webapp_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}

app = FastAPI(title="VirtuosoNet Performance Scorer")

_model = None
_model_lock = threading.Lock()
_jobs = {}  # job_id -> {"status": ..., "result": ..., "error": ...}


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = load_model(str(CHECKPOINT_PATH), device="cpu")
    return _model


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/pieces")
def list_pieces():
    return [{"id": p.id, "title": p.title, "composer": p.composer} for p in LIBRARY]


def _extract_audio_if_video(path: Path) -> Path:
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        return path
    audio_path = path.with_suffix(".extracted.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "22050", str(audio_path)],
        check=True, capture_output=True,
    )
    return audio_path


def _run_job(job_id: str, piece_id: str, upload_path: Path):
    try:
        piece = LIBRARY_BY_ID[piece_id]
        audio_path = _extract_audio_if_video(upload_path)
        result = score_audio_performance(
            xml_path=str(piece.xml_path),
            score_midi_path=str(piece.score_midi_path),
            composer=piece.composer,
            performance_audio_path=str(audio_path),
            model=get_model(),
        )
        _jobs[job_id] = {
            "status": "done",
            "result": {
                "piece": piece.title,
                "composer": piece.composer,
                "overall_score": result["overall_score"],
                "tempo_score": result["tempo_score"],
                "dynamics_score": result["dynamics_score"],
                "articulation_score": result["articulation_score"],
                "feedback": result["feedback"],
            },
        }
    except Exception as e:
        _jobs[job_id] = {"status": "error", "error": str(e)}
    finally:
        upload_path.unlink(missing_ok=True)
        extracted = upload_path.with_suffix(".extracted.wav")
        extracted.unlink(missing_ok=True)


@app.post("/api/analyze")
async def analyze(piece_id: str = Form(...), file: UploadFile = File(...)):
    if piece_id not in LIBRARY_BY_ID:
        raise HTTPException(400, f"Unknown piece_id: {piece_id}")

    job_id = uuid.uuid4().hex
    suffix = Path(file.filename).suffix or ".bin"
    upload_path = UPLOAD_DIR / f"{job_id}{suffix}"
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    _jobs[job_id] = {"status": "processing"}
    thread = threading.Thread(target=_run_job, args=(job_id, piece_id, upload_path), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job_id")
    return JSONResponse(job)
