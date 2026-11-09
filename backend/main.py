from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
import json
import uuid

app = FastAPI(
    title="Danger Sound AI API",
    version="1.0.0"
)

DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "detections.jsonl"
FEEDBACK_FILE = DATA_DIR / "feedback.jsonl"

DATA_DIR.mkdir(exist_ok=True)


class Detection(BaseModel):
    sound_class: str
    confidence: float
    latency_ms: float


class Feedback(BaseModel):
    detection_id: str
    is_correct: bool


@app.get("/")
def root():
    return {
        "message": "Danger Sound AI API"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "danger-sound-api"
    }


@app.post("/detections")
def create_detection(detection: Detection):

    data = {
        "id": str(uuid.uuid4()),
        "sound_class": detection.sound_class,
        "confidence": detection.confidence,
        "latency_ms": detection.latency_ms,
        "timestamp": datetime.now().isoformat()
    }

    with open(DATA_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(data) + "\n")

    return {
        "status": "success",
        "message": "Detection saved",
        "data": data
    }


@app.get("/detections")
def get_detections():

    detections = []

    if not DATA_FILE.exists():
        return {
            "status": "success",
            "count": 0,
            "data": []
        }

    with open(DATA_FILE, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:
                detection = json.loads(line)
                detections.append(detection)

            except json.JSONDecodeError:
                print(
                    "Baris JSON rusak dilewati:",
                    line
                )

    return {
        "status": "success",
        "count": len(detections),
        "data": detections
    }


@app.post("/feedback")
def create_feedback(feedback: Feedback):

    detection_found = False

    if DATA_FILE.exists():
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if line:
                    detection = json.loads(line)

                    if detection.get("id") == feedback.detection_id:
                        detection_found = True
                        break

    if not detection_found:
        raise HTTPException(
            status_code=404,
            detail="Detection ID not found"
        )

    data = {
        "detection_id": feedback.detection_id,
        "is_correct": feedback.is_correct,
        "timestamp": datetime.now().isoformat()
    }

    with open(FEEDBACK_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(data) + "\n")

    return {
        "status": "success",
        "message": "Feedback saved",
        "data": data
    }