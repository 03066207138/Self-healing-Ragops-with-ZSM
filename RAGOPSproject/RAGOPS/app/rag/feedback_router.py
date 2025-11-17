# app/rag/feedback_router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import csv
import os

from app.self_healing_cp.learner import LearningMemory

router = APIRouter(prefix="/feedback", tags=["feedback"])

# Global instance
learner = LearningMemory()


# -----------------------------
# Feedback Models
# -----------------------------
class FeedbackIn(BaseModel):
    request_id: str
    is_correct: bool
    correct_answer: str = ""


class CorrectOnly(BaseModel):
    request_id: str
    correct_answer: str


class WrongOnly(BaseModel):
    request_id: str


# -----------------------------
# Helper: get request row from log
# -----------------------------
def load_log_row(request_id: str):
    log_path = "metrics_log.csv"
    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="metrics_log.csv not found.")

    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("request_id") == request_id:
                return row

    raise HTTPException(status_code=404, detail="request_id not found.")


# ============================================================
# ⭐ MAIN FEEDBACK ENDPOINT
# ============================================================
@router.post("/")
def submit_feedback(payload: FeedbackIn):
    row = load_log_row(payload.request_id)

    query = row.get("query", "")
    answer = row.get("model_answer", "") or row.get("answer", "")

    # ------------------ IS CORRECT ------------------
    if payload.is_correct:
        learner.add_verified_qa(query, answer)
        return {"status": "ok", "message": "👍 Marked correct. Added to verified memory."}

    # ------------------ IS WRONG ------------------
    learner.add_failure(query, answer, reason="user_marked_wrong")

    if payload.correct_answer.strip():
        learner.add_verified_qa(query, payload.correct_answer.strip())
        return {
            "status": "ok",
            "message": "👎 Wrong stored + corrected answer saved.",
        }

    return {
        "status": "ok",
        "message": "👎 Wrong answer stored (no correction provided)."
    }


# ============================================================
# ⭐ SIMPLE UI ENDPOINT FOR CORRECT ANSWERS
# ============================================================
@router.post("/correct")
def feedback_correct(payload: CorrectOnly):
    row = load_log_row(payload.request_id)
    query = row.get("query", "")
    corrected = payload.correct_answer.strip()

    if not corrected:
        raise HTTPException(status_code=400, detail="Correct answer is empty")

    learner.add_verified_qa(query, corrected)

    return {
        "status": "ok",
        "message": "Correct answer added to verified memory.",
        "correct_answer": corrected
    }


# ============================================================
# ⭐ SIMPLE UI ENDPOINT FOR WRONG ANSWERS
# ============================================================
@router.post("/wrong")
def feedback_wrong(payload: WrongOnly):
    row = load_log_row(payload.request_id)
    query = row.get("query", "")
    wrong = row.get("model_answer", "")

    learner.add_failure(query, wrong, reason="user_marked_wrong")

    return {
        "status": "ok",
        "message": "Wrong answer stored."
    }
