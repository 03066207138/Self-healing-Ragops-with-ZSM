# app/metrics_router.py
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

# You can override this with env var METRICS_LOG_PATH if you save it elsewhere
METRICS_PATH = os.getenv("METRICS_LOG_PATH", "metrics_log.csv")

router = APIRouter(prefix="/metrics", tags=["metrics"])

@router.get("/download")
def download_metrics():
    """Download the metrics log as a CSV file."""
    if not os.path.exists(METRICS_PATH) or os.path.getsize(METRICS_PATH) == 0:
        raise HTTPException(status_code=404, detail="metrics_log.csv not found or empty.")
    # FileResponse sets Content-Disposition so the browser downloads it
    return FileResponse(
        METRICS_PATH,
        media_type="text/csv",
        filename="metrics_log.csv",
    )
