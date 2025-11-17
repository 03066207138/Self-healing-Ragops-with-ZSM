import csv
import os
from datetime import datetime, timezone
from typing import Dict, Any


class MetricsLogger:
    def __init__(self, path: str = "metrics_log.csv"):
        self.path = path
        # NOTE: retriever_type is included; keep this in sync with summary() "expected" list
        self.fieldnames = [
            "request_id", "ts_start", "ts_end", "latency_ms",
            "tokens_in", "tokens_out", "cost_usd",
            "k", "retrieved_doc_ids", "coverage_at_k",
            "grounding_faithfulness", "retriever_type",
            "anomaly_detected", "anomaly_type", "ts_detection",
            "ts_recovery_end", "mttr_s", "healing_action", "status_after_heal",
        ]
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self.fieldnames, quoting=csv.QUOTE_ALL)
                w.writeheader()

    @staticmethod
    def _sanitize(v: Any) -> Any:
        """
        Prevent raw newlines/commas from corrupting the CSV when rows were previously unquoted.
        We also stringify datetimes.
        """
        if hasattr(v, "isoformat"):
            return v.isoformat()
        if isinstance(v, (list, tuple)):
            # If someone passes a list of ids, join with '|' (NOT comma)
            return "|".join(str(x) for x in v)
        if isinstance(v, str):
            # Replace hard newlines to keep single-line CSV rows; commas are fine because we quote all fields
            return v.replace("\r\n", " ").replace("\n", " ").strip()
        return v

    def log(self, row: Dict[str, Any]):
        # ensure presence of all fields and sanitize
        out = {}
        for k in self.fieldnames:
            v = row.get(k, "")
            out[k] = self._sanitize(v)

        # Special case: if retrieved_doc_ids is a comma-joined string, swap commas -> pipes
        if isinstance(out.get("retrieved_doc_ids", ""), str):
            out["retrieved_doc_ids"] = out["retrieved_doc_ids"].replace(",", "|")

        # write quoted
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames, quoting=csv.QUOTE_ALL)
            w.writerow(out)
