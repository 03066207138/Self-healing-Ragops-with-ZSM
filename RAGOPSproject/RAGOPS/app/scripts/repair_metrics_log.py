import csv

INPUT = "metrics_log.csv"
OUTPUT = "metrics_log.fixed.csv"

FIELDS = [
    "request_id","ts_start","ts_end","latency_ms",
    "tokens_in","tokens_out","cost_usd",
    "k","retrieved_doc_ids","coverage_at_k",
    "grounding_faithfulness","retriever_type",
    "anomaly_detected","anomaly_type","ts_detection",
    "ts_recovery_end","mttr_s","healing_action","status_after_heal"
]

def main():
    rows = []
    with open(INPUT, "r", encoding="utf-8", newline="") as f:
        # Try to read loosely; if header mismatches, we still collect cells
        reader = csv.reader(f)
        for r in reader:
            # Normalize row length to FIELDS
            if len(r) < len(FIELDS):
                r += [""] * (len(FIELDS) - len(r))
            elif len(r) > len(FIELDS):
                r = r[:len(FIELDS)]
            rows.append(r)

    # If first row looks like an old header (missing retriever_type), drop it
    if rows and "request_id" in rows[0][0]:
        rows = rows[1:]

    with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in rows:
            w.writerow({k: v.replace("\n"," ").replace("\r"," ") for k, v in zip(FIELDS, r)})

    print(f"Rewrote to {OUTPUT}. Replace the original when satisfied.")

if __name__ == "__main__":
    main()
