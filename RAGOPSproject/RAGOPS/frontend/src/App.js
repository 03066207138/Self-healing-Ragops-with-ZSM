// src/App.js
import React, { useState, useEffect, useMemo } from "react";
import axios from "axios";

/**
 * FINAL FIXED VERSION — Self-Healing RAGOps Dashboard (Stable)
 * - Handles backend errors
 * - Shows fallback UI when no chunks or empty answers
 * - Fully safe metric access
 */

const API_BASE =
  (typeof import.meta !== "undefined" && import.meta.env?.VITE_API_BASE) ||
  process.env.REACT_APP_API_BASE ||
  "https://self-healing-ragops-with-zsm.onrender.com";

export default function App() {
  // ---------------- State ----------------
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [metrics, setMetrics] = useState({});
  const [history, setHistory] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const [file, setFile] = useState(null);
  const [uploads, setUploads] = useState([]);

  const [showFeedback, setShowFeedback] = useState(false);
  const [feedbackMode, setFeedbackMode] = useState(null);
  const [correctAnswer, setCorrectAnswer] = useState("");

  const [theme, setTheme] = useState(
    () => localStorage.getItem("theme") || "dark"
  );

  // ---------------- Effects ----------------
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
    fetchUploads();
  }, [theme]);

  useEffect(() => {
    const handler = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") submit();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [query]);

  // ---------------- API Calls ----------------
  async function fetchUploads() {
    try {
      const res = await axios.get(`${API_BASE}/uploads`);
      setUploads(res.data || []);
    } catch {
      setUploads([]);
    }
  }

  // ============================================================
  // 📌 QDRANT-INTEGRATED UPLOAD + INDEXING
  // ============================================================
  async function handleUpload() {
    if (!file) return;

    try {
      const form = new FormData();
      form.append("file", file);

      await axios.post(`${API_BASE}/upload`, form);
      await axios.post(
        `${API_BASE}/index_pdf/${encodeURIComponent(file.name)}`
      );

      fetchUploads();
      setFile(null);
      alert("📚 PDF Uploaded + Indexed in Qdrant!");
    } catch (e) {
      console.error(e);
      setError("❌ Upload or Index failed.");
    }
  }

  async function handleDelete(filename) {
    try {
      await axios.delete(`${API_BASE}/delete/${filename}`);
      fetchUploads();
    } catch {
      setError("⚠ Delete failed.");
    }
  }

  // ============================================================
  // 📌 MAIN QUERY — FIXED VERSION WITH ERROR LOGGING
  // ============================================================
  async function submit() {
    if (!query.trim()) return;

    setLoading(true);
    setError("");
    setShowFeedback(false);

    try {
      const res = await axios.post(`${API_BASE}/query`, { query });

      console.log("DEBUG /query response:", res.data);

      // ❌ Backend returned explicit error
      if (res.data?.error) {
        setAnswer(`⚠ ${res.data.error}`);
        setMetrics({});
        setError(res.data.error);
        return;
      }

      // ❌ No answer received (likely no chunks)
      if (!res.data?.answer || res.data.answer.trim() === "") {
        setAnswer(
          "⚠ No relevant answer found.\nUpload more PDFs or ask a different question."
        );
        setMetrics({});
        return;
      }

      // 🟢 Normal success
      const M = res.data.metrics || {};
      setAnswer(res.data.answer);
      setMetrics(M);

      if (Object.keys(M).length > 0) {
        updateHistory(M);
      }
    } catch (e) {
      console.error("QUERY ERROR:", e);
      setError("⚠ Backend unreachable or Qdrant has no indexed data");
      setAnswer("⚠ Could not reach backend");
    } finally {
      setLoading(false);
    }
  }

  // ============================================================
  // 📌 TIMELINE UPDATE (SAFE ACCESS)
  // ============================================================
  function updateHistory(m) {
    setHistory((prev) => [
      ...prev,
      {
        idx: (prev[prev.length - 1]?.idx ?? 0) + 1,

        latency_ms: m?.latency_ms ?? 0,
        coverage_at_k: m?.coverage_at_k ?? 0,
        faithfulness: m?.faithfulness ?? 0,
        semantic_drift: m?.semantic_drift ?? 0,
        hallucination_rate: m?.hallucination_rate ?? 0,

        anomaly_type: m?.anomaly_type ?? "—",
        healing_action: m?.healing_action ?? "—",
        reward: m?.reward ?? 0,

        status_after_heal: m?.status_after_heal ?? "unknown",
      },
    ]);
  }

  // ============================================================
  // 📌 FEEDBACK
  // ============================================================
  async function sendFeedback(isCorrect, corrected = null) {
    try {
      await axios.post(`${API_BASE}/feedback`, {
        request_id: metrics?.request_id || "",
        is_correct: isCorrect,
        correct_answer: corrected,
      });

      setShowFeedback(false);
      setCorrectAnswer("");
      setError(
        isCorrect
          ? "✔ Correct — added to learning memory"
          : corrected
          ? "✔ Improved answer saved"
          : "⚠ Marked wrong — system noted"
      );
    } catch {
      setError("⚠ Feedback failed");
    }
  }

  // ============================================================
  // 📌 KPIs
  // ============================================================
  const kpis = useMemo(() => {
    if (!metrics) return [];

    return [
      {
        label: "Governance Score",
        value: metrics?.governance_score?.toFixed?.(2) ?? "--",
      },
      {
        label: "Faithfulness",
        value: metrics?.faithfulness?.toFixed?.(2) ?? "--",
      },
      {
        label: "Coverage@K",
        value: metrics?.coverage_at_k?.toFixed?.(2) ?? "--",
      },
      {
        label: "Semantic Drift",
        value: metrics?.semantic_drift?.toFixed?.(2) ?? "--",
      },
      {
        label: "Latency (ms)",
        value: metrics?.latency_ms?.toFixed?.(1) ?? "--",
      },
    ];
  }, [metrics]);

  const StatusBadge = ({ status }) => {
    const cls =
      status === "healthy" ? "ok" : status === "failed" ? "bad" : "warn";
    return (
      <span className={`chip ${cls}`}>
        <span className="dot" /> {status}
      </span>
    );
  };

  // ============================================================
  // ============================= UI =============================
  // ============================================================
  return (
    <div className="app">
      <style>{styles}</style>

      {/* HEADER */}
      <header className="header glassy">
        <div className="brand">
          <span className="logo">🧠</span>
          <div>
            <h1 className="title">Agentic Self-Healing RAGOps 2.0</h1>
            <p className="subtitle">Autonomous AI • Multi-Agent Control Plane</p>
          </div>
        </div>

        <button
          className="btn toggle"
          onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
        >
          {theme === "dark" ? "☀ Light" : "🌙 Dark"}
        </button>
      </header>

      {/* LAYOUT */}
      <div className="layout">
        {/* SIDEBAR */}
        <aside className="sidebar glassy">
          <h3>📁 Uploaded Documents</h3>

          <label className="file">
            <input
              type="file"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
            <span>{file ? file.name : "Choose file…"}</span>
          </label>

          <button
            className="btn run"
            onClick={handleUpload}
            disabled={!file}
            style={{ marginTop: "8px" }}
          >
            ⬆ Upload + Index
          </button>

          <ul className="file-list" style={{ marginTop: "14px" }}>
            {uploads.length === 0 && <li className="muted">No files</li>}

            {uploads.map((f) => (
              <li key={f.doc_id} className="file-item">
                <div className="file-info">
                  <span className="file-name">{f.doc_id}</span>
                  <span className="file-size">{f.size_kb} KB</span>
                </div>
                <button
                  className="icon-btn danger"
                  onClick={() => handleDelete(f.doc_id)}
                >
                  🗑
                </button>
              </li>
            ))}
          </ul>

          <hr />

          {/* KPIs */}
          <div className="kpis">
            {kpis.map((k) => (
              <div key={k.label} className="kpi glassy">
                <div className="kpi-label">{k.label}</div>
                <div className="kpi-value">{k.value}</div>
              </div>
            ))}
          </div>
        </aside>

        {/* MAIN CONTENT */}
        <main className="content">
          {/* Query Panel */}
          <section className="panel glassy">
            <h3>💬 Ask a Question</h3>

            <textarea
              rows={4}
              className="input textarea"
              placeholder="Ask anything..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />

            <div style={{ display: "flex", gap: "10px", marginTop: "10px" }}>
              <button className="btn run" disabled={loading} onClick={submit}>
                {loading ? "⏳ Running..." : "🚀 Run"}
              </button>

              <button
                className="btn clear"
                onClick={() => {
                  setQuery("");
                  setAnswer("");
                  setMetrics({});
                }}
              >
                Clear
              </button>
            </div>
          </section>

          {/* Answer Panel */}
          {answer && (
            <section className="panel glassy">
              <div className="panel-title">
                <div className="title-inline">
                  <span className="badge success">Agent Response</span>
                  {metrics?.status_after_heal && (
                    <StatusBadge status={metrics.status_after_heal} />
                  )}
                </div>
              </div>

              <pre className="answer">{answer}</pre>

              <div className="panel glassy" style={{ marginTop: "10px" }}>
                {!showFeedback && (
                  <>
                    <h4>Was this answer correct?</h4>
                    <div style={{ display: "flex", gap: "10px" }}>
                      <button
                        className="btn run"
                        onClick={() => sendFeedback(true)}
                      >
                        👍 Yes
                      </button>

                      <button
                        className="btn clear"
                        onClick={() => {
                          setShowFeedback(true);
                          setFeedbackMode("wrong");
                        }}
                      >
                        👎 No
                      </button>

                      <button
                        className="btn ghost"
                        onClick={() => {
                          setShowFeedback(true);
                          setFeedbackMode("improve");
                        }}
                      >
                        ✍ Improve
                      </button>
                    </div>
                  </>
                )}

                {showFeedback && feedbackMode === "wrong" && (
                  <>
                    <h4>Mark as wrong?</h4>
                    <button
                      className="btn clear"
                      onClick={() => sendFeedback(false)}
                    >
                      Confirm Wrong
                    </button>
                  </>
                )}

                {showFeedback && feedbackMode === "improve" && (
                  <>
                    <h4>Provide the correct answer:</h4>

                    <textarea
                      rows={3}
                      className="input textarea"
                      value={correctAnswer}
                      onChange={(e) => setCorrectAnswer(e.target.value)}
                    />

                    <button
                      className="btn run"
                      onClick={() =>
                        sendFeedback(false, correctAnswer.trim())
                      }
                    >
                      Save Correct Answer
                    </button>
                  </>
                )}
              </div>
            </section>
          )}

          {/* Healing Timeline */}
          <section className="panel glassy">
            <h3>🔄 Healing Timeline</h3>

            <table className="tbl">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Latency</th>
                  <th>Coverage</th>
                  <th>Faithfulness</th>
                  <th>Drift</th>
                  <th>Hallucination</th>
                  <th>Anomaly</th>
                  <th>Action</th>
                  <th>Reward</th>
                  <th>Status</th>
                </tr>
              </thead>

              <tbody>
                {history.map((h, i) => (
                  <tr key={i}>
                    <td>{h.idx}</td>
                    <td>{h.latency_ms?.toFixed?.(1) ?? "--"}</td>
                    <td>{h.coverage_at_k?.toFixed?.(2) ?? "--"}</td>
                    <td>{h.faithfulness?.toFixed?.(2) ?? "--"}</td>
                    <td>{h.semantic_drift?.toFixed?.(2) ?? "--"}</td>
                    <td>{h.hallucination_rate?.toFixed?.(2) ?? "--"}</td>
                    <td>{h.anomaly_type}</td>
                    <td>{h.healing_action}</td>
                    <td>{h.reward?.toFixed?.(2) ?? "--"}</td>
                    <td>{h.status_after_heal}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <footer className="footer">
            © {new Date().getFullYear()} Self-Healing AI
          </footer>
        </main>
      </div>

      {error && (
        <div className="toast glassy">
          <div className="toast-body">{error}</div>
          <button className="icon-btn" onClick={() => setError("")}>
            ✖
          </button>
        </div>
      )}
    </div>
  );
}

/* ---------------- Styles (same as your version) ---------------- */
const styles = `
:root{
  --bg:#060913;
  --bg-accent:#0b1226;
  --surface:rgba(255,255,255,0.06);
  --surface-strong:rgba(255,255,255,0.1);
  --text:#eaf0ff;
  --muted:#8fa2c8;
  --ok:#16a34a;
  --warn:#f59e0b;
  --bad:#ef4444;
  --border:rgba(255,255,255,0.08);
  --grad: linear-gradient(120deg,#6d28d9 0%, #4f46e5 35%, #06b6d4 100%);
}
[data-theme="light"]{
  --bg:#f5f7fb;
  --bg-accent:#eff3fb;
  --surface:#ffffffd9;
  --surface-strong:#ffffff;
  --text:#0b1020;
  --muted:#5b6b8c;
  --border:rgba(0,0,0,0.08);
  --grad: linear-gradient(120deg,#6d28d9 0%, #4f46e5 35%, #06b6d4 100%);
}

/* Global */
*{box-sizing:border-box}
html,body,#root{height:100%}
body{
  margin:0;
  color:var(--text);
  background:
    radial-gradient(1200px 800px at 10% -10%, rgba(93,63,211,0.25), transparent 60%),
    radial-gradient(900px 600px at 110% 10%, rgba(6,182,212,0.25), transparent 60%),
    linear-gradient(180deg, var(--bg), var(--bg-accent));
  font-family: ui-sans-serif, system-ui, -apple-system, "Inter", "Segoe UI", Roboto, "Helvetica Neue", Arial, "Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol";
  background-attachment: fixed;
}

/* Containers */
.app { max-width: 1300px; margin: 0 auto; padding: 20px; }
.header{
  display:flex;align-items:center;justify-content:space-between;
  padding:14px 18px;margin-bottom:18px;border-radius:16px;color:white;
  background: var(--grad);
  box-shadow: 0 10px 30px rgba(36,42,116,0.35);
  position: relative; overflow: hidden;
}
.header::after{
  content:""; position:absolute; inset:-1px;
  background: radial-gradient(1000px 300px at -10% 0%, rgba(255,255,255,0.18), transparent 30%);
  pointer-events:none;
}
.brand{display:flex;gap:12px;align-items:center}
.logo{font-size:30px;filter: drop-shadow(0 2px 6px rgba(0,0,0,0.25));}
.title{margin:0;font-size:18px;letter-spacing:0.3px}
.subtitle{margin:2px 0 0 0; font-size:12.5px; opacity:.95}
.header-actions{display:flex;gap:10px;align-items:center}
.hint{font-size:12px;opacity:0.85}

.layout{
  display:grid;
  grid-template-columns: 320px 1fr;
  gap:16px;
}
@media (max-width: 980px){
  .layout{grid-template-columns:1fr}
  .sidebar{order:2}
  .content{order:1}
}

.glassy{
  background: var(--surface);
  backdrop-filter: blur(14px);
  border: 1px solid var(--border);
  border-radius: 16px;
}

.sidebar{
  position: sticky; top: 16px; height: fit-content;
  padding: 14px;
}
.sidebar-header{
  display:flex;justify-content:space-between;align-items:center;margin-bottom:8px
}
.tag{
  font-size:11px; padding:4px 8px; border-radius:999px; border:1px solid var(--border);
  background: rgba(255,255,255,0.06)
}
.tag.faint{opacity:.8}

.panel{ padding:16px; border-radius:16px; margin-bottom:16px; }
.panel-title{
  display:flex;justify-content:space-between;align-items:center;margin-bottom:10px
}
.title-inline{display:flex;gap:10px;align-items:center}
.badge{
  font-size:12px;font-weight:700;padding:6px 10px;border-radius:999px;
  background:linear-gradient(90deg,#22c55e22,#22c55e33); color:#22c55e;
  border:1px solid #22c55e55;
}
.badge.success{}

/* Buttons */
.btn{ padding:8px 14px; border:none; border-radius:10px; cursor:pointer; font-weight:700; transition:transform .06s ease }
.btn:active{ transform: translateY(1px) }
.btn.run{ background:linear-gradient(90deg,#6a5acd,#8b5cf6); color:#fff; box-shadow:0 6px 16px rgba(139,92,246,.35) }
.btn.clear{ background:var(--surface-strong); color:var(--text); border:1px solid var(--border) }
.btn.toggle{ background: #ffffff22; border:1px solid #ffffff55; color:#fff }
.btn.ghost{ background:transparent; color:var(--text); border:1px dashed var(--border) }
.icon-btn{
  background:var(--surface-strong); border:1px solid var(--border); color:var(--text);
  border-radius:10px; padding:6px 10px; cursor:pointer
}
.icon-btn.danger{ border-color:#ef444466; color:#ef4444; background:rgba(239,68,68,0.08) }

/* Inputs */
.input.textarea{
  width:100%; padding:12px 12px; border-radius:12px; border:1px solid var(--border);
  background: rgba(0,0,0,0.08); color: var(--text); resize: vertical; min-height: 100px;
}
.file{
  display:flex; align-items:center; gap:10px; border:1px dashed var(--border); padding:10px; border-radius:12px;
  background:rgba(255,255,255,0.04); cursor:pointer; width:100%;
}
.file input{ display:none }
.file span{ font-size:14px; color:var(--muted) }

.uploader{ display:flex; gap:10px; align-items:center; margin:10px 0 14px }
.uploads-head{ display:flex; justify-content:space-between; margin:6px 2px 10px; font-weight:600 }
.file-list{ list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:8px; max-height:260px; overflow:auto }
.file-item{
  display:flex; align-items:center; justify-content:space-between; gap:8px;
  padding:10px 12px; border-radius:12px; border:1px solid var(--border);
  background: rgba(255,255,255,0.035)
}
.file-info{ display:flex; flex-direction:column; gap:4px; min-width:0 }
.file-name{ font-size:13px; white-space:nowrap; text-overflow:ellipsis; overflow:hidden; max-width:220px }
.file-size{ font-size:12px; color:var(--muted) }

/* Content Tables & Answer */
.tbl{ width:100%; border-collapse:separate; border-spacing:0 8px }
.tbl thead th{
  font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted);
  text-align:center; padding:6px 8px;
}
.tbl tbody td{
  background: rgba(255,255,255,0.03);
  border:1px solid var(--border);
  padding:10px 8px; text-align:center; vertical-align:middle;
}
.tbl tbody tr td:first-child{
  border-top-left-radius:12px; border-bottom-left-radius:12px;
}
.tbl tbody tr td:last-child{
  border-top-right-radius:12px; border-bottom-right-radius:12px;
}
.tbl.comparison tbody td{ padding:12px 10px }

.answer{
  white-space:pre-wrap; background:rgba(0,0,0,0.15);
  padding:14px; border-radius:14px; overflow:auto; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size:14px; line-height:1.5;
  border:1px solid var(--border);
}

/* KPI Cards */
.kpis{
  display:grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap:12px; margin-top:14px;
}
.kpi{
  padding:14px; border-radius:14px; text-align:center;
  background:linear-gradient(180deg,#1e1b4baa,#0d0b2b99);
  border:1px solid var(--border)
}
.kpi-label{ font-size:12px; color:var(--muted) }
.kpi-value{ font-weight:800; font-size:22px; padding-top:6px }

/* Status Chip */
.chip{
  display:inline-flex; align-items:center; gap:8px;
  padding:6px 10px; border-radius:999px; font-weight:700; letter-spacing:.02em;
  border:1px solid var(--border); background:rgba(255,255,255,0.06);
  text-transform:capitalize;
}
.chip .dot{ width:8px; height:8px; border-radius:999px; background: currentColor; display:inline-block }
.chip.ok{ color:#16a34a; background:rgba(22,163,74,.12); border-color:#16a34a55 }
.chip.warn{ color:#f59e0b; background:rgba(245,158,11,.12); border-color:#f59e0b55 }
.chip.bad{ color:#ef4444; background:rgba(239,68,68,.12); border-color:#ef444455 }

/* Bars */
.bar-container{
  position:relative; width:100%; height:14px; background:rgba(255,255,255,0.06);
  border:1px solid var(--border); border-radius:999px; overflow:hidden;
}
.bar-fill{
  height:100%;
  animation: grow 600ms ease-out;
  background: linear-gradient(90deg, #22c55e, #06b6d4);
}
.bar-fill.loss{
  background: linear-gradient(90deg, #ef4444, #f59e0b);
}
.delta{
  position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
  font-size:11px; font-weight:800; color:#ffffffee; text-shadow:0 1px 2px rgba(0,0,0,.35);
  mix-blend-mode: plus-lighter;
}
.delta.loss{ color:#fff }
.score-badge{
  font-weight:800; font-size:12px; padding:6px 10px; border-radius:999px; border:1px solid var(--border)
}
.score-badge.gain{ color:#16a34a; background:rgba(22,163,74,.12); border-color:#16a34a55 }
.score-badge.loss{ color:#ef4444; background:rgba(239,68,68,.12); border-color:#ef444455 }
@keyframes grow{ from{ width:0 } to{ } }

/* Skeleton */
.skeleton{
  background: linear-gradient(90deg, rgba(255,255,255,0.08), rgba(255,255,255,0.18), rgba(255,255,255,0.08));
  background-size: 300% 100%;
  animation: shimmer 1.2s ease-in-out infinite;
  border-radius: 12px; border:1px solid var(--border); height: 120px; margin-top: 10px;
}
@keyframes shimmer{ 0%{ background-position: 0% 50% } 100%{ background-position: -200% 50% } }

/* Footer */
.footer{
  text-align:center; color:var(--muted); padding: 8px 0; margin-top: 6px;
}

/* Toast */
.toast{
  position:fixed; right:16px; bottom:16px; display:flex; gap:10px; align-items:center;
  padding:12px 14px; border-radius:12px; background:var(--surface); border:1px solid var(--border);
  box-shadow:0 10px 30px rgba(0,0,0,.25); z-index:50
}
.toast .toast-body{ max-width: 260px }
`;
