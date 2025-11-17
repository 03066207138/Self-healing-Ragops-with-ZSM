import React from "react";

export default function Dashboard({ summary, last, onHeal }) {
  const healActions = [
    "increase_k",
    "reindex",
    "switch_retriever",
    "tighten_prompt",
    "fallback_llm",
    "shrink_context",
    "scale_out"
  ];

  return (
    <div>
      <h2 className="text-2xl font-semibold mb-3">📊 Metrics Summary</h2>
      <div className="grid grid-cols-2 gap-4 mb-6">
        {Object.entries(summary).map(([k, v]) => (
          <div key={k} className="border rounded p-3 shadow-sm">
            <strong>{k}</strong>: {v?.toFixed ? v.toFixed(3) : v}
          </div>
        ))}
      </div>

      <h3 className="text-xl font-semibold mb-3">🧩 Last Metrics</h3>
      <pre className="bg-gray-100 p-3 rounded mb-4">
        {JSON.stringify(last, null, 2)}
      </pre>

      <h3 className="text-xl font-semibold mb-2">🛠️ Trigger Healing Action</h3>
      <div className="flex flex-wrap gap-2">
        {healActions.map(a => (
          <button
            key={a}
            className="bg-green-500 hover:bg-green-600 text-white px-3 py-2 rounded"
            onClick={() => onHeal(a)}
          >
            {a}
          </button>
        ))}
      </div>
    </div>
  );
}
