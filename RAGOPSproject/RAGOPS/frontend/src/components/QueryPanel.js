import React, { useState } from "react";

export default function QueryPanel({ onQuery, loading }) {
  const [query, setQuery] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    onQuery(query);
    setQuery("");
  };

  return (
    <form onSubmit={handleSubmit} className="mb-6">
      <input
        type="text"
        placeholder="Enter your query..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        className="border p-2 w-2/3 rounded"
      />
      <button
        type="submit"
        className="ml-2 bg-blue-600 text-white px-4 py-2 rounded"
        disabled={loading}
      >
        {loading ? "Processing..." : "Ask"}
      </button>
    </form>
  );
}
