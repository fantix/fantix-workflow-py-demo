"use client";

import { useState } from "react";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

interface ResultResponse {
  status: string;
  result: string[] | null;
}

export default function Home() {
  const [token, setToken] = useState("");
  const [runId, setRunId] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("");
  const [resultRunId, setResultRunId] = useState("");
  const [resultResponse, setResultResponse] = useState<ResultResponse | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [loading, setLoading] = useState({
    start: false,
    draft: false,
    finish: false,
    result: false,
  });

  const addLog = (msg: string) => setLogs((prev) => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`]);

  const startWorkflow = async () => {
    if (!token.trim()) return;
    setLoading((prev) => ({ ...prev, start: true }));
    try {
      const res = await fetch(`${BACKEND_URL}/start_workflow?token=${encodeURIComponent(token)}`);
      const data = await res.json();
      setRunId(data.runId);
      setResultRunId(data.runId);
      addLog(`Workflow started — runId: ${data.runId}`);
    } catch (error) {
      console.error("Failed to start workflow:", error);
      addLog("Failed to start workflow");
    } finally {
      setLoading((prev) => ({ ...prev, start: false }));
    }
  };

  const sendDraftRequest = async () => {
    if (!prompt.trim() || !token.trim()) return;
    setLoading((prev) => ({ ...prev, draft: true }));
    try {
      await fetch(`${BACKEND_URL}/draft_request?prompt=${encodeURIComponent(prompt)}&token=${encodeURIComponent(token)}`);
      addLog(`Draft request sent: "${prompt}"`);
      setPrompt("");
    } catch (error) {
      console.error("Failed to send draft request:", error);
      addLog("Failed to send draft request");
    } finally {
      setLoading((prev) => ({ ...prev, draft: false }));
    }
  };

  const finishWorkflow = async () => {
    if (!token.trim()) return;
    setLoading((prev) => ({ ...prev, finish: true }));
    try {
      await fetch(`${BACKEND_URL}/finish_workflow?token=${encodeURIComponent(token)}`);
      addLog("Workflow finish signal sent");
    } catch (error) {
      console.error("Failed to finish workflow:", error);
      addLog("Failed to finish workflow");
    } finally {
      setLoading((prev) => ({ ...prev, finish: false }));
    }
  };

  const getResult = async () => {
    if (!resultRunId.trim()) return;
    setLoading((prev) => ({ ...prev, result: true }));
    try {
      const res = await fetch(`${BACKEND_URL}/get_result`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: resultRunId }),
      });
      const data: ResultResponse = await res.json();
      setResultResponse(data);
      addLog(`Result fetched — status: ${data.status}`);
    } catch (error) {
      console.error("Failed to fetch result:", error);
      addLog("Failed to fetch result");
    } finally {
      setLoading((prev) => ({ ...prev, result: false }));
    }
  };

  return (
    <div className="container">
      <header>
        <h1>Multi-Drafter Workflow</h1>
        <p>Interactive workflow demo with Vercel Workflow + FastAPI</p>
      </header>

      {/* Global token input */}
      <div className="token-bar">
        <label htmlFor="token">Workflow Token</label>
        <input
          id="token"
          type="text"
          placeholder="Enter a unique token (e.g. my-session-1)..."
          value={token}
          onChange={(e) => setToken(e.target.value)}
        />
      </div>

      <div className="cards-grid">
        {/* 1. Start Workflow */}
        <div className="card">
          <h2>
            <svg className="card-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Start Workflow
          </h2>
          <div className="card-content">
            <p>Launch the multi-drafter workflow with the token above.</p>
            <button onClick={startWorkflow} disabled={loading.start || !token.trim()}>
              {loading.start ? <span className="loading"></span> : "Start"}
            </button>
            {runId && (
              <div className="response-box">
                <span className="label">Run ID:</span> {runId}
              </div>
            )}
          </div>
        </div>

        {/* 2. Send Draft Request */}
        <div className="card">
          <h2>
            <svg className="card-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
            </svg>
            Send Draft Request
          </h2>
          <div className="card-content">
            <p>Send a prompt to both the thinking &amp; fast drafters.</p>
            <div className="input-group">
              <input
                type="text"
                placeholder="Enter your prompt..."
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && sendDraftRequest()}
              />
              <button onClick={sendDraftRequest} disabled={loading.draft || !prompt.trim() || !token.trim()}>
                {loading.draft ? <span className="loading"></span> : "Send"}
              </button>
            </div>
          </div>
        </div>

        {/* 3. Finish Workflow */}
        <div className="card">
          <h2>
            <svg className="card-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Finish Workflow
          </h2>
          <div className="card-content">
            <p>Signal both drafters to stop and finalize the workflow.</p>
            <button onClick={finishWorkflow} disabled={loading.finish || !token.trim()}>
              {loading.finish ? <span className="loading"></span> : "Finish"}
            </button>
          </div>
        </div>

        {/* 4. Get Result */}
        <div className="card">
          <h2>
            <svg className="card-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Get Result
          </h2>
          <div className="card-content">
            <p>Fetch the workflow result by run ID.</p>
            <div className="input-group">
              <input
                type="text"
                placeholder="Run ID..."
                value={resultRunId}
                onChange={(e) => setResultRunId(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && getResult()}
              />
              <button onClick={getResult} disabled={loading.result || !resultRunId.trim()}>
                {loading.result ? <span className="loading"></span> : "Fetch"}
              </button>
            </div>
            {resultResponse && (
              <div className="response-box">
                <div><span className="label">Status:</span> <span className={`status-tag ${resultResponse.status}`}>{resultResponse.status}</span></div>
                {resultResponse.result && (
                  <ul className="result-list">
                    {resultResponse.result.map((item, i) => (
                      <li key={i}>{item}</li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Activity Log */}
      {logs.length > 0 && (
        <div className="env-info">
          <h3>Activity Log</h3>
          <div className="log-box">
            {logs.map((log, i) => (
              <div key={i} className="log-line">{log}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
