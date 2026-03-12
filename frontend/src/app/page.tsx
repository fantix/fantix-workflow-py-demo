"use client";

import { useState, useEffect } from "react";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

interface HealthResponse {
  status: string;
  timestamp: string;
}

interface GreetingResponse {
  runId: string,
}

interface GreetingsResponse {
  status: string;
  result: string[] | null;
}

interface Item {
  id: number;
  name: string;
  price: number;
}

interface ItemsResponse {
  items: Item[];
}

export default function Home() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [greeting, setGreeting] = useState<GreetingResponse | null>(null);
  const [runIdInput, setRunIdInput] = useState("");
  const [greetingsResponse, setGreetingsResponse] = useState<GreetingsResponse | null>(null);
  const [items, setItems] = useState<Item[]>([]);
  const [loading, setLoading] = useState({
    health: false,
    greeting: false,
    greetings: false,
    items: false,
  });
  const [isConnected, setIsConnected] = useState<boolean | null>(null);

  // Check backend health on mount
  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    setLoading((prev) => ({ ...prev, health: true }));
    try {
      const res = await fetch(`${BACKEND_URL}/health`);
      const data = await res.json();
      setHealth(data);
      setIsConnected(true);
    } catch (error) {
      console.error("Health check failed:", error);
      setHealth(null);
      setIsConnected(false);
    } finally {
      setLoading((prev) => ({ ...prev, health: false }));
    }
  };

  const fetchGreeting = async () => {
    setLoading((prev) => ({ ...prev, greeting: true }));
    try {
      const res = await fetch(`${BACKEND_URL}/greeting`);
      const data: GreetingResponse = await res.json();
      setGreeting(data);
    } catch (error) {
      console.error("Failed to fetch greeting:", error);
    } finally {
      setLoading((prev) => ({ ...prev, greeting: false }));
    }
  };

  const fetchGreetingsResult = async () => {
    if (!runIdInput.trim()) return;
    setLoading((prev) => ({ ...prev, greetings: true }));
    try {
      const res = await fetch(`${BACKEND_URL}/get_greetings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: runIdInput }),
      });
      const data: GreetingsResponse = await res.json();
      setGreetingsResponse(data);
    } catch (error) {
      console.error("Failed to fetch greetings result:", error);
    } finally {
      setLoading((prev) => ({ ...prev, greetings: false }));
    }
  };

  const fetchItems = async () => {
    setLoading((prev) => ({ ...prev, items: true }));
    try {
      const res = await fetch(`${BACKEND_URL}/items`);
      const data: ItemsResponse = await res.json();
      setItems(data.items);
    } catch (error) {
      console.error("Failed to fetch items:", error);
    } finally {
      setLoading((prev) => ({ ...prev, items: false }));
    }
  };

  return (
    <div className="container">
      <header>
        <h1>Next.js + FastAPI</h1>
        <p>Frontend-backend communication demo</p>
        <div className={`status-badge ${isConnected === true ? "connected" : isConnected === false ? "disconnected" : ""}`}>
          <span className="status-dot"></span>
          {isConnected === null
            ? "Checking connection..."
            : isConnected
            ? "Backend connected"
            : "Backend disconnected"}
        </div>
      </header>

      <div className="cards-grid">
        {/* Health Check Card */}
        <div className="card">
          <h2>
            <svg className="card-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Health Check
          </h2>
          <div className="card-content">
            <p>Check if the backend API is running and responsive.</p>
            <button onClick={checkHealth} disabled={loading.health}>
              {loading.health ? <span className="loading"></span> : "Check Health"}
            </button>
            {health && (
              <div className="response-box">
                {JSON.stringify(health, null, 2)}
              </div>
            )}
          </div>
        </div>

        {/* Greeting Card */}
        <div className="card">
          <h2>
            <svg className="card-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
            </svg>
            Get Greeting
          </h2>
          <div className="card-content">
            <p>Fetch a greeting message from the backend.</p>
            <button onClick={fetchGreeting} disabled={loading.greeting}>
              {loading.greeting ? <span className="loading"></span> : "Fetch Greeting"}
            </button>
            {greeting && (
              <div className="response-box">
                {JSON.stringify(greeting, null, 2)}
              </div>
            )}
          </div>
        </div>

        {/* Get Greetings Card */}
        <div className="card">
          <h2>
            <svg className="card-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
            </svg>
            Get Greetings
          </h2>
          <div className="card-content">
            <p>Enter a workflow run ID to get the greetings result.</p>
            <div className="input-group">
              <input
                type="text"
                placeholder="Paste run ID here..."
                value={runIdInput}
                onChange={(e) => setRunIdInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && fetchGreetingsResult()}
              />
              <button onClick={fetchGreetingsResult} disabled={loading.greetings || !runIdInput.trim()}>
                {loading.greetings ? <span className="loading"></span> : "Fetch"}
              </button>
            </div>
            {greetingsResponse && (
              <div className="response-box">
                {JSON.stringify(greetingsResponse, null, 2)}
              </div>
            )}
          </div>
        </div>

        {/* Items Card */}
        <div className="card">
          <h2>
            <svg className="card-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
            </svg>
            Items List
          </h2>
          <div className="card-content">
            <p>Fetch a list of items from the backend.</p>
            <button onClick={fetchItems} disabled={loading.items}>
              {loading.items ? <span className="loading"></span> : "Load Items"}
            </button>
            {items.length > 0 && (
              <ul className="items-list">
                {items.map((item) => (
                  <li key={item.id}>
                    <span className="item-name">{item.name}</span>
                    <span className="item-price">${item.price.toFixed(2)}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>

      <div className="env-info">
        <h3>Environment Configuration</h3>
        <code>NEXT_PUBLIC_BACKEND_URL={BACKEND_URL}</code>
      </div>
    </div>
  );
}
