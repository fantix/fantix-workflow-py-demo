"use client";

import { useState, useEffect } from "react";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

interface HealthResponse {
  status: string;
  timestamp: string;
}

interface GreetingResponse {
  message: string;
  timestamp: string;
}

interface EchoResponse {
  message: string;
  timestamp: string;
  reversed: string;
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
  const [echoInput, setEchoInput] = useState("");
  const [echoResponse, setEchoResponse] = useState<EchoResponse | null>(null);
  const [items, setItems] = useState<Item[]>([]);
  const [loading, setLoading] = useState({
    health: false,
    greeting: false,
    echo: false,
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
      const res = await fetch(`${BACKEND_URL}/api/health`);
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
      const res = await fetch(`${BACKEND_URL}/api/greeting`);
      const data: GreetingResponse = await res.json();
      setGreeting(data);
    } catch (error) {
      console.error("Failed to fetch greeting:", error);
    } finally {
      setLoading((prev) => ({ ...prev, greeting: false }));
    }
  };

  const sendEcho = async () => {
    if (!echoInput.trim()) return;
    setLoading((prev) => ({ ...prev, echo: true }));
    try {
      const res = await fetch(`${BACKEND_URL}/api/echo`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: echoInput }),
      });
      const data: EchoResponse = await res.json();
      setEchoResponse(data);
    } catch (error) {
      console.error("Echo request failed:", error);
    } finally {
      setLoading((prev) => ({ ...prev, echo: false }));
    }
  };

  const fetchItems = async () => {
    setLoading((prev) => ({ ...prev, items: true }));
    try {
      const res = await fetch(`${BACKEND_URL}/api/items`);
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

        {/* Echo Card */}
        <div className="card">
          <h2>
            <svg className="card-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
            </svg>
            Echo Service
          </h2>
          <div className="card-content">
            <p>Send a message and get it echoed back (with reversal).</p>
            <div className="input-group">
              <input
                type="text"
                placeholder="Type a message..."
                value={echoInput}
                onChange={(e) => setEchoInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && sendEcho()}
              />
              <button onClick={sendEcho} disabled={loading.echo || !echoInput.trim()}>
                {loading.echo ? <span className="loading"></span> : "Send"}
              </button>
            </div>
            {echoResponse && (
              <div className="response-box">
                {JSON.stringify(echoResponse, null, 2)}
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
