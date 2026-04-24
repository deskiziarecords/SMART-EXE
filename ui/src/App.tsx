import React, { useState, useEffect, useRef } from 'react';
import clsx from 'clsx';

interface LogEntry {
  time: string;
  msg: string;
  type: string;
}

interface MarketState {
  action: string;
  confidence: number;
  state_vector: {
    entropy: number;
    rho_c: number;
    delta_r: number;
    eta_trend: number;
    P_stop: number;
    Q_session: number;
  };
}

interface UpdateMessage {
  type: 'update';
  symbol: string;
  price: number;
  state: MarketState;
  sequence: string;
  timestamp: number;
}

const SYMBOLS = ['B', 'X', 'I', 'W', 'D', 'S', 'U'];

export default function App() {
  const [history, setHistory] = useState<string[]>([]);
  const [price, setPrice] = useState(1.08500);
  const [state, setState] = useState<MarketState | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [connected, setConnected] = useState(false);
  const [mode, setMode] = useState<'manual' | 'auto'>('manual');
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const connect = () => {
      const socket = new WebSocket('ws://localhost:8000/ws');
      socketRef.current = socket;

      socket.onopen = () => {
        setConnected(true);
        console.log('Connected to Hyperion Server');
      };

      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        switch (data.type) {
          case 'init':
            setHistory(data.history);
            setLogs(data.logs);
            setPrice(data.price);
            break;
          case 'update':
            setHistory(prev => [...prev, data.symbol].slice(-60));
            setPrice(data.price);
            setState(data.state);
            break;
          case 'price':
            setPrice(data.price);
            break;
          case 'log':
            setLogs(prev => [data.data, ...prev].slice(0, 20));
            break;
        }
      };

      socket.onclose = () => {
        setConnected(false);
        setTimeout(connect, 3000);
      };
    };

    connect();
    return () => socketRef.current?.close();
  }, []);

  const renderBlocks = (seq: string[]) => {
    return seq.map((s, i) => (
      <div 
        key={i} 
        className={clsx('block', `block-${s}`, i === seq.length - 1 && 'latest')}
      >
        {s}
      </div>
    ));
  };

  const getMetricWidth = (val: number) => {
    // Basic normalization for visualization
    return Math.min(Math.max(val * 100, 5), 100) + '%';
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="topbar glass">
        <div className="brand">
          HYPERION <span className="brand-accent">SENTINEL</span>
        </div>
        <div className={clsx('status-badge', !connected && 'disconnected')}>
          {connected ? 'LIVE STREAM' : 'OFFLINE'}
        </div>
        <div className="price-display">
          {price.toFixed(5)}
        </div>
      </header>

      {/* Main Grid */}
      <main className="main-content">
        <section className="stream-panel glass">
          <div className="panel-header">
            <span>Pattern Stream · 1m</span>
            <span>History: {history.length}</span>
          </div>
          <div className="pattern-rows">
            <div className="pattern-row">
              <div className="pattern-time">CUR SESSION</div>
              <div className="blocks-container">
                {renderBlocks(history.slice(-20))}
              </div>
            </div>
            {/* Historical rows would go here */}
          </div>
        </section>

        <aside className="sidebar">
          {/* Signal Indicator */}
          <div className="card glass">
            <div className="card-title">Trading Signal</div>
            <div className="signal-box">
              <div className="signal-text">Current Recommendation</div>
              <div className={clsx('signal-value', state?.action || 'HOLD')}>
                {state?.action || 'WAITING'}
              </div>
              {state && state.action !== 'HOLD' && (
                <div style={{fontSize: '10px', marginTop: '4px', color: 'var(--text-secondary)'}}>
                  Confidence: {(state.confidence * 10).toFixed(1)}%
                </div>
              )}
            </div>
            
            <div style={{display: 'flex', gap: '8px'}}>
              <button 
                onClick={() => { setMode('manual'); socketRef.current?.send(JSON.stringify({command: 'toggle_mode', mode: 'manual'})); }}
                className={clsx('act-btn', mode === 'manual' ? 'active-mode' : 'inactive-mode')}
              >MANUAL</button>
              <button 
                onClick={() => { setMode('auto'); socketRef.current?.send(JSON.stringify({command: 'toggle_mode', mode: 'auto'})); }}
                className={clsx('act-btn', mode === 'auto' ? 'active-mode' : 'inactive-mode')}
              >AUTO</button>
            </div>
          </div>

          {/* Market Intelligence */}
          <div className="card glass">
            <div className="card-title">Market State Intelligence</div>
            {[
              { label: 'Entropy H(S)', key: 'entropy' },
              { label: 'Momentum \u03C1', key: 'rho_c' },
              { label: 'Reversal \u03B4', key: 'delta_r' },
              { label: 'Trend \u03B7', key: 'eta_trend' },
              { label: 'Stop Hunt P', key: 'P_stop' }
            ].map(m => (
              <div key={m.key} className="metric-item">
                <div className="metric-info">
                  <span>{m.label}</span>
                  <span>{state?.state_vector[m.key as keyof typeof state.state_vector]?.toFixed(2) || '0.00'}</span>
                </div>
                <div className="metric-bar-bg">
                  <div 
                    className="metric-bar-fill" 
                    style={{ width: getMetricWidth(state?.state_vector[m.key as keyof typeof state.state_vector] || 0) }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </aside>
      </main>

      {/* Footer Logs */}
      <footer className="log-panel glass">
        {logs.length === 0 && <span className="text-muted" style={{fontSize: '10px'}}>NO RECENT ACTIVITY</span>}
        {logs.map((log, i) => (
          <div key={i} className="log-entry">
            <span className="text-secondary">[{log.time}]</span>
            <span className="log-type">{log.type.toUpperCase()}</span>
            <span>{log.msg}</span>
          </div>
        ))}
      </footer>
    </div>
  );
}
