import clsx from 'clsx';
import { type PatternAlert, getPatternColor } from '../lib/engine';

interface SignalPanelProps {
  signalState: 'WAIT' | 'WATCH' | 'BUY' | 'SELL';
  mode: 'manual' | 'auto';
  patterns: PatternAlert[];
  onBuy: () => void;
  onSell: () => void;
  onClose: () => void;
}

export function SignalPanel({ signalState, mode, patterns, onBuy, onSell, onClose }: SignalPanelProps) {
  const signalText = signalState === 'BUY'
    ? '\u25B2  buy signal'
    : signalState === 'SELL'
      ? '\u25BC  sell signal'
      : '\u2014  waiting \u2014';

  const signalColor = signalState === 'BUY'
    ? '#4a8a4a'
    : signalState === 'SELL'
      ? '#8a4a4a'
      : '#2a2a2a';

  const buyGlow = signalState === 'BUY';
  const sellGlow = signalState === 'SELL';

  return (
    <div className="signal-card">
      <div className="signal-title">signal</div>
      <div className="signal-state" style={{ color: signalColor }}>{signalText}</div>

      {patterns.length > 0 && (
        <div className="pattern-alerts">
          {patterns.map((p, i) => (
            <div
              key={i}
              className="pattern-alert-item"
              style={{ color: getPatternColor(p.urgency), borderLeftColor: getPatternColor(p.urgency) }}
            >
              {p.name}
            </div>
          ))}
        </div>
      )}

      {mode === 'manual' ? (
        <div className="manual-controls">
          <div className="action-row">
            <button className={clsx('act-btn', 'btn-buy', buyGlow && 'glow')} onClick={onBuy}>
              BUY
            </button>
            <button className={clsx('act-btn', 'btn-sell', sellGlow && 'glow')} onClick={onSell}>
              SELL
            </button>
          </div>
          <button className={clsx('close-btn', signalState !== 'WAIT' && 'active')} onClick={onClose}>
            close position
          </button>
        </div>
      ) : (
        <div className="auto-controls">
          <div className="auto-status-line">model active  ·  monitoring</div>
          <div className="auto-sub-status">local llm  ·  rag ready</div>
          <button className="close-btn emergency" onClick={onClose}>
            emergency stop
          </button>
        </div>
      )}
    </div>
  );
}
