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
    ? '\u25B2  BUY SIGNAL'
    : signalState === 'SELL'
      ? '\u25BC  SELL SIGNAL'
      : '\u2014  WAITING  \u2014';

  return (
    <div className="card glass">
      <div className="card-title">Trading Signal</div>
      <div className="signal-box">
        <div className="signal-text">Current Recommendation</div>
        <div className={clsx('signal-value', signalState)}>
          {signalText}
        </div>
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
      </div>

      {mode === 'manual' ? (
        <div className="manual-controls">
          <div className="action-row">
            <button
              className={clsx('act-btn', 'btn-buy', signalState === 'BUY' && 'glow')}
              onClick={onBuy}
            >
              BUY
            </button>
            <button
              className={clsx('act-btn', 'btn-sell', signalState === 'SELL' && 'glow')}
              onClick={onSell}
            >
              SELL
            </button>
          </div>
          <button
            className={clsx('close-btn', signalState !== 'WAIT' && 'active')}
            onClick={onClose}
          >
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
