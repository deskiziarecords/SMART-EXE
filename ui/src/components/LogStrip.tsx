import clsx from 'clsx';
import type { LogEntry } from '../hooks/useHyperion';

interface LogStripProps {
  logs: LogEntry[];
  entropy: number;
  sequence: string;
}

export function LogStrip({ logs, entropy, sequence }: LogStripProps) {
  const entNorm = entropy / 2.8;
  const entClass = entNorm < 0.4 ? 'ent-low' : entNorm < 0.7 ? 'ent-med' : 'ent-high';

  return (
    <footer className="log-panel glass">
      {logs.length === 0 && (
        <span className="text-muted" style={{ fontSize: '10px' }}>NO RECENT ACTIVITY</span>
      )}
      {logs.map((log, i) => (
        <div key={i} className="log-entry">
          <span className="text-secondary">[{log.time}]</span>
          <span className={clsx('log-type', log.type === 'warn' && 'log-warn', log.type === 'signal' && 'log-signal')}>
            {log.msg}
          </span>
        </div>
      ))}
      <div className="entropy-wrap">
        <span className="entropy-label">H(S)</span>
        <span className={clsx('entropy-val', entClass)}>{entropy.toFixed(2)}</span>
        <span className="seq-display">{sequence.slice(-10)}</span>
      </div>
    </footer>
  );
}
