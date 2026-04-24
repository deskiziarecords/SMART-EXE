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
    <div className="log-strip">
      {logs.length === 0 && (
        <span className="log-item" style={{ color: '#252525' }}>NO RECENT ACTIVITY</span>
      )}
      {logs.map((log, i) => (
        <span
          key={i}
          className={clsx('log-item', log.type === 'signal' && 'recent', log.type === 'warn' && 'warn')}
        >
          {log.time}  {log.msg}
        </span>
      ))}
      <div className="entropy-wrap">
        <span className="entropy-label">H(S)</span>
        <span className={clsx('entropy-val', entClass)}>{entropy.toFixed(2)}</span>
        <span className="seq-display">{sequence.slice(-10)}</span>
      </div>
    </div>
  );
}
