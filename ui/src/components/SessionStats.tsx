interface SessionStatsProps {
  tradesCount: number;
  blockedCount: number;
  pnl: number;
}

export function SessionStats({ tradesCount, blockedCount, pnl }: SessionStatsProps) {
  const pnlColor = pnl >= 0 ? '#5a9a5a' : '#9a4a4a';
  const pnlText = (pnl >= 0 ? '+' : '') + pnl.toFixed(4);

  return (
    <div className="card glass">
      <div className="card-title">Session</div>
      <div className="session-row">
        <span className="session-label">trades today</span>
        <span className="session-val">{tradesCount}</span>
      </div>
      <div className="session-row">
        <span className="session-label">blocked</span>
        <span className="session-val blocked">{blockedCount}</span>
      </div>
      <div className="session-row">
        <span className="session-label">p&l</span>
        <span className="session-val" style={{ color: pnlColor }}>{pnlText}</span>
      </div>
    </div>
  );
}
