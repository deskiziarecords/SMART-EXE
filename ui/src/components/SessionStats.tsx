interface SessionStatsProps {
  tradesCount: number;
  blockedCount: number;
  pnl: number;
}

export function SessionStats({ tradesCount, blockedCount, pnl }: SessionStatsProps) {
  const pnlColor = pnl >= 0 ? '#3a5a3a' : '#7a3a3a';
  const pnlText = (pnl >= 0 ? '+' : '') + pnl.toFixed(4);

  return (
    <div className="metric-card">
      <div className="metric-title">session</div>
      <div className="metric-row">
        <span className="metric-name">trades today</span>
        <span className="metric-val" style={{ color: '#444' }}>{tradesCount}</span>
      </div>
      <div className="metric-row">
        <span className="metric-name">blocked</span>
        <span className="metric-val" style={{ color: '#4a3a2a' }}>{blockedCount}</span>
      </div>
      <div className="metric-row">
        <span className="metric-name">p&l</span>
        <span className="metric-val" style={{ color: pnlColor }}>{pnlText}</span>
      </div>
    </div>
  );
}
