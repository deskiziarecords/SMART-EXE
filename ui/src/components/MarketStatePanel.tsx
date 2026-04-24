import { type MarketMetrics } from '../lib/engine';

interface MarketStatePanelProps {
  metrics: MarketMetrics;
}

function MetricRow({
  label,
  value,
  barWidth,
  barColor,
  valueColor,
  format,
}: {
  label: string;
  value: number;
  barWidth: number;
  barColor: string;
  valueColor: string;
  format?: (v: number) => string;
}) {
  const display = format ? format(value) : value.toFixed(2);
  return (
    <div className="metric-row">
      <span className="metric-name">{label}</span>
      <div className="metric-bar-wrap">
        <div
          className="metric-bar"
          style={{ width: `${Math.round(barWidth * 100)}%`, background: barColor }}
        />
      </div>
      <span className="metric-val" style={{ color: valueColor }}>{display}</span>
    </div>
  );
}

export function MarketStatePanel({ metrics }: MarketStatePanelProps) {
  const entNorm = metrics.entropy / 2.8;

  return (
    <div className="metric-card">
      <div className="metric-title">market state</div>
      <MetricRow
        label="entropy H"
        value={metrics.entropy}
        barWidth={entNorm}
        barColor={entNorm < 0.4 ? '#2a3a2a' : entNorm < 0.7 ? '#3a3a1a' : '#3a1a1a'}
        valueColor={entNorm < 0.4 ? '#4a7a4a' : entNorm < 0.7 ? '#7a7a3a' : '#7a3a3a'}
      />
      <MetricRow
        label="momentum p"
        value={metrics.momentum}
        barWidth={Math.max(0, (metrics.momentum + 1) / 2)}
        barColor={metrics.momentum > 0 ? '#1a3a1a' : '#3a1a1a'}
        valueColor={metrics.momentum > 0.2 ? '#4a6a4a' : metrics.momentum < -0.2 ? '#7a4a4a' : '#5a5a3a'}
        format={v => (v >= 0 ? '+' : '') + v.toFixed(2)}
      />
      <MetricRow
        label="doji density"
        value={metrics.dojiDensity}
        barWidth={metrics.dojiDensity}
        barColor="#2a2a2a"
        valueColor={metrics.dojiDensity > 0.3 ? '#7a5a3a' : '#333'}
      />
      <MetricRow
        label="lambda7 causal"
        value={metrics.lambda7}
        barWidth={metrics.lambda7}
        barColor="#1a2a3a"
        valueColor="#3a5a6a"
      />
      <MetricRow
        label="MI predict"
        value={metrics.miPredict}
        barWidth={metrics.miPredict}
        barColor="#2a1a3a"
        valueColor="#5a3a7a"
      />
    </div>
  );
}
