import { type MarketMetrics } from '../lib/engine';

interface MarketStatePanelProps {
  metrics: MarketMetrics;
}

function MetricItem({
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
    <div className="metric-item">
      <div className="metric-info">
        <span>{label}</span>
        <span style={{ color: valueColor }}>{display}</span>
      </div>
      <div className="metric-bar-bg">
        <div
          className="metric-bar-fill"
          style={{
            width: `${Math.min(Math.max(barWidth * 100, 5), 100)}%`,
            background: barColor,
          }}
        />
      </div>
    </div>
  );
}

export function MarketStatePanel({ metrics }: MarketStatePanelProps) {
  const entNorm = metrics.entropy / 2.8;

  return (
    <div className="card glass">
      <div className="card-title">Market State Intelligence</div>
      <MetricItem
        label="Entropy H(S)"
        value={metrics.entropy}
        barWidth={entNorm}
        barColor={entNorm < 0.4 ? '#1a4020' : entNorm < 0.7 ? '#3a3a1a' : '#401a1a'}
        valueColor={entNorm < 0.4 ? '#5a9a5a' : entNorm < 0.7 ? '#9a9a3a' : '#9a4a4a'}
      />
      <MetricItem
        label="Momentum p"
        value={metrics.momentum}
        barWidth={Math.max(0, (metrics.momentum + 1) / 2)}
        barColor={metrics.momentum > 0 ? '#1a4020' : '#401a1a'}
        valueColor={metrics.momentum > 0.2 ? '#5a9a5a' : metrics.momentum < -0.2 ? '#9a4a4a' : '#9a9a3a'}
        format={v => (v >= 0 ? '+' : '') + v.toFixed(2)}
      />
      <MetricItem
        label="Doji Density"
        value={metrics.dojiDensity}
        barWidth={metrics.dojiDensity}
        barColor="#2a2a2a"
        valueColor={metrics.dojiDensity > 0.3 ? '#9a7a2a' : '#666'}
      />
      <MetricItem
        label="Lambda7 Causal"
        value={metrics.lambda7}
        barWidth={metrics.lambda7}
        barColor="#152840"
        valueColor="#4a7a9a"
      />
      <MetricItem
        label="MI Predict"
        value={metrics.miPredict}
        barWidth={metrics.miPredict}
        barColor="#2a1a3a"
        valueColor="#6a4a8a"
      />
    </div>
  );
}
