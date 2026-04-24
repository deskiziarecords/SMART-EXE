import clsx from 'clsx';

interface TopBarProps {
  price: number;
  mode: 'manual' | 'auto';
  connected: boolean;
  onModeChange: (mode: 'manual' | 'auto') => void;
}

export function TopBar({ price, mode, connected, onModeChange }: TopBarProps) {
  return (
    <header className="topbar glass">
      <div className="brand">
        HYPERION <span className="brand-accent">SENTINEL</span>
      </div>
      <div className="topbar-badges">
        <span className="broker-badge">OANDA</span>
        <span className="broker-badge">demo</span>
      </div>
      <div className="mode-toggle">
        <button
          className={clsx('mode-btn', mode === 'manual' && 'active')}
          onClick={() => onModeChange('manual')}
        >
          manual
        </button>
        <button
          className={clsx('mode-btn', mode === 'auto' && 'active')}
          onClick={() => onModeChange('auto')}
        >
          auto
        </button>
      </div>
      <div className={clsx('status-badge', !connected && 'disconnected')}>
        {connected ? 'LIVE STREAM' : 'OFFLINE'}
      </div>
      <div className="price-display">
        {price.toFixed(5)}
      </div>
      <span className="spread-label">0.8 pip</span>
    </header>
  );
}
