import clsx from 'clsx';

interface TopBarProps {
  price: number;
  mode: 'manual' | 'auto';
  connected: boolean;
  onModeChange: (mode: 'manual' | 'auto') => void;
}

export function TopBar({ price, mode, connected, onModeChange }: TopBarProps) {
  return (
    <header className="topbar">
      <span className="asset-label">EUR / USD</span>
      <span className="broker-badge">OANDA</span>
      <span className="broker-badge">demo</span>
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
      <span className={clsx('conn-badge', connected ? 'conn-live' : 'conn-off')}>
        {connected ? 'LIVE' : 'OFFLINE'}
      </span>
      <span className="price-display">{price.toFixed(5)}</span>
      <span className="spread-label">0.8 pip</span>
    </header>
  );
}
