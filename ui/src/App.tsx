import { TopBar } from './components/TopBar';
import { PatternStream } from './components/PatternStream';
import { MarketStatePanel } from './components/MarketStatePanel';
import { SignalPanel } from './components/SignalPanel';
import { SessionStats } from './components/SessionStats';
import { LogStrip } from './components/LogStrip';
import { useHyperion } from './hooks/useHyperion';

export default function App() {
  const { state, setMode, executeTrade, closePosition } = useHyperion();

  return (
    <div className="app-container">
      <TopBar
        price={state.price}
        mode={state.mode}
        connected={state.connected}
        onModeChange={setMode}
      />

      <main className="main-content">
        <PatternStream
          sequence={state.sequence}
          position={state.position}
          patterns={state.patterns}
        />

        <aside className="sidebar">
          <SignalPanel
            signalState={state.signalState}
            mode={state.mode}
            patterns={state.patterns}
            onBuy={() => executeTrade('BUY')}
            onSell={() => executeTrade('SELL')}
            onClose={closePosition}
          />
          <MarketStatePanel metrics={state.metrics} />
          <SessionStats
            tradesCount={state.sessionStats.tradesCount}
            blockedCount={state.sessionStats.blockedCount}
            pnl={state.sessionStats.pnl}
          />
        </aside>
      </main>

      <LogStrip
        logs={state.logs}
        entropy={state.metrics.entropy}
        sequence={state.sequence}
      />
    </div>
  );
}
