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
    <div className="app">
      <TopBar
        price={state.price}
        mode={state.mode}
        connected={state.connected}
        onModeChange={setMode}
      />

      <div className="main-row">
        <PatternStream
          sequence={state.sequence}
          position={state.position}
          patterns={state.patterns}
        />

        <div className="side-panel">
          <MarketStatePanel metrics={state.metrics} />
          <SignalPanel
            signalState={state.signalState}
            mode={state.mode}
            patterns={state.patterns}
            onBuy={() => executeTrade('BUY')}
            onSell={() => executeTrade('SELL')}
            onClose={closePosition}
          />
          <SessionStats
            tradesCount={state.sessionStats.tradesCount}
            blockedCount={state.sessionStats.blockedCount}
            pnl={state.sessionStats.pnl}
          />
        </div>
      </div>

      <LogStrip
        logs={state.logs}
        entropy={state.metrics.entropy}
        sequence={state.sequence}
      />
    </div>
  );
}
