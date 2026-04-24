import { useState, useEffect, useCallback, useRef } from 'react';
import { generateSymbol, computeMetrics, detectPatterns, type MarketMetrics, type PatternAlert } from '../lib/engine';
import { logTrade, upsertSession, getSession, type TradeLog } from '../lib/supabase';

const SESSION_ID = `session_${Date.now()}`;

export interface LogEntry {
  time: string;
  msg: string;
  type: 'signal' | 'warn' | 'info';
}

export interface HyperionState {
  sequence: string;
  position: number;
  price: number;
  metrics: MarketMetrics;
  patterns: PatternAlert[];
  signalState: 'WAIT' | 'WATCH' | 'BUY' | 'SELL';
  mode: 'manual' | 'auto';
  logs: LogEntry[];
  sessionStats: {
    tradesCount: number;
    blockedCount: number;
    pnl: number;
  };
  connected: boolean;
}

export function useHyperion() {
  const [state, setState] = useState<HyperionState>({
    sequence: 'IIBXXIBBUBD',
    position: 47,
    price: 1.08421,
    metrics: { entropy: 0, momentum: 0, dojiDensity: 0, lambda7: 0, miPredict: 0 },
    patterns: [],
    signalState: 'WAIT',
    mode: 'manual',
    logs: [],
    sessionStats: { tradesCount: 0, blockedCount: 0, pnl: 0 },
    connected: false,
  });

  const tickRef = useRef(0);
  const stateRef = useRef(state);
  stateRef.current = state;

  // Initialize session from Supabase
  useEffect(() => {
    (async () => {
      const existing = await getSession(SESSION_ID);
      if (existing) {
        setState(prev => ({
          ...prev,
          sessionStats: {
            tradesCount: existing.trades_count,
            blockedCount: existing.blocked_count,
            pnl: existing.pnl,
          },
          connected: true,
        }));
      } else {
        await upsertSession({
          session_id: SESSION_ID,
          pair: 'EUR_USD',
          mode: 'manual',
          started_at: new Date().toISOString(),
          trades_count: 0,
          blocked_count: 0,
          pnl: 0,
          status: 'active',
        });
        setState(prev => ({ ...prev, connected: true }));
      }
    })();
  }, []);

  const addLog = useCallback((msg: string, type: LogEntry['type']) => {
    const time = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' });
    setState(prev => ({
      ...prev,
      logs: [{ time, msg, type }, ...prev.logs].slice(0, 20),
    }));
  }, []);

  const tick = useCallback(() => {
    const sym = generateSymbol();
    const newSeq = (stateRef.current.sequence + sym).slice(-60);
    const newPos = stateRef.current.position + 1;
    const newPrice = stateRef.current.price + (Math.random() - 0.498) * 0.00015;
    const metrics = computeMetrics(newSeq);
    const patterns = detectPatterns(newSeq.slice(-20));

    const entNorm = metrics.entropy / 2.8;
    const highPat = patterns.some(p => p.urgency === 'HIGH');
    const buySignal = metrics.momentum > 0.3 && entNorm < 0.6 && metrics.miPredict > 0.4 && highPat;
    const sellSignal = metrics.momentum < -0.3 && entNorm < 0.6 && metrics.miPredict > 0.4;
    const signalState: HyperionState['signalState'] = buySignal ? 'BUY' : sellSignal ? 'SELL' : 'WAIT';

    setState(prev => ({
      ...prev,
      sequence: newSeq,
      position: newPos,
      price: newPrice,
      metrics,
      patterns,
      signalState,
    }));

    tickRef.current++;

    // Auto-execute in auto mode
    if (stateRef.current.mode === 'auto' && signalState !== 'WAIT') {
      const direction = signalState === 'BUY' ? 'BUY' : 'SELL';
      addLog(`${direction} 0.01  ·  ${newSeq.slice(-5)} -> auto`, 'signal');
      logTrade({
        session_id: SESSION_ID,
        timestamp: new Date().toISOString(),
        event_type: 'EXECUTED',
        direction,
        size: 0.01,
        price: newPrice,
        reason: 'auto-mode',
        entropy: metrics.entropy,
        memory_bias: metrics.momentum,
        state_vector: metrics as unknown as Record<string, number>,
      });
      setState(prev => ({
        ...prev,
        sessionStats: {
          ...prev.sessionStats,
          tradesCount: prev.sessionStats.tradesCount + 1,
          pnl: prev.sessionStats.pnl + (Math.random() - 0.45) * 0.001,
        },
      }));
    } else if (signalState === 'WAIT' && tickRef.current % 5 === 0) {
      // Periodically log blocks
      const reasons: string[] = [];
      if (entNorm >= 0.6) reasons.push(`entropy ${metrics.entropy.toFixed(2)} > threshold`);
      if (metrics.dojiDensity > 0.3) reasons.push(`doji density ${metrics.dojiDensity.toFixed(2)}`);
      if (metrics.miPredict < 0.4) reasons.push('low MI predict');
      if (reasons.length > 0) {
        addLog(`BLOCKED  ·  ${reasons.join(', ')}`, 'warn');
        logTrade({
          session_id: SESSION_ID,
          timestamp: new Date().toISOString(),
          event_type: 'BLOCKED',
          direction: 'N/A',
          size: 0,
          price: newPrice,
          reason: reasons.join('; '),
          entropy: metrics.entropy,
          memory_bias: metrics.momentum,
          state_vector: metrics as unknown as Record<string, number>,
        });
        setState(prev => ({
          ...prev,
          sessionStats: {
            ...prev.sessionStats,
            blockedCount: prev.sessionStats.blockedCount + 1,
          },
        }));
      }
    }
  }, [addLog]);

  // Main tick loop
  useEffect(() => {
    const interval = setInterval(tick, 1800);
    return () => clearInterval(interval);
  }, [tick]);

  const setMode = useCallback((mode: 'manual' | 'auto') => {
    setState(prev => ({ ...prev, mode }));
    upsertSession({
      session_id: SESSION_ID,
      pair: 'EUR_USD',
      mode,
      started_at: new Date().toISOString(),
      trades_count: stateRef.current.sessionStats.tradesCount,
      blocked_count: stateRef.current.sessionStats.blockedCount,
      pnl: stateRef.current.sessionStats.pnl,
      status: 'active',
    });
  }, []);

  const executeTrade = useCallback((direction: 'BUY' | 'SELL') => {
    const s = stateRef.current;
    addLog(`${direction} 0.01  ·  ${s.sequence.slice(-5)} -> manual`, 'signal');
    logTrade({
      session_id: SESSION_ID,
      timestamp: new Date().toISOString(),
      event_type: 'EXECUTED',
      direction,
      size: 0.01,
      price: s.price,
      reason: 'manual',
      entropy: s.metrics.entropy,
      memory_bias: s.metrics.momentum,
      state_vector: s.metrics as unknown as Record<string, number>,
    });
    setState(prev => ({
      ...prev,
      sessionStats: {
        ...prev.sessionStats,
        tradesCount: prev.sessionStats.tradesCount + 1,
        pnl: prev.sessionStats.pnl + (Math.random() - 0.45) * 0.001,
      },
    }));
  }, [addLog]);

  const closePosition = useCallback(() => {
    addLog('position closed', 'info');
  }, [addLog]);

  return { state, setMode, executeTrade, closePosition };
}
