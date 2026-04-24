// Client-side pattern engine matching the Python encoder logic

export const SYMBOLS = ['B', 'I', 'X', 'U', 'D', 'W', 'w'] as const;
export type SymbolType = typeof SYMBOLS[number];

export const SYMBOL_COLORS: Record<string, { bg: string; fg: string; border: string }> = {
  B: { bg: '#0f2010', fg: '#5a9a5a', border: '#1a4020' },
  I: { bg: '#200f0f', fg: '#9a4a4a', border: '#401a1a' },
  X: { bg: '#151510', fg: '#7a7a50', border: '#2a2a18' },
  U: { bg: '#0a1820', fg: '#4a7a9a', border: '#152840' },
  w: { bg: '#1a1220', fg: '#6a4a8a', border: '#2a1a3a' },
  D: { bg: '#111111', fg: '#4a4a4a', border: '#222222' },
  W: { bg: '#201a08', fg: '#9a7a2a', border: '#3a2a10' },
};

export interface PatternAlert {
  name: string;
  urgency: 'HIGH' | 'MEDIUM' | 'IGNORE';
}

const PATTERN_URGENCY_COLORS: Record<string, string> = {
  HIGH: '#9a5020',
  MEDIUM: '#3a6a8a',
  IGNORE: '#2a2a2a',
};

export function getPatternColor(urgency: string) {
  return PATTERN_URGENCY_COLORS[urgency] ?? '#2a2a2a';
}

export function detectPatterns(seq: string): PatternAlert[] {
  const found: PatternAlert[] = [];
  if (seq.includes('IIII')) found.push({ name: 'VOLATILITY SQZ', urgency: 'HIGH' });
  if (seq.includes('BBBBB')) found.push({ name: 'MOMENTUM CAP', urgency: 'HIGH' });
  if (seq.includes('XXXXX')) found.push({ name: 'MOMENTUM CAP', urgency: 'HIGH' });
  if (seq.includes('XB')) found.push({ name: 'FALSE BRKDN', urgency: 'MEDIUM' });
  if (seq.includes('BX')) found.push({ name: 'FALSE BRKUP', urgency: 'MEDIUM' });
  if (seq.includes('DD')) found.push({ name: 'DOJI CLUSTER', urgency: 'MEDIUM' });
  if (seq.slice(-3).includes('w')) found.push({ name: 'WEAK WICK', urgency: 'IGNORE' });
  if (seq.slice(-3).includes('W')) found.push({ name: 'WEAK WICK', urgency: 'IGNORE' });
  return found;
}

export interface MarketMetrics {
  entropy: number;
  momentum: number;
  dojiDensity: number;
  lambda7: number;
  miPredict: number;
}

export function computeMetrics(seq: string): MarketMetrics {
  const n = seq.length || 1;
  const cnt: Record<string, number> = {};
  for (const s of 'BIXUwDW') cnt[s] = (seq.split(s).length - 1);

  const ent = -Object.values(cnt)
    .filter(v => v > 0)
    .reduce((a, v) => {
      const p = v / n;
      return a + p * Math.log2(p);
    }, 0);

  const mom = (cnt.B - cnt.I) / n;
  const doji = cnt.D / n;
  const lam = 0.5 + Math.random() * 0.4;
  const mi = Math.max(0, 1 - ent / 2.8);

  return {
    entropy: Math.min(ent, 2.8),
    momentum: mom,
    dojiDensity: doji,
    lambda7: lam,
    miPredict: mi,
  };
}

export function generateSymbol(): string {
  const weights = [0.18, 0.18, 0.15, 0.12, 0.15, 0.11, 0.11];
  const r = Math.random();
  let cum = 0;
  for (let i = 0; i < SYMBOLS.length; i++) {
    cum += weights[i];
    if (r < cum) return SYMBOLS[i];
  }
  return 'D';
}
