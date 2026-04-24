import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_SUPABASE_ANON_KEY;

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

export interface TradeLog {
  id?: string;
  session_id: string;
  timestamp: string;
  event_type: 'EXECUTED' | 'BLOCKED' | 'SIGNAL';
  direction: string;
  size: number;
  price: number;
  reason: string;
  entropy: number;
  memory_bias: number;
  state_vector: Record<string, number>;
}

export interface Session {
  id?: string;
  session_id: string;
  pair: string;
  mode: string;
  started_at: string;
  trades_count: number;
  blocked_count: number;
  pnl: number;
  status: string;
}

export async function logTrade(log: Omit<TradeLog, 'id'>) {
  const { error } = await supabase.from('trade_logs').insert(log);
  if (error) console.error('Failed to log trade:', error);
}

export async function getRecentLogs(sessionId: string, limit = 20) {
  const { data, error } = await supabase
    .from('trade_logs')
    .select('*')
    .eq('session_id', sessionId)
    .order('timestamp', { ascending: false })
    .limit(limit);
  if (error) { console.error('Failed to fetch logs:', error); return []; }
  return data ?? [];
}

export async function upsertSession(session: Omit<Session, 'id'>) {
  const { error } = await supabase
    .from('sessions')
    .upsert(session, { onConflict: 'session_id' });
  if (error) console.error('Failed to upsert session:', error);
}

export async function getSession(sessionId: string) {
  const { data, error } = await supabase
    .from('sessions')
    .select('*')
    .eq('session_id', sessionId)
    .maybeSingle();
  if (error) { console.error('Failed to fetch session:', error); return null; }
  return data;
}
