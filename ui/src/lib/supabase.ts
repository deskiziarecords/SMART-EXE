import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_SUPABASE_ANON_KEY;

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

export async function logTrade(log: {
  session_id: string;
  timestamp: string;
  event_type: string;
  direction: string;
  size: number;
  price: number;
  reason: string;
  entropy: number;
  memory_bias: number;
  state_vector: Record<string, number>;
}) {
  const { error } = await supabase.from('trade_logs').insert(log);
  if (error) console.error('Failed to log trade:', error);
}

export async function upsertSession(session: {
  session_id: string;
  pair: string;
  mode: string;
  started_at: string;
  trades_count: number;
  blocked_count: number;
  pnl: number;
  status: string;
}) {
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
