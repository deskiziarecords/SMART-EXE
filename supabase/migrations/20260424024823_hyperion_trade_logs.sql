/*
  # Hyperion Sentinel - Trade Logs and Session Data

  1. New Tables
    - `trade_logs`
      - `id` (uuid, primary key)
      - `session_id` (text, identifies the trading session)
      - `timestamp` (timestamptz, when the event occurred)
      - `event_type` (text: 'EXECUTED', 'BLOCKED', 'SIGNAL')
      - `direction` (text: 'BUY', 'SELL', 'HOLD', 'N/A')
      - `size` (float, position size, default 0)
      - `price` (float, execution price, default 0)
      - `reason` (text, block reason or signal details)
      - `entropy` (float, entropy value at event time)
      - `memory_bias` (float, memory bias at event time)
      - `state_vector` (jsonb, full market state snapshot)
      - `created_at` (timestamptz, record creation time)

    - `sessions`
      - `id` (uuid, primary key)
      - `session_id` (text, unique session identifier)
      - `pair` (text, trading pair e.g. 'EUR_USD')
      - `mode` (text: 'manual' or 'auto')
      - `started_at` (timestamptz)
      - `trades_count` (int, default 0)
      - `blocked_count` (int, default 0)
      - `pnl` (float, default 0)
      - `status` (text: 'active', 'closed')
      - `created_at` (timestamptz)

  2. Security
    - Enable RLS on both tables
    - Public read/write for demo (no auth required for this trading dashboard)
    - In production, these would be restricted to authenticated users
*/

CREATE TABLE IF NOT EXISTS trade_logs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id text NOT NULL DEFAULT 'default',
  timestamp timestamptz NOT NULL DEFAULT now(),
  event_type text NOT NULL DEFAULT 'BLOCKED',
  direction text NOT NULL DEFAULT 'N/A',
  size float DEFAULT 0,
  price float DEFAULT 0,
  reason text DEFAULT '',
  entropy float DEFAULT 0,
  memory_bias float DEFAULT 0,
  state_vector jsonb DEFAULT '{}'::jsonb,
  created_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sessions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id text UNIQUE NOT NULL,
  pair text NOT NULL DEFAULT 'EUR_USD',
  mode text NOT NULL DEFAULT 'manual',
  started_at timestamptz DEFAULT now(),
  trades_count int DEFAULT 0,
  blocked_count int DEFAULT 0,
  pnl float DEFAULT 0,
  status text DEFAULT 'active',
  created_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_trade_logs_session ON trade_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_trade_logs_timestamp ON trade_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);

ALTER TABLE trade_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;

-- For this demo dashboard, allow public read/write
-- In production, replace with authenticated user policies
CREATE POLICY "Public read trade logs"
  ON trade_logs FOR SELECT
  TO anon, authenticated
  USING (true);

CREATE POLICY "Public insert trade logs"
  ON trade_logs FOR INSERT
  TO anon, authenticated
  WITH CHECK (true);

CREATE POLICY "Public read sessions"
  ON sessions FOR SELECT
  TO anon, authenticated
  USING (true);

CREATE POLICY "Public insert sessions"
  ON sessions FOR INSERT
  TO anon, authenticated
  WITH CHECK (true);

CREATE POLICY "Public update sessions"
  ON sessions FOR UPDATE
  TO anon, authenticated
  USING (true)
  WITH CHECK (true);
