import clsx from 'clsx';
import { SYMBOL_COLORS, type PatternAlert, getPatternColor } from '../lib/engine';

interface PatternStreamProps {
  sequence: string;
  position: number;
  patterns: PatternAlert[];
}

const ROW_SIZE = 12;
const NUM_ROWS = 3;

export function PatternStream({ sequence, position, patterns }: PatternStreamProps) {
  const seqToShow = sequence.slice(-(ROW_SIZE * NUM_ROWS));

  const rows = [];
  for (let r = 0; r < NUM_ROWS; r++) {
    const chunk = seqToShow.slice(r * ROW_SIZE, (r + 1) * ROW_SIZE);
    if (!chunk) break;
    const posStart = position - seqToShow.length + r * ROW_SIZE;

    const blocks = chunk.split('').map((s, i) => {
      const isLatest = r === NUM_ROWS - 1 && i === chunk.length - 1;
      const colors = SYMBOL_COLORS[s] ?? SYMBOL_COLORS['D'];
      return (
        <div
          key={`${r}-${i}`}
          className={clsx('symbol-block', isLatest && 'block-active')}
          style={{
            background: colors.bg,
            color: colors.fg,
            border: `0.5px solid ${colors.border}`,
            ...(isLatest ? { boxShadow: `0 0 6px ${colors.fg}` } : {}),
          }}
        >
          {s}
        </div>
      );
    });

    const rowPatterns = r === NUM_ROWS - 1 ? patterns : [];
    const tags = rowPatterns.map((p, i) => (
      <span
        key={i}
        className={clsx('pattern-tag', `tag-${p.urgency}`)}
        style={{ color: getPatternColor(p.urgency) }}
      >
        {p.name}
      </span>
    ));

    rows.push(
      <div key={r} className="block-row">
        <span className="pos-label">{posStart + ROW_SIZE}</span>
        {blocks}
        {tags}
      </div>
    );
  }

  return (
    <div className="stream-panel">
      <div className="stream-header">
        <span>pattern stream  ·  1m</span>
        <span>pos {position}</span>
      </div>
      {rows}
    </div>
  );
}
