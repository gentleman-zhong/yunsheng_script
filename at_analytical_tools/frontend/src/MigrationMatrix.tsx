import {useEffect, useMemo, useState} from 'react';
import {fmt} from './api';

export interface Migration {raw: number; final: number; count: number}
const PAGE_SIZE = 20;

export function migrationIndex(migrations: Migration[]) {
  const columns = [...new Set(migrations.map(item => item.raw))].sort((a, b) => a - b);
  const rows = [...new Set(migrations.map(item => item.final))].sort((a, b) => a - b);
  const counts = new Map<string, number>();
  let maximum = 1;
  for (const item of migrations) {
    const key = `${item.raw}:${item.final}`;
    const count = (counts.get(key) || 0) + item.count;
    counts.set(key, count);
    maximum = Math.max(maximum, count);
  }
  return {columns, rows, counts, maximum};
}

export default function MigrationMatrix({migrations, onSelect}: {
  migrations: Migration[];
  onSelect: (raw: number, final: number) => void;
}) {
  const data = useMemo(() => migrationIndex(migrations), [migrations]);
  const [columnOffset, setColumnOffset] = useState(0);
  const [rowOffset, setRowOffset] = useState(0);
  useEffect(() => {setColumnOffset(0); setRowOffset(0)}, [migrations]);
  const columns = data.columns.slice(columnOffset, columnOffset + PAGE_SIZE);
  const rows = data.rows.slice(rowOffset, rowOffset + PAGE_SIZE);

  return <div className="chart-card">
    <h3>Track 长度迁移 · 点击格子筛选</h3>
    <p className="hint">横轴：输入长度；纵轴：最终长度。最终长度 0 表示未保留。只列出实际出现的长度，格子表示精确长度，不是区间。</p>
    {!migrations.length ? <p className="empty">没有可比较的长度迁移记录</p> : <>
      {(data.columns.length > PAGE_SIZE || data.rows.length > PAGE_SIZE) &&
        <div className="table-filters">
          <span>输入长度列 {columnOffset + 1}–{Math.min(columnOffset + PAGE_SIZE, data.columns.length)} / {data.columns.length}</span>
          <button disabled={!columnOffset} onClick={() => setColumnOffset(n => Math.max(0, n - PAGE_SIZE))}>前一组输入长度</button>
          <button disabled={columnOffset + PAGE_SIZE >= data.columns.length} onClick={() => setColumnOffset(n => n + PAGE_SIZE)}>后一组输入长度</button>
          <span>最终长度行 {rowOffset + 1}–{Math.min(rowOffset + PAGE_SIZE, data.rows.length)} / {data.rows.length}</span>
          <button disabled={!rowOffset} onClick={() => setRowOffset(n => Math.max(0, n - PAGE_SIZE))}>前一组最终长度</button>
          <button disabled={rowOffset + PAGE_SIZE >= data.rows.length} onClick={() => setRowOffset(n => n + PAGE_SIZE)}>后一组最终长度</button>
        </div>}
      <div className="migration-scroll"><div className="migration-grid" style={{gridTemplateColumns: `50px repeat(${columns.length}, minmax(45px, 1fr))`}}>
        <span>终 / 原</span>
        {columns.map(raw => <span key={raw}>{raw}</span>)}
        {rows.map(final => <div className="migration-row" key={final}>
          <span>{final}</span>
          {columns.map(raw => {
            const count = data.counts.get(`${raw}:${final}`) || 0;
            return <button key={raw} disabled={!count}
              style={{background: count ? `rgba(76,197,153,${.15 + .8 * Math.log1p(count) / Math.log1p(data.maximum)})` : undefined}}
              title={`输入 ${raw} → 最终 ${final}：${fmt(count)} 条`}
              onClick={() => onSelect(raw, final)}>{count ? fmt(count) : ''}</button>;
          })}
        </div>)}
      </div></div>
    </>}
  </div>;
}
