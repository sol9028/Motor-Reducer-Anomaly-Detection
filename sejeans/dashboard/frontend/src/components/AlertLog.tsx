import React, { useState } from 'react';
import { Alert } from '../types';

interface AlertLogProps {
  alerts: Alert[];
}

type FilterLevel = 'all' | 'info' | 'caution' | 'warning';

const LEVEL_BADGE: Record<string, { label: string; cls: string }> = {
  info: { label: '정보', cls: 'bg-blue-900 text-blue-300 border border-blue-700' },
  caution: { label: '주의', cls: 'bg-yellow-900 text-yellow-300 border border-yellow-700' },
  warning: { label: '경고', cls: 'bg-red-900 text-red-300 border border-red-700' },
};

function formatTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString('ko-KR', { hour12: false });
  } catch {
    return iso;
  }
}

export const AlertLog: React.FC<AlertLogProps> = ({ alerts }) => {
  const [filter, setFilter] = useState<FilterLevel>('all');

  const filtered = filter === 'all'
    ? alerts
    : alerts.filter((a) => a.level === filter);

  const warningCount = alerts.filter((a) => a.level === 'warning').length;
  const cautionCount = alerts.filter((a) => a.level === 'caution').length;

  return (
    <div className="bg-slate-900 border-t border-slate-700 flex flex-col flex-shrink-0" style={{ height: '160px' }}>
      {/* Log header */}
      <div className="flex items-center justify-between px-4 py-1.5 border-b border-slate-800 flex-shrink-0">
        <div className="flex items-center gap-3">
          <span className="text-slate-300 text-xs font-semibold">알림 로그</span>
          <span className="text-slate-600 text-xs">(실시간)</span>
          {warningCount > 0 && (
            <span className="bg-red-900 text-red-300 text-xs px-1.5 py-0.5 rounded-full border border-red-700 font-semibold animate-pulse">
              경고 {warningCount}
            </span>
          )}
          {cautionCount > 0 && (
            <span className="bg-yellow-900 text-yellow-300 text-xs px-1.5 py-0.5 rounded-full border border-yellow-700 font-semibold">
              주의 {cautionCount}
            </span>
          )}
        </div>

        {/* Filter buttons */}
        <div className="flex items-center gap-1">
          {(['all', 'info', 'caution', 'warning'] as FilterLevel[]).map((level) => (
            <button
              key={level}
              onClick={() => setFilter(level)}
              className={`text-xs px-2 py-0.5 rounded transition-colors ${
                filter === level
                  ? 'bg-slate-600 text-white'
                  : 'text-slate-500 hover:text-slate-300'
              }`}
            >
              {level === 'all' ? '전체' : level === 'info' ? '정보' : level === 'caution' ? '주의' : '경고'}
            </button>
          ))}
          <span className="text-slate-600 text-xs ml-2">{filtered.length}건</span>
        </div>
      </div>

      {/* Log entries */}
      <div className="flex-1 overflow-y-auto">
        {filtered.length === 0 ? (
          <div className="flex items-center justify-center h-full text-slate-600 text-xs">
            알림 없음
          </div>
        ) : (
          <table className="w-full text-xs">
            <tbody>
              {filtered.map((alert, idx) => {
                const badge = LEVEL_BADGE[alert.level] ?? LEVEL_BADGE.info;
                return (
                  <tr
                    key={alert.id ?? idx}
                    className={`border-b border-slate-800 hover:bg-slate-800 transition-colors ${
                      idx === 0 && alert.level === 'warning' ? 'bg-red-950' : ''
                    }`}
                  >
                    <td className="px-4 py-1 text-slate-500 font-mono whitespace-nowrap w-24">
                      {formatTimestamp(alert.timestamp)}
                    </td>
                    <td className="px-2 py-1 w-14">
                      <span className={`px-1.5 py-0.5 rounded text-xs font-semibold ${badge.cls}`}>
                        {badge.label}
                      </span>
                    </td>
                    <td className="px-2 py-1 text-blue-400 font-mono whitespace-nowrap w-28">
                      {alert.vehicle_id}
                    </td>
                    <td className="px-2 py-1 text-slate-300">
                      {alert.message}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};
