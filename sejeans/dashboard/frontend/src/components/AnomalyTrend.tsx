import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceLine,
  Tooltip,
  ResponsiveContainer,
  TooltipProps,
} from 'recharts';
import { TrendDataPoint } from '../types';

interface AnomalyTrendProps {
  data: TrendDataPoint[];
  threshold: number;
}

const CustomTooltip: React.FC<TooltipProps<number, string>> = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;

  const score = (payload[0]?.value ?? 0) as number;

  return (
    <div className="bg-slate-800 border border-slate-600 rounded px-3 py-2 text-xs shadow-lg">
      <div className="text-slate-400 mb-1">{label}</div>
      <div className="text-blue-400">
        이상 점수: <span className="font-mono font-bold">{score.toFixed(2)}</span>
      </div>
    </div>
  );
};

export const AnomalyTrend: React.FC<AnomalyTrendProps> = ({ data, threshold }) => {
  const isExceeded = data.length > 0 && data[data.length - 1].score > threshold;
  const latestScore = data.length > 0 ? data[data.length - 1].score : 0;

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-3 flex flex-col">
      <div className="flex items-center justify-between mb-2 flex-shrink-0">
        <span className="text-slate-400 text-xs font-semibold uppercase tracking-wider">
          이상 점수 트렌드
        </span>
        <div className="flex items-center gap-2">
          {isExceeded && (
            <span className="text-red-400 text-xs font-semibold animate-pulse">
              임계값 초과!
            </span>
          )}
          <span
            className={`font-mono text-sm font-bold ${
              isExceeded ? 'text-red-400' : 'text-emerald-400'
            }`}
          >
            {latestScore.toFixed(2)}
          </span>
        </div>
      </div>

      {data.length === 0 ? (
        <div className="flex items-center justify-center text-slate-600 text-xs" style={{ height: 160 }}>
          데이터 없음 — 데모를 시작하거나 데이터를 요청하세요
        </div>
      ) : (
        <div style={{ height: 160 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                dataKey="time"
                tick={{ fontSize: 9, fill: '#64748b' }}
                interval="preserveStartEnd"
                tickLine={false}
              />
              <YAxis
                tick={{ fontSize: 9, fill: '#64748b' }}
                tickLine={false}
                axisLine={false}
                domain={[0, 'auto']}
              />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine
                y={threshold}
                stroke="#ef4444"
                strokeDasharray="4 2"
                strokeWidth={1.5}
                label={{
                  value: `임계 ${threshold}`,
                  fill: '#ef4444',
                  fontSize: 9,
                  position: 'right',
                }}
              />
              <Line
                type="monotone"
                dataKey="score"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 3, fill: '#3b82f6' }}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="flex items-center gap-4 mt-2 flex-shrink-0">
        <div className="flex items-center gap-1.5">
          <div className="w-4 h-0.5 bg-blue-500" />
          <span className="text-slate-500 text-xs">이상 점수</span>
        </div>
        <div className="flex items-center gap-1.5">
          <svg width="16" height="4" className="flex-shrink-0">
            <line x1="0" y1="2" x2="16" y2="2" stroke="#ef4444" strokeWidth="2" strokeDasharray="4 2" />
          </svg>
          <span className="text-slate-500 text-xs">임계값 {threshold}</span>
        </div>
        <span className="text-slate-600 text-xs ml-auto">최근 {data.length}개</span>
      </div>
    </div>
  );
};
