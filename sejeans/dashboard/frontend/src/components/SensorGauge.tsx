import React from 'react';
import { SensorScores, StatusLevel } from '../types';

interface SensorGaugeProps {
  scores: SensorScores;
}

interface GaugeBarProps {
  label: string;
  value: number;
  max?: number;
}

function getStatusFromScore(value: number): StatusLevel {
  if (value < 30) return 'normal';
  if (value <= 70) return 'caution';
  return 'warning';
}

function getBarColor(value: number): string {
  if (value < 30) return 'bg-emerald-500';
  if (value <= 70) return 'bg-yellow-500';
  return 'bg-red-500';
}

function getTextColor(value: number): string {
  if (value < 30) return 'text-emerald-400';
  if (value <= 70) return 'text-yellow-400';
  return 'text-red-400';
}

const STATUS_LABELS: Record<StatusLevel, string> = {
  normal: '정상',
  caution: '주의',
  warning: '경고',
};

const GaugeBar: React.FC<GaugeBarProps> = ({ label, value, max = 100 }) => {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  const status = getStatusFromScore(value);

  return (
    <div className="flex items-center gap-2">
      <span className="text-slate-400 text-xs w-20 flex-shrink-0">{label}</span>
      <div className="flex-1 bg-slate-700 rounded-full h-2.5 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${getBarColor(value)}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className={`text-xs font-mono font-semibold w-7 text-right ${getTextColor(value)}`}>
        {value.toFixed(0)}
      </span>
      <span
        className={`text-xs w-8 flex-shrink-0 ${
          status === 'normal'
            ? 'text-emerald-400'
            : status === 'caution'
            ? 'text-yellow-400'
            : 'text-red-400'
        }`}
      >
        {STATUS_LABELS[status]}
      </span>
    </div>
  );
};

export const SensorGauge: React.FC<SensorGaugeProps> = ({ scores }) => {
  const sensors = [
    { label: '모터 U상', value: scores.current_u },
    { label: '감속기', value: scores.vib_motor },
    { label: '구동 시스템', value: scores.vib_tm },
  ];

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-3">
      <div className="flex items-center gap-2 mb-3">
        <span className="text-slate-400 text-xs font-semibold uppercase tracking-wider">
          부품 상태 게이지
        </span>
      </div>
      <div className="space-y-2.5">
        {sensors.map((sensor) => (
          <GaugeBar key={sensor.label} label={sensor.label} value={sensor.value} />
        ))}
      </div>
      {/* Scale labels */}
      <div className="mt-2 flex text-slate-600 text-xs gap-0">
        <span className="ml-[5.5rem] mr-auto">0</span>
        <span className="mr-[10%]">30</span>
        <span className="mr-[7%]">70</span>
        <span>100</span>
      </div>
    </div>
  );
};
