import React from 'react';
import { BatteryMetricsData } from '../types';

interface BatteryMetricsProps {
  metrics: BatteryMetricsData;
}

interface MetricCardProps {
  label: string;
  value: string;
  unit: string;
  alert?: boolean;
}

const MetricCard: React.FC<MetricCardProps> = ({ label, value, unit, alert }) => (
  <div className={`bg-slate-900 rounded border ${alert ? 'border-red-800' : 'border-slate-700'} px-3 py-2`}>
    <div className="text-slate-500 text-xs mb-1">{label}</div>
    <div className="flex items-baseline gap-1">
      <span className={`font-mono text-lg font-bold ${alert ? 'text-red-400' : 'text-slate-100'}`}>
        {value}
      </span>
      <span className="text-slate-500 text-xs">{unit}</span>
    </div>
  </div>
);

export const BatteryMetrics: React.FC<BatteryMetricsProps> = ({ metrics }) => {
  const voltageAlert = metrics.voltage_deviation > 0.01;
  const tempAlert = metrics.temperature !== null && metrics.temperature > 50;

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-3">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-slate-400 text-xs font-semibold uppercase tracking-wider">
          배터리 수치
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <MetricCard
          label="셀 전압 편차"
          value={metrics.voltage_deviation.toFixed(3)}
          unit="V"
          alert={voltageAlert}
        />
        <MetricCard
          label="최저 셀 전압"
          value={metrics.min_cell_voltage.toFixed(2)}
          unit="V"
        />
        <MetricCard
          label="온도"
          value={metrics.temperature !== null ? metrics.temperature.toFixed(1) : '--'}
          unit="°C"
          alert={tempAlert}
        />
        <MetricCard
          label="셀 개수"
          value={metrics.cell_count.toString()}
          unit="셀"
        />
      </div>

      {(voltageAlert || tempAlert) && (
        <div className="mt-2 text-red-400 text-xs flex items-center gap-1">
          <span className="animate-pulse">⚠</span>
          <span>
            {voltageAlert && '셀 편차 임계 초과'}
            {voltageAlert && tempAlert && ' · '}
            {tempAlert && '과온 감지'}
          </span>
        </div>
      )}
    </div>
  );
};
