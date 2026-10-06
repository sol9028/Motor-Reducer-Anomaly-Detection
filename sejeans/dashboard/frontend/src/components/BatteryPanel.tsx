import React from 'react';
import { BatteryData } from '../types';
import { HealthGauge } from './HealthGauge';
import { CellGrid } from './CellGrid';
import { ClassificationBar } from './ClassificationBar';
import { BatteryMetrics } from './BatteryMetrics';

interface BatteryPanelProps {
  battery: BatteryData;
}

const STATUS_HEADER: Record<string, { dot: string; text: string; border: string }> = {
  normal: {
    dot: 'bg-emerald-400',
    text: 'text-emerald-400',
    border: 'border-emerald-800',
  },
  caution: {
    dot: 'bg-yellow-400',
    text: 'text-yellow-400',
    border: 'border-yellow-800',
  },
  warning: {
    dot: 'bg-red-400',
    text: 'text-red-400',
    border: 'border-red-800',
  },
};

const FAULT_BADGE: Record<string, { label: string; color: string }> = {
  none:           { label: '결함없음',   color: 'text-emerald-400 bg-emerald-950 border-emerald-800' },
  cell_voltage:   { label: '셀전압결함', color: 'text-orange-400  bg-orange-950  border-orange-800'  },
  cell_deviation: { label: '셀편차결함', color: 'text-purple-400  bg-purple-950  border-purple-800'  },
};

export const BatteryPanel: React.FC<BatteryPanelProps> = ({ battery }) => {
  const style = STATUS_HEADER[battery.status];
  const faultBadge = FAULT_BADGE[battery.fault_type] ?? FAULT_BADGE['none'];

  return (
    <div className="flex flex-col gap-2 pb-2">
      {/* Panel Header */}
      <div className={`bg-slate-800 rounded-lg border ${style.border} px-3 py-2 flex items-center justify-between flex-shrink-0`}>
        <div className="flex items-center gap-2">
          <span className="text-lg">🔋</span>
          <span className="text-white font-semibold text-sm">배터리</span>
          <span className={`text-xs font-semibold px-1.5 py-0.5 rounded border ${faultBadge.color}`}>
            {faultBadge.label}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className={`inline-block w-2 h-2 rounded-full ${style.dot} ${battery.status === 'warning' ? 'animate-pulse' : ''}`} />
          <span className={`text-xs font-semibold ${style.text}`}>
            {battery.status === 'normal' ? '정상' : battery.status === 'caution' ? '주의' : '경고'}
          </span>
          <span className="text-slate-500 text-xs">|</span>
          <span className="text-slate-400 text-xs">SOH</span>
          <span className={`font-mono text-sm font-bold ${style.text}`}>
            {battery.soh_proxy.toFixed(0)}
          </span>
        </div>
      </div>

      {/* Health Gauge */}
      <div className="flex-shrink-0">
        <HealthGauge value={battery.soh_proxy} status={battery.status} />
      </div>

      {/* Cell Grid */}
      <div className="flex-shrink-0">
        <CellGrid
          cellScores={battery.cell_scores}
          faultyCellIndices={battery.faulty_cell_indices}
          cellMeanVoltages={battery.metrics.cell_mean_voltages}
          cellCount={battery.metrics.cell_count}
        />
      </div>

      {/* Classification Bar */}
      <div className="flex-shrink-0">
        <ClassificationBar
          probabilities={battery.probabilities}
          classification={battery.classification}
          faultProbabilities={battery.fault_probabilities}
          faultType={battery.fault_type}
        />
      </div>

      {/* Battery Metrics */}
      <div>
        <BatteryMetrics metrics={battery.metrics} />
      </div>
    </div>
  );
};
