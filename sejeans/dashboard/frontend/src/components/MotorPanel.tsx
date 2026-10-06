import React from 'react';
import { MotorData, TrendDataPoint } from '../types';
import { SensorGauge } from './SensorGauge';
import { SpectrogramViewer } from './SpectrogramViewer';
import { AnomalyTrend } from './AnomalyTrend';

interface MotorPanelProps {
  motor: MotorData;
  vehicleId: string;
  trendData: TrendDataPoint[];
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

export const MotorPanel: React.FC<MotorPanelProps> = ({ motor, vehicleId, trendData }) => {
  const style = STATUS_HEADER[motor.status];

  return (
    <div className="flex flex-col gap-2 pb-2">
      {/* Panel Header */}
      <div className={`bg-slate-800 rounded-lg border ${style.border} px-3 py-2 flex items-center justify-between flex-shrink-0`}>
        <div className="flex items-center gap-2">
          <span className="text-lg">⚙️</span>
          <span className="text-white font-semibold text-sm">모터-감속기</span>
        </div>
        <div className="flex items-center gap-2">
          <span className={`inline-block w-2 h-2 rounded-full ${style.dot} ${motor.status === 'warning' ? 'animate-pulse' : ''}`} />
          <span className={`text-xs font-semibold ${style.text}`}>
            {motor.status === 'normal' ? '정상' : motor.status === 'caution' ? '주의' : '경고'}
          </span>
          <span className="text-slate-500 text-xs">|</span>
          <span className="text-slate-400 text-xs">MSE</span>
          <span className={`font-mono text-sm font-bold ${style.text}`}>
            {motor.anomaly_score.toFixed(1)}
          </span>
        </div>
      </div>

      {/* Sensor Gauge */}
      <div className="flex-shrink-0">
        <SensorGauge scores={motor.sensor_scores} />
      </div>

      {/* Spectrogram */}
      <div className="flex-shrink-0">
        <SpectrogramViewer
          faultType={motor.fault_type}
          anomalyScore={motor.anomaly_score}
          threshold={motor.threshold}
          status={motor.status}
          vehicleId={vehicleId}
          spectrogramUrl={motor.spectrogram_url}
          localization={motor.localization}
        />
      </div>

      {/* Anomaly Trend */}
      <div>
        <AnomalyTrend data={trendData} threshold={motor.threshold} />
      </div>
    </div>
  );
};
