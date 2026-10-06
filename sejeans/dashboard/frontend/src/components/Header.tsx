import React, { useState, useEffect } from 'react';
import { CarModel, StatusLevel } from '../types';

interface HeaderProps {
  selectedCar: CarModel;
  onCarChange: (car: CarModel) => void;
  overallStatus: StatusLevel;
  fps: number;
  wsStatus: 'disconnected' | 'connecting' | 'connected' | 'error';
  onConnect: () => void;
  onDisconnect: () => void;
  onFetchSample: () => void;
}

const CAR_MODELS: CarModel[] = ['IONIQ', 'KONA', 'NIRO'];

const STATUS_LABELS: Record<StatusLevel, string> = {
  normal: '정상',
  caution: '주의',
  warning: '경고',
};

const STATUS_COLORS: Record<StatusLevel, string> = {
  normal: 'text-emerald-400',
  caution: 'text-yellow-400',
  warning: 'text-red-400',
};

const STATUS_DOT_COLORS: Record<StatusLevel, string> = {
  normal: 'bg-emerald-400',
  caution: 'bg-yellow-400',
  warning: 'bg-red-400',
};

const WS_STATUS_LABELS = {
  disconnected: '연결 끊김',
  connecting: '연결 중...',
  connected: '연결됨',
  error: '오류',
};

const WS_STATUS_COLORS = {
  disconnected: 'text-slate-400',
  connecting: 'text-yellow-400',
  connected: 'text-emerald-400',
  error: 'text-red-400',
};

export const Header: React.FC<HeaderProps> = ({
  selectedCar,
  onCarChange,
  overallStatus,
  fps,
  wsStatus,
  onConnect,
  onDisconnect,
  onFetchSample,
}) => {
  const [currentTime, setCurrentTime] = useState(() =>
    new Date().toLocaleTimeString('ko-KR', { hour12: false })
  );

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date().toLocaleTimeString('ko-KR', { hour12: false }));
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <header className="bg-slate-900 border-b border-slate-700 px-4 py-2 flex items-center justify-between flex-shrink-0">
      {/* Left: Logo + Car selector */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-blue-600 rounded flex items-center justify-center">
            <span className="text-white text-xs font-bold">EV</span>
          </div>
          <span className="text-white font-semibold text-sm tracking-wide">이상 탐지 시스템</span>
        </div>

        <div className="h-4 w-px bg-slate-600" />

        {/* Car model selector */}
        <div className="flex items-center gap-2">
          <span className="text-slate-400 text-xs">차량 선택</span>
          <select
            value={selectedCar}
            onChange={(e) => onCarChange(e.target.value as CarModel)}
            className="bg-slate-800 border border-slate-600 text-white text-sm rounded px-2 py-1 focus:outline-none focus:border-blue-500 cursor-pointer"
          >
            {CAR_MODELS.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Center: Status indicators */}
      <div className="flex items-center gap-6">
        {/* Overall status */}
        <div className="flex items-center gap-2">
          <span className="text-slate-400 text-xs">전체 상태</span>
          <div className="flex items-center gap-1.5">
            <span
              className={`inline-block w-2.5 h-2.5 rounded-full ${STATUS_DOT_COLORS[overallStatus]} ${
                overallStatus === 'warning' ? 'animate-pulse' : ''
              }`}
            />
            <span className={`text-sm font-semibold ${STATUS_COLORS[overallStatus]}`}>
              {STATUS_LABELS[overallStatus]}
            </span>
          </div>
        </div>

        {/* FPS */}
        <div className="flex items-center gap-1.5">
          <span className="text-slate-400 text-xs">FPS</span>
          <span className="text-slate-200 text-sm font-mono font-semibold">
            {fps.toFixed(1)}
          </span>
        </div>

        {/* WebSocket status */}
        <div className="flex items-center gap-1.5">
          <span className="text-slate-400 text-xs">WebSocket</span>
          <span className={`text-xs font-semibold ${WS_STATUS_COLORS[wsStatus]}`}>
            {WS_STATUS_LABELS[wsStatus]}
          </span>
        </div>
      </div>

      {/* Right: Controls + Clock */}
      <div className="flex items-center gap-3">
        {/* Control buttons */}
        <button
          onClick={onFetchSample}
          className="bg-slate-700 hover:bg-slate-600 text-slate-200 text-xs px-3 py-1.5 rounded border border-slate-600 transition-colors"
        >
          데이터 요청
        </button>

        {wsStatus === 'connected' ? (
          <button
            onClick={onDisconnect}
            className="bg-red-700 hover:bg-red-600 text-white text-xs px-3 py-1.5 rounded transition-colors"
          >
            스트리밍 중지
          </button>
        ) : (
          <button
            onClick={onConnect}
            disabled={wsStatus === 'connecting'}
            className="bg-blue-600 hover:bg-blue-500 disabled:bg-blue-800 disabled:cursor-not-allowed text-white text-xs px-3 py-1.5 rounded transition-colors"
          >
            {wsStatus === 'connecting' ? '연결 중...' : '데모 시작'}
          </button>
        )}

        <div className="h-4 w-px bg-slate-600" />

        {/* Clock */}
        <span className="text-slate-300 text-sm font-mono">{currentTime}</span>
      </div>
    </header>
  );
};
