import React, { useMemo } from 'react';
import { LocalizationResult, StatusLevel } from '../types';

interface SpectrogramViewerProps {
  faultType: string;
  anomalyScore: number;
  threshold: number;
  status: StatusLevel;
  vehicleId: string;
  spectrogramUrl?: string;
  localization?: LocalizationResult;
}

const STATUS_BADGE: Record<StatusLevel, { label: string; cls: string }> = {
  normal:  { label: '정상', cls: 'bg-emerald-900 text-emerald-300 border border-emerald-700' },
  caution: { label: '주의', cls: 'bg-yellow-900  text-yellow-300  border border-yellow-700'  },
  warning: { label: '경고', cls: 'bg-red-900     text-red-300     border border-red-700'     },
};

const FAULT_KO: Record<string, string> = {
  ECC10:  '편심 10%',
  ECC20:  '편심 20%',
  DEMAG:  '감자(소자)',
  REDUC:  '감속기 결함',
  NORMAL: '정상',
};

/**
 * 모델이 없을 때만 사용하는 fallback: 결함 유형 + 이상 점수 기반 시뮬레이션
 */
function simulateCamBox(faultType: string, anomalyScore: number) {
  const faultBands: Record<string, { top: number; height: number; left: number; width: number }> = {
    ECC10:  { top: 10, height: 30, left: 15, width: 40 },
    ECC20:  { top:  5, height: 45, left: 10, width: 55 },
    DEMAG:  { top: 55, height: 25, left: 20, width: 35 },
    REDUC:  { top: 30, height: 30, left: 50, width: 40 },
    NORMAL: { top: 40, height: 20, left: 35, width: 25 },
  };
  const band      = faultBands[faultType] ?? faultBands['ECC10'];
  const intensity = Math.min(anomalyScore / 80, 1);
  const jitter    = (anomalyScore * 7.3) % 8 - 4;
  return {
    left:    Math.max(2, band.left   + jitter),
    top:     Math.max(2, band.top    + jitter * 0.5),
    width:   Math.min(90, band.width  + intensity * 15),
    height:  Math.min(85, band.height + intensity * 10),
    opacity: 0.5 + intensity * 0.45,
  };
}

export const SpectrogramViewer: React.FC<SpectrogramViewerProps> = ({
  faultType,
  anomalyScore,
  threshold,
  status,
  spectrogramUrl,
  localization,
}) => {
  const badge       = STATUS_BADGE[status];
  const isAnomalous = anomalyScore > threshold;
  const useModel    = localization?.model_available && localization?.has_bbox;

  // 원본 이미지 해상도 (2000×1280) 기준 SVG 좌표로 변환
  // SVG viewBox를 이미지와 동일하게 맞추면 object-contain letterbox와 자동 정렬됨
  const IMG_W = 2000;
  const IMG_H = 1280;

  const modelSvgBox = useMemo(() => {
    if (!useModel || !localization) return null;
    return {
      x:  localization.x1 * IMG_W,
      y:  localization.y1 * IMG_H,
      w:  (localization.x2 - localization.x1) * IMG_W,
      h:  (localization.y2 - localization.y1) * IMG_H,
      label: `${localization.predicted_class} ${((localization.probabilities[localization.predicted_class] ?? 0) * 100).toFixed(0)}%`,
      color: '#f97316',  // 주황 (모델)
    };
  }, [localization, useModel]);

  // fallback: 시뮬레이션 박스도 SVG 좌표로
  const simSvgBox = useMemo(() => {
    if (!isAnomalous) return null;
    const s = simulateCamBox(faultType, anomalyScore);
    return {
      x:     (s.left / 100) * IMG_W,
      y:     (s.top  / 100) * IMG_H,
      w:     (s.width  / 100) * IMG_W,
      h:     (s.height / 100) * IMG_H,
      label: faultType,
      color: '#ef4444',  // 빨강 (시뮬레이션)
    };
  }, [faultType, anomalyScore, isAnomalous]);

  const svgBox = modelSvgBox ?? simSvgBox;

  const predClass = localization?.predicted_class;
  const predProbs = localization?.probabilities ?? {};
  const topProb   = predClass ? (predProbs[predClass] ?? 0) : 0;

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-3">
      {/* 헤더 */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-slate-400 text-xs font-semibold uppercase tracking-wider">
            {useModel ? 'Grad-CAM (모델)' : 'Grad-CAM (시뮬레이션)'}
          </span>
          {useModel && (
            <span className="text-xs bg-blue-900 text-blue-300 border border-blue-700 px-1.5 py-0.5 rounded">
              AI 탐지
            </span>
          )}
        </div>
        <span className={`text-xs px-2 py-0.5 rounded-full font-semibold ${badge.cls}`}>
          {badge.label}
        </span>
      </div>

      {/* 스펙트로그램 + SVG 오버레이 */}
      <div className="relative w-full h-44 bg-slate-900 rounded border border-slate-700 overflow-hidden">
        {/* 이미지 */}
        {spectrogramUrl ? (
          <img
            src={spectrogramUrl}
            alt="STFT 스펙트로그램"
            className="absolute inset-0 w-full h-full object-contain"
          />
        ) : (
          <div
            className="absolute inset-0 opacity-60"
            style={{
              background: isAnomalous
                ? 'linear-gradient(135deg,#0f172a 0%,#2d1b69 40%,#dc2626 70%,#f59e0b 85%,#0f172a 100%)'
                : 'linear-gradient(135deg,#0f172a 0%,#0e7490 60%,#0369a1 80%,#1e3a5f 100%)',
            }}
          />
        )}

        {/*
          SVG 오버레이: viewBox를 원본 이미지(2000×1280)와 동일하게,
          preserveAspectRatio="xMidYMid meet" 으로 object-contain과 완벽히 정렬됨
          → bbox가 절대로 이미지 밖으로 나가지 않음
        */}
        {svgBox && (
          <svg
            className="absolute inset-0 w-full h-full"
            viewBox={`0 0 ${IMG_W} ${IMG_H}`}
            preserveAspectRatio="xMidYMid meet"
          >
            {/* 반투명 배경 채우기 */}
            <rect
              x={svgBox.x} y={svgBox.y}
              width={svgBox.w} height={svgBox.h}
              fill={svgBox.color}
              fillOpacity={0.12}
            />
            {/* 테두리 */}
            <rect
              x={svgBox.x} y={svgBox.y}
              width={svgBox.w} height={svgBox.h}
              fill="none"
              stroke={svgBox.color}
              strokeWidth={14}
              strokeOpacity={0.9}
              rx={6}
            />
            {/* 라벨 배경 */}
            <rect
              x={svgBox.x}
              y={Math.max(0, svgBox.y - 44)}
              width={svgBox.label.length * 14 + 12}
              height={40}
              fill={svgBox.color}
              rx={4}
            />
            {/* 라벨 텍스트 */}
            <text
              x={svgBox.x + 6}
              y={Math.max(0, svgBox.y - 44) + 28}
              fill="white"
              fontSize={28}
              fontWeight="bold"
              fontFamily="monospace"
            >
              {svgBox.label}
            </text>
          </svg>
        )}

        {/* 정상 메시지 */}
        {!svgBox && !isAnomalous && (
          <div className="absolute inset-0 flex items-center justify-center text-slate-500 text-xs">
            이상 없음
          </div>
        )}
      </div>

      {/* 하단 정보 */}
      <div className="mt-2 space-y-1">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-slate-500 text-xs">결함 유형</span>
            <span className={`text-xs font-semibold ${isAnomalous ? 'text-red-400' : 'text-emerald-400'}`}>
              {FAULT_KO[faultType] ?? faultType}
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-slate-500 text-xs">MSE</span>
            <span className={`font-mono text-xs font-bold ${isAnomalous ? 'text-red-400' : 'text-emerald-400'}`}>
              {anomalyScore.toFixed(2)}
            </span>
            <span className="text-slate-600 text-xs">/ {threshold.toFixed(1)}</span>
          </div>
        </div>

        {/* 모델 예측 확률 미니 바 */}
        {useModel && localization && (
          <div className="mt-1">
            <div className="text-slate-500 text-xs mb-0.5">모델 예측 확률</div>
            <div className="flex gap-0.5 h-3">
              {Object.entries(predProbs)
                .sort((a, b) => b[1] - a[1])
                .map(([cls, prob]) => (
                  <div
                    key={cls}
                    className="relative group flex-shrink-0 rounded-sm"
                    style={{
                      width:  `${prob * 100}%`,
                      background: cls === predClass ? '#f97316' : '#334155',
                    }}
                    title={`${cls}: ${(prob * 100).toFixed(1)}%`}
                  />
                ))}
            </div>
            <div className="text-slate-400 text-xs mt-0.5">
              예측: <span className="text-orange-400 font-semibold">{predClass}</span>
              {' '}({(topProb * 100).toFixed(1)}%)
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
