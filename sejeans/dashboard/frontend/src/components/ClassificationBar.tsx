import React from 'react';
import { BatteryProbabilities, FaultProbabilities } from '../types';

interface ClassificationBarProps {
  probabilities: BatteryProbabilities;
  classification: string;
  faultProbabilities: FaultProbabilities;
  faultType: string;
}

const FAULT_LABEL: Record<string, string> = {
  none:             '결함 없음',
  cell_voltage:     '셀전압 결함',
  cell_deviation:   '셀편차 결함',
};

const FAULT_COLOR: Record<string, string> = {
  none:           'text-emerald-400',
  cell_voltage:   'text-orange-400',
  cell_deviation: 'text-purple-400',
};

export const ClassificationBar: React.FC<ClassificationBarProps> = ({
  probabilities,
  classification,
  faultProbabilities,
  faultType,
}) => {
  // ── 상태 확률 (대소문자 모두 허용) ────────────────────────────────
  const p = probabilities as Record<string, number>;
  const pNormal  = p['normal']  ?? p['NORMAL']  ?? 0;
  const pCaution = p['caution'] ?? p['CAUTION'] ?? 0;
  const pDefect  = p['defect']  ?? p['DEFECT']  ?? 0;
  const total    = pNormal + pCaution + pDefect || 1;
  const normalPct  = (pNormal  / total) * 100;
  const cautionPct = (pCaution / total) * 100;
  const defectPct  = (pDefect  / total) * 100;

  // ── 결함 유형 확률 (undefined 방어) ───────────────────────────────
  const fp     = faultProbabilities ?? { none: 1, cell_voltage: 0, cell_deviation: 0 };
  const fTotal = (fp.none + fp.cell_voltage + fp.cell_deviation) || 1;
  const nonePct  = (fp.none           / fTotal) * 100;
  const cvPct    = (fp.cell_voltage   / fTotal) * 100;
  const cdPct    = (fp.cell_deviation / fTotal) * 100;

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-3 flex flex-col gap-3">

      {/* ── 상태 분류 ── */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-slate-400 text-xs font-semibold uppercase tracking-wider">
            상태 분류
          </span>
          <span className="text-xs font-semibold text-slate-300 bg-slate-700 px-2 py-0.5 rounded">
            {classification}
          </span>
        </div>

        <div className="w-full h-5 rounded-full overflow-hidden flex">
          <div
            className="bg-emerald-600 flex items-center justify-center transition-all duration-500"
            style={{ width: `${normalPct}%` }}
          >
            {normalPct >= 10 && (
              <span className="text-white text-xs font-semibold">{Math.round(normalPct)}%</span>
            )}
          </div>
          <div
            className="bg-yellow-500 flex items-center justify-center transition-all duration-500"
            style={{ width: `${cautionPct}%` }}
          >
            {cautionPct >= 10 && (
              <span className="text-white text-xs font-semibold">{Math.round(cautionPct)}%</span>
            )}
          </div>
          <div
            className="bg-red-600 flex items-center justify-center transition-all duration-500"
            style={{ width: `${defectPct}%` }}
          >
            {defectPct >= 10 && (
              <span className="text-white text-xs font-semibold">{Math.round(defectPct)}%</span>
            )}
          </div>
          <div className="flex-1 bg-slate-700" />
        </div>

        <div className="flex justify-between mt-1.5">
          <div className="flex items-center gap-1">
            <div className="w-2.5 h-2.5 rounded-sm bg-emerald-600" />
            <span className="text-slate-400 text-xs">정상</span>
            <span className="text-emerald-400 text-xs font-mono font-bold ml-1">
              {normalPct.toFixed(1)}%
            </span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2.5 h-2.5 rounded-sm bg-yellow-500" />
            <span className="text-slate-400 text-xs">주의</span>
            <span className="text-yellow-400 text-xs font-mono font-bold ml-1">
              {cautionPct.toFixed(1)}%
            </span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2.5 h-2.5 rounded-sm bg-red-600" />
            <span className="text-slate-400 text-xs">결함</span>
            <span className="text-red-400 text-xs font-mono font-bold ml-1">
              {defectPct.toFixed(1)}%
            </span>
          </div>
        </div>
      </div>

      {/* 구분선 */}
      <div className="border-t border-slate-700" />

      {/* ── 결함 유형 분류 ── */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-slate-400 text-xs font-semibold uppercase tracking-wider">
            결함 유형
          </span>
          <span className={`text-xs font-semibold px-2 py-0.5 rounded bg-slate-700 ${FAULT_COLOR[faultType] ?? 'text-slate-300'}`}>
            {FAULT_LABEL[faultType] ?? faultType}
          </span>
        </div>

        <div className="w-full h-5 rounded-full overflow-hidden flex">
          <div
            className="bg-emerald-700 flex items-center justify-center transition-all duration-500"
            style={{ width: `${nonePct}%` }}
          >
            {nonePct >= 12 && (
              <span className="text-white text-xs font-semibold">{Math.round(nonePct)}%</span>
            )}
          </div>
          <div
            className="bg-orange-500 flex items-center justify-center transition-all duration-500"
            style={{ width: `${cvPct}%` }}
          >
            {cvPct >= 12 && (
              <span className="text-white text-xs font-semibold">{Math.round(cvPct)}%</span>
            )}
          </div>
          <div
            className="bg-purple-600 flex items-center justify-center transition-all duration-500"
            style={{ width: `${cdPct}%` }}
          >
            {cdPct >= 12 && (
              <span className="text-white text-xs font-semibold">{Math.round(cdPct)}%</span>
            )}
          </div>
          <div className="flex-1 bg-slate-700" />
        </div>

        <div className="flex justify-between mt-1.5">
          <div className="flex items-center gap-1">
            <div className="w-2.5 h-2.5 rounded-sm bg-emerald-700" />
            <span className="text-slate-400 text-xs">정상</span>
            <span className="text-emerald-400 text-xs font-mono font-bold ml-1">
              {nonePct.toFixed(1)}%
            </span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2.5 h-2.5 rounded-sm bg-orange-500" />
            <span className="text-slate-400 text-xs">셀전압</span>
            <span className="text-orange-400 text-xs font-mono font-bold ml-1">
              {cvPct.toFixed(1)}%
            </span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2.5 h-2.5 rounded-sm bg-purple-600" />
            <span className="text-slate-400 text-xs">셀편차</span>
            <span className="text-purple-400 text-xs font-mono font-bold ml-1">
              {cdPct.toFixed(1)}%
            </span>
          </div>
        </div>
      </div>

    </div>
  );
};
