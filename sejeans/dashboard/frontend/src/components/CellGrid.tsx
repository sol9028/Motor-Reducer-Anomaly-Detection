import React from 'react';

interface CellGridProps {
  cellScores: number[];
  faultyCellIndices: number[];
  cellMeanVoltages: number[];
  cellCount: number;
}

function getCellColor(score: number, isFaulty: boolean): string {
  if (isFaulty || score > 0.7) return 'bg-red-600 border-red-400';
  if (score > 0.4) return 'bg-orange-600 border-orange-400';
  if (score > 0.2) return 'bg-yellow-700 border-yellow-600';
  return 'bg-blue-900 border-blue-700';
}

function getCellTextColor(score: number, isFaulty: boolean): string {
  if (isFaulty || score > 0.7) return 'text-red-100';
  if (score > 0.4) return 'text-orange-100';
  if (score > 0.2) return 'text-yellow-100';
  return 'text-blue-300';
}

export const CellGrid: React.FC<CellGridProps> = ({
  cellScores,
  faultyCellIndices,
  cellMeanVoltages,
  cellCount,
}) => {
  // Determine grid layout — aim for roughly square-ish, prefer 12 cols
  const cols = 12;
  const rows = Math.ceil(cellCount / cols);

  const cells = Array.from({ length: cellCount }, (_, i) => ({
    index: i,
    score: cellScores[i] ?? 0,
    isFaulty: faultyCellIndices.includes(i),
    voltage: cellMeanVoltages[i] ?? 0,
  }));

  const faultyCount = faultyCellIndices.length;

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-slate-400 text-xs font-semibold uppercase tracking-wider">
          셀별 결함 시각화
        </span>
        {faultyCount > 0 && (
          <span className="text-red-400 text-xs font-semibold">
            결함 셀: {faultyCellIndices.map((i) => `#${i + 1}`).join(', ')}
          </span>
        )}
      </div>

      {/* Cell grid */}
      <div
        className="grid gap-0.5 relative"
        style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
      >
        {cells.map((cell) => (
          <div
            key={cell.index}
            className={`
              relative aspect-square rounded-sm border cursor-pointer
              flex items-center justify-center
              transition-transform hover:scale-110 hover:z-10
              ${getCellColor(cell.score, cell.isFaulty)}
              ${cell.isFaulty ? 'ring-1 ring-red-400 ring-offset-0' : ''}
            `}
            title={`셀 #${cell.index + 1} | 점수: ${cell.score.toFixed(3)} | 전압: ${cell.voltage.toFixed(3)} V${cell.isFaulty ? ' | 결함' : ''}`}
          >
            <span
              className={`text-center leading-none select-none ${getCellTextColor(cell.score, cell.isFaulty)}`}
              style={{ fontSize: '6px' }}
            >
              {cell.index + 1}
            </span>
          </div>
        ))}
      </div>

      {/* Row labels hint */}
      <div className="mt-2 flex items-center gap-3 flex-wrap">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-sm bg-blue-900 border border-blue-700" />
          <span className="text-slate-500 text-xs">정상</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-sm bg-yellow-700 border border-yellow-600" />
          <span className="text-slate-500 text-xs">주의</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-sm bg-orange-600 border border-orange-400" />
          <span className="text-slate-500 text-xs">위험</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-sm bg-red-600 border border-red-400 ring-1 ring-red-400" />
          <span className="text-slate-500 text-xs">결함</span>
        </div>
        <span className="text-slate-600 text-xs ml-auto">{rows}행 × {cols}열 = {cellCount}셀</span>
      </div>
    </div>
  );
};
