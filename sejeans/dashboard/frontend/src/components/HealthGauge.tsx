import React from 'react';
import { StatusLevel } from '../types';

interface HealthGaugeProps {
  value: number; // 0–100
  status: StatusLevel;
}

const STATUS_INFO: Record<StatusLevel, { label: string; color: string }> = {
  normal: { label: '정상', color: '#10b981' },
  caution: { label: '주의', color: '#f59e0b' },
  warning: { label: '경고', color: '#ef4444' },
};

function getStatusFromValue(value: number): StatusLevel {
  if (value >= 80) return 'normal';
  if (value >= 60) return 'caution';
  return 'warning';
}

/** Describe an SVG arc from angle a1 to a2 (radians, 0 = right, PI = left). */
function describeArc(cx: number, cy: number, r: number, a1: number, a2: number): string {
  const x1 = cx + r * Math.cos(a1);
  const y1 = cy - r * Math.sin(a1);
  const x2 = cx + r * Math.cos(a2);
  const y2 = cy - r * Math.sin(a2);
  const sweep = a1 > a2 ? 1 : 0; // clockwise when going from larger to smaller angle
  const large = Math.abs(a1 - a2) > Math.PI ? 1 : 0;
  return `M ${x1} ${y1} A ${r} ${r} 0 ${large} ${sweep} ${x2} ${y2}`;
}

export const HealthGauge: React.FC<HealthGaugeProps> = ({ value, status }) => {
  const derivedStatus = getStatusFromValue(value);
  const activeStatus: StatusLevel = status || derivedStatus;
  const info = STATUS_INFO[activeStatus];

  const cx = 100;
  const cy = 90;
  const r = 70;
  const sw = 12; // stroke width

  // Semicircle: from PI (left) to 0 (right), top half
  const bgPath = describeArc(cx, cy, r, Math.PI, 0);

  // Value arc: from PI to the angle corresponding to value
  const valueAngle = Math.PI * (1 - value / 100);
  const valuePath = value > 0 ? describeArc(cx, cy, r, Math.PI, valueAngle) : '';

  // Zone arcs (background hints)
  const redPath = describeArc(cx, cy, r, Math.PI, Math.PI * 0.4);       // 0-60
  const yellowPath = describeArc(cx, cy, r, Math.PI * 0.4, Math.PI * 0.2); // 60-80
  const greenPath = describeArc(cx, cy, r, Math.PI * 0.2, 0);           // 80-100

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-3">
      <div className="flex items-center gap-2 mb-1">
        <span className="text-slate-400 text-xs font-semibold uppercase tracking-wider">
          건강 지수 (Health Index)
        </span>
      </div>

      <div className="flex flex-col items-center">
        <svg viewBox="0 0 200 115" width="220" height="126" className="block mx-auto">
          {/* Background track */}
          <path d={bgPath} fill="none" stroke="#1e293b" strokeWidth={sw} strokeLinecap="round" />

          {/* Zone hint arcs */}
          <path d={redPath} fill="none" stroke="#7f1d1d" strokeWidth={sw - 3} strokeLinecap="butt" opacity="0.35" />
          <path d={yellowPath} fill="none" stroke="#78350f" strokeWidth={sw - 3} strokeLinecap="butt" opacity="0.35" />
          <path d={greenPath} fill="none" stroke="#14532d" strokeWidth={sw - 3} strokeLinecap="butt" opacity="0.35" />

          {/* Value fill */}
          {value > 0 && (
            <path
              d={valuePath}
              fill="none"
              stroke={info.color}
              strokeWidth={sw}
              strokeLinecap="round"
              style={{ filter: `drop-shadow(0 0 6px ${info.color})` }}
            />
          )}

          {/* Tick marks */}
          {[0, 20, 40, 60, 80, 100].map((tick) => {
            const angle = Math.PI * (1 - tick / 100);
            const inner = r - sw / 2 - 3;
            const outer = r + sw / 2 + 3;
            return (
              <line
                key={tick}
                x1={cx + inner * Math.cos(angle)}
                y1={cy - inner * Math.sin(angle)}
                x2={cx + outer * Math.cos(angle)}
                y2={cy - outer * Math.sin(angle)}
                stroke="#475569"
                strokeWidth={1}
              />
            );
          })}

          {/* Center value */}
          <text x={cx} y={cy - 10} textAnchor="middle" fontSize="28" fontWeight="700" fill={info.color} fontFamily="'JetBrains Mono', monospace">
            {value.toFixed(0)}
          </text>
          <text x={cx} y={cy + 6} textAnchor="middle" fontSize="11" fill="#64748b">
            / 100
          </text>

          {/* Min/Max labels */}
          <text x="18" y={cy + 14} fontSize="10" fill="#475569" textAnchor="middle">0</text>
          <text x="182" y={cy + 14} fontSize="10" fill="#475569" textAnchor="middle">100</text>
        </svg>

        {/* Status badge */}
        <div className="flex items-center gap-1.5 mt-1">
          <span
            className="inline-block w-2.5 h-2.5 rounded-full"
            style={{ backgroundColor: info.color, boxShadow: `0 0 6px ${info.color}` }}
          />
          <span className="text-sm font-semibold" style={{ color: info.color }}>
            {info.label}
          </span>
        </div>
      </div>
    </div>
  );
};
