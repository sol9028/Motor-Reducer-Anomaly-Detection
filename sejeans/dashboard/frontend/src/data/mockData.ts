import { InferenceResult, Alert } from '../types';

export const MOCK_INFERENCE: InferenceResult = {
  timestamp: new Date().toISOString(),
  vehicle_id: 'MOB-X001',
  car_model: 'IONIQ',
  motor: {
    anomaly_score: 27.3,
    threshold: 10.0,
    status: 'warning',
    fault_type: '편심 결함',
    sensor_scores: {
      current_u: 73,
      vib_motor: 61,
      vib_tm: 45,
    },
  },
  battery: {
    soh_proxy: 78,
    status: 'normal',
    classification: '정상',
    probabilities: {
      normal: 0.72,
      caution: 0.21,
      defect: 0.07,
    },
    fault_type: 'none',
    fault_probabilities: {
      none: 0.85,
      cell_voltage: 0.10,
      cell_deviation: 0.05,
    },
    faulty_cell_indices: [46, 47, 71],
    cell_scores: Array.from({ length: 96 }, (_, i) => {
      if ([46, 47, 71].includes(i)) return 0.8 + Math.random() * 0.2;
      return Math.random() * 0.3;
    }),
    metrics: {
      min_cell_voltage: 3.91,
      voltage_deviation: 0.012,
      temperature: 45.8,
      cell_count: 96,
      cell_mean_voltages: Array.from({ length: 96 }, () => 3.9 + Math.random() * 0.2),
    },
  },
};

export const MOCK_ALERTS: Alert[] = [
  {
    id: '1',
    timestamp: new Date(Date.now() - 0).toISOString(),
    level: 'warning',
    vehicle_id: 'MOB-X001',
    message: '모터 이상: MSE 27.3 (임계값 초과)',
  },
  {
    id: '2',
    timestamp: new Date(Date.now() - 30000).toISOString(),
    level: 'caution',
    vehicle_id: 'MOB-X002',
    message: '배터리 셀 편차 증가: 0.015 V',
  },
  {
    id: '3',
    timestamp: new Date(Date.now() - 120000).toISOString(),
    level: 'info',
    vehicle_id: 'MOB-X001',
    message: '시스템 진단 완료 — 정상 범위',
  },
  {
    id: '4',
    timestamp: new Date(Date.now() - 300000).toISOString(),
    level: 'caution',
    vehicle_id: 'MOB-X003',
    message: '모터 진동 센서 주의: vib_motor 61',
  },
  {
    id: '5',
    timestamp: new Date(Date.now() - 600000).toISOString(),
    level: 'info',
    vehicle_id: 'MOB-X002',
    message: '데이터 수집 시작',
  },
];
