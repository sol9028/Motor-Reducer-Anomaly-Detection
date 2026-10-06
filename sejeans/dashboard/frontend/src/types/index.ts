export type CarModel = 'IONIQ' | 'KONA' | 'NIRO';
export type StatusLevel = 'normal' | 'caution' | 'warning';

export interface SensorScores {
  current_u: number;
  vib_motor: number;
  vib_tm: number;
}

export interface LocalizationResult {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  predicted_class: string;
  probabilities: Record<string, number>;
  has_bbox: boolean;
  model_available: boolean;
}

export interface MotorData {
  anomaly_score: number;
  threshold: number;
  status: StatusLevel;
  fault_type: string;
  sensor_scores: SensorScores;
  spectrogram_url?: string;
  localization?: LocalizationResult;
}

export interface BatteryMetricsData {
  min_cell_voltage: number;
  voltage_deviation: number;
  temperature: number | null;
  cell_count: number;
  cell_mean_voltages: number[];
}

export interface BatteryProbabilities {
  normal: number;
  caution: number;
  defect: number;
}

export interface FaultProbabilities {
  none: number;
  cell_voltage: number;
  cell_deviation: number;
}

export interface BatteryData {
  soh_proxy: number;
  status: StatusLevel;
  classification: string;
  probabilities: BatteryProbabilities;
  fault_type: string;
  fault_probabilities: FaultProbabilities;
  faulty_cell_indices: number[];
  cell_scores: number[];
  metrics: BatteryMetricsData;
}

export interface InferenceResult {
  timestamp: string;
  vehicle_id: string;
  car_model: CarModel;
  motor: MotorData;
  battery: BatteryData;
}

export interface Alert {
  id: string;
  timestamp: string;
  level: 'info' | 'caution' | 'warning';
  vehicle_id: string;
  message: string;
}

export interface TrendDataPoint {
  time: string;
  score: number;
  threshold: number;
}

export interface WebSocketMessage {
  type: 'inference' | 'alert' | 'ping';
  data?: InferenceResult;
  alert?: Alert;
}
