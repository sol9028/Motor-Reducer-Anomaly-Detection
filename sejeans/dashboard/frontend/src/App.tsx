import { useState, useCallback, useEffect } from 'react';
import { CarModel, InferenceResult, Alert, TrendDataPoint, StatusLevel } from './types';
import { useWebSocket } from './hooks/useWebSocket';
import { Header } from './components/Header';
import { MotorPanel } from './components/MotorPanel';
import { BatteryPanel } from './components/BatteryPanel';
import { AlertLog } from './components/AlertLog';
import { MOCK_INFERENCE, MOCK_ALERTS } from './data/mockData';

const MAX_TREND_POINTS = 30;
const API_BASE = '/api';

function computeOverallStatus(data: InferenceResult | null): StatusLevel {
  if (!data) return 'normal';
  const levels: StatusLevel[] = [data.motor.status, data.battery.status];
  if (levels.includes('warning')) return 'warning';
  if (levels.includes('caution')) return 'caution';
  return 'normal';
}

function nowLabel(): string {
  return new Date().toLocaleTimeString('ko-KR', { hour12: false });
}

export default function App() {
  const [selectedCar, setSelectedCar] = useState<CarModel>('IONIQ');
  const [currentData, setCurrentData] = useState<InferenceResult | null>(MOCK_INFERENCE);
  const [localAlerts, setLocalAlerts] = useState<Alert[]>(MOCK_ALERTS);
  const [trendData, setTrendData] = useState<TrendDataPoint[]>([]);
  const [isFetching, setIsFetching] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);

  const { latestData, alerts: wsAlerts, status: wsStatus, fps, connect, disconnect } = useWebSocket();

  // Merge WebSocket data into local state
  useEffect(() => {
    if (!latestData) return;
    setCurrentData(latestData);

    // Update trend
    setTrendData((prev) => {
      const newPoint: TrendDataPoint = {
        time: nowLabel(),
        score: latestData.motor.anomaly_score,
        threshold: latestData.motor.threshold,
      };
      return [...prev, newPoint].slice(-MAX_TREND_POINTS);
    });
  }, [latestData]);

  // Merge WebSocket alerts
  useEffect(() => {
    if (wsAlerts.length === 0) return;
    setLocalAlerts((prev) => {
      const existingIds = new Set(prev.map((a) => a.id));
      const newOnes = wsAlerts.filter((a) => !existingIds.has(a.id));
      if (newOnes.length === 0) return prev;
      return [...newOnes, ...prev].slice(0, 100);
    });
  }, [wsAlerts]);

  // Fetch a single sample from the REST API
  const handleFetchSample = useCallback(async () => {
    setIsFetching(true);
    setFetchError(null);
    try {
      const res = await fetch(`${API_BASE}/demo/sample?car_model=${selectedCar}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: InferenceResult = await res.json();
      setCurrentData(data);

      setTrendData((prev) => {
        const newPoint: TrendDataPoint = {
          time: nowLabel(),
          score: data.motor.anomaly_score,
          threshold: data.motor.threshold,
        };
        return [...prev, newPoint].slice(-MAX_TREND_POINTS);
      });

      // Auto-add alert if anomalous
      if (data.motor.status !== 'normal' || data.battery.status !== 'normal') {
        const autoAlert: Alert = {
          id: `fetch-${Date.now()}`,
          timestamp: data.timestamp,
          level: data.motor.status === 'warning' || data.battery.status === 'warning' ? 'warning' : 'caution',
          vehicle_id: data.vehicle_id,
          message: `[수동 조회] ${data.motor.status !== 'normal'
            ? `모터: MSE ${data.motor.anomaly_score.toFixed(1)}`
            : `배터리 SOH ${data.battery.soh_proxy.toFixed(0)}`}`,
        };
        setLocalAlerts((prev) => [autoAlert, ...prev].slice(0, 100));
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : '알 수 없는 오류';
      setFetchError(msg);
      // Fall back to mock data variation
      const mockCopy: InferenceResult = {
        ...MOCK_INFERENCE,
        timestamp: new Date().toISOString(),
        car_model: selectedCar,
        motor: {
          ...MOCK_INFERENCE.motor,
          anomaly_score: 5 + Math.random() * 30,
          sensor_scores: {
            current_u: 20 + Math.random() * 70,
            vib_motor: 20 + Math.random() * 70,
            vib_tm: 20 + Math.random() * 70,
          },
        },
        battery: {
          ...MOCK_INFERENCE.battery,
          soh_proxy: 60 + Math.random() * 40,
        },
      };
      setCurrentData(mockCopy);
      setTrendData((prev) => {
        const newPoint: TrendDataPoint = {
          time: nowLabel(),
          score: mockCopy.motor.anomaly_score,
          threshold: mockCopy.motor.threshold,
        };
        return [...prev, newPoint].slice(-MAX_TREND_POINTS);
      });
    } finally {
      setIsFetching(false);
    }
  }, [selectedCar]);

  // On car change, reset to mock for that car
  const handleCarChange = useCallback((car: CarModel) => {
    setSelectedCar(car);
    setCurrentData({ ...MOCK_INFERENCE, car_model: car });
    setTrendData([]);
  }, []);

  const overallStatus = computeOverallStatus(currentData);

  return (
    <div className="flex flex-col h-screen bg-dash-bg text-white overflow-hidden font-sans">
      {/* Header */}
      <Header
        selectedCar={selectedCar}
        onCarChange={handleCarChange}
        overallStatus={overallStatus}
        fps={fps}
        wsStatus={wsStatus}
        onConnect={connect}
        onDisconnect={disconnect}
        onFetchSample={handleFetchSample}
      />

      {/* Fetch status bar */}
      {(isFetching || fetchError) && (
        <div
          className={`px-4 py-1 text-xs flex items-center gap-2 flex-shrink-0 ${
            fetchError
              ? 'bg-red-950 text-red-400 border-b border-red-900'
              : 'bg-blue-950 text-blue-400 border-b border-blue-900'
          }`}
        >
          {isFetching && (
            <>
              <span className="inline-block w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
              <span>데이터 요청 중...</span>
            </>
          )}
          {fetchError && (
            <>
              <span>백엔드 연결 실패 ({fetchError}) — 목 데이터 사용 중</span>
              <button
                onClick={() => setFetchError(null)}
                className="ml-auto text-red-500 hover:text-red-300"
              >
                ✕
              </button>
            </>
          )}
        </div>
      )}

      {/* Main panels */}
      {currentData ? (
        <div className="flex-1 grid grid-cols-2 gap-2 p-2 min-h-0 overflow-hidden">
          {/* Left: Motor Panel */}
          <div className="overflow-y-auto pr-1 scrollbar-thin h-full">
            <MotorPanel
              motor={currentData.motor}
              vehicleId={currentData.vehicle_id}
              trendData={trendData}
            />
          </div>

          {/* Right: Battery Panel */}
          <div className="overflow-y-auto pl-1 scrollbar-thin h-full">
            <BatteryPanel battery={currentData.battery} />
          </div>
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="text-slate-500 text-lg mb-3">데이터 없음</div>
            <button
              onClick={handleFetchSample}
              className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded text-sm"
            >
              데이터 요청
            </button>
          </div>
        </div>
      )}

      {/* Alert Log */}
      <AlertLog alerts={localAlerts} />
    </div>
  );
}
