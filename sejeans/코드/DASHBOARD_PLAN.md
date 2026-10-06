# 자율주행 전기차 이상 탐지 통합 대시보드 구현 계획

> **프로젝트**: 자율주행 전기차 모터 및 배터리 이상 징후 탐지 기술 경량화
> **팀**: 7조 (박세진 C321027, 박솔 C321028)
> **문서 목적**: Claude Code가 단계적으로 대시보드를 구현할 수 있도록 작성된 기술 명세서

---

## 1. 프로젝트 개요

### 1.1 대시보드의 역할
학습된 YOLOv11-Seg(모터-감속기)와 CNN(배터리) 경량화 모델의 추론 결과를 **실시간 모니터링 형태로 시각화**하는 프론트엔드 + 백엔드 시스템이다. 새로운 센서 데이터가 입력되면 모델 추론을 수행하고, 결함 유형 / 위치 / 심각도를 사용자에게 즉시 전달한다.

### 1.2 화면 구성 원칙
- **좌/우 2분할 레이아웃**: 좌측 모터-감속기, 우측 배터리
- **상단 공통 헤더**: 차량 선택 드롭다운(IONIQ / KONA / NIRO), 전체 상태 요약, 현재 시각
- **하단 공통 푸터**: 알림 로그(모터-감속기 + 배터리 통합)

---

## 2. 시스템 아키텍처

### 2.1 전체 구조
```
[데이터 소스]                  [백엔드]                        [프론트엔드]
PNG/NPY 파일       ──►    FastAPI 서버                ──►    React 대시보드
(시뮬레이션 입력)          ├── 모델 추론 엔진                   ├── 좌측: 모터-감속기
                           │   ├─ YOLOv11-Seg (ONNX)           │   ├─ Grad-CAM 히트맵
                           │   └─ CNN 양자화 모델 (ONNX)        │   └─ 부품 상태 게이지
                           ├── 후처리 (셀별 이상 점수,           └── 우측: 배터리
                           │    SOH proxy 계산)                     ├─ 셀별 결함 시각화
                           └── WebSocket 스트리밍                    └─ SOH / 편차 수치
```

### 2.2 기술 스택
| 영역 | 기술 | 비고 |
|------|------|------|
| 백엔드 | FastAPI + Uvicorn | 추론 API + WebSocket |
| 모델 런타임 | ONNX Runtime | TensorRT 변환은 추후(엣지 배포 단계) |
| 실시간 통신 | WebSocket | MQTT는 IoT 연동 단계 이후 |
| 프론트엔드 | React + TypeScript | Vite 기반 |
| 차트 | Recharts 또는 Chart.js | 이상 점수 트렌드, SOH 게이지 |
| 스타일 | Tailwind CSS | 빠른 프로토타이핑 |
| Grad-CAM | pytorch-grad-cam 라이브러리 | YOLO 호환 위해 EigenCAM 사용 |

---

## 3. 데이터 처리 파이프라인

### 3.1 입력 데이터 형식
| 구분 | 형식 | shape / 예시 | 출처 |
|------|------|--------------|------|
| 모터-감속기 | PNG | STFT 스펙트로그램 이미지 | Current_U / Vib_Motor / Vib_TM |
| 배터리 | NPY | (20, 96) 또는 (20, 98) 전압 행렬 | 셀별 시계열 전압 |
| 메타데이터 | JSON | car_model, speed, temperature, measure_date | 라벨링 데이터 |

### 3.2 실시간 입력 시뮬레이션
**실제 차량 연결이 불가능하므로 아래 방식 중 하나로 시뮬레이션한다:**

**방식 A - 폴더 감시(권장)**: `watchdog` 라이브러리로 `/data/incoming/` 폴더를 감시하다가 새 파일이 추가되면 자동 추론 → WebSocket으로 프론트 전송

**방식 B - 수동 업로드**: 대시보드 상단에 파일 업로드 버튼을 두고 사용자가 직접 테스트

**방식 C - 스트림 시뮬레이터**: validation 셋에서 1~2초 간격으로 파일을 순차 재생하는 스크립트(`scripts/stream_simulator.py`)

→ **초기 구현은 방식 B, 데모 단계에서 방식 C 추가 권장**

### 3.3 배터리 파생 지표 계산 (NPY → 대시보드 수치)
대시보드에 표시되는 **셀 전압 편차, 최저 셀 전압, 배터리 온도**는 JSON 메타데이터와 NPY에서 각각 계산한다.

```python
import numpy as np

def compute_battery_metrics(npy_array: np.ndarray, metadata: dict) -> dict:
    """
    npy_array: shape (20, 96) 또는 (20, 98), 각 셀의 시간별 전압
    metadata: JSON 메타데이터
    """
    # 셀별 평균 전압 (시간축 평균)
    cell_mean_voltage = npy_array.mean(axis=0)  # shape (96,) 또는 (98,)

    # 전체 시간 구간에서의 최저 셀 전압
    min_cell_voltage = float(npy_array.min())

    # 셀 전압 편차 (동일 시점에서 셀 간 표준편차의 시간 평균)
    cell_std_per_time = npy_array.std(axis=1)  # 각 시점에서의 셀 간 편차
    voltage_deviation = float(cell_std_per_time.mean())

    # 배터리 온도는 메타데이터에서 직접
    temperature = metadata.get("temperature", None)

    return {
        "min_cell_voltage": round(min_cell_voltage, 3),   # 예: 3.91 V
        "voltage_deviation": round(voltage_deviation, 4), # 예: 0.012 V
        "temperature": round(temperature, 1) if temperature else None,
        "cell_count": npy_array.shape[1],
        "cell_mean_voltages": cell_mean_voltage.tolist(), # 셀별 시각화용
    }
```

### 3.4 SOH Proxy (⚠️ 원본 데이터에 SOH 실측값 없음)
**문제점**: AI Hub 자율주행 고장진단 데이터는 스냅샷 기반이라 누적 충방전 이력이 없어 **실제 SOH를 계산할 수 없다.**

**해결책 - SOH Proxy Score 제안**:
```
soh_proxy = 100 - w1 * (voltage_deviation_normalized)
              - w2 * (min_voltage_deficit_normalized)
              - w3 * (anomaly_score_normalized)
```
- 가중치는 정상 데이터 분포 기준으로 calibration
- 대시보드 표기는 **"배터리 건강 지수 (Health Index)"**로 변경 권장 (SOH라는 용어의 오해 방지)
- 또는 학계 용어 그대로 유지하되 문서에 proxy임을 명시

---

## 4. 화면 상세 명세

### 4.1 전체 레이아웃
```
┌──────────────────────────────────────────────────────────────────────┐
│ [차량 선택 ▼ IONIQ] [전체 상태: ● 경고] [23.4 FPS]        09:45:12   │ ← 헤더
├───────────────────────────────┬──────────────────────────────────────┤
│  🔧 모터-감속기                │  🔋 배터리                           │
│                               │                                      │
│  ┌─ 부품 상태 게이지 ──────┐   │  ┌─ 건강 지수 게이지 ──────┐         │
│  │ 모터 U상    [███░] 73  │   │  │      78 / 100           │         │
│  │ 감속기     [██░░] 61   │   │  │      ● 정상             │         │
│  │ 구동 시스템 [████] 45  │   │  └─────────────────────────┘         │
│  └─────────────────────────┘   │                                      │
│                               │  ┌─ 셀별 결함 시각화 ──────┐         │
│  ┌─ Grad-CAM 히트맵 ──────┐    │  │ [셀 1][셀 2]...[셀 96] │         │
│  │  [스펙트로그램 + CAM]   │   │  │ 결함 셀: #47 (전압)    │         │
│  │  편심 결함 탐지         │   │  │ 결함 유형: 셀편차      │         │
│  └─────────────────────────┘   │  │ 확률: 정상 8% / 주의   │         │
│                               │  │       15% / 결함 77%   │         │
│  ┌─ 이상 점수 트렌드 ─────┐    │  └─────────────────────────┘         │
│  │  [실시간 라인 차트]     │   │  ┌─ 배터리 수치 ──────────┐          │
│  │  임계값 초과 알림       │   │  │ 셀 편차: 0.012 V       │          │
│  └─────────────────────────┘   │  │ 최저 전압: 3.91 V      │          │
│                               │  │ 온도: 45.8 °C          │          │
│                               │  └─────────────────────────┘          │
├───────────────────────────────┴──────────────────────────────────────┤
│ 📋 알림 로그 (실시간)                                                 │
│  09:45:12 [경고] MOB-X001 모터 이상: MSE 27.3 (임계값 초과)           │
│  09:30:05 [주의] MOB-X001 감속기 상태 변화 감지 (편심_주의)           │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 좌측 패널 - 모터-감속기

#### 4.2.1 부품 상태 게이지
- **표시 항목**: 모터 U상, 감속기, 구동 시스템 (3개 센서별 이상 점수)
- **데이터 소스**: YOLOv11-Seg의 각 센서 입력별 segmentation confidence
- **시각화**: 가로 bar + 수치(0~100) + 상태 라벨(정상/주의/경고)
- **임계값**: 정상 < 30 < 주의 < 70 < 경고 (ROC 기반 튜닝 후 확정)

#### 4.2.2 Grad-CAM 히트맵 ⚠️ 기술적 주의
- **표시 항목**: 입력 스펙트로그램 + Grad-CAM 오버레이
- **⚠️ YOLO 호환성 이슈**:
  - 표준 Grad-CAM은 classification head를 가정하나, YOLO는 multi-scale anchor-free detection head
  - **해결책**: `pytorch-grad-cam` 라이브러리의 `EigenCAM` 또는 `EigenGradCAM` 사용 (YOLO 공식 호환)
  - 또는 Ultralytics 커뮤니티의 `yolo-gradcam` 구현체 참조
- **구현 시점**: 모델 학습 완료 후 (2학기 초, 8월)

#### 4.2.3 이상 점수 트렌드
- **표시 항목**: 최근 N초간의 이상 점수 라인 차트
- **데이터 소스**: 프레임별 최대 segmentation confidence
- **임계값 선**: ROC 곡선 기반으로 사전 결정 (예: 10.0)
- **알림 트리거**: 임계값 초과 시 빨간 마커 + 알림 로그 자동 추가

### 4.3 우측 패널 - 배터리

#### 4.3.1 건강 지수 게이지 (SOH Proxy)
- **표시 항목**: 0~100 반원 게이지 + 현재 값 + 상태 라벨
- **계산**: §3.4의 SOH Proxy 공식
- **색상**: 80+ 녹색, 60~80 황색, <60 적색

#### 4.3.2 셀별 결함 시각화 ⭐ 핵심 요구사항
> "몇 번째 셀 결함인지" 요구사항 대응

**문제**: 단순 CNN 분류 모델은 "어느 셀이 이상한지"를 localization하지 못한다.

**해결 옵션 3가지**:

**옵션 A - Attention 기반 CNN (권장)**
- CNN 마지막 conv layer의 feature map을 셀 축 방향으로 global average pooling
- 각 셀 위치별 activation을 "셀별 이상 기여도"로 사용
- Grad-CAM을 배터리 CNN에도 적용 가능 → 일관성 있는 XAI

**옵션 B - 규칙 기반 localization**
- 분류는 CNN이 수행(정상/주의/결함)
- 결함으로 판정된 경우, **규칙 기반으로 어느 셀이 문제인지 특정**:
  - 셀전압 결함 → 각 셀의 최저 전압이 threshold를 가장 많이 벗어난 셀
  - 셀편차 결함 → 다른 셀 평균 대비 가장 크게 벗어난 셀
- 구현 난이도 낮음, 해석 명확

**옵션 C - Autoencoder 재구성 오차**
- 정상 데이터만으로 AE 학습
- 셀별 재구성 오차가 큰 셀을 결함 셀로 특정
- 이상 점수 트렌드와 자연스럽게 연동

→ **초기 구현은 옵션 B, 모델 고도화 단계에서 옵션 A 병행 권장**

**시각화**:
- 96(또는 98)개 셀을 그리드로 표시 (예: 12×8)
- 각 셀의 색상을 이상 점수로 매핑 (파랑=정상, 빨강=결함)
- 결함 셀 클릭 시 해당 셀의 시계열 전압 그래프 팝업

#### 4.3.3 분류 결과 표시
- **결함 유형**: 셀전압 결함 / 셀편차 결함 / 정상
- **확률**: 정상 / 주의 / 결함 3-class softmax 결과
- **시각화**: 가로 누적 bar + 퍼센트

#### 4.3.4 배터리 수치 카드
- 셀 편차 (V), 최저 셀 전압 (V), 온도 (°C), 셀 개수
- §3.3의 `compute_battery_metrics` 함수 결과 직결

### 4.4 헤더 - 차량 선택
**요구사항 반영**: 차량 3종(IONIQ / KONA / NIRO) 선택 가능

**구현**:
```typescript
interface VehicleState {
  car_model: "IONIQ" | "KONA" | "NIRO";
  cell_count: 96 | 98;  // IONIQ=96, KONA/NIRO=98
  current_data: BatteryData | MotorData;
}
```

**다른 의견 (권장사항)**:
1. **차량 선택 + 다중 차량 모니터링 병행**: 현업 시나리오(플릿 관리)를 고려하면 좌측 사이드바에 차량 리스트(MOB-X001, MOB-X002, ...)를 두고 클릭 시 상세 화면으로 전환하는 구조가 현실적
2. **차량 모델은 데이터에서 자동 감지**: 업로드된 NPY의 shape으로 셀 개수를 확인하거나, JSON 메타데이터의 `car_model` 필드를 읽어 자동 설정 → 사용자 실수 방지

### 4.5 알림 로그
- **형식**: `[시각] [레벨] [차량ID] [메시지]`
- **레벨**: 정보(🔵) / 주의(🟡) / 경고(🔴)
- **저장**: 메모리(초기) → SQLite(확장)
- **필터**: 레벨별, 차량별 필터링 UI
- **알림 우선순위 시스템**: ROC 기반 임계값 + 최근 N분간 동일 알림 중복 억제(deduplication)

---

## 5. 백엔드 API 명세

### 5.1 REST 엔드포인트
| Method | Path | 설명 |
|--------|------|------|
| POST | `/api/inference/motor` | 모터-감속기 PNG 업로드 및 추론 |
| POST | `/api/inference/battery` | 배터리 NPY 업로드 및 추론 |
| GET | `/api/vehicles` | 등록된 차량 리스트 |
| GET | `/api/alerts?level=&limit=` | 알림 로그 조회 |
| GET | `/api/health` | 서버 상태 확인 |

### 5.2 WebSocket 채널
| 채널 | 메시지 포맷 | 용도 |
|------|-------------|------|
| `/ws/stream` | JSON | 실시간 추론 결과 푸시 |

### 5.3 응답 스키마 예시
```json
{
  "timestamp": "2026-04-19T09:45:12Z",
  "vehicle_id": "MOB-X001",
  "car_model": "IONIQ",
  "motor": {
    "anomaly_score": 27.3,
    "threshold": 10.0,
    "status": "warning",
    "fault_type": "ECC20",
    "sensor_scores": {"current_u": 73, "vib_motor": 45, "vib_tm": 61},
    "gradcam_url": "/static/cam/MOB-X001_20260419094512.png"
  },
  "battery": {
    "soh_proxy": 78,
    "status": "normal",
    "probabilities": {"normal": 0.08, "caution": 0.15, "defect": 0.77},
    "fault_type": "cell_deviation",
    "faulty_cell_indices": [47],
    "cell_scores": [0.02, 0.03, ..., 0.89, ...],
    "metrics": {
      "min_cell_voltage": 3.91,
      "voltage_deviation": 0.012,
      "temperature": 45.8,
      "cell_count": 96
    }
  }
}
```

---

## 6. 디렉토리 구조 (권장)

```
dashboard/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI 엔트리
│   │   ├── routers/
│   │   │   ├── inference.py        # 추론 엔드포인트
│   │   │   ├── alerts.py
│   │   │   └── websocket.py
│   │   ├── models/
│   │   │   ├── motor_model.py      # YOLOv11-Seg 래퍼
│   │   │   └── battery_model.py    # CNN 래퍼
│   │   ├── services/
│   │   │   ├── battery_metrics.py  # §3.3 구현
│   │   │   ├── soh_proxy.py        # §3.4 구현
│   │   │   ├── cell_localizer.py   # §4.3.2 옵션 B 구현
│   │   │   └── gradcam_yolo.py     # EigenCAM 래퍼
│   │   └── schemas/                # Pydantic 모델
│   ├── weights/
│   │   ├── yolov11seg_int8.onnx
│   │   └── battery_cnn_int8.onnx
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.tsx
│   │   │   ├── MotorPanel.tsx
│   │   │   ├── BatteryPanel.tsx
│   │   │   ├── CellGrid.tsx        # 96/98셀 그리드
│   │   │   ├── GaugeChart.tsx
│   │   │   ├── AnomalyTrend.tsx
│   │   │   ├── GradCamViewer.tsx
│   │   │   └── AlertLog.tsx
│   │   ├── hooks/
│   │   │   └── useWebSocket.ts
│   │   ├── types/
│   │   └── App.tsx
│   └── package.json
├── scripts/
│   └── stream_simulator.py         # §3.2 방식 C
└── data/
    ├── incoming/                   # 폴더 감시용
    └── samples/                    # 테스트용 샘플
```

---

## 7. 구현 순서 (Claude Code 작업 단위)

### Phase 1 - 기반 (1~2주)
1. [ ] FastAPI 서버 스캐폴딩 + 헬스체크
2. [ ] Vite + React + Tailwind 프론트 스캐폴딩
3. [ ] 좌/우 2분할 레이아웃 + 차량 선택 헤더 (mock 데이터)
4. [ ] `compute_battery_metrics` 함수 구현 및 단위 테스트

### Phase 2 - 추론 통합 (2주)
5. [ ] 학습된 모델을 ONNX로 변환 (`torch.onnx.export`)
6. [ ] 모터/배터리 추론 REST 엔드포인트
7. [ ] 응답 스키마(§5.3) 확정 및 프론트 타입 정의
8. [ ] 셀 localizer 옵션 B 구현

### Phase 3 - 시각화 (2주)
9. [ ] 부품 상태 게이지
10. [ ] 이상 점수 트렌드 (Recharts)
11. [ ] 건강 지수 게이지
12. [ ] 셀별 결함 그리드 (96/98셀)
13. [ ] 알림 로그 + 레벨별 필터

### Phase 4 - 실시간화 (1주)
14. [ ] WebSocket 서버 + 클라이언트
15. [ ] 폴더 감시 또는 스트림 시뮬레이터
16. [ ] 알림 중복 억제 로직

### Phase 5 - XAI 통합 (1~2주)
17. [ ] EigenCAM for YOLOv11-Seg
18. [ ] 배터리 CNN Grad-CAM (선택)
19. [ ] Grad-CAM 이미지 렌더링 뷰어

### Phase 6 - 마무리
20. [ ] 데모 시나리오 3종 (정상 / 주의 / 경고) 준비
21. [ ] README + 실행 가이드
22. [ ] 데모 영상 녹화

---

## 8. ⚠️ 원 계획 대비 리스크 및 권고사항

| # | 항목 | 원 계획 | 리스크 | 권고사항 |
|---|------|---------|--------|----------|
| 1 | Grad-CAM | YOLO에 직접 적용 | 표준 Grad-CAM은 YOLO 비호환 | EigenCAM 사용, 사전에 MVP 검증 |
| 2 | SOH | SOH 게이지 표시 | 원본 데이터에 SOH 실측 없음 | "건강 지수(Health Index) Proxy"로 명시 |
| 3 | 셀 localization | "몇 번째 셀 결함" 요구 | 분류 모델 단독으로 불가 | 옵션 B(규칙 기반) 먼저 구현 |
| 4 | 실시간성 | 실시간 스트리밍 | 실제 차량 연결 불가 | 폴더 감시 + 시뮬레이터로 대체 |
| 5 | 배터리 Grad-CAM | 누락 | 일관성 저하 | 2학기 확장 시 추가 |
| 6 | TensorRT | 사용 계획 | 엣지 디바이스 미확보 | ONNX Runtime 먼저, TensorRT는 검증 단계에서 |
| 7 | 차량 선택 UX | 드롭다운 | 플릿 관리 시나리오 미반영 | 사이드바 차량 리스트 추가 권장 |

---

## 9. 성능 목표

| 지표 | 목표값 | 측정 방법 |
|------|--------|-----------|
| 엔드투엔드 지연 | < 200ms | 파일 입력 → 화면 렌더 |
| 모델 추론 시간 | < 50ms (YOLO), < 10ms (배터리 CNN) | ONNX Runtime 벤치마크 |
| WebSocket 업데이트 주기 | 1~2 Hz | 프론트 타이머 |
| mIoU (모터-감속기) | ≥ 0.85 | Validation set |
| 배터리 3-class F1 | ≥ 0.90 | Validation set |

---

## 10. 참고 자료
- Ultralytics YOLOv11-Seg 공식 문서
- pytorch-grad-cam (jacobgil): YOLO 호환 CAM 구현체
- FastAPI WebSocket 공식 튜토리얼
- AI Hub 자율주행 고장진단 데이터 활용 가이드 (2022)

---

**문서 버전**: v1.0
**작성일**: 2026-04-19
**다음 리뷰**: Phase 1 완료 후 (~2주)
