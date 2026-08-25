# SO-101 동적 파라미터 정합 (Dynamics Matching) 결과

## 개요

Real-to-Sim 동역학 정합. 실물 SO-101 팔의 관절 궤적을 측정하고, Isaac Sim 시뮬레이션의 물리 파라미터를 비선형 최적화로 조정해 실물 궤적에 정합.

- **방법**: 위치 궤적 RMSE 최소화 (비선형 최소제곱, Powell)
- **입력 신호**: 계단(-15°->0°->+15°) + 사인(20°, 0.2 Hz)
- **최적화 파라미터**: stiffness, damping, armature, friction(정적·점성)
- **대상**: shoulder_lift, elbow_flex, wrist_flex (관절별 개별 정합)

## RMSE 결과

| 관절 | base (deg) | tuned (deg) | 감소율 |
|------|-----------|------------|--------|
| shoulder_lift | 2.99 | 1.16 | 61.2% |
| elbow_flex | 2.60 | 1.01 | 61.2% |
| wrist_flex | 11.69 | 0.91 | 92.2% |
| **평균** | **5.76** | **1.03** | **82.1%** |

세 관절 모두 tuned RMSE ~1° 수준으로 정합. 육안상 실물과 시뮬 궤적이 거의 일치.

## 최적 파라미터

| 관절 | stiffness | damping | armature | friction_static | friction_viscous |
|------|-----------|---------|----------|-----------------|------------------|
| shoulder_lift | 170 -> 404 | 65 -> 59 | 0 -> 0.197 | 0.001 | 0.007 |
| elbow_flex | 120 -> 238 | 45 -> 35 | 0 -> 0.073 | 0.867 | 0.023 |
| wrist_flex | 80 -> 272 | 30 -> 45 | 0 -> 0.111 | 0.386 | 0.109 |

(base -> tuned. armature, friction은 base에서 미설정=0)

## 관절별 물리 특성 해석

**shoulder_lift** - 강한 모터, 깨끗한 응답
- stiffness 최대(404), 마찰 거의 0
- 큰 관절이라 강한 제어, 마찰 영향 미미

**elbow_flex** - 정적 마찰 존재 (데드존)
- friction_static 큼(0.867)
- 계단 출발 시 데드존 재현
- 정적 토크 측정에서 관찰된 데드존과 일치 (교차 검증)

**wrist_flex** - 점성 마찰, base 부실
- friction_viscous 최대(0.109)
- base stiffness=80이 실물 대비 크게 낮아 base gap 최대(11.69)
- 최적화로 개선폭 최대(92%)

## 방법론 요약

### 파이프라인
1. 실물 궤적 측정 -> CSV (record_real.py)
2. 시뮬 base 궤적 비교 -> base RMSE (compare_base.py)
3. 파라미터 최적화 -> tuned RMSE (optimize_sim.py)
4. 종합 (summary_plot.py)

### 최적화 정식화
목적함수: RMSE(theta) = sqrt( (1/N) sum (y_real - y_sim(theta))^2 )

- theta = [stiffness, damping, armature, friction_s, friction_v]
- 오프셋 상쇄 (초기값 빼기, 상대 정렬)
- 파라미터 0~1 정규화 후 Powell (bounds 지원)
- 관절별 개별 최적화 (관절 간 상호작용 회피)

### 왜 비선형 최소제곱인가
- 위치 궤적은 파라미터에 대해 비선형 (시뮬레이터가 미분방정식 해)
- 토크 회귀(tau = Y·Theta, 선형/QP)는 정확한 토크·가속도 측정 필요
- 저가 서보(STS3215)는 토크 부정확(Load 기반)·가속도 노이즈 -> 위치 궤적 매칭 선택

## 시간 동기화 검증
- 시뮬 env.step_dt = sim.dt(0.01) x decimation(2) = 0.02s = 50 Hz
- 실물 RATE = 50 Hz (0.02s)
- 시간축 일치 확인 -> 궤적 gap은 진짜 동역학 차이

## 확보된 정합 지표 (포트폴리오)
- 정적: 질량 검증 3.5%, Load-토크 캘리브레이션 (shoulder_lift, elbow_flex)
- 동적: 궤적 RMSE 평균 82% 감소 (5.76° -> 1.03°)
- 관절별 물리 특성 정량화 (stiffness, armature, friction)
