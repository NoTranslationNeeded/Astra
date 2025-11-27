# TensorBoard 메트릭 가이드 📊

**모델:** deepstack_7actions_dense_v2_fair  
**환경:** Variable Stack (1-250 BB), 7 Actions, Dense Reward, **Fair Dealer Rotation**

이 문서는 Texas Hold'em AI 학습 중 TensorBoard에서 확인할 수 있는 모든 메트릭에 대한 설명을 제공합니다.

---

## 🎯 현재 학습 환경

- **보상 시스템:** Dense Reward (매 핸드마다 보상)
- **보상 공식:** `(chip_payoff / BB) / 250.0`
- **보상 범위:** `-0.4 ~ +0.4` (일반적으로 `-0.1 ~ +0.1`)
- **액션 공간:** 7가지 (Fold, Check/Call, 33%, 75%, 100%, 150%, All-in)
- **스택 깊이:** 1 BB ~ 250 BB (랜덤)
- **블라인드:** 250 ~ 5000 (랜덤)

---

## 📈 Episode 메트릭 (게임 성과)

### `episode_reward_mean`
- **설명:** 토너먼트당 평균 보상 (Dense Reward 누적)
- **정상 범위:** `-0.1 ~ +0.1` (Self-Play 균형)
- **해석:**
  - **+0.05:** Player 0이 평균적으로 약간 우세
  - **0 근처:** 이상적인 Self-Play 균형
  - **-0.05:** Player 1이 평균적으로 약간 우세
- **목표:** **0 근처 유지** (양쪽 플레이어 균형 학습)

### `episode_reward_max` / `episode_reward_min`
- **설명:** 최고/최저 토너먼트 보상
- **정상 범위:** `max ≈ +0.4`, `min ≈ -0.4`
- **의미:** 전체 스택(250 BB) 획득/손실 시 최대값

### `episode_len_mean`
- **설명:** 토너먼트당 평균 핸드 수
- **정상 범위:** 20-100 핸드
- **해석:**
  - **10-30 핸드:** 숏 스택 게임 (1-50 BB 시작)
  - **50-100 핸드:** 미들 스택 (50-150 BB)
  - **100-200 핸드:** 딥 스택 (150-250 BB)
- **참고:** 가변 스택 깊이 때문에 자연스럽게 큰 변동

---

## 🧠 학습 메트릭 (AI 성능)

### `learner/default_policy/learner_stats/policy_loss`
- **설명:** 정책 네트워크 손실 (Policy Loss)
- **해석:**
  - AI가 "더 나은 행동"을 배우기 위한 오차
  - **감소:** 정책이 개선되고 있음
  - **증가:** 학습이 불안정하거나 새로운 전략 탐색 중
- **정상 범위:** -0.1 ~ -0.001 (음수)

### `learner/default_policy/learner_stats/vf_loss`
- **설명:** 가치 함수 손실 (Value Function Loss)
- **해석:**
  - AI가 "이 상황이 얼마나 좋은지" 예측하는 오차
  - **감소:** 가치 추정이 정확해지고 있음
- **정상 범위:** 0.1 ~ 100 (환경에 따라 다름)

### `learner/default_policy/learner_stats/entropy`
- **설명:** 행동 선택의 무작위성 (탐색 정도)
- **해석:**
  - **높음(>1.5):** 다양한 액션 시도 (탐색 단계)
  - **낮음(<0.5):** 확신 있는 액션 선택 (수렴 단계)
- **목표:** 초반에는 높고, 후반에는 **서서히 감소**

### `learner/default_policy/learner_stats/kl`
- **설명:** KL Divergence (정책 업데이트 크기)
- **해석:**
  - 이전 정책과 새 정책의 차이
  - **너무 높음(>0.1):** 정책이 급격히 변경 (불안정)
  - **적절함(0.01-0.05):** 안정적인 학습
- **PPO 제약:** 일반적으로 낮게 유지됨

### `learner/default_policy/learner_stats/total_loss`
- **설명:** 전체 손실 (Policy + Value + Entropy)
- **해석:**
  - 모든 손실의 가중 합
  - **감소 추세:** 전반적인 학습 진행 중
- **목표:** 시간이 지나면서 **안정적으로 감소**

---

## ⚙️ 학습 설정 메트릭

### `learner/default_policy/learner_stats/curr_lr`
- **설명:** 현재 학습률 (Learning Rate)
- **고정값:** 0.0003
- **의미:** 파라미터 업데이트 속도

### `learner/default_policy/learner_stats/cur_kl_coeff`
- **설명:** KL Divergence 계수
- **의미:** PPO에서 정책 변화를 제한하는 파라미터

### `clip_param`
- **설명:** PPO Clipping Range
- **설정값:** 0.2 (보수적, Dense Reward에 최적화)
- **의미:** 정책 업데이트 시 최대 변화량 제한

### `lambda_` (GAE Lambda)
- **설명:** Generalized Advantage Estimation Lambda
- **설정값:** 0.95
- **의미:** Bias-Variance 균형 (1.0=Monte Carlo, 0.0=TD)

---

## 🎯 환경 메트릭 (토너먼트 통계)

### `custom_metrics/tournament_hands_mean`
- **설명:** 토너먼트당 평균 핸드 수
- **해석:** `episode_len_mean`과 유사 (토너먼트 길이)

### `custom_metrics/win_rate_mean`
- **설명:** 승률 (0.0 ~ 1.0)
- **해석:**
  - **0.5:** 50% 승률 (이상적)
  - **>0.5:** Player 0이 더 강함
  - **<0.5:** Player 1이 더 강함
- **목표:** 자가 대결(Self-Play)에서 **0.5 유지** (균형)

### `custom_metrics/final_chips_mean`
- **설명:** 최종 칩 평균 (승자)
- **해석:** 토너먼트 종료 시 승자가 가진 칩 (약 2배의 시작 칩)

---

## ⏱️ 성능 메트릭

### `timers/learn_time_ms`
- **설명:** 학습(Gradient Update)에 걸린 시간 (밀리초)
- **정상 범위:** 100-500ms
- **주의:** 너무 높으면 병목

### `timers/sample_time_ms`
- **설명:** 환경에서 데이터 수집 시간 (밀리초)
- **정상 범위:** 500-2000ms
- **참고:** 복잡한 환경일수록 증가

### `num_env_steps_sampled_this_iter`
- **설명:** 이번 iteration에서 수집한 타임스텝 수
- **정상값:** 약 4000-8000 (4 runners × 각 runner의 샘플)
- **의미:** Batch size와 연관

### `num_env_steps_sampled_lifetime`
- **설명:** 전체 학습에서 수집한 총 타임스텝 수
- **목표:** 1,000,000까지 증가 (학습 종료 조건)

---

## 🎲 Multi-Agent 메트릭

### `policy_reward_mean/player_0` / `policy_reward_mean/player_1`
- **설명:** 각 플레이어별 평균 보상
- **해석:**
  - Self-Play에서는 **두 값이 거의 반대**여야 함 (제로섬 게임)
  - Player 0: +X → Player 1: -X
- **균형 확인:** 두 값의 합이 **0에 가까움**
- TensorBoard 왼쪽 상단 `Smoothing` 슬라이더
- **0.0:** 원본 데이터 (노이즈 많음)
- **0.6-0.8:** 추세 확인에 적합 (추천)
- **0.95:** 매끄럽지만 지연

### 2. 비교 모드
- 여러 실험을 동시에 시각화
- 다른 Reward Function이나 하이퍼파라미터 비교

### 3. 주요 체크 포인트
- **초반(0-100k 타임스텝):** Entropy 높고, Loss 감소 시작
- **중반(100k-500k):** Reward 증가, Entropy 감소
- **후반(500k-1M):** 안정화, Reward 수렴

---

## 📌 요약

| 메트릭 | 목표 방향 | 정상 범위 |
|--------|----------|----------|
| `episode_reward_mean` | ➡️ 0 근처 | -0.1 ~ +0.1 |
| `episode_reward_max` | - | +0.3 ~ +0.4 |
| `episode_reward_min` | - | -0.3 ~ -0.4 |
| `policy_loss` | ↘️ 감소 | -0.05 ~ -0.001 |
| `vf_loss` | ↘️ 감소 | 0.01 ~ 10 |
| `entropy` | ↘️ 서서히 감소 | 1.5 → 0.8 |
| `kl` | ➡️ 안정 | 0.01 ~ 0.05 |
| `win_rate_mean` | ➡️ 0.5 유지 | 0.45 ~ 0.55 |

**현재 설정 (deepstack_7actions_dense_v1):**
- Batch Size: 4000
- Entropy Coeff: 0.02
- Clip Param: 0.2
- GAE Lambda: 0.95
- Env Runners: 4
- Action Space: 7

**학습 모니터링 우선순위:**
1. `episode_reward_mean` (Self-Play 균형, 0 근처 목표)
2. `entropy` (탐색 vs 활용, 1.5 → 0.8)
3. `policy_loss` / `vf_loss` (학습 진행)
4. `win_rate_mean` (Self-Play 균형, 0.5 목표)
