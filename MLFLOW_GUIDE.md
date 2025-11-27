# MLflow 사용 가이드

## MLflow란?

MLflow는 머신러닝 실험을 추적하고 관리하는 오픈소스 플랫폼입니다.

**핵심 기능:**
- 하이퍼파라미터 자동 로깅
- 메트릭 추적 (Reward, Loss 등)
- 모델 버전 관리
- 실험 비교

---

## 시작하기

### 1. 학습 실행
```bash
python train_mlflow.py
```

### 2. MLflow UI 실행 (새 터미널)
```bash
mlflow ui
```

### 3. 브라우저에서 확인
http://localhost:5000

---

## MLflow UI 사용법

### 📊 **Experiments 페이지**

**왼쪽 패널: 실험 목록**
- `poker-ai-independent-policies`: 현재 프로젝트

**중앙 패널: Run 목록**
- 각 학습 세션이 하나의 "Run"
- Start Time, Duration, Metrics 표시

**클릭하면:**
- Parameters (하이퍼파라미터)
- Metrics (성능 지표)
- Artifacts (저장된 모델)

---

### 🔍 **Run 비교**

1. 여러 Run 체크박스 선택
2. "Compare" 버튼 클릭
3. 그래프로 비교:
   - Parallel Coordinates Plot
   - Scatter Plot
   - Contour Plot

---

### 📈 **주요 메트릭**

MLflow에 자동 로깅되는 지표들:

| 메트릭 | 설명 | 목표 |
|---|---|---|
| `episode_reward_mean` | 평균 보상 | 0에 수렴 |
| `episode_len_mean` | 게임 길이 | 2~4턴 |
| `policy_loss` | Policy Loss | 감소 |
| `vf_loss` | Value Function Loss | 감소 |
| `entropy` | 탐험 정도 | 초반 높음 → 후반 낮음 |

---

### 🏷️ **Parameters 확인**

다음 하이퍼파라미터들이 자동 기록됩니다:

```
lr: 0.0003
gamma: 0.99
train_batch_size: 8000
entropy_coeff: 0.01
fcnet_hiddens: [256, 256]
use_lstm: True
lstm_cell_size: 256
```

---

## TensorBoard vs MLflow

### **TensorBoard (실시간 모니터링)**
- 학습 중 실시간 그래프
- 자세한 메트릭 추적
- 단일 실험에 최적

### **MLflow (실험 관리)**
- 여러 실험 비교
- 하이퍼파라미터 추적
- 모델 버전 관리
- 재현성 보장

**결론: 둘 다 사용하세요!**

---

## 실전 활용

### **실험 비교 예시**

```bash
# 실험 1: 기본 설정
python train_mlflow.py

# 실험 2: Learning Rate 변경
# train_mlflow.py에서 lr=0.0005로 수정 후
python train_mlflow.py

# 실험 3: Entropy 증가
# entropy_coeff=0.05로 수정 후
python train_mlflow.py
```

MLflow UI에서 3개 실험을 동시에 비교하여 최적 설정 찾기!

---

## 모델 로드

### **최고 성능 모델 찾기**

1. MLflow UI에서 `episode_reward_mean`으로 정렬
2. 가장 0에 가까운 Run 선택
3. Artifacts 탭에서 모델 다운로드

### **재현하기**

```python
import mlflow

# Run ID로 로드
run_id = "abc123..."
model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
```

---

## 팁

### **실험 정리**
```bash
# 오래된 실험 삭제
mlflow gc --backend-store-uri file:./mlruns
```

### **원격 추적**
```python
mlflow.set_tracking_uri("http://your-server:5000")
```

### **자동 로깅 비활성화**
```python
mlflow.autolog(disable=True)
```
