# TensorBoard 사용 가이드

## 시작하기

Ray/RLlib은 자동으로 TensorBoard 형식으로 로그를 저장합니다.

### 1. 학습 시작
```bash
python train_ray.py
```

### 2. TensorBoard 실행 (새 터미널에서)
```bash
tensorboard --logdir=./ray_results
```

### 3. 브라우저에서 확인
http://localhost:6006 접속

---

## 주요 메트릭 설명

### 📊 **Scalars 탭에서 확인할 것들**

1. **episode_reward_mean**
   - Zero-Sum 게임이므로 0에 수렴해야 정상
   - 한쪽으로 치우치면 학습 불균형

2. **episode_len_mean**
   - 평균 게임 길이
   - 너무 짧으면: 한쪽이 항상 폴드
   - 너무 길면: 아무도 폴드 안 함

3. **info/learner/default_policy/policy_loss**
   - Policy 네트워크 Loss
   - 감소 추세여야 함

4. **info/learner/default_policy/vf_loss**
   - Value Function Loss
   - 감소 추세여야 함

5. **info/learner/default_policy/entropy**
   - 탐험 정도
   - 초반: 높음 (랜덤 탐험)
   - 후반: 낮음 (활용)

6. **info/learner/default_policy/cur_lr**
   - 현재 Learning Rate
   - Learning Rate Scheduler 확인

---

## 팁

### 여러 실험 비교
```bash
# 실험1 실행
python train_ray.py

# 실험2 실행 (다른 설정으로)
python train_ray.py

# TensorBoard에서 두 실험 동시 비교
tensorboard --logdir=./ray_results
```

### 원격 접속
```bash
tensorboard --logdir=./ray_results --host=0.0.0.0 --port=6006
```

### 로그 정리
```bash
# 오래된 로그 삭제
rm -rf ./ray_results/*
```
