# 텍사스 홀덤 AI 구현 계획

# 목표 (Goal)
**강화학습(Deep Q-Learning 또는 PPO)**을 사용하여 헤즈업 노리밋 텍사스 홀덤 AI를 구축합니다. AI는 자가 플레이(Self-Play)를 통해 학습하며 GTO 전략에 근접하는 것을 목표로 합니다.

## 사용자 검토 필요 (User Review Required)
> [!IMPORTANT]
> **기술 스택 선정**: 강화학습의 표준인 **Python**과 **PyTorch** 사용을 제안합니다.
> - **엔진**: 커스텀 Python 클래스 vs `treys`/`poker-engine` 라이브러리? (학습 목적상 커스텀 구현 제안)
> - **알고리즘**: PPO보다 시작하기 쉬운 DQN(Deep Q-Network)을 제안합니다.

## 제안 아키텍처 (Proposed Architecture)

### 1. 게임 환경 (The "World")
AI가 상호작용할 수 있는 시뮬레이션 환경이 필요합니다.
- **상태 표현 (State Representation)**: AI가 게임을 인식하는 방식
    - 핸드 카드 (One-hot 인코딩)
    - 커뮤니티 카드
    - 팟 크기, 스택 크기
    - 딜러 위치
    - 이전 행동 기록
- **행동 공간 (Action Space)**: AI가 할 수 있는 행동
    - 폴드, 체크/콜, 레이즈 (이산적 금액 또는 연속적 금액)
- **보상 시스템 (Reward System)**:
    - 핸드 종료 시 획득한 칩(+) / 잃은 칩(-)

### 2. 에이전트 (The "Brain")
**상태(State)**를 입력받아 최적의 **행동(Action)**을 출력하는 신경망입니다.
- **입력층**: 상태 벡터의 크기 (예: 약 50-100개 입력)
- **은닉층**: 완전 연결 계층 (예: 128개 노드의 2개 층)
- **출력층**: 각 가능한 행동에 대한 Q값 (폴드, 콜, 최소 레이즈, 팟 레이즈, 올인)

### 3. 학습 루프 (The "Gym")
- **자가 플레이 (Self-Play)**: 에이전트 A vs 에이전트 B (A의 복제본)
- **경험 리플레이 (Experience Replay)**: 게임을 메모리에 저장하고 무작위 배치로 학습하여 상관관계 제거
- **타겟 네트워크 (Target Network)**: 학습 안정화를 위한 기술 (표준 DQN 기법)

## 변경 사항 (Proposed Changes)

### 프로젝트 구조
#### [NEW] [main.py](file:///C:/Users/99san/.gemini/antigravity/brain/b8d92e35-4398-49f0-baeb-b6fc33b47c16/main.py)
학습 또는 플레이를 위한 진입점입니다.

#### [NEW] [poker_env.py](file:///C:/Users/99san/.gemini/antigravity/brain/b8d92e35-4398-49f0-baeb-b6fc33b47c16/poker_env.py)
게임 로직 (덱, 카드, 핸드, 게임 상태)을 담당합니다.

#### [NEW] [agent.py](file:///C:/Users/99san/.gemini/antigravity/brain/b8d92e35-4398-49f0-baeb-b6fc33b47c16/agent.py)
신경망 모델(PyTorch)과 의사결정 로직을 포함합니다.

#### [NEW] [trainer.py](file:///C:/Users/99san/.gemini/antigravity/brain/b8d92e35-4398-49f0-baeb-b6fc33b47c16/trainer.py)
학습 루프를 관리하고 모델을 저장합니다.

## 검증 계획 (Verification Plan)
### 자동화 테스트
- **단위 테스트**: 포커 규칙 검증 (예: 플러시가 스트레이트를 이기는지, 팟 계산이 정확한지 등)
- **승률 확인**:
    1.  1,000 핸드 학습 진행
    2.  "무작위 에이전트"와 100 핸드 대결 (승률 80% 이상 목표)
    3.  "무조건 콜 에이전트"와 100 핸드 대결 (밸류 베팅 학습 여부 확인)
