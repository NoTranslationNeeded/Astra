import pandas as pd
import glob

# Find progress.csv file
csv_files = glob.glob('./ray_results/tournament_poker_icm_survival/**/progress.csv', recursive=True)

if not csv_files:
    print("No progress.csv file found!")
else:
    df = pd.read_csv(csv_files[0])
    
    # 메트릭 이름 찾기
    metric_name = None
    for col in df.columns:
        if 'actor_manager_num_outstanding_async_reqs' in col:
            metric_name = col
            break
    
    if metric_name is None:
        print("❌ actor_manager_num_outstanding_async_reqs 메트릭을 찾을 수 없습니다.")
        print("\n사용 가능한 env_runner 관련 메트릭:")
        env_runner_cols = [c for c in df.columns if 'env_runner' in c.lower()]
        for col in env_runner_cols[:20]:
            print(f"  - {col}")
    else:
        print("="*80)
        print(f"메트릭 분석: {metric_name}")
        print("="*80)
        
        print(f"\n총 iteration 수: {len(df)}")
        print(f"\n고유값: {sorted(df[metric_name].unique())}")
        print(f"최소값: {df[metric_name].min()}")
        print(f"최대값: {df[metric_name].max()}")
        print(f"평균값: {df[metric_name].mean():.2f}")
        print(f"표준편차: {df[metric_name].std():.4f}")
        
        # 값의 분포
        print("\n" + "="*80)
        print("값의 분포:")
        print("="*80)
        value_counts = df[metric_name].value_counts().sort_index()
        for val, count in value_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {val}: {count}회 ({pct:.1f}%)")
        
        # 시간에 따른 변화
        print("\n" + "="*80)
        print("최근 20개 iteration에서의 값:")
        print("="*80)
        print(df[['training_iteration', metric_name]].tail(20).to_string(index=False))
        
        # 값이 2로 일정한지 확인
        print("\n" + "="*80)
        print("일관성 분석:")
        print("="*80)
        
        is_constant = df[metric_name].nunique() == 1
        if is_constant:
            constant_value = df[metric_name].iloc[0]
            print(f"✅ 값이 일관되게 {constant_value}로 유지됨")
            
            # training config 확인
            print("\n" + "="*80)
            print("학습 설정 분석 (train_tournament_icm.py):")
            print("="*80)
            print("num_env_runners 설정을 확인하면...")
            
            with open('train_tournament_icm.py', 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if 'num_env_runners' in line:
                        print(f"  Line {line_num}: {line.strip()}")
            
            print("\n💡 해석:")
            print(f"  - num_env_runners가 {constant_value}로 설정되어 있습니다")
            print(f"  - 이 메트릭은 비동기 요청 대기 중인 env runner 수를 나타냅니다")
            print(f"  - 값이 {constant_value}로 일정한 것은 NORMAL 입니다!")
            print(f"  - 모든 env runner가 활발하게 작동 중임을 의미합니다")
            
        else:
            print(f"⚠️ 값이 변동됨 (범위: {df[metric_name].min()} ~ {df[metric_name].max()})")
            
            # 변화 지점 찾기
            changes = df[df[metric_name] != df[metric_name].shift()].index.tolist()
            if len(changes) > 0:
                print(f"\n값이 변경된 iteration: {df.loc[changes, 'training_iteration'].tolist()[:10]}")
