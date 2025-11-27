import pandas as pd
import glob

# Find progress.csv file
csv_files = glob.glob('./ray_results/tournament_poker_icm_survival/**/progress.csv', recursive=True)

if not csv_files:
    print("No progress.csv file found!")
else:
    df = pd.read_csv(csv_files[0])
    
    print("="*80)
    print("ray/tune/done 메트릭 분석")
    print("="*80)
    print(f"\n총 iteration 수: {len(df)}")
    print(f"\ndone 컬럼의 고유값: {df['done'].unique()}")
    print(f"\ndone=True 카운트: {(df['done'] == True).sum()}")
    print(f"done=False 카운트: {(df['done'] == False).sum()}")
    
    # 사용 가능한 에피소드 관련 컬럼 찾기
    episode_cols = [c for c in df.columns if 'episode' in c.lower() and ('mean' in c.lower() or 'len' in c.lower())]
    
    print("\n" + "="*80)
    print("최근 10개 iteration 상태:")
    print("="*80)
    
    # 표시할 컬럼 선택
    display_cols = ['training_iteration', 'done']
    if 'env_runners/episode_return_mean' in df.columns:
        display_cols.append('env_runners/episode_return_mean')
    if 'env_runners/episode_len_mean' in df.columns:
        display_cols.append('env_runners/episode_len_mean')
    
    print(df[display_cols].tail(10).to_string())
    
    # done 값의 변화 분석
    if 'done' in df.columns:
        print("\n" + "="*80)
        print("done 값 변화 분석:")
        print("="*80)
        
        # done이 True인 지점 찾기
        done_true_indices = df[df['done'] == True].index.tolist()
        
        if done_true_indices:
            print(f"\ndone=True가 발생한 iteration: {df.loc[done_true_indices, 'training_iteration'].tolist()}")
            print("\n⚠️ WARNING: Trial이 중간에 종료되었습니다!")
        else:
            print("\n✅ 모든 iteration에서 done=False (정상 진행 중)")
            print("   학습이 계속 진행되고 있으며, 목표 iteration(200)에 도달하면 done=True가 됩니다.")
    
    print("\n" + "="*80)
    print("학습 진행도:")
    print("="*80)
    current_iter = df['training_iteration'].max()
    target_iter = 200
    progress_pct = (current_iter / target_iter) * 100
    print(f"현재 iteration: {current_iter} / {target_iter}")
    print(f"진행도: {progress_pct:.1f}%")
    print(f"남은 iteration: {target_iter - current_iter}")

