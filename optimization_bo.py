import numpy as np
from bayes_opt import BayesianOptimization
import sim_pipeline as sim
import os
import sys
import argparse
import imageio
import plotly.graph_objects as go

# =========================================================
# 0. 전역 설정
# =========================================================
# [목표 설정]
TARGET_DEFLECTION_MM = 0.0212  
HISTORY = []                  
SCALE_FACTOR = None           

print(f"--- Optimization Setup ---")
print(f"Target Max Deflection: {TARGET_DEFLECTION_MM} mm")
print(f"Strategy: Run full iterations to minimize error.")

# =========================================================
# 1. 목적 함수 정의
# =========================================================
def fem_objective_function(t_val):
    try:
        t = float(t_val)
        print(f"\n[BO] Probing Thickness t = {t:.4f}")

        # 1. 시뮬레이션 실행
        msh_file = sim.generate_mesh(t)
        mesh, u, basis = sim.run_simulation(msh_file, t)
        
        if u is None:
            return -1e5 

        # 2. 결과 시각화 (Dashboard)
        img_path = sim.visualize_dashboard(mesh, u, basis, t, fixed_scale=SCALE_FACTOR)

        # 3. Max Deflection 계산 (P2 Node Interpolation)
        M = basis.probes(mesh.p)
        u_at_nodes = M @ u
        n_vertices = mesh.p.shape[1]
        displacement = np.vstack((
            u_at_nodes[0*n_vertices : 1*n_vertices],
            u_at_nodes[1*n_vertices : 2*n_vertices],
            u_at_nodes[2*n_vertices : 3*n_vertices]
        )).T
        
        max_deflection_m = np.max(np.linalg.norm(displacement, axis=1))
        max_deflection_mm = max_deflection_m * 1000.0
        
        print(f"[BO] Result -> Max Deflection: {max_deflection_mm:.5f} mm")

        # 4. 점수 및 오차 계산
        # Bayesian Optimization은 Maximize를 하므로, Error에 음수를 붙여 Score로 사용
        error = abs(max_deflection_mm - TARGET_DEFLECTION_MM)
        score = -error
        
        # 이력 저장
        HISTORY.append({
            'step': len(HISTORY) + 1,
            't': t,
            'deflection': max_deflection_mm,
            'error': error,
            'img_path': img_path
        })
        
        return score

    except Exception as e:
        print(f"[BO] Error during simulation: {e}")
        return -1e5 

# =========================================================
# 2. Plotly 그래프 생성 (시뮬레이션 순서대로 변화 추이)
# =========================================================
def generate_plotly_history(history, target_val, output_dir):
    print("\n[System] Generating Optimization History Plot...")
    
    # Step 순서대로 정렬 (이미 순서대로지만 보장 차원)
    history_by_step = sorted(history, key=lambda x: x['step'])
    
    steps = [h['step'] for h in history_by_step]
    deflections = [h['deflection'] for h in history_by_step]
    thicknesses = [h['t'] for h in history_by_step]
    errors = [h['error'] for h in history_by_step]

    fig = go.Figure()

    # 결과값 변화 (파란 실선)
    fig.add_trace(go.Scatter(
        x=steps,
        y=deflections,
        mode='lines+markers',
        name='Max Deflection',
        marker=dict(size=8, color='blue'),
        line=dict(color='royalblue', width=2),
        text=[f"t={t:.3f}mm<br>Err={e:.5f}mm" for t, e in zip(thicknesses, errors)],
        hovertemplate="<b>Step %{x}</b><br>Def: %{y:.5f} mm<br>%{text}<extra></extra>"
    ))

    # 목표값 (빨간 점선)
    fig.add_trace(go.Scatter(
        x=[min(steps), max(steps)],
        y=[target_val, target_val],
        mode='lines',
        name='Target',
        line=dict(color='red', width=2, dash='dash'),
        hovertemplate="Target: %{y} mm<extra></extra>"
    ))

    fig.update_layout(
        title=f"Optimization History (Target: {target_val} mm)",
        xaxis_title="Simulation Step",
        yaxis_title="Max Deflection (mm)",
        template="plotly_white",
        hovermode="x unified"
    )

    html_path = os.path.join(output_dir, "optimization_history.html")
    fig.write_html(html_path)
    print(f"[System] Interactive Graph saved to {html_path}")

# =========================================================
# 3. Main Execution
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BO Optimization for FEM")
    parser.add_argument("--scale", type=float, default=None, help="Set fixed visual scale.")
    args = parser.parse_args()
    
    SCALE_FACTOR = args.scale
    if SCALE_FACTOR:
        print(f"[System] Visualization: Fixed Scale x{SCALE_FACTOR}")
    else:
        print(f"[System] Visualization: Auto Scale")

    # BO 초기화
    # 탐색 범위: -4.0 ~ 3.0 (실제 두께 약 0.1mm ~ 8mm)
    pbounds = {'t_val': (-4.0, 3.0)}

    optimizer = BayesianOptimization(
        f=fem_objective_function,
        pbounds=pbounds,
        random_state=42, # 결과 재현성을 위해 시드 고정
        verbose=2
    )

    print("\n[System] Starting Bayesian Optimization (Full Iterations)...")
    
    # [전략] 최대한 정밀하게 찾기 위해 횟수를 넉넉히 설정하고 중단 없이 수행
    # init_points: 초기 랜덤 탐색 (전역 탐색)
    # n_iter: 모델 기반 최적화 탐색 (정밀 탐색)
    optimizer.maximize(
        init_points=5,
        n_iter=15,
    )

    # -----------------------------------------------------
    # 최종 결과 리포트
    # -----------------------------------------------------
    best_params = optimizer.max['params']
    best_t = best_params['t_val']
    best_score = optimizer.max['target']
    best_error = -best_score

    print("\n============================================")
    print(f" [Optimization Finished] ")
    print(f" Target Deflection     : {TARGET_DEFLECTION_MM} mm")
    print(f" Best Parameter (t)    : {best_t:.5f}")
    print(f" Physical Thickness    : {5.0 + best_t:.5f} mm")
    print(f" Minimum Error         : {best_error:.6f} mm")
    print("============================================")

    if HISTORY:
        # 1. GIF 생성 (오차 기준 정렬: Worst -> Best)
        print("\n[System] Creating GIF sorted by Error (Convergence process)...")
        sorted_history = sorted(HISTORY, key=lambda x: x['error'], reverse=True)
        
        suffix = "fixed" if SCALE_FACTOR else "auto"
        gif_path = os.path.join(sim.OUTPUT_DIR, f"optimization_best_convergence_{suffix}.gif")
        
        image_list = []
        for item in sorted_history:
            try:
                img = imageio.v2.imread(item['img_path'])
                image_list.append(img)
            except Exception:
                pass

        if image_list:
            # 마지막(최적해) 프레임은 3초, 나머지는 0.5초 유지
            durations = [500] * (len(image_list) - 1) + [1000]
            
            print(f"[System] Saving GIF to {gif_path} ...")
            imageio.mimsave(gif_path, image_list, loop=0, duration=durations) # type: ignore
            print("[System] GIF creation complete!")

        # 2. Plotly 그래프 생성 (시뮬레이션 스텝별 추이)
        generate_plotly_history(HISTORY, TARGET_DEFLECTION_MM, sim.OUTPUT_DIR)