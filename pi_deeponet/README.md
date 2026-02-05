# PI-DeepONet (Stable) Full Code

이 패키지는 3D 선형탄성(고정단 + 원형 분포하중) 빔 문제에 대해
DeepONet을 데이터(fem) + 물리(Dirichlet + Traction + Energy-balance)로 학습하는
PI-DeepONet 구현입니다.

이번 버전은 다음 안정화(발산 방지) 솔루션이 반영되어 있습니다.

1) **Hard Dirichlet BC (u=0)**  
   - 길이축 정규화 좌표 `x_hat[...,0] == -1`에서 변위가 항상 0이 되도록
     `u = g(x)*u_free`,  g(x) = (x_len_hat + 1)/2 를 적용합니다.
   - soft penalty만 쓸 때 발생 가능한 rigid-body/치팅을 구조적으로 차단합니다.

2) **물리 loss의 단위/스케일 정합**
   - traction: `(σn - t)/|t|` 형태로 무차원화
   - energy: (U, W)를 에너지(J) 단위로 적분 근사한 뒤,
     Clapeyron 정리 기반 `2U - W = 0` 잔차를 무차원화하여 제곱 페널티로 사용합니다.
   - 좌표 입력이 [-1,1] 정규화이므로, autograd로 얻은 ∂u/∂x_hat 을
     체인룰로 ∂u/∂x_phys 로 변환합니다. (**이게 빠지면 내부에너지(U)가 과소추정되어 발산합니다.**)

3) **2-stage 학습**
   - Stage-1: data-only로 먼저 맞춘 뒤
   - Stage-2: physics 항을 ramp-up으로 천천히 켭니다.

4) **샘플링 개선 + 안정화**
   - 내부점(interior) 샘플을 고정단 근처로 bias 줄 수 있음
   - gradient clipping 지원

---

## Quickstart

### 1) 데이터 생성

```bash
python generate_data_pi.py --num_samples 80 --t_min -4 --t_max 3 --p_points 6000 --out data/beam_dataset_pi.npz --use_onshape
python check_data_pi.py --dataset data/beam_dataset_pi.npz
```

### 2) 학습

기본(권장): 2-stage

```bash
python train_pi_deeponet.py --dataset data/beam_dataset_pi.npz --epochs 2000 --stage1_epochs 400 --batch_size 8 --device cuda
```

### 3) 정량 평가

```bash
python test_accuracy_pi.py --t -1.5 --ckpt checkpoints_pi/best_pi_deeponet.pth --device cpu
```

---

## Onshape 설정(권장)

코드에 API key를 하드코딩하지 않습니다.
환경변수로 설정하세요.

- ONSHAPE_API_KEY
- ONSHAPE_SECRET_KEY
- ONSHAPE_DOC_ID
- ONSHAPE_WSP_ID
- ONSHAPE_ELM_ID

Windows 예시:

```bat
set ONSHAPE_API_KEY=...
set ONSHAPE_SECRET_KEY=...
set ONSHAPE_DOC_ID=...
set ONSHAPE_WSP_ID=...
set ONSHAPE_ELM_ID=...
```

---

## Notes

- 이 구현은 geometry가 “직육면체 빔에 가까운 형상”이라는 가정 하에
  canonical box([-1,1]^3)에서 physics collocation을 수행합니다.
- 형상이 많이 복잡해지면(큰 필렛/홀 등) geometry mask 또는 SDF 기반 collocation이 필요할 수 있습니다.
