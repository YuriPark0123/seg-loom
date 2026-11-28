# 데이터셋 형식과 생성 방법

> 데이터셋 버전과 변경 내역은 Hugging Face [`yrpark/seg-loom`](https://huggingface.co/datasets/yrpark/seg-loom) README의 "변경 내역"에 있다. 현재 v1 (렌즈 접힘 수정 반영), 이전 버전은 태그 `v0-pre-lens-fix`.

[ARCHITECTURE.md](ARCHITECTURE.md) A.4(데이터 엔진, M8)를 구현한 결과물이다. MuJoCo EGL이 되는 곳이면 어디서나 생성할 수 있다. RTX 3090과 B200 모두에서 동작을 확인했다.

## 1. 구성 요소

| 파일 | 역할 |
|---|---|
| `shadow/procgen.py` | 절차적 로봇 생성. 관절 3~7개, 링크 길이·반경·축, 측면 오프셋, 프레임 재배향, 관절 한계, 외형(원기둥/박스, 색), 그리퍼(평행 prismatic / 회전형, mimic)를 무작위로 정한다. URDF와 spec을 쓰고, "그리퍼가 아래를 향하는 기본 자세"를 IK로 구한다. 작업 공간 도달률이 75% 미만인 설계는 버린다 |
| `shadow/urdf_perturb.py` | 오차가 섞인 근사 URDF(`approx.urdf`)를 만든다. 링크 길이 ±10%, 관절 영점 ±5°, 도형 크기 ±15%, 원기둥↔박스 교체, 메시→경계 박스, 링크 시각 정보 누락. 적용한 오차는 모두 `perturb.json`에 기록한다 |
| `shadow/datagen.py` | 장면 무작위화, 카메라 모델·배치·움직임, 액추에이터 오차, 동작 생성(motion / ref / task), 렌더링과 기록 |
| `scripts/gen_dataset.py` | 병렬 생성. 로봇 하나가 작업 하나이고, 중단 후 다시 실행하면 이어서 생성한다 |
| `scripts/view_episode.py` | 에피소드 시각 점검. RGB, visible 라벨, amodal 로봇, amodal 객체, skeleton과 객체 중심을 나란히 그린다 |

## 2. 생성 명령

```bash
conda activate shadow   # mujoco 2.3.7, numpy<2, opencv-python<4.11, scipy, trimesh, pycollada, pyyaml
export MUJOCO_GL=egl

# 평가셋: 실제 로봇 4종 + 처음 보는 절차적 로봇 50종
python scripts/gen_dataset.py --out data/eval --robots so101 ur7e piper panda --procgen 50 --seed 100000 \
    --modes motion:12 task:8 --workers 12
# 렌더러 검증 기준셋: 정적인 무작위 자세, 이상적인 액추에이터
python scripts/gen_dataset.py --out data/ref --robots so101 ur7e piper panda --procgen 196 --seed 200000 \
    --modes ref:5 --T 10 --workers 12
# 학습셋 (B200): seed 범위가 평가셋과 겹치지 않게 한다
python scripts/gen_dataset.py --out data/train_000 --procgen 2000 --seed 0 --modes motion:10 task:10 \
    --workers 32 --gpus 0 1 2 3 4 5 6 7
# 3090에서 실제로 만든 학습셋: 로봇 200종씩 생성 → HF 업로드 → 확인 → 로컬 삭제 (디스크 부족 대응)
bash scripts/gen_train_chunks.sh 0 9 24                                    # train/r0000..r1800 (seed 0~1999, 단순 장면)
SPLIT=train_clutter CLUTTER=1.0 bash scripts/gen_train_chunks.sh 10 14 24  # train_clutter/r2000..r2800 (seed 2000~2999, 복잡한 장면)
# 공개 실제 로봇 (2.1절): 로봇마다 인스턴스(근사 URDF·부착물·색) 여러 개
SPLIT=eval_real  ROBOTS="<extra_splits.yaml의 eval 7종>"  K=4  SEED=100000 MODES="motion:12 task:8" bash scripts/gen_train_chunks.sh 0 0 24
SPLIT=train_real ROBOTS="<extra_splits.yaml의 train 21종>" K=10 SEED=0 RECOLOR=0.5 CLUTTER=0.5 bash scripts/gen_train_chunks.sh 0 3 24
```

### 2.1 공개 실제 로봇 (28종)

- `python scripts/onboard_robots.py`가 공개 URDF/xacro를 받아 `assets/robots_train/<이름>/<이름>.urdf`(메시는 `package://extra/...`)와 `configs/robots/<이름>.yaml`을 만든다. 하는 일: xacro 전개(`$(find pkg)`는 package.xml로 해석), 팔 관절과 그리퍼 관절 자동 분리, 공구 방향(마지막 관절 축)과 TCP 추정 후 `tcp_frame` 링크 추가(z = 접근 방향), 그리퍼 열림/닫힘 방향 확인, reach와 "그리퍼 아래로" 기본 자세 탐색(작업 공간 75% 이상 도달), 손목 카메라 위치 자동 조정(로봇 안이 아니고, 화면의 1~30%에 공구가 보이는 가장 가까운 위치).
- 출처는 `scripts/fetch_assets.sh`에 커밋으로 고정했고, 저장소·라이선스·변환 결과는 `configs/robots/extra_report.json`에 있다. 라이선스는 BSD-3 / Apache-2.0 / MIT만 썼다(GPL인 poppy, 라이선스 표기가 없는 Unitree Z1은 뺐다).
- 분할은 `configs/robots/extra_splits.yaml`. **평가용 7종은 계열째로 뺐다**: KUKA(iiwa7, iiwa14), Flexiv(rizon4, rizon10), Rethink Sawyer, I2RT YAM, Comau e.DO. 학습용 21종: Fanuc 5, Kinova 4(gen2, gen3 lite, Jaco2, Mico), UFactory 5(xArm5/6/7, Lite6, 850), ROBOTIS 3(OMX-F, OMY-3M, OpenManipulator-X), Interbotix 4(PX150, RX200, WX250s, VX300s).
- 원래 평가 로봇 4종과 그 사촌 기종(SO-ARM 계열, UR 전 계열, Franka FER/FR3, AgileX Piper)은 학습에 넣지 않았다.
- `--instances A B`: 실제 로봇마다 인스턴스 A..B-1을 만든다. 인스턴스마다 근사 URDF 오차, 부착물, 손목 카메라 오차가 따로 뽑힌다(`<이름>_iNNN`). `--recolor p`: 확률 p로 로봇을 다시 칠한다(전체 색조 / 두 가지 색 / 링크마다 무작위). 같은 로봇의 인스턴스들은 메시 파일을 하드링크로 공유해 tar에는 한 번만 들어간다.

- `--seed`는 절차적 로봇 번호의 시작값이다. 로봇 이름은 `pg{seed+i:07d}`이다. 분할(split)끼리 범위가 겹치면 안 된다. 기본 배정은 학습 0~99,999, 평가 100,000~, 기준셋 200,000~, smoke 900,000~이다.
- `--clutter p`를 주면 에피소드마다 확률 p로 복잡한 장면(5절)을 만든다. 0이면 난수를 쓰지 않아 기존과 같은 장면·궤적·라벨이 나온다. RGB는 GPU 렌더 특성상 같은 코드로 다시 돌려도 몇십 픽셀이 밝기 1~3 정도 달라질 수 있다.
- `--level 0`을 주면 근사 URDF가 정답과 같아진다. 오차의 영향을 측정하는 실험에 쓴다.
- `--gpus`를 주면 worker들을 해당 GPU(nvidia-smi 번호)에 나눠 배정한다. EGL 장치 번호는 nvidia-smi 번호와 순서가 다를 수 있어서 (B200 서버: EGL 0 = GPU 4, EGL 4 = GPU 0) `shadow/egl_devices.py`가 CUDA 번호로 대응을 찾아 `MUJOCO_EGL_DEVICE_ID`를 정한다. 직접 `MUJOCO_EGL_DEVICE_ID`를 쓸 때는 `python -m shadow.egl_devices`로 표를 먼저 확인한다.

## 3. 디렉터리 구조

```
<out>/
  config.json            생성 인자
  manifest.jsonl         완료된 에피소드 한 줄씩: episode, robot, kind(real/procgen), mode, T, instruction, success
  errors.log
  procgen/<robot>/       생성기 출력 (robot.urdf, spec.yaml, mj/ 변환 캐시)
  robots/<robot>/
    true.urdf            세계가 쓰는 정답 URDF (메시 경로는 상대 경로)
    approx.urdf          모델에 주는 근사 URDF (A.4의 "URDF 오차")
    meshes/              두 URDF가 참조하는 메시 사본 (폴더째 옮겨도 동작)
    perturb.json         근사 URDF에 넣은 오차 (관절별 길이 비율, 영점 deg, 도형 변경, 누락 링크)
    extras.json          URDF에 없는 부착물 (케이블, 박스), 손목 카메라 장착 오차, 로봇 body 목록
    spec.yaml            팔/그리퍼 관절, ee body, TCP, 기본 자세, reach
    render_geoms.npz     MuJoCo가 그린 로봇 시각 geom 전체 (부착물 포함): body 인덱스, 종류, 크기, 지역 pose, rgba, 메시 정점/면
  episodes/<robot>_<mode>_<k>/
    meta.json  arrays.npz  frames.npz
```

## 4. 에피소드 내용

### 4.1 모드

| mode | 내용 | 액추에이터 | 카메라 움직임 |
|---|---|---|---|
| `motion` | 1단계: task 없음. 무작위 EEF 경유점(대부분 그리퍼 하향, 기울기 0~0.5rad) + 관절 흔들기 + 그리퍼 열고 닫기 | 오차 있음 | static / drift / shake / jump 무작위 |
| `task` | 2단계: 스크립트 keyframe "pick up X" / "put X on Y". 도달할 수 있는 객체와 자세만 고른다. 잡기는 기구학적으로 붙이는 방식 | 오차 있음 | 무작위 |
| `ref` | 렌더러 검증용. 프레임마다 독립적인 무작위 자세 | 이상적 | static |

### 4.2 좌표와 규약

- **기준 좌표계는 로봇 베이스**(`robot_mount`)다. 모든 3D 값은 베이스 기준이다.
- **카메라는 OpenCV 규약**(x 오른쪽, y 아래, z 앞)이다. `T_base_cam`은 카메라→베이스 변환(4×4)이다.
- **쿼터니언 순서**
  - `arrays.npz`의 pose(7) = `[x, y, z, qx, qy, qz, qw]` (scipy 순서)
  - `meta.json`/`extras.json`의 `quat`과 `render_geoms.npz`의 `quat` = `[w, x, y, z]` (MuJoCo 순서)
- **명령과 프레임 주기:** 10Hz. 액추에이터는 50Hz 하위 스텝으로 적분한다.

### 4.3 `arrays.npz` (T = 프레임 수, n = 팔 관절 수, g = 그리퍼 관절 수)

| 키 | 형태 | 설명 |
|---|---|---|
| `q_cmd` | (T, n) | 팔 관절 명령 (action, 관절 공간) |
| `grip_cmd` | (T, 1) | 그리퍼 명령, 열림 비율 0(닫힘)~1(열림) |
| `tcp_cmd_pose` | (T, 7) | `q_cmd`의 FK로 구한 명령 EEF pose (**로봇과 무관한 action**) |
| `q_true` | (T, n+g) | 실제 관절값 (지연, 1차 지연, 백래시 적용 후) |
| `q_enc` | (T, n+g) | encoder 읽기값 = 실제값 + 노이즈 (모델 입력은 이것) |
| `tcp_pose` | (T, 7) | 실제 TCP pose |
| `body_pose` | (T, Nb+No, 7) | 로봇 body(Nb개, `meta.robot_bodies` 순서) 다음에 객체(No개)의 pose |
| `obj_pose` | (T, No, 7) | 객체 pose |
| `joint_anchor`, `joint_axis` | (T, J, 3) | 로봇 모든 관절(`meta.joint_names`)의 위치와 축 |
| `skel3d` | (T, K, 3) | skeleton 점 (`meta.skeleton_names`: base, 팔 관절, ee, tcp, 손가락 끝) |
| `skel2d_{cam}` | (T, K, 3) | 위 점의 카메라 투영 `[u, v, 화면 안 여부]` (렌즈 왜곡 적용) |
| `obj2d_{cam}` | (T, No, 3) | 객체 중심의 투영 |
| `T_base_cam_{cam}` | (T, 4, 4) | 카메라 pose (움직이는 카메라, 손목 카메라 포함) |
| `attached` | (T, 1) | 그리퍼에 붙은 객체 인덱스, 없으면 -1 |

`{cam}` ∈ `top`, `front`, `wrist`.

### 4.4 `frames.npz`

카메라와 종류별로 인코딩된 바이트를 이어 붙인 배열과 오프셋이 들어 있다. `datagen.unpack_image(F[k], F[k+"_off"], t, kind)`로 꺼낸다.

| 키 | 형식 | 내용 |
|---|---|---|
| `{cam}_rgb` | JPEG q92 | 깨끗한 RGB. 화질 저하(`camera_model.degrade`)는 학습할 때 실시간으로 적용한다 |
| `{cam}_vis` | PNG uint8 | 보이는 라벨 |
| `{cam}_am_r` | PNG uint8 | amodal 로봇 라벨 (다른 모든 것을 숨기고 렌더) |
| `{cam}_am_o` | PNG uint8 | amodal 객체 라벨 (로봇을 숨기고 렌더; 객체끼리의 가림은 남음) |

**라벨 값:** 0 배경 · 1..Nb 로봇 body(`robot_bodies` 인덱스+1; 부착물은 붙은 링크의 값) · 100+k 객체 k · 200 테이블 · 201 테이블 다리 · 202 바닥 · 210+k 테이블 위 방해 물체 k (복잡한 장면) · 250 벽·가구·가림 물체 (복잡한 장면)

### 4.5 `meta.json`

| 키 | 내용 |
|---|---|
| `robot`, `mode`, `fps`, `T` | 에피소드 기본 정보 |
| `cameras.{cam}.model` | width, height, fx, fy, cx, cy, kind(pinhole/brown/fisheye), dist |
| `cameras.{cam}.motion` | static / drift / shake / jump / mounted |
| `scene` | 테이블·바닥 색, 조명, 객체(이름, 색 이름, rgba, 반 크기 half, 초기 위치, yaw) |
| `task` | kind(pick/stack), src, dst, instruction, grasp_width, success, keyframes[] |
| `task.keyframes[]` | name(pregrasp/grasp/close/lift/preplace/place/release/retreat/hold), frame, tcp_base, tcp_world, R_world, grip, ik_err |
| `actuator` | delay_ms, tau_s, backlash_rad(관절별), enc_noise_rad |
| `arm_joints`, `gripper_joints`, `robot_bodies`, `joint_names`, `skeleton_names` | 배열 인덱스의 의미 |
| `gripper_width_table` | 열림 비율 ↔ 손가락 끝 간격(m). 정책의 "그리퍼 폭"을 로봇별 관절값으로 바꿀 때 쓴다 |

## 5. 무작위화 범위 (현재 값)

| 축 | 범위 |
|---|---|
| 로봇 | 절차적: 관절 3~7개, 크기 S 0.3~1.0m, 링크 스타일 2종, 색 10종, 그리퍼 2종 / 실제 평가: so101, ur7e, piper, panda + 공개 7종 / 실제 학습: 공개 21종 × 인스턴스 40 (2.1절) |
| URDF 오차 | 길이 N(0, 5%)를 ±10%로 자름, 영점 U(±5°), 도형 ±15%, 원기둥↔박스 20%, 메시→박스 30%, 링크 누락 10% |
| URDF에 없는 부착물 | 케이블(캡슐) / 박스 0~3개, 손목 카메라 장착 오차 위치 σ6mm, 각도 σ2.5° |
| 카메라 모델 | pinhole / Brown / fisheye(KB), 화각 45~100°(fisheye 100~175°), 주점 오차, 출력 폭 320px |
| 카메라 배치 | top: 높이 1.4~2.4R, 기울기 약 0~15°, 임의 yaw / front: 방위 ±75°, 고각 12~60°, 거리 1.1~2.1R (광각일수록 가깝게), roll σ4° |
| 카메라 움직임 | static, drift(부드러운 이동), shake(흔들림+이동), jump(중간에 순간 이동) |
| 장면 | 테이블 색 (30%는 흰색), 바닥 체커 색, 방향광 1~3개, 주변광, 블럭 2~5개 (흰색 50%, 같은 색 쌍 30%) |
| 복잡한 장면 (`--clutter`) | 절차적 텍스처(나무·노이즈·대리석·타일·줄무늬·카펫·포스터·단색)를 테이블(80%)·바닥·벽에 입힘, 하늘색 무작위, 조명 색조 ±12%, 스포트라이트 50% / 테이블 위 방해 물체 3~10개(공·원기둥·캡슐·타원체·머그·병·캔, 블럭 모양은 없음, 높은 것은 로봇에서 0.8R 밖) / 바닥 가구 0~4개(카메라 시선은 피함) / 벽 1~3개(모든 카메라보다 바깥) / 가림 물체 0~2개(기둥·막대·판, 카메라와 로봇 사이. 로봇이 50% 미만만 보이면 제거) |
| 액추에이터 | 지연 0~40ms, 1차 지연 20~120ms, 백래시 0~8mrad, encoder 노이즈 0~3mrad |

## 6. 비용 (RTX 3090, 28코어, worker 12개)

- 에피소드 하나: worker 하나 기준 약 5~13초 (카메라 3대 × 렌더 4종, 어안 렌즈는 큐브맵 5면).
- 용량: 약 3MB/에피소드 (T≈60, 320px).
- smoke 검증 (실제 4종 + 절차적 8종, task 48개): 성공 46/48. 실패 2개는 그 로봇의 손이 블럭에 닿지 않는 경우였고, `success: false`로 기록했다.

**생성 완료 (2026-11-26, 3090)**

| 분할 | 로봇 | 에피소드 | 프레임 | 용량 | 소요 시간 | task 성공 |
|---|---|---|---|---|---|---|
| `data/eval` | 실제 4 + 절차적 50 (seed 100000~) | 1080 (motion 648, task 432) | 59,843 × 카메라 3대 | 2.4GB | 15분 | 397/432 (절차적 366/400, so101·panda·piper 8/8, ur7e 7/8) |
| `data/ref` | 실제 4 + 절차적 196 (seed 200000~) | 1000 (ref, 각 10프레임) | 10,000 × 카메라 3대 | 0.5GB | 7분 | — |
| HF `train/r0000`~`r1800` (2026-11-27, 3090) | 절차적 2,000 (seed 0~1999), 단순 장면 | 40,000 (motion 20,000, task 20,000) | 2,170,860 × 카메라 3대 | 약 90GB | 6시간 (worker 24) | 18,677/20,000 (93.4%) |
| HF `train_clutter/r2000`~`r2800` (2026-11-27, 3090) | 절차적 1,000 (seed 2000~2999), 복잡한 장면 | 20,000 (motion 10,000, task 10,000) | 1,087,588 × 카메라 3대 | 약 63GB | 3시간 (worker 24) | 9,318/10,000 (93.2%) |
| HF `eval_real/i000` (2026-11-27, 3090) | 공개 실제 평가 7종 × 인스턴스 4 (원래 색, 단순 장면, seed 100000) | 560 (motion 336, task 224) | 30,969 × 카메라 3대 | 1.5GB | 7분 | 205/224 (91.5%) |
| HF `train_real/i000`~`i030` (2026-11-28, 3090) | 공개 실제 학습 21종 × 인스턴스 40 (recolor 0.5, clutter 0.5) | 16,800 (motion 8,400, task 8,400) | 907,359 × 카메라 3대 | 51.9GB | 2.5시간 (worker 24) | 7,890/8,400 (93.9%) |

학습셋 HF 형식: `train/r0000`~`r0600`은 에피소드 파일이 개별로 올라가 있고, `r0800`부터와 `train_clutter/`는 폴더마다 `robots.tar` + `episodes_000.tar`~`007.tar`(에피소드 500개씩)로 묶여 있다(HF 파일 단위 다운로드 제한 대응). 풀기: `for f in <폴더>/*.tar; do tar xf $f -C <폴더>; done`.

`eval_real`/`train_real` 폴더도 같은 tar 형식이다(`robots.tar`에 `robots/<이름>_iNNN/` 인스턴스별 근사 URDF·메시). 로봇 이름은 `<로봇>_iNNN`, 분할은 `configs/robots/extra_splits.yaml`, 라벨 값은 위와 같다. 실제 로봇 task 성공률은 로봇마다 다르다: 큰 Fanuc 2종(m710ic70 236/400, m10ia 290/400)과 crx10ial 318/400이 낮고 나머지는 89~100%. 2026-11-28: 공구 방향(tcp_frame)이 뒤집혀 있던 Fanuc 5종·xarm5(train_real)와 sawyer(eval_real)를 고쳐서(04eb5cf) 같은 시드로 다시 만들어 교체했다(`gen_train_chunks.sh`의 `FIX=`). 예전엔 fanuc_cr7ia가 0/400이었다.

task 실패는 모두 "도달 가능한 블럭 배치가 없는" 경우다. 실패로 기록해 두었으므로, 실행 가능성 피드백(M7)을 학습하는 데 그대로 쓸 수 있다.

## 7. B200 환경 메모 (확인된 사항)

- 드라이버 570.211 / CUDA 12.8 툴킷: 정상.
- MuJoCo 2.3.7 EGL: B200에서 렌더링 정상 → **데이터 생성도 B200에서 가능**.
- 기존 env의 torch 2.7.1+cu126은 sm_90까지만 지원 → B200(sm_100)에서 `no kernel image` 오류. **cu128 빌드가 필요**하다:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  python -c "import torch; print(torch.cuda.get_arch_list()); torch.zeros(1, device='cuda')"   # sm_100 포함 확인
  ```
- nvdiffrast를 쓸 경우 `TORCH_CUDA_ARCH_LIST="10.0"`으로 소스 빌드한다. 학습 중 렌더 비교 층(M3)은 순수 torch 점 splatting으로도 만들 수 있다.

## 8. 아직 하지 않은 것

- 접촉 물리로 잡기 검증: 지금은 기구학적으로 붙이는 방식이다. MuJoCo 물리 검증은 CPU 작업이라 B200 서버에서 할 수 있다.
- 사람 손에 의한 가림, 사진 텍스처(지금은 절차적 텍스처만).
- 로봇 여러 대가 함께 있는 장면.
- 형태가 블럭이 아닌 조작 대상(방해 물체로는 원기둥·복합 형상이 있지만 잡는 대상은 블럭뿐).

## 9. 버그 기록: 그리퍼 연동 관절이 항상 0으로 그려지던 문제 (2026-11-28)

**증상.** 일부 로봇에서 그리퍼 손가락 하나(또는 여러 링크)가 열고 닫는 명령과 상관없이 항상 0 위치에 그려졌다. 이미지와 라벨(visible/amodal 마스크, 자세)도 모두 그 상태로 만들어졌다. B200 세션이 xarm, px150, rx200, wx250s, vx300s, kinova_gen3_lite, open_manipulator_x, omx_f에서 처음 발견했다.

**원인 1: 연동(mimic) 관절을 아무도 채우지 않았다.**
- URDF의 `<mimic>` 관절(다른 관절을 따라 움직이는 관절)은 MuJoCo로 바꿀 때 `<equality joint>` 제약이 된다(`shadow/urdf_robot.py`).
- MuJoCo에서 이 제약은 물리 계산(`mj_step`)을 돌릴 때만 맞춰진다.
- 데이터 생성은 물리를 돌리지 않는다. 관절각을 `qpos`에 직접 넣고 `mj_kinematics`로 자세만 계산한다.
- 그래서 config의 `gripper.joints`에 적힌 관절만 움직이고, 그것을 따라가야 할 관절은 처음 값 0에 그대로 남았다.
- onboarding 코드에는 "mimic 관절은 MuJoCo에서 자동으로 따라간다"는 잘못된 가정이 주석으로 남아 있었다.

**원인 2: 그리퍼 open/close가 거꾸로 저장되어 있었다.**
- onboarding은 관절 한계의 위쪽 값을 open, 아래쪽 값을 close로 정한다. 그다음 `finalize`에서 손가락 사이 거리를 재서 open 쪽이 더 좁으면 둘을 바꾼다.
- 하지만 이 거리를 재는 `gripper_width_table`은 "닫힌 상태 대비 변화량의 절댓값을 누적한 값"을 돌려준다. 그래서 첫 값이 항상 0이고, 바꾸는 조건이 한 번도 참이 될 수 없었다.
- 또 관절 하나로 구동하는 그리퍼(xarm, Robotiq, UMI 등)는 측정할 손가락 끝이 하나뿐이라 거리가 늘 0이었다.
- 결과: xarm6, xarm7, uf850, kinova_gen3_lite, arx_l5와 모든 Robotiq/UMI 조합에서 `open` 값이 실제로는 닫힌 자세였다.
- omx_f는 URDF의 그리퍼 한계가 ±2π(한 바퀴)라서 open/close 값 자체가 의미가 없었다.

**영향 범위.**
- 연동 관절이 0에 고정된 로봇:
  - xarm6, xarm7, uf850, px150, rx200, wx250s, vx300s, kinova_gen3_lite, open_manipulator_x, omx_f (`train_real` 21종 중 10종)
  - arx_l5, seeed_devarm, trossen_wxai
  - 36개 그리퍼 조합 전부 (`<팔>__robotiq`, `<팔>__umi`)
- 영향 없음:
  - 절차적 로봇(`train`, `train_clutter`, `eval`, `ref`): 두 손가락 관절을 모두 직접 지정한다.
  - `eval_real` 7종, 원래 평가 4종(panda, piper, so101, ur7e).
- 점검 방법: 모든 `configs/robots/*.yaml`에 대해 MuJoCo 모델의 joint equality를 훑고, 따라가는 관절이 `arm_joints`나 `gripper.joints`에 없는 로봇을 찾았다.

**해결 (커밋 526502f).**
1. `shadow/kinematic_mask.py`에 `apply_joint_couplings(model, data, explicit)`를 추가했다.
   - joint equality의 다항식 `q1 - q1_0 = c0 + c1·(q2 - q2_0) + …`로 따라가는 관절 값을 계산해 넣는다.
   - 따라가는 관절의 따라가는 관절이 있을 수 있어 여러 번 반복한다.
   - 직접 지정한 관절은 건드리지 않는다.
   - 관절을 설정하는 모든 곳에서 호출한다: `datagen.RenderWorld.set_q`, `mj_world.World.set_joints`, `kinematic_mask.RobotKinematics.set_joints`, `calibration`의 `set_joints`.
2. `gripper_width_table`이 연동 관절이 있는 로봇에서는 그리퍼 관절(구동 + 연동)이 움직이는 모든 geom 사이의 최대 거리로 폭을 잰다. `raw=True`는 절대 거리를 돌려준다.
3. `scripts/onboard_robots.py`의 `finalize`는 open/close를 비교할 때 이 절대 거리(`raw=True`)를 쓴다.
   - URDF 한계가 실제 행정이 아닌 경우를 위해 `GRIP_RANGE`를 두었다. omx_f는 MoveIt srdf에 따라 open 1.0, close 0.0이다.
4. 기존 config 41개의 open/close를 고쳤다. 다른 값(rest 자세 등)은 그대로 두었다.
5. 검증:
   - 그리퍼 근접 렌더에서 수정 전에는 한쪽 손가락만, 수정 후에는 양쪽이 같이 열리고 닫혔다.
   - 시험 생성(10종 × 인스턴스 2)에서 손가락이 로봇 마스크에 포함되었다. task 성공은 112/120이었다.

**데이터 교체.**
- `train_real/i000`~`i030`: 위 10종만 같은 시드로 다시 만들어 덮어썼다(`FIX=`).
- `eval_real2/i000`: 전체를 다시 만들었다.
- `train_real2`: 별도 저장소 `yrpark/seg-loom-real2`에 처음부터 다시 만든다. 수정 전 v2 데이터는 쓰지 않는다.
- `grip_cmd`(1 = open)는 고친 config를 따른다. 수정 전 데이터에서 위 open/close가 뒤바뀐 로봇의 `grip_cmd`는 의미가 반대였다.

## 10. 링크 길이 오차 상한 1cm (2026-11-28 15시부터)

근사 URDF의 링크 길이 오차를 비율(`--len-frac`)로 뽑은 뒤 **관절 원점마다 절대값 1cm를 넘지 않도록 자른다**(`--len-max-m 0.01`, `gen_train_chunks.sh`의 기본값 `LENMAX=0.01`). 난수 뽑기 횟수는 같으므로 같은 시드에서 다른 무작위 값(영점, 형상, 색 등)은 그대로이고 길이 오차 크기만 바뀐다. `perturb.json`에 `len_max_m`이 기록된다.

- 모든 세트를 같은 시드로 다시 만들어 HF에서 교체한다(2026-11-28 15:24 시작). 순서: `eval_real2` → `eval` → `ref` → `eval_real` → `train_real` → `train` → `train_clutter` → `train_real2`(별도 저장소 `yrpark/seg-loom-real2`).
- 에피소드 이름과 구성은 같고, 근사 URDF의 링크 길이 오차만 달라진다(`eval_real2`, `train_real2`, `train_real`은 그리퍼 연동 관절 수정도 함께 들어간다).
- `train/r0000`~`r0600`은 낱개 파일에서 tar 묶음 형식으로 바뀐다(`REPLACE=1`이 예전 파일을 지운다). `eval`, `ref`는 예전처럼 낱개 파일이다(`scripts/regen_loose.sh`).
- 교체 전 기존 세트의 링크 길이 오차 분포: 관절 원점 길이는 실제 로봇 중간값 11.9cm(최대 101.6cm, 10cm 이상 59%), 절차적 로봇 중간값 6.1cm(최대 50.7cm, 10cm 이상 27%). 10% 설정에서 1cm를 넘던 원점은 실제 로봇 18%(최대 10.2cm), 절차적 로봇 12%(최대 5.1cm).

## 11. 버그 기록: 어안·강한 왜곡 카메라의 화면 바깥쪽이 실제 렌즈와 달랐던 문제 (2026-11-28)

**증상.** 어안 카메라 이미지의 바깥쪽 고리가 장면이 뭉개진 띠로 그려졌다. `data/eval`에서 카메라의 약 1/4이 어안이고, 그 이미지에서 평균 약 25%(중간값 15%, 최대 95%)의 픽셀이 이 상태였다. 강한 배럴 왜곡(Brown, k1 ≈ -0.35) 카메라도 구석 일부에서 광선이 틀렸다. A4 ChArUco 렌즈 추정을 sim에서 시험하다가 발견했다.

**원인.**
1. `random_camera`의 어안 초점거리 식이 일반 렌즈용 `tan()` 식이었다. Kannala-Brandt(등거리형) 렌즈에서는 픽셀 반지름이 각도에 비례하므로, 뽑은 화각(100~175°)보다 실제 화각이 훨씬 넓었다(최대 약 290°, 화면 끝 광선이 광축에서 145°).
2. 픽셀→광선 변환(`cv2.fisheye.undistortPoints`)은 90°를 넘는 광선을 표현하지 못해 그 부분이 90° 근처로 뭉개졌다. 렌더러(`ModelRenderer`)와 학습 입력 ray map이 모두 이 변환을 쓴다.
3. Brown 렌즈는 강한 배럴 값에서 왜곡 다항식이 화면 구석 전에 꺾이거나 거의 평평해져서 구석에 맞는 광선이 없었다. `cv2.undistortPointsIter`도 그런 구석에서 수렴하지 않았다(재투영 오차 최대 수백 px).
4. 넓은 화각을 큐브맵 5면으로 그릴 때, 면 경계의 bilinear 보간이 옆 면 이미지를 섞어 가는 점선 이음매가 생겼다.

**해결.**
- `random_camera` (난수 뽑는 순서는 같고 계산만 바꿈):
  - 어안: 뽑은 화각을 **대각선 화각**(최대 170°)으로 본다. 주점까지 뽑은 뒤 가장 먼 구석의 광선이 그 화각의 절반이 되도록 Kannala-Brandt 식으로 초점거리를 정한다.
  - Brown: 화면 구석까지 왜곡 다항식의 기울기가 0.3 이상이 되도록 k1, k2를 0.9배씩 줄인다.
- `CameraModel.unproject`를 직접 구현했다: 반지름 방향은 조회표와 Newton으로 정확히 풀고, Brown의 접선 왜곡은 Newton 4회로 푼다. 카메라 5,000대에서 픽셀→광선→픽셀 왕복 오차가 0.01px 이하다.
- 큐브맵 면을 92.8°로 조금 넓게 렌더해 이음매를 없앴다.
- 결과: 대각선 화각은 pinhole 51~120°, Brown 50~135°, 어안 84~170°.

**영향.** 2026-11-28 이전에 만든 모든 세트. 11장의 1cm 재생성에 이 수정이 함께 들어간다. 학습 코드의 `ray_map_np`가 `CameraModel.ray_map`을 쓰므로, 예전 데이터를 새 코드로 읽으면 ray map이 그 데이터의 렌더와 어긋난다. 재생성된 데이터를 쓴다.
