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
```

- `--seed`는 절차적 로봇 번호의 시작값이다. 로봇 이름은 `pg{seed+i:07d}`이다. 분할(split)끼리 범위가 겹치면 안 된다. 기본 배정은 학습 0~99,999, 평가 100,000~, 기준셋 200,000~, smoke 900,000~이다.
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

**라벨 값:** 0 배경 · 1..Nb 로봇 body(`robot_bodies` 인덱스+1; 부착물은 붙은 링크의 값) · 100+k 객체 k · 200 테이블 · 201 테이블 다리 · 202 바닥

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
| 로봇 | 절차적: 관절 3~7개, 크기 S 0.3~1.0m, 링크 스타일 2종, 색 10종, 그리퍼 2종 / 실제: so101, ur7e, piper, panda |
| URDF 오차 | 길이 N(0, 5%)를 ±10%로 자름, 영점 U(±5°), 도형 ±15%, 원기둥↔박스 20%, 메시→박스 30%, 링크 누락 10% |
| URDF에 없는 부착물 | 케이블(캡슐) / 박스 0~3개, 손목 카메라 장착 오차 위치 σ6mm, 각도 σ2.5° |
| 카메라 모델 | pinhole / Brown / fisheye(KB), 화각 45~100°(fisheye 100~175°), 주점 오차, 출력 폭 320px |
| 카메라 배치 | top: 높이 1.4~2.4R, 기울기 약 0~15°, 임의 yaw / front: 방위 ±75°, 고각 12~60°, 거리 1.1~2.1R (광각일수록 가깝게), roll σ4° |
| 카메라 움직임 | static, drift(부드러운 이동), shake(흔들림+이동), jump(중간에 순간 이동) |
| 장면 | 테이블 색 (30%는 흰색), 바닥 체커 색, 방향광 1~3개, 주변광, 블럭 2~5개 (흰색 50%, 같은 색 쌍 30%) |
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
- 사람 손이나 방해물에 의한 가림, 다양한 배경과 텍스처.
- 로봇 여러 대가 함께 있는 장면.
- 형태가 블럭이 아닌 객체(원기둥, 복합 형상).
