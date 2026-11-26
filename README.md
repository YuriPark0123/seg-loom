# seg-loom

URDF와 카메라 영상만으로 처음 보는 로봇과 카메라에서도 동작하는 cross-embodiment 조작 연구 코드입니다.

- 로봇의 기구학과 카메라 기하를 학습 모델 안의 구조로 넣습니다 (inductive bias). 근사 URDF와 렌즈를 모르는 카메라에서도 로봇을 찾고 추적합니다.
- 정책은 EEF 기준으로 로봇과 무관하게 동작합니다. 실제 실행은 로봇별 IK가 맡습니다.

설계는 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), 데이터 형식은 [docs/DATASET.md](docs/DATASET.md)에 있습니다.
데이터셋은 Hugging Face [`yrpark/seg-loom`](https://huggingface.co/datasets/yrpark/seg-loom)에 있습니다.

## 구성

| 경로 | 내용 |
|---|---|
| `shadow/urdf_robot.py` | URDF → MuJoCo 변환 (메시 변환, mimic → equality) |
| `shadow/mj_world.py` | reach에 맞춰 크기를 정하는 장면과 top/front/wrist 카메라, robot-only twin |
| `shadow/kinematic_mask.py`, `shadow/skeleton.py` | 기구학 twin 마스크, skeleton |
| `shadow/calibration.py` | 체커보드 eye-to-hand / eye-in-hand 캘리브레이션 + BA |
| `shadow/ik.py` | DLS IK (위치 우선) |
| `shadow/camera_model.py` | pinhole / Brown / fisheye 카메라 모델, 광각 큐브맵 렌더, 화질 저하 |
| `shadow/procgen.py` | 절차적 로봇 팔 생성 |
| `shadow/urdf_perturb.py` | 오차를 섞은 근사 URDF 생성 |
| `shadow/datagen.py`, `scripts/gen_dataset.py` | 데이터셋 생성 (motion / task / ref 에피소드) |
| `webapp/` | 대화형 데모: EEF 드래그 IK, 카메라 이동, 재캘리브레이션 |
| `configs/robots/` | 실제 로봇 spec (so101, ur7e, piper, panda) |

## 설치

```bash
conda create -n shadow python=3.10 && conda activate shadow
pip install mujoco==2.3.7 "numpy<2" "opencv-python<4.11" scipy trimesh pycollada pyyaml fastapi uvicorn websockets
# numpy<2: mujoco 2.3.7 + numpy 2.x는 분할(id color) 렌더가 깨져 라벨이 전부 0으로 저장됨 (오류 없음)
# opencv-python<4.11: numpy 1.x와 호환되는 마지막 버전 (5.x에는 cv2.undistortPointsIter도 없음)
# pycollada: ur7e/panda의 .dae 메시용
bash scripts/fetch_assets.sh          # 실제 로봇 URDF/메시 (고정 커밋)
export MUJOCO_GL=egl
```

## 사용

```bash
python scripts/urdf_check.py --robot so101          # 캘리브레이션 + twin 겹침 확인
bash webapp/run.sh panda                            # 웹 데모
python scripts/gen_dataset.py --out data/smoke --robots so101 --procgen 4 --seed 900000 --modes motion:2 task:2
python scripts/view_episode.py data/smoke/episodes/<episode>
```
