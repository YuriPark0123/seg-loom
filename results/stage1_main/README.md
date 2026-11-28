# stage1_main — incremental stage-1 training (2026-11-27, B200 GPU 4)

Code: `scripts/train_stage1.py` (Stage1Net: resnet18-FPN, seg + 2D displacement heads; camera pose by LM over lens reprojection with a Gaussian prior),
driven by `scripts/incremental_train.py`, which added HF train folders as they were uploaded and resumed from the last checkpoint at every switch.

Data (HF dataset `yrpark/seg-loom`): `train/r0000..r1800` (10 folders, simple scenes, 40k episodes) +
`train_clutter/r2000..r2800` (5 folders, cluttered scenes, 20k episodes), batches sampled simple:clutter = 50:50 once clutter existed.
Eval: lens-fixed `eval/` (54 held-out robots: 4 real + 50 procgen), every 20th frame, fixed seed 20261126 (`scripts/eval_stage1.py`).

**Known-lens condition.** The lens (K, distortion) of every camera is the TRUE one, both in the ray map the network
sees and in the projection of the pose solve. All numbers here are "lens known"; an unknown / approximate lens is not
measured yet.

**Pose floor = approx-URDF joint zero offsets** (`results/pose_diag/dq_k10.*`, 3090 finding reproduced on B200):
the training target (approx point seen by the true camera) is self-consistent with the approx URDF, so the "oracle"
of `diag_pose.py` (0.03 deg) hid the model mismatch. With the pixel where the link point REALLY is (true FK), a
camera-only solve stays at ~4.0 deg / 6 cm even over 10 frames; with the true zero offsets 0.42 deg / 2.2 cm; solving
camera + per-joint offsets jointly over 10 frames 0.91 deg / 2.3 cm (offsets recovered to 0.55 deg).

Checkpoints (model, optimizer, scheduler, step): HF model repo [`yrpark/seg-loom-models`](https://huggingface.co/yrpark/seg-loom-models), `stage1_main/`.

| file | content |
|---|---|
| `report.md` | one row per stage switch: time, folders, step, IoU amodal/visible, rot deg, trans cm (median, after solve) |
| `stages/*_eval.json` | full eval metrics per stage-end checkpoint (all / real / procgen / per camera, before -> after) |
| `log.jsonl`, `train.log` | training loss / lr / it/s every 100 steps |
| `controller.log`, `controller_state.json` | download, verify, switch and eval events |

Final (step 209628): IoU all 0.784 / 0.782, real 0.572 / 0.567 (best real: stage3.4_end 0.579 / 0.571);
camera rot 4.56 -> 3.26 deg, trans 4.00 -> 3.61 cm (median) while reprojection drops 7.2 -> 1.8 px —
the pose solve, not the correspondences, is the current bottleneck.
