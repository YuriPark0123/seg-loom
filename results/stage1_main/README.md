# stage1_main — incremental stage-1 training (2026-11-27, B200 GPU 4)

Code: `scripts/train_stage1.py` (Stage1Net: resnet18-FPN, seg + 2D displacement heads; camera pose by LM over lens reprojection with a Gaussian prior),
driven by `scripts/incremental_train.py`, which added HF train folders as they were uploaded and resumed from the last checkpoint at every switch.

Data (HF dataset `yrpark/seg-loom`): `train/r0000..r1800` (10 folders, simple scenes, 40k episodes) +
`train_clutter/r2000..r2800` (5 folders, cluttered scenes, 20k episodes), batches sampled simple:clutter = 50:50 once clutter existed.
Eval: lens-fixed `eval/` (54 held-out robots: 4 real + 50 procgen), every 20th frame, fixed seed 20261126 (`scripts/eval_stage1.py`).

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
