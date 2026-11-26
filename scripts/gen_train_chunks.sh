#!/bin/bash
# Generate the training set in 200-robot chunks, upload each to HF (train/rNNNN), then free local disk.
# Usage: bash scripts/gen_train_chunks.sh [first_chunk] [last_chunk] [workers]
#   env SPLIT=train (HF folder / local data/<SPLIT>), CLUTTER=0.0 (probability of a cluttered scene)
set -u
cd "$(dirname "$0")/.."
C0=${1:-0}; C1=${2:-9}; W=${3:-20}; N=200; SPLIT=${SPLIT:-train}; CLUTTER=${CLUTTER:-0.0}
source "$(conda info --base)/etc/profile.d/conda.sh"
for c in $(seq $C0 $C1); do
  s=$((c * N)); tag=$(printf "r%04d" $s); out=data/$SPLIT/$tag
  echo "=== chunk $c ($tag) start $(date)"
  conda activate shadow
  MUJOCO_GL=egl timeout -k 60 6h python scripts/gen_dataset.py --out $out --procgen $N --seed $s \
      --modes motion:10 task:10 --workers $W --gpus 0 --clutter $CLUTTER || { echo "gen failed chunk $c"; exit 1; }
  conda activate OpenViTac
  python - "$out" "$tag" "$SPLIT" "$CLUTTER" "$(git rev-parse --short HEAD)" <<'PY' || { echo "upload failed chunk $c"; exit 1; }
import os, sys
from huggingface_hub import HfApi
out, tag, split, clutter, rev = sys.argv[1:6]
api = HfApi(); repo = "yrpark/seg-loom"
local = sorted(os.path.relpath(os.path.join(d, f), out) for d, _, fs in os.walk(out) for f in fs)

api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=out, path_in_repo=f"{split}/{tag}",
                  commit_message=f"{split}/{tag}: procgen seeds {int(tag[1:])}+200, motion:10 task:10, clutter {clutter} (code {rev})")
remote = set(f[len(f"{split}/{tag}/"):] for f in api.list_repo_files(repo, repo_type="dataset") if f.startswith(f"{split}/{tag}/"))
miss = [f for f in local if f not in remote]
print(f"local {len(local)} remote {len(remote)} missing {len(miss)}")
sys.exit(1 if miss else 0)
PY
  rm -rf "$out"/episodes "$out"/procgen "$out"/robots
  echo "=== chunk $c done $(date); kept $out/{manifest.jsonl,errors.log,config.json}"
done
echo ALL_DONE
