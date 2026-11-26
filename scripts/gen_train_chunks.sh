#!/bin/bash
# Generate the training set in 200-robot chunks, upload each to HF (<SPLIT>/rNNNN) as a few tar shards
# (HF rate-limits per-file downloads: 14k loose files ~45 min, 10 shards ~minutes), then free local disk.
# Usage: bash scripts/gen_train_chunks.sh [first_chunk] [last_chunk] [workers]
#   env SPLIT=train (HF folder / local data/<SPLIT>), CLUTTER=0.0 (probability of a cluttered scene)
# Uploaded per chunk:  manifest.jsonl, config.json, errors.log, robots.tar (robots/ + procgen/),
#                      episodes_000.tar ... (500 episode folders each, uncompressed; members are <ep>/{meta.json,arrays.npz,frames.npz})
# Unpack on the reader side:  for f in rNNNN/*.tar; do tar xf $f -C rNNNN; done   (episodes land in rNNNN/episodes/)
set -u
act() { set +u; conda activate "$1"; set -u; }
cd "$(dirname "$0")/.."
C0=${1:-0}; C1=${2:-9}; W=${3:-20}; N=200; SPLIT=${SPLIT:-train}; CLUTTER=${CLUTTER:-0.0}; SHARD=500
source "$(conda info --base)/etc/profile.d/conda.sh"
for c in $(seq $C0 $C1); do
  s=$((c * N)); tag=$(printf "r%04d" $s); out=data/$SPLIT/$tag; up=data/${SPLIT}_upload/$tag
  echo "=== chunk $c ($tag) start $(date)"
  act shadow
  MUJOCO_GL=egl timeout -k 60 6h python scripts/gen_dataset.py --out $out --procgen $N --seed $s \
      --modes motion:10 task:10 --workers $W --gpus 0 --clutter $CLUTTER || { echo "gen failed chunk $c"; exit 1; }
  # ---- pack
  rm -rf $up; mkdir -p $up
  cp $out/manifest.jsonl $out/config.json $out/errors.log $up/ 2>/dev/null
  tar cf $up/robots.tar -C $out robots procgen || { echo "pack failed chunk $c"; exit 1; }
  ls $out/episodes | sort > $up/.eplist
  split -l $SHARD -d -a 3 $up/.eplist $up/.part_
  for p in $up/.part_*; do
    k=${p##*_}
    sed 's#^#episodes/#' $p | tar cf $up/episodes_$k.tar -C $out -T - || { echo "pack failed chunk $c"; exit 1; }
  done
  n_ep=$(wc -l < $up/.eplist); n_tar=$(for t in $up/episodes_*.tar; do tar tf $t; done | grep -c "/frames.npz$")
  rm -f $up/.eplist $up/.part_*
  [ "$n_ep" -eq "$n_tar" ] || { echo "pack failed chunk $c: $n_ep episodes, $n_tar in tars"; exit 1; }
  echo "packed $n_ep episodes into $(ls $up/episodes_*.tar | wc -l) shards"
  # ---- upload + verify (names and sizes)
  act OpenViTac
  python - "$up" "$tag" "$SPLIT" "$CLUTTER" "$(git rev-parse --short HEAD)" <<'PY' || { echo "upload failed chunk $c"; exit 1; }
import os, sys
from huggingface_hub import HfApi
up, tag, split, clutter, rev = sys.argv[1:6]
api = HfApi(); repo = "yrpark/seg-loom"; pre = f"{split}/{tag}"
local = {f: os.path.getsize(os.path.join(up, f)) for f in os.listdir(up)}
api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=up, path_in_repo=pre,
                  commit_message=f"{pre}: procgen seeds {int(tag[1:])}+200, motion:10 task:10, clutter {clutter}, tar shards (code {rev})")
remote = {i.path[len(pre) + 1:]: i.size for i in api.list_repo_tree(repo, path_in_repo=pre, repo_type="dataset")}
bad = [f for f, sz in local.items() if remote.get(f) != sz]
print(f"local {len(local)} remote {len(remote)} missing_or_size_mismatch {len(bad)} {bad[:5]}")
sys.exit(1 if bad else 0)
PY
  rm -rf "$out"/episodes "$out"/procgen "$out"/robots "$up"
  echo "=== chunk $c done $(date); kept $out/{manifest.jsonl,errors.log,config.json}"
done
echo ALL_DONE
