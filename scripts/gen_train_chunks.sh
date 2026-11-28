#!/bin/bash
# Generate the training set in 200-robot chunks, upload each to HF (<SPLIT>/rNNNN) as a few tar shards
# (HF rate-limits per-file downloads: 14k loose files ~45 min, 10 shards ~minutes), then free local disk.
# Usage: bash scripts/gen_train_chunks.sh [first_chunk] [last_chunk] [workers]
#   env SPLIT=train (HF folder / local data/<SPLIT>), CLUTTER=0.0 (probability of a cluttered scene)
#   real robots instead of procedural ones: ROBOTS="xarm6 wx250s ..." K=10 (instances per chunk) RECOLOR=0.5
#     chunk c = instances [c*K, (c+1)*K) of every robot, folder <SPLIT>/iNNN (NNN = first instance)
#   MODES="motion:10 task:10"   LENFRAC=0.10 (approx-URDF link-length error, see gen_dataset.py --len-frac)
#   LENMAX=0.01 (absolute cap on that error in metres, 1 cm; LENMAX=none: no cap = sets made before 2026-11-28 15:00)
#   REPO=yrpark/seg-loom (HF dataset repo; created public if missing)
#   REPLACE=1: after the verified upload, delete remote entries of the folder that this upload does not have
#     (old loose-file folders train/r0000..r0600, a larger old shard count)
#   FIX="robot_a robot_b": re-make only these robots in already uploaded chunks (download, drop, regenerate, re-upload)
# Uploaded per chunk:  manifest.jsonl, config.json, errors.log, robots.tar (robots/ + procgen/),
#                      episodes_000.tar ... (500 episode folders each, uncompressed; members are <ep>/{meta.json,arrays.npz,frames.npz})
# Unpack on the reader side:  for f in rNNNN/*.tar; do tar xf $f -C rNNNN; done   (episodes land in rNNNN/episodes/)
set -u
act() { set +u; conda activate "$1"; set -u; }
cd "$(dirname "$0")/.."
C0=${1:-0}; C1=${2:-9}; W=${3:-20}; N=200; SPLIT=${SPLIT:-train}; CLUTTER=${CLUTTER:-0.0}; SHARD=500
ROBOTS=${ROBOTS:-}; K=${K:-10}; RECOLOR=${RECOLOR:-0.0}; MODES=${MODES:-motion:10 task:10}; LENFRAC=${LENFRAC:-0.10}; export REPO=${REPO:-yrpark/seg-loom}; LENMAX=${LENMAX:-0.01}
source "$(conda info --base)/etc/profile.d/conda.sh"
for c in $(seq $C0 $C1); do
  if [ -n "$ROBOTS" ]; then
    s=$((c * K)); tag=$(printf "i%03d" $s)
    GEN="--robots $ROBOTS --instances $s $((s + K)) --recolor $RECOLOR --seed ${SEED:-0}"
  else
    s=$((c * N)); tag=$(printf "r%04d" $s)
    GEN="--procgen $N --seed $s"
  fi
  out=data/$SPLIT/$tag; up=data/${SPLIT}_upload/$tag
  echo "=== chunk $c ($tag) start $(date)"
  if [ -n "${FIX:-}" ]; then
    # replace some robots in an uploaded chunk: download it, drop their episodes, regenerate only those
    # (same args => same seeds; finished episodes of the other robots are skipped), then repack + upload as usual
    rm -rf $out data/_dl; act OpenViTac
    python - "$SPLIT/$tag" <<'PY' || { echo "download failed chunk $c"; exit 1; }
import os, sys
from huggingface_hub import snapshot_download
snapshot_download(os.environ["REPO"], repo_type="dataset", allow_patterns=[sys.argv[1] + "/*"], local_dir="data/_dl")
PY
    mkdir -p "$(dirname $out)"; mv data/_dl/$SPLIT/$tag $out; rm -rf data/_dl
    for t in $out/*.tar; do { tar xf $t -C $out && rm $t; } || { echo "unpack failed chunk $c"; exit 1; }; done
    for r in $FIX; do rm -rf $out/episodes/${r}_i[0-9][0-9][0-9]_* $out/robots/${r}_i[0-9][0-9][0-9]; done
    python - $out "$FIX" <<'PY'
import json, sys
out, fix = sys.argv[1], sys.argv[2].split()
p = out + "/manifest.jsonl"; L = open(p).read().splitlines()
keep = [l for l in L if json.loads(l)["robot"].rsplit("_i", 1)[0] not in fix]
open(p, "w").write("".join(l + "\n" for l in keep)); print(f"fix: kept {len(keep)}/{len(L)} manifest lines")
PY
  fi
  act shadow
  MUJOCO_GL=egl timeout -k 60 6h python scripts/gen_dataset.py --out $out $GEN \
      --modes $MODES --workers $W --gpus 0 --clutter $CLUTTER --len-frac $LENFRAC $( [ "$LENMAX" != none ] && echo --len-max-m $LENMAX ) || { echo "gen failed chunk $c"; exit 1; }
  # ---- pack
  rm -rf $up; mkdir -p $up
  cp $out/manifest.jsonl $out/config.json $out/errors.log $up/ 2>/dev/null
  tar cf $up/robots.tar -C $out robots $( [ -d $out/procgen ] && echo procgen ) || { echo "pack failed chunk $c"; exit 1; }
  ls $out/episodes | sort > $up/.eplist
  split -l $SHARD -d -a 3 $up/.eplist $up/.part_
  for p in $up/.part_*; do
    k=${p##*_}
    sed 's#^#episodes/#' $p | tar cf $up/episodes_$k.tar -C $out -T - || { echo "pack failed chunk $c"; exit 1; }
  done
  n_ep=$(wc -l < $up/.eplist); n_tar=$(for t in $up/episodes_*.tar; do tar tf $t; done | grep -c "/frames.npz$")
  rm -f $up/.eplist $up/.part_*
  [ "$n_ep" -eq "$n_tar" ] || { echo "pack failed chunk $c: $n_ep episodes, $n_tar in tars"; exit 1; }
  n_man=$(wc -l < $up/manifest.jsonl)
  [ "$n_ep" -eq "$n_man" ] || { echo "pack failed chunk $c: $n_ep episodes, $n_man manifest lines"; exit 1; }
  echo "packed $n_ep episodes into $(ls $up/episodes_*.tar | wc -l) shards"
  # ---- upload + verify (names and sizes)
  act OpenViTac
  python - "$up" "$tag" "$SPLIT" "$CLUTTER" "$(git rev-parse --short HEAD)" <<'PY' || { echo "upload failed chunk $c"; exit 1; }
import os, sys
from huggingface_hub import HfApi
up, tag, split, clutter, rev = sys.argv[1:6]
api = HfApi(); repo = os.environ["REPO"]; pre = f"{split}/{tag}"
api.create_repo(repo, repo_type="dataset", exist_ok=True)
local = {f: os.path.getsize(os.path.join(up, f)) for f in os.listdir(up)}
api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=up, path_in_repo=pre,
                  commit_message=f"{pre}: {os.environ.get('DESC', 'procgen seeds ' + tag[1:] + '+200')}, clutter {clutter}, tar shards (code {rev})")
remote = {i.path[len(pre) + 1:]: i.size for i in api.list_repo_tree(repo, path_in_repo=pre, repo_type="dataset")}
bad = [f for f, sz in local.items() if remote.get(f) != sz]
print(f"local {len(local)} remote {len(remote)} missing_or_size_mismatch {len(bad)} {bad[:5]}")
if bad:
    sys.exit(1)
if os.environ.get("REPLACE") == "1":
    from huggingface_hub import CommitOperationDelete
    top = list(api.list_repo_tree(repo, path_in_repo=pre, repo_type="dataset"))
    stale = [i for i in top if i.path[len(pre) + 1:] not in local]
    if stale:
        ops = [CommitOperationDelete(path_in_repo=i.path + ("/" if not hasattr(i, "size") else ""), is_folder=not hasattr(i, "size")) for i in stale]
        api.create_commit(repo, repo_type="dataset", operations=ops, commit_message=f"{pre}: remove entries replaced by the new upload")
    print(f"replace: removed {len(stale)} stale entries {[i.path[len(pre) + 1:] for i in stale][:6]}")
PY
  rm -rf "$out"/episodes "$out"/procgen "$out"/robots "$up"
  echo "=== chunk $c done $(date); kept $out/{manifest.jsonl,errors.log,config.json}"
done
echo ALL_DONE
