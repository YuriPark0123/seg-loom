#!/bin/bash
# Re-make a loose-file split (data/eval, data/ref: one HF folder of plain files, not tar shards) and replace it on HF.
# Usage: bash scripts/regen_loose.sh <name> <gen_dataset args...>     (HF folder <name>, local data/<name>)
# Generates into data/<name>_new, checks episode count, uploads, verifies every file (name + size), deletes remote
# files the new set does not have, then swaps the local copy.
set -u
act() { set +u; conda activate "$1"; set -u; }
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
name=$1; shift
new=data/${name}_new
rm -rf $new; act shadow
MUJOCO_GL=egl timeout -k 60 6h python scripts/gen_dataset.py --out $new "$@" || { echo "gen failed $name"; exit 1; }
n_ep=$(ls $new/episodes | wc -l); n_man=$(wc -l < $new/manifest.jsonl)
[ "$n_ep" -eq "$n_man" ] || { echo "gen failed $name: $n_ep episodes, $n_man manifest lines"; exit 1; }
act OpenViTac
python - "$new" "$name" "$(git rev-parse --short HEAD)" <<'PY' || { echo "upload failed $name"; exit 1; }
import os, sys
from huggingface_hub import HfApi, CommitOperationDelete
new, pre, rev = sys.argv[1:4]
api = HfApi(); repo = os.environ.get("REPO", "yrpark/seg-loom")
local = {}
for d, _, fs in os.walk(new):
    for f in fs:
        p = os.path.join(d, f); local[os.path.relpath(p, new)] = os.path.getsize(p)
api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=new, path_in_repo=pre,
                  commit_message=f"{pre}: re-made, link-length error capped at 1 cm (code {rev})")
remote = {i.path[len(pre) + 1:]: i.size for i in api.list_repo_tree(repo, path_in_repo=pre, repo_type="dataset", recursive=True)
          if hasattr(i, "size")}
bad = [f for f, sz in local.items() if remote.get(f) != sz]
print(f"local {len(local)} remote {len(remote)} missing_or_size_mismatch {len(bad)} {bad[:5]}")
if bad:
    sys.exit(1)
stale = [f for f in remote if f not in local]
for k in range(0, len(stale), 2000):
    api.create_commit(repo, repo_type="dataset", operations=[CommitOperationDelete(path_in_repo=f"{pre}/{f}") for f in stale[k:k + 2000]],
                      commit_message=f"{pre}: remove files not in the re-made set")
print(f"replace: removed {len(stale)} stale files")
PY
rm -rf data/$name && mv $new data/$name && echo "=== $name replaced $(date)"
