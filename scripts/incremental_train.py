"""Train stage 1 while the training set is still being uploaded to Hugging Face, growing the data in steps.

Loop (internal tick 60 s, HF poll every --poll-min minutes):
  * a folder <split>/<tag> on HF is complete once <split>/<tag>/manifest.jsonl exists (one commit per folder)
  * complete folders are downloaded one at a time (oldest first) into --local, then verified:
    file count = remote, manifest = episode dirs, label maps of sampled episodes not empty
  * training starts once the GPU is free (no foreign compute process on it and the OpenViTac queue has exited)
  * switches (training saves at its next checkpoint, exits 3, restarts with --resume and the larger data):
      stage2      all --train-folders simple folders verified
      stage3.k    each verified cluttered folder (mixed --clutter-frac per batch)
  * at every switch: copy the checkpoint to <out>/stages/, evaluate it on --eval (fixed frames), append to report.md
  * planned total steps: the cosine runs --interim-hours past each switch, --final-hours after the last folder

    CUDA_DEVICE_ORDER=PCI_BUS_ID python scripts/incremental_train.py --gpu 4 --out out/stage1_main
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time

REPO = "yrpark/seg-loom"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Ctl:
    def __init__(self, a):
        self.a = a
        os.makedirs(os.path.join(a.out, "stages"), exist_ok=True)
        self.state_path = os.path.join(a.out, "controller_state.json")
        self.st = json.load(open(self.state_path)) if os.path.exists(self.state_path) else dict(
            verified=[], bad=[], stage=None, stages=[], started=None)
        self.log_f = open(os.path.join(a.out, "controller.log"), "a")
        self.dl = None              # (folder, Popen)
        self.train = None           # Popen
        self.train_cfg = None       # (simple folders, clutter folders, stage name)
        self.evals = []             # (stage, Popen, json path)
        self.remote = {}            # folder -> remote file count
        self.last_poll = 0.0
        self.crashes = 0

    # ------------------------------------------------------------------ utils
    def log(self, msg):
        line = f"{now()} {msg}"
        print(line, flush=True)
        self.log_f.write(line + "\n"); self.log_f.flush()

    def save(self):
        json.dump(self.st, open(self.state_path + ".tmp", "w"), indent=1)
        os.replace(self.state_path + ".tmp", self.state_path)

    def report(self, text):
        with open(os.path.join(self.a.out, "report.md"), "a") as f:
            f.write(text + "\n")

    # ------------------------------------------------------------------ HF / download
    def poll(self):
        from huggingface_hub import HfApi
        try:
            fs = HfApi().list_repo_files(REPO, repo_type="dataset")
        except Exception as e:                                   # network / rate limit: try next tick
            self.log(f"poll failed: {e}"); return
        cnt = {}
        for f in fs:
            m = re.match(r"^(train(?:_clutter)?/r\d{4})/", f)
            if m:
                cnt[m.group(1)] = cnt.get(m.group(1), 0) + 1
        done = {k for k in cnt if f"{k}/manifest.jsonl" in fs}
        new = sorted(done - set(self.remote))
        self.remote = {k: cnt[k] for k in done}
        if new:
            self.log(f"HF complete folders: {sorted(done)} (new: {new})")

    def local(self, folder):
        return os.path.join(self.a.local, folder)

    def n_local(self, folder):
        return sum(len(fs) for _, _, fs in os.walk(self.local(folder)))

    def verify(self, folder):
        d = self.local(folder)
        eps = [json.loads(l)["episode"] for l in open(os.path.join(d, "manifest.jsonl"))]
        dirs = [e for e in os.listdir(os.path.join(d, "episodes")) if not e.endswith(".tmp")]
        if set(eps) != set(dirs):
            return f"manifest {len(eps)} != episode dirs {len(dirs)}"
        sys.path.insert(0, ROOT)
        import numpy as np
        from shadow.datagen import unpack_image
        for e in random.Random(0).sample(eps, min(12, len(eps))):
            F = np.load(os.path.join(d, "episodes", e, "frames.npz"))
            if not any(unpack_image(F[f"{c}_vis"], F[f"{c}_vis_off"], t, "png").any()
                       for c in ("top", "front", "wrist") for t in (0, len(F[f"{c}_vis_off"]) - 2)):
                return f"empty label maps in {e}"
        return None

    def tick_download(self):
        if self.dl is not None:
            folder, p = self.dl
            if p.poll() is None:
                return
            n, want = self.n_local(folder), self.remote.get(folder, -1)
            if n < want:
                self.log(f"download {folder}: {n}/{want} files (exit {p.returncode}), retrying")
                self.start_download(folder)
                return
            err = self.verify(folder)
            if err:
                self.log(f"VERIFY FAILED {folder}: {err} -> not used")
                self.st["bad"].append(folder); self.report(f"- {now()} **{folder} 검증 실패, 학습에서 제외:** {err}")
            else:
                self.st["verified"].append(folder)
                self.log(f"verified {folder} ({n} files)")
            self.save()
            self.dl = None
        todo = [f for f in sorted(self.remote, key=lambda k: (k.startswith("train_clutter"), k))
                if f not in self.st["verified"] and f not in self.st["bad"]]
        if todo:
            self.start_download(todo[0])

    def start_download(self, folder):
        cmd = [os.path.join(os.path.dirname(sys.executable), "hf"), "download", REPO, "--repo-type", "dataset",
               "--include", f"{folder}/*", "--local-dir", self.a.local, "--max-workers", "16"]
        logf = open(os.path.join(self.a.out, "download.log"), "a")
        self.dl = (folder, subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT))
        self.log(f"download {folder} started ({self.n_local(folder)}/{self.remote.get(folder)} files local)")

    # ------------------------------------------------------------------ GPU / training
    def gpu_free(self):
        uuid = subprocess.run(["nvidia-smi", "-i", str(self.a.gpu), "--query-gpu=uuid", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout
        mine = {self.train.pid} if self.train else set()
        foreign = [l for l in apps.splitlines() if l.startswith(uuid) and int(l.split(",")[1]) not in mine]
        queue = subprocess.run(["pgrep", "-u", os.environ.get("USER", "guest1"), "-f", "scripts/b200/queue[.]sh"],
                               capture_output=True, text=True).stdout.strip()
        return not foreign and not queue

    def it_per_s(self):
        try:
            rates = [json.loads(l)["it_s"] for l in open(os.path.join(self.a.out, "log.jsonl")) if '"it_s"' in l]
            return rates[-1] if rates else self.a.default_its
        except FileNotFoundError:
            return self.a.default_its

    def current_step(self):
        ck = os.path.join(self.a.out, "last.pt")
        if not os.path.exists(ck):
            return 0
        import torch
        return torch.load(ck, map_location="cpu", weights_only=False)["step"]

    def start_train(self, simple, clutter, stage, final):
        step = self.current_step()
        hours = self.a.final_hours if final else self.a.interim_hours
        total = int(step + self.it_per_s() * 3600 * hours)
        cmd = [sys.executable, os.path.join(HERE, "train_stage1.py"),
               "--data", *[self.local(f) for f in simple], "--val-data", self.a.eval, "--out", self.a.out,
               "--steps", str(total), "--batch", str(self.a.batch), "--workers", str(self.a.workers),
               "--ckpt-every", "1000", "--eval-every", "5000", "--eval-batches", "20", "--warmup", "1000"]
        if clutter:
            cmd += ["--clutter-data", *[self.local(f) for f in clutter], "--clutter-frac", str(self.a.clutter_frac)]
        if step > 0 or os.path.exists(os.path.join(self.a.out, "last.pt")):
            cmd.append("--resume")
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(self.a.gpu))
        logf = open(os.path.join(self.a.out, "train.log"), "a")
        logf.write(f"\n===== {now()} {stage}: {' '.join(cmd)}\n"); logf.flush()
        self.train = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        self.train_cfg = (list(simple), list(clutter), stage, final)
        self.st["stage"] = stage
        self.st["stages"].append(dict(stage=stage, start=now(), step=step, simple=simple, clutter=clutter,
                                      planned_total=total, it_s=self.it_per_s()))
        self.save()
        self.log(f"training {stage} started at step {step}: {len(simple)} simple + {len(clutter)} cluttered folders, "
                 f"plan {total} steps ({hours} h at {self.it_per_s():.2f} it/s)")

    def snapshot_and_eval(self, label):
        ck = os.path.join(self.a.out, "last.pt")
        step = self.current_step()
        dst = os.path.join(self.a.out, "stages", f"{label}_step{step}.pt")
        shutil.copy2(ck, dst)
        js = dst.replace(".pt", "_eval.json")
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(self.a.gpu))
        logf = open(os.path.join(self.a.out, "eval.log"), "a")
        p = subprocess.Popen([sys.executable, os.path.join(HERE, "eval_stage1.py"), dst, "--data", self.a.eval,
                              "--out", js, "--workers", "8"], stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        self.evals.append((label, p, js))
        self.log(f"{label}: checkpoint step {step} -> {dst}, eval started")

    def tick_evals(self):
        for item in list(self.evals):
            label, p, js = item
            if p.poll() is None:
                continue
            self.evals.remove(item)
            if p.returncode != 0 or not os.path.exists(js):
                self.log(f"eval of {label} failed (exit {p.returncode})"); continue
            r = json.load(open(js)); m = r["metrics"]; d = r.get("data") or {}
            nf = (len(d.get("simple", [])), len(d.get("clutter", [])))
            row = lambda g: (f"{m[g]['iou_amodal']:.3f} / {m[g]['iou_visible']:.3f}",
                             f"{m[g]['rot_deg_after']:.2f}", f"{m[g]['trans_cm_after']:.2f}")
            a_, r_ = row("all"), row("real")
            self.report(f"| {r['eval_time'][5:16]} | {label} | 단순 {nf[0]} / 복잡 {nf[1]} | {r['step']} | {a_[0]} | "
                        f"{a_[1]} | {a_[2]} | {r_[0]} | {r_[1]} | {r_[2]} |")
            self.log(f"eval {label}: all {a_} real {r_}")

    def target(self):
        """(simple folders, clutter folders, stage name, final?) the training should run on now."""
        v = self.st["verified"]
        simple = sorted(f for f in v if f.startswith("train/"))
        clutter = sorted(f for f in v if f.startswith("train_clutter/"))
        all_simple = len(simple) >= self.a.train_folders
        cur = self.train_cfg
        if cur is None:                                         # first start: whatever simple data is there
            return simple, [], "stage1", False
        cs, cc, cname, _ = cur
        if not all_simple:
            return cs, cc, cname, False                         # stage 1 keeps its data until all simple folders are in
        if not clutter:
            return simple, [], "stage2", False
        final = all_simple and len(clutter) >= self.a.clutter_folders
        return simple, clutter, f"stage3.{len(clutter)}", final

    def tick_train(self):
        if self.train is None:
            if not self.st["verified"] or not self.gpu_free():
                return
            s, c, name, final = self.target()
            if not s:
                return
            if self.st["started"] is None:
                self.st["started"] = now()
                self.report("| 시각 | 전환 시점 | 폴더 | step | IoU 가림 포함 / 보이는 부분 (전체) | 회전 ° | 위치 cm | "
                            "IoU (실제 로봇) | 회전 ° | 위치 cm |\n|---|---|---|---|---|---|---|---|---|---|")
                self.report(f"<!-- {now()} stage1 시작: train 폴더 {len(s)}개 -->")
            self.start_train(s, c, name, final)
            return
        rc = self.train.poll()
        cs, cc, cname, cfinal = self.train_cfg
        if rc is None:
            s, c, name, final = self.target()
            switch = os.path.join(self.a.out, "SWITCH")
            if (s, c) != (cs, cc) and not os.path.exists(switch):
                open(switch, "w").write(f"{now()} -> {name}\n")
                self.log(f"SWITCH requested: {cname} -> {name} (waits for the next 1000-step checkpoint)")
            return
        if rc == 3:                                             # switched at a checkpoint
            if os.path.exists(os.path.join(self.a.out, "SWITCH")):
                os.remove(os.path.join(self.a.out, "SWITCH"))
            s, c, name, final = self.target()
            self.snapshot_and_eval(f"{cname}_end")
            self.report(f"<!-- {now()} 전환 {cname} -> {name}: 단순 {len(s)} / 복잡 {len(c)} 폴더 -->")
            self.train = None
            self.start_train(s, c, name, final)
        elif rc == 0:
            self.log(f"training finished ({cname})")
            self.snapshot_and_eval(f"{cname}_final")
            self.train = None
            self.st["finished"] = now(); self.save()
        else:
            self.crashes += 1
            self.log(f"training crashed (exit {rc}), restart {self.crashes}/3 from the last checkpoint")
            self.train = None
            if self.crashes <= 3:
                s, c, name, final = self.train_cfg[0], self.train_cfg[1], cname, cfinal
                self.start_train(s, c, name, final)

    # ------------------------------------------------------------------ main loop
    def run(self):
        self.log(f"controller start: {vars(self.a)}")
        while True:
            if time.time() - self.last_poll > self.a.poll_min * 60:
                self.poll(); self.last_poll = time.time()
            self.tick_download()
            self.tick_train()
            self.tick_evals()
            if self.st.get("finished") and not self.evals:
                self.log("all done"); break
            time.sleep(60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=4, help="nvidia-smi index of OUR GPU (CUDA_DEVICE_ORDER=PCI_BUS_ID)")
    ap.add_argument("--out", default=os.path.join(ROOT, "out/stage1_main"))
    ap.add_argument("--local", default=os.path.join(ROOT, "data/hf"))
    ap.add_argument("--eval", default=os.path.join(ROOT, "data/eval"))
    ap.add_argument("--train-folders", type=int, default=10)
    ap.add_argument("--clutter-folders", type=int, default=5)
    ap.add_argument("--clutter-frac", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--poll-min", type=float, default=10)
    ap.add_argument("--interim-hours", type=float, default=6)
    ap.add_argument("--final-hours", type=float, default=4)
    ap.add_argument("--default-its", type=float, default=6.0)
    a = ap.parse_args()
    a.out, a.local, a.eval = map(os.path.abspath, (a.out, a.local, a.eval))
    Ctl(a).run()


if __name__ == "__main__":
    main()
