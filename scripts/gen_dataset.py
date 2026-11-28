"""Generate a dataset split (runs anywhere MuJoCo EGL works: RTX box or B200).

  # held-out evaluation set: 4 real robots + 50 unseen procedural robots
  python scripts/gen_dataset.py --out data/eval --robots so101 ur7e piper panda --procgen 50 --seed 100000 \
      --modes motion:12 task:8 --workers 8
  # renderer-validation reference: static random poses, ideal actuators
  python scripts/gen_dataset.py --out data/ref --robots so101 ur7e piper panda --procgen 196 --seed 200000 \
      --modes ref:5 --T 10 --workers 8
  # training data (B200): many procedural robots
  python scripts/gen_dataset.py --out data/train_000 --procgen 2000 --seed 0 --modes motion:10 task:10 --workers 32

Layout:
  <out>/robots/<robot>/   true.urdf, approx.urdf, meshes/, spec.yaml, perturb.json, extras.json, render_geoms.npz
  <out>/procgen/<robot>/  generator output (URDF + spec used to build the world)
  <out>/episodes/<robot>_<mode>_<k>/  meta.json, arrays.npz, frames.npz
  <out>/manifest.jsonl    one line per finished episode
Re-running skips episodes that already exist (resumable).
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def robot_job(args):
    name, kind, seed, out, modes, T, level, clutter = args[:8]
    inst, recolor = (args[8], args[9]) if len(args) > 8 else (None, 0.0)
    len_frac = args[10] if len(args) > 10 else 0.10
    from shadow.datagen import EpisodeGen, robot_render_geoms, sample_robot_extras, save_episode
    from shadow.mj_world import Layout
    from shadow.procgen import generate_robot
    from shadow.urdf_perturb import export_robot
    from shadow.urdf_robot import RobotSpec
    rng = np.random.default_rng(seed)
    t0 = time.time()
    try:
        if kind == "procgen":
            p = os.path.join(out, "procgen", name, "spec.yaml")
            if not os.path.exists(p):
                p = generate_robot(rng, name, os.path.join(out, "procgen"))
                if p is None:
                    return dict(robot=name, error="procgen validation failed")
            spec = RobotSpec.load(p)
        else:
            spec = RobotSpec.load(name)
            if inst is not None:                       # instance k of a real robot: own approx URDF / extras / colours
                spec.asset_dir = spec.out_dir              # keep the converted-model cache of the base robot
                spec.name = f"{name}_i{inst:03d}"
        rdir = os.path.join(out, "robots", spec.name)
        layout = Layout(spec)
        if not os.path.exists(os.path.join(rdir, "extras.json")):
            rec = export_robot(spec, rdir, rng, level=level, len_frac=len_frac)
            extras = sample_robot_extras(spec, layout, rng, recolor=recolor)
            json.dump(rec, open(os.path.join(rdir, "perturb.json"), "w"), indent=1)
            json.dump(extras, open(os.path.join(rdir, "extras.json"), "w"), indent=1)
            y = {k: v for k, v in spec.__dict__.items() if not k.startswith("_")}
            y["urdf"] = "true.urdf"; y.pop("package_dirs", None); y.pop("asset_dir", None)
            yaml.safe_dump(_plain(y), open(os.path.join(rdir, "spec.yaml"), "w"), sort_keys=False)
        extras = json.load(open(os.path.join(rdir, "extras.json")))
        gen = EpisodeGen(spec, extras, rng, clutter=clutter)
        done = []
        for mode, n in modes:
            for k in range(n):
                ed = os.path.join(out, "episodes", f"{spec.name}_{mode}_{k:03d}")
                if os.path.exists(os.path.join(ed, "frames.npz")):
                    # finished earlier (episodes are renamed into place only when complete): keep it in the
                    # manifest, which is only written when the whole robot job returns
                    meta = json.load(open(os.path.join(ed, "meta.json")))
                else:
                    erng = np.random.default_rng([seed, MODE_ID[mode], k])
                    gen.rng = erng
                    meta, arrays, frames, world = gen.run(mode, T=T)
                    if not any(np.any(v) for f in frames.values() for v in f["vis"]):
                        # the table / floor are always in view: all-zero labels mean the segmentation render broke
                        raise RuntimeError("all label maps are empty (segmentation render broken, e.g. numpy>=2 "
                                           "with mujoco 2.3.7); nothing saved")
                    tmp = ed + ".tmp"
                    shutil.rmtree(tmp, ignore_errors=True)
                    save_episode(tmp, meta, arrays, frames)
                    shutil.rmtree(ed, ignore_errors=True)
                    os.rename(tmp, ed)                               # atomic: no half-written episode survives a kill
                    if not os.path.exists(os.path.join(rdir, "render_geoms.npz")):
                        np.savez_compressed(os.path.join(rdir, "render_geoms.npz"), **robot_render_geoms(world))
                done.append(dict(episode=os.path.basename(ed), robot=spec.name, kind=kind, mode=mode, T=meta["T"],
                                 instruction=(meta["task"] or {}).get("instruction"),
                                 success=(meta["task"] or {}).get("success")))
        return dict(robot=spec.name, episodes=done, sec=time.time() - t0)
    except Exception as e:
        return dict(robot=name, error=f"{e}\n{traceback.format_exc()}")


MODE_ID = {"motion": 1, "task": 2, "ref": 3}


def _selftest():
    from shadow.kinematic_mask import segmentation_selftest
    segmentation_selftest()


def _init_worker(gpus):
    """Spread workers over EGL devices (EGL indices; must happen before mujoco is imported in the worker)."""
    if gpus:
        k = mp.current_process()._identity[0] - 1
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpus[k % len(gpus)])
    import cv2
    cv2.setNumThreads(1)


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--robots", nargs="*", default=[], help="real robots from configs/robots")
    ap.add_argument("--procgen", type=int, default=0, help="number of procedural robots")
    ap.add_argument("--seed", type=int, default=0, help="procedural robot seeds start here (keep splits disjoint)")
    ap.add_argument("--modes", nargs="+", default=["motion:10", "task:10"], help="mode:episodes_per_robot")
    ap.add_argument("--T", type=int, default=60, help="frames per motion/ref episode (task length is scripted)")
    ap.add_argument("--level", type=float, default=1.0, help="URDF error level (0 = exact URDF)")
    ap.add_argument("--len-frac", type=float, default=0.10,
                    help="approx URDF link-length error: per joint offset scaled by 1+e, e~N(0,f/2) clipped to +-f "
                         "(0.10 = sets made before 2026-11-28; 0.01 = manufacturer URDF, mm-level)")
    ap.add_argument("--clutter", type=float, default=0.0, help="probability of a cluttered scene per episode")
    ap.add_argument("--instances", type=int, nargs=2, default=None, metavar=("START", "END"),
                    help="real robots: generate instances START..END-1 of each (own approx URDF, extras, colours)")
    ap.add_argument("--recolor", type=float, default=0.0, help="probability of repainting a robot instance")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gpus", type=int, nargs="*", default=None,
                    help="GPUs (nvidia-smi / CUDA indices) to spread the EGL workers over")
    ap.add_argument("--render", choices=["gpu", "cpu"], default="gpu",
                    help="cpu: Mesa llvmpipe through EGL, one thread per worker. Same labels/arrays as gpu, RGB within "
                         "~1/255. Datacenter GPUs (B200) rasterize slowly: there one GPU saturates at ~8 workers "
                         "(~830 episodes/h) while a cpu worker makes ~40 episodes/h, so run both on disjoint ranges")
    ap.add_argument("--robot-range", type=int, nargs=2, default=None, metavar=("START", "END"),
                    help="only robot jobs [START, END) of this split, so several processes can fill one output dir")
    a = ap.parse_args()
    # one compute thread per worker: on a many-core box (B200 server: 192 cores) every worker otherwise starts
    # BLAS/OpenMP pools as wide as the machine (~450 threads each) and 64 workers crawl. Inherited by spawn.
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    egl_ids = None
    if a.render == "cpu":
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"
        os.environ["LP_NUM_THREADS"] = "1"
        os.environ.setdefault("EGL_LOG_LEVEL", "fatal")              # no /dev/dri permission warnings per worker
        a.gpus = None
    if a.gpus:
        # EGL numbers devices in its own order (B200 server: EGL 0 = GPU 4, EGL 4 = GPU 0): translate
        from shadow.egl_devices import egl_index_for_gpu
        egl_ids = [egl_index_for_gpu(g) for g in a.gpus]
        print("GPU -> EGL device:", dict(zip(a.gpus, egl_ids)), flush=True)
    # fail fast if label rendering is broken in this environment (checked in a worker-like spawned process)
    with mp.get_context("spawn").Pool(1, initializer=_init_worker, initargs=(egl_ids,)) as p:
        p.apply(_selftest)
    out = os.path.abspath(a.out)
    os.makedirs(os.path.join(out, "episodes"), exist_ok=True)
    modes = [(m.split(":")[0], int(m.split(":")[1])) for m in a.modes]
    if a.instances:
        jobs = [(r, "real", a.seed + 10_000_000 + i + 100_000 * k, out, modes, a.T, a.level, a.clutter, k, a.recolor, a.len_frac)
                for k in range(*a.instances) for i, r in enumerate(a.robots)]
    else:
        jobs = [(r, "real", a.seed + 10_000_000 + i, out, modes, a.T, a.level, a.clutter, None, 0.0, a.len_frac)
                for i, r in enumerate(a.robots)]
    jobs += [(f"pg{a.seed + i:07d}", "procgen", a.seed + i, out, modes, a.T, a.level, a.clutter, None, 0.0, a.len_frac)
             for i in range(a.procgen)]
    if a.robot_range:
        jobs = jobs[a.robot_range[0]:a.robot_range[1]]
    cfg = os.path.join(out, "config.json") if not a.robot_range else \
        os.path.join(out, f"config_{a.robot_range[0]}_{a.robot_range[1]}.json")
    json.dump(dict(args=vars(a), started=time.ctime()), open(cfg, "w"), indent=1)
    mpath = os.path.join(out, "manifest.jsonl")
    listed = {json.loads(l)["episode"] for l in open(mpath)} if os.path.exists(mpath) else set()
    t0, n_ep, n_err = time.time(), 0, 0
    with mp.get_context("spawn").Pool(a.workers, initializer=_init_worker, initargs=(egl_ids,), maxtasksperchild=4) as pool, \
            open(os.path.join(out, "manifest.jsonl"), "a") as man, open(os.path.join(out, "errors.log"), "a") as err:
        for i, r in enumerate(pool.imap_unordered(robot_job, jobs)):
            if "error" in r:
                n_err += 1
                err.write(f"== {r['robot']}\n{r['error']}\n"); err.flush()
            else:
                for e in r["episodes"]:
                    if e["episode"] not in listed:                   # resumed runs report finished episodes again
                        man.write(json.dumps(e) + "\n")
                        listed.add(e["episode"])
                n_ep += len(r["episodes"]); man.flush()
            el = time.time() - t0
            stat = "ERROR" if "error" in r else f"{len(r['episodes'])} ep, {r['sec']:.0f}s"
            print(f"[{i + 1}/{len(jobs)}] {r['robot']} {stat}"
                  f" | total {n_ep} ep, {n_err} errors, {el / 60:.1f} min, eta {el / (i + 1) * (len(jobs) - i - 1) / 60:.1f} min",
                  flush=True)


if __name__ == "__main__":
    main()
