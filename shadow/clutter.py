"""Cluttered scenes: what a real lab image has and the plain scene does not.

  textures    : procedural table / floor / wall textures (wood, noise, marble, tiles, stripes, carpet, posters)
  sky         : random gradient colours
  distractors : 3-10 static non-block shapes on the table (sphere, cylinder, capsule, ellipsoid, mug, bottle, can)
  furniture   : 0-4 big boxes / cylinders on the floor around the table (cabinets, stools, bins)
  walls       : 1-3 textured walls beyond every camera
  occluders   : 0-2 poles / bars / panels between a world camera and the robot (partial occlusion)
  lights      : colour tint per light, optional positional spot light

Labels (visible map only; amodal robot / object maps are unaffected):
  LABEL_DIS0 + k  table distractor k           LABEL_CLUTTER  walls, furniture, occluders
Everything is described in scene["clutter"] (texture kinds + seeds, final poses) so an episode can be rebuilt.
Walls, furniture and occluders depend on the sampled cameras, so they are placed by place_after_cams().
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import cv2
import numpy as np

LABEL_DIS0, LABEL_CLUTTER = 210, 250
MAX_DIS = 10
TEX_KINDS = ["wood", "noise", "marble", "tiles", "stripes", "carpet", "posters", "flat"]
DIS_SHAPES = ["sphere", "cylinder", "capsule", "ellipsoid", "mug", "bottle", "can"]
TEX_RES = 256


def _s(v):
    return " ".join(f"{float(x):.6g}" for x in np.atleast_1d(v))


# ------------------------------------------------------------------------------------------- textures
def _value_noise(rng, res, cells):
    g = rng.random((cells + 1, cells + 1)).astype(np.float32)
    return cv2.resize(g, (res, res), interpolation=cv2.INTER_CUBIC)


def _fbm(rng, res, octaves=4, base=4):
    out, amp, tot = np.zeros((res, res), np.float32), 1.0, 0.0
    for o in range(octaves):
        out += amp * _value_noise(rng, res, base * 2 ** o); tot += amp; amp *= 0.5
    return out / tot


def _mix(a, b, t):
    return (np.asarray(a)[None, None] * (1 - t[..., None]) + np.asarray(b)[None, None] * t[..., None])


def make_texture(kind: str, seed: int, res=TEX_RES):
    """RGB uint8 image, reproducible from (kind, seed)."""
    rng = np.random.default_rng(seed)
    c1 = rng.uniform(0.05, 0.95, 3); c2 = np.clip(c1 * rng.uniform(0.5, 1.3) + rng.normal(0, 0.12, 3), 0, 1)
    y, x = np.mgrid[0:res, 0:res].astype(np.float32) / res
    if kind == "wood":
        c1 = np.array([0.55, 0.36, 0.2]) * rng.uniform(0.6, 1.4); c2 = c1 * rng.uniform(0.55, 0.8)
        n = _fbm(rng, res, 3, 3)
        t = 0.5 + 0.5 * np.sin((x * rng.uniform(15, 40) + 6 * n) * 2 * np.pi / 4)
        img = _mix(c1, c2, t ** 2)
    elif kind == "noise":
        img = _mix(c1, c2, _fbm(rng, res, 5, int(rng.integers(2, 8))))
    elif kind == "marble":
        n = _fbm(rng, res, 5, 4)
        t = 0.5 + 0.5 * np.sin((x + y) * rng.uniform(4, 12) + 8 * n)
        img = _mix(c1, np.clip(c1 + 0.35, 0, 1), t ** 3)
    elif kind == "tiles":
        k = int(rng.integers(2, 9)); w = rng.uniform(0.02, 0.08)
        gx, gy = (x * k) % 1, (y * k) % 1
        grout = (gx < w) | (gy < w)
        tile_col = rng.uniform(-0.08, 0.08, (k, k))[np.minimum((y * k).astype(int), k - 1), np.minimum((x * k).astype(int), k - 1)]
        img = np.clip(c1[None, None] + tile_col[..., None], 0, 1)
        img[grout] = c2 * 0.6
    elif kind == "stripes":
        k = rng.uniform(3, 20); a = rng.uniform(0, np.pi)
        t = (np.sin((x * np.cos(a) + y * np.sin(a)) * k * 2 * np.pi) > rng.uniform(-0.5, 0.5)).astype(np.float32)
        img = _mix(c1, c2, t)
    elif kind == "carpet":
        t = 0.7 * rng.random((res, res)).astype(np.float32) + 0.3 * _fbm(rng, res, 3, 6)
        img = _mix(c1, c2, t)
    elif kind == "posters":
        img = _mix(c1, c1 * 0.9, _fbm(rng, res, 3, 3))
        for _ in range(int(rng.integers(2, 7))):
            w, h = rng.uniform(0.1, 0.4, 2); px, py = rng.uniform(0, 1 - w), rng.uniform(0, 1 - h)
            m = (x > px) & (x < px + w) & (y > py) & (y < py + h)
            pc = rng.uniform(0, 1, 3)
            if rng.random() < 0.5:
                img[m] = _mix(pc, rng.uniform(0, 1, 3), _fbm(rng, res, 3, 8))[m]
            else:
                img[m] = pc
    else:  # flat
        img = np.broadcast_to(c1, (res, res, 3)).copy()
    img = np.clip(img + rng.normal(0, 0.015, img.shape), 0, 1)
    return (img * 255).astype(np.uint8)


def texture_png(kind, seed):
    ok, b = cv2.imencode(".png", make_texture(kind, seed)[..., ::-1])
    return b.tobytes()


def _tex(rng, kinds=TEX_KINDS):
    return dict(kind=str(rng.choice(kinds)), seed=int(rng.integers(2 ** 31)), repeat=float(rng.uniform(1, 6)),
                reflectance=float(rng.uniform(0, 0.15)))


# ------------------------------------------------------------------------------------------- sampling
def sample_clutter(layout, scene, rng):
    R, th = layout.R, layout.table_h
    tc, th_half = np.asarray(layout.table_center), np.asarray(layout.table_half)
    base = np.asarray(layout.base_pos)
    cube = layout.cube_half
    C = dict(
        table_tex=_tex(rng, ["wood", "noise", "marble", "tiles", "stripes", "flat", "carpet"]) if rng.random() < 0.8 else None,
        floor_tex=_tex(rng, ["wood", "noise", "tiles", "carpet", "marble", "stripes"]),
        sky=dict(rgb1=rng.uniform(0.2, 1.0, 3).tolist(), rgb2=rng.uniform(0.0, 0.6, 3).tolist()),
        light_tint=[np.clip(1 + rng.normal(0, 0.12, 3), 0.6, 1.3).tolist() for _ in scene["lights"]],
        spot=(dict(pos=(tc + np.r_[rng.uniform(-1, 1, 2) * R, rng.uniform(1.0, 2.5) * R]).tolist(),
                   diffuse=float(rng.uniform(0.1, 0.35)), cutoff=float(rng.uniform(25, 70)))
              if rng.random() < 0.5 else None),
        distractors=[], walls=[], furniture=[], occluders=[])
    # --- table distractors: keep clear of the blocks and the robot base; tall ones only far from the base
    placed = [(np.asarray(o["pos"][:2]), float(np.linalg.norm(o["half"][:2]))) for o in scene["objects"]]
    placed.append((base[:2], 0.15 * R))
    for k in range(int(rng.integers(3, MAX_DIS + 1))):
        shape = str(rng.choice(DIS_SHAPES))
        r = float(rng.uniform(0.5, 1.6) * cube)
        h = float(rng.uniform(0.6, 2.0) * cube)
        if shape in ("bottle", "can", "mug"):
            h = float(rng.uniform(1.5, 4.0 if shape == "bottle" else 2.5) * cube)
        for _ in range(60):
            p = tc[:2] + rng.uniform(-1, 1, 2) * (th_half[:2] - r - 0.01)
            far = np.linalg.norm(p - base[:2]) > 0.8 * R
            if h > 2.2 * cube and not far:
                continue
            if all(np.linalg.norm(p - q) > (r + rq) + 0.05 * R for q, rq in placed):
                break
        else:
            continue
        placed.append((p, r))
        C["distractors"].append(dict(shape=shape, r=r, h=h, pos=[float(p[0]), float(p[1]), th],
                                     yaw=float(rng.uniform(-np.pi, np.pi)), rgba=rng.uniform(0.05, 0.95, 3).tolist() + [1],
                                     tex=_tex(rng) if rng.random() < 0.3 else None))
    # --- camera-dependent items: only their sizes / looks are drawn here, poses in place_after_cams
    for _ in range(int(rng.integers(1, 4))):
        C["walls"].append(dict(az=float(rng.uniform(-np.pi, np.pi)), gap=float(rng.uniform(0.2, 1.5)),
                               height=float(rng.uniform(2.0, 3.5)), tex=_tex(rng, ["posters", "noise", "tiles", "flat", "stripes", "marble"])))
    for _ in range(int(rng.integers(0, 5))):
        C["furniture"].append(dict(shape=str(rng.choice(["box", "cylinder"])),
                                   size=[float(rng.uniform(0.15, 0.5)), float(rng.uniform(0.15, 0.5)), float(rng.uniform(0.2, 1.0))],
                                   rgba=rng.uniform(0.05, 0.9, 3).tolist() + [1], tex=_tex(rng) if rng.random() < 0.5 else None))
    for _ in range(int(rng.choice([0, 1, 1, 2]))):
        kind = str(rng.choice(["pole", "bar", "panel"]))
        C["occluders"].append(dict(kind=kind, cam=str(rng.choice(["top", "front", "front"])),
                                   frac=float(rng.uniform(0.25, 0.6)), side=float(rng.uniform(0.3, 1.2)) * rng.choice([-1, 1]),
                                   size=float(rng.uniform(0.01, 0.04) if kind != "panel" else rng.uniform(0.04, 0.12)) * max(R, 0.3) / 0.3,
                                   half_len=float(rng.uniform(0.6, 1.5) * max(R, 0.3)), aspect=float(rng.uniform(0.6, 1.8)),
                                   rgba=rng.uniform(0.05, 0.9, 3).tolist() + [1]))
    return C


# ------------------------------------------------------------------------------------------- XML
def _material(asset, name, tex, assets: dict):
    fn = f"{name}.png"
    assets[fn] = texture_png(tex["kind"], tex["seed"])
    ET.SubElement(asset, "texture", name=f"{name}_tex", type="2d", file=fn)
    ET.SubElement(asset, "material", name=name, texture=f"{name}_tex", texrepeat=_s([tex["repeat"]] * 2),
                  reflectance=f"{tex['reflectance']:.3g}", texuniform="true")
    return name


def apply_xml(x: ET.Element, layout, C: dict, scene: dict):
    """Edit the world XML in place; returns the texture assets dict for MjModel.from_xml_string."""
    assets = {}
    asset, wb = x.find("asset"), x.find("worldbody")
    for t in asset.iter("texture"):
        if t.get("name") == "sky":
            t.set("rgb1", _s(C["sky"]["rgb1"])); t.set("rgb2", _s(C["sky"]["rgb2"]))
    for g in wb.iter("geom"):
        if g.get("name") == "floor":
            g.set("material", _material(asset, "floor_mat2", C["floor_tex"], assets))
        if g.get("name") == "table" and C["table_tex"]:
            g.set("material", _material(asset, "table_mat", C["table_tex"], assets)); g.set("rgba", "1 1 1 1")
    lights = [l for l in wb.findall("light")]
    for L, tint in zip(lights, C["light_tint"]):
        d = float(L.get("diffuse").split()[0])
        L.set("diffuse", _s(np.asarray(tint) * d))
    if C["spot"]:
        s = C["spot"]
        ET.SubElement(wb, "light", pos=_s(s["pos"]), dir=_s(np.asarray(layout.table_center) - s["pos"]), directional="false",
                      diffuse=_s([s["diffuse"]] * 3), cutoff=f"{s['cutoff']:.3g}", castshadow="true", attenuation="1 0 0")
    st = dict(contype="0", conaffinity="0", group="1")

    def geom(b, nm, **kw):
        ET.SubElement(b, "geom", name=nm, **st, **kw)

    for k, d in enumerate(C["distractors"]):
        q = [np.cos(d["yaw"] / 2), 0, 0, np.sin(d["yaw"] / 2)]
        b = ET.SubElement(wb, "body", name=f"distractor{k}", pos=_s(d["pos"]), quat=_s(q))
        look = dict(rgba=_s(d["rgba"]))
        if d["tex"]:
            look = dict(material=_material(asset, f"dis{k}_mat", d["tex"], assets), rgba="1 1 1 1")
        r, h, n = d["r"], d["h"], f"distractor{k}_"
        if d["shape"] == "sphere":
            geom(b, n + "0", type="sphere", size=_s(r), pos=_s([0, 0, r]), **look)
        elif d["shape"] == "cylinder":
            geom(b, n + "0", type="cylinder", size=_s([r, h / 2]), pos=_s([0, 0, h / 2]), **look)
        elif d["shape"] == "capsule":                                   # lying on its side
            geom(b, n + "0", type="capsule", size=_s([r * 0.6, h / 2]), pos=_s([0, 0, r * 0.6]), quat="0.7071 0.7071 0 0", **look)
        elif d["shape"] == "ellipsoid":
            geom(b, n + "0", type="ellipsoid", size=_s([r, r * 0.7, h / 2]), pos=_s([0, 0, h / 2]), **look)
        elif d["shape"] == "mug":
            geom(b, n + "0", type="cylinder", size=_s([r, h / 2]), pos=_s([0, 0, h / 2]), **look)
            geom(b, n + "1", type="box", size=_s([0.25 * r, 0.12 * r, 0.3 * h]), pos=_s([r * 1.15, 0, h / 2]), **look)
        elif d["shape"] == "bottle":
            geom(b, n + "0", type="cylinder", size=_s([r, 0.35 * h]), pos=_s([0, 0, 0.35 * h]), **look)
            geom(b, n + "1", type="capsule", size=_s([0.4 * r, 0.15 * h]), pos=_s([0, 0, 0.8 * h]), **look)
        else:  # can
            geom(b, n + "0", type="cylinder", size=_s([r * 0.8, h / 2]), pos=_s([0, 0, h / 2]), **look)
    for k, w in enumerate(C["walls"]):
        b = ET.SubElement(wb, "body", name=f"clutter_wall{k}", pos="0 0 -20")
        geom(b, f"clutter_wall{k}", type="box", size=_s([0.03, 4.0, w["height"] / 2]),
             material=_material(asset, f"wall{k}_mat", w["tex"], assets), rgba="1 1 1 1")
    for k, f in enumerate(C["furniture"]):
        b = ET.SubElement(wb, "body", name=f"clutter_furn{k}", pos="0 0 -20")
        look = dict(rgba=_s(f["rgba"]))
        if f["tex"]:
            look = dict(material=_material(asset, f"furn{k}_mat", f["tex"], assets), rgba="1 1 1 1")
        sx, sy, sz = f["size"]
        if f["shape"] == "box":
            geom(b, f"clutter_furn{k}", type="box", size=_s([sx / 2, sy / 2, sz / 2]), pos=_s([0, 0, sz / 2]), **look)
        else:
            geom(b, f"clutter_furn{k}", type="cylinder", size=_s([sx / 2, sz / 2]), pos=_s([0, 0, sz / 2]), **look)
    for k, o in enumerate(C["occluders"]):
        b = ET.SubElement(wb, "body", name=f"clutter_occ{k}", pos="0 0 -20")
        s = o["size"]
        if o["kind"] in ("pole", "bar"):   # capsule along body z; placement aims z (vertical pole / sideways bar)
            geom(b, f"clutter_occ{k}", type="capsule", size=_s([s, o["half_len"]]), rgba=_s(o["rgba"]))
        else:                              # thin panel, body z faces the camera
            geom(b, f"clutter_occ{k}", type="box", size=_s([s, s * o["aspect"], 0.005]), rgba=_s(o["rgba"]))
    return assets


def label_for(name: str):
    if name.startswith("distractor"):
        return LABEL_DIS0 + int(name[len("distractor"):].split("_")[0])
    if name.startswith("clutter_"):
        return LABEL_CLUTTER
    return None


# ------------------------------------------------------------------------------------------- placement
def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip((p - a) @ ab / max(ab @ ab, 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def _look_quat(z_axis):
    """Quaternion (wxyz) rotating the body z axis onto z_axis."""
    from scipy.spatial.transform import Rotation
    z = z_axis / np.linalg.norm(z_axis)
    a = np.cross([0, 0, 1.0], z); s = np.linalg.norm(a)
    if s < 1e-8:
        return np.array([1.0, 0, 0, 0])
    q = Rotation.from_rotvec(a / s * np.arctan2(s, z[2])).as_quat()
    return q[[3, 0, 1, 2]]


def place_after_cams(world, C: dict, tracks: dict, rng):
    """Put walls beyond all cameras, furniture off the view lines, occluders partly across them."""
    m, L = world.model, world.layout
    R, tc = L.R, np.asarray(L.table_center)
    robot_c = np.asarray(L.base_pos) + np.r_[0.3 * R, 0, 0.25 * R]
    cam_pts = np.array([p.pos for c in ("top", "front") for p in tracks[c]])
    cam0 = {c: tracks[c][0].pos for c in ("top", "front")}
    rmax = float(np.max(np.linalg.norm(cam_pts[:, :2] - tc[:2], axis=1)))
    rmax = max(rmax, float(np.linalg.norm(L.table_half[:2]))) + 0.3

    def setb(name, pos, quat=(1, 0, 0, 0)):
        bid = m.body(name).id
        m.body_pos[bid] = pos; m.body_quat[bid] = quat

    for k, w in enumerate(C["walls"]):
        d = rmax + w["gap"]
        p = np.r_[tc[:2] + d * np.r_[np.cos(w["az"]), np.sin(w["az"])], w["height"] / 2]
        q = [np.cos(w["az"] / 2), 0, 0, np.sin(w["az"] / 2)]
        setb(f"clutter_wall{k}", p, q); w["pos"] = p.tolist()
    for k, f in enumerate(C["furniture"]):
        rad = float(np.hypot(f["size"][0], f["size"][1]) / 2)
        f["pos"] = None
        for _ in range(40):
            a = rng.uniform(-np.pi, np.pi)
            p = np.r_[tc[:2] + rng.uniform(np.linalg.norm(L.table_half[:2]) + rad, rmax + 0.5) * np.r_[np.cos(a), np.sin(a)], 0.0]
            if np.min(np.linalg.norm(cam_pts[:, :2] - p[:2], axis=1)) < rad + 0.15:
                continue
            if any(_seg_dist(p + np.r_[0, 0, f["size"][2] / 2], cam0[c], robot_c) < rad + 0.1 * R for c in cam0):
                continue
            yaw = rng.uniform(-np.pi, np.pi)
            setb(f"clutter_furn{k}", p, [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]); f["pos"] = p.tolist(); f["yaw"] = float(yaw)
            break
    for k, o in enumerate(C["occluders"]):
        o["pos"] = None
        a = cam0[o["cam"]]
        for _ in range(30):
            tgt = robot_c + rng.normal(0, 0.15 * R, 3)
            v = tgt - a
            pt = a + o["frac"] * v
            side = np.cross(v / np.linalg.norm(v), [0, 0, 1.0])
            if np.linalg.norm(side) < 1e-3:
                side = np.r_[1.0, 0, 0]
            side /= np.linalg.norm(side)
            pt = pt + side * o["side"] * 0.12 * R
            if np.linalg.norm(pt - np.asarray(L.base_pos)) < 1.05 * R or pt[2] < L.table_h + 0.02:
                continue
            if np.min(np.linalg.norm(cam_pts - pt, axis=1)) < o["size"] + 0.08:
                continue
            if o["kind"] == "pole":
                quat = _look_quat(np.r_[rng.normal(0, 0.15, 2), 1.0])
            elif o["kind"] == "bar":
                quat = _look_quat(side + rng.normal(0, 0.25, 3))
            else:
                quat = _look_quat(-v)
            setb(f"clutter_occ{k}", pt, quat); o["pos"] = pt.tolist(); o["quat"] = np.asarray(quat).tolist()
            break


def limit_occlusion(world, C: dict, render_fn, min_visible=0.5):
    """Drop occluders (last first) from a camera while less than min_visible of the robot stays visible there."""
    nb = len(world.robot_bodies)
    for c in ("top", "front"):
        while True:
            _, vis, am_r, _ = render_fn(c)
            ratio = ((vis >= 1) & (vis <= nb)).sum() / max((am_r > 0).sum(), 1)
            mine = [k for k, o in enumerate(C["occluders"]) if o["cam"] == c and o.get("pos") is not None]
            if ratio >= min_visible or not mine:
                break
            k = mine[-1]
            world.model.body_pos[world.model.body(f"clutter_occ{k}").id] = [0, 0, -20]
            C["occluders"][k]["pos"] = None; C["occluders"][k]["dropped"] = f"robot visible {ratio:.2f} < {min_visible}"
            world.refresh()
