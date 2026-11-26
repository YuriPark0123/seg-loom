"""torch side of seg-loom: differentiable kinematics, camera projection, point rendering, data loading.

Everything here is batched torch and runs on any device. Nothing reads MuJoCo at training time:
the robot comes from a URDF (true.urdf for checks, approx.urdf as the model input) and the cameras
from meta.json, exactly as docs/ARCHITECTURE.md (M3, M5) assumes.
"""
