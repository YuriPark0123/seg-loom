"""Map CUDA / nvidia-smi GPU indices to EGL device indices (MUJOCO_EGL_DEVICE_ID).

EGL enumerates devices in its own order. On the B200 server EGL device 0 is nvidia-smi GPU 4 and EGL device 4 is
GPU 0, so MUJOCO_EGL_DEVICE_ID=<nvidia-smi index> silently renders on someone else's GPU. Every EGL device of the
NVIDIA driver reports its CUDA ordinal through EGL_NV_device_cuda (EGL_CUDA_DEVICE_NV); this module uses that.
CUDA ordinals follow nvidia-smi order when CUDA_DEVICE_ORDER=PCI_BUS_ID (identical GPUs: the default order agrees).

    python -m shadow.egl_devices        # print the table
"""
from __future__ import annotations

import ctypes
import os

EGL_CUDA_DEVICE_NV = 0x323A


def egl_cuda_table() -> dict[int, int]:
    """{EGL device index: CUDA ordinal} for the devices that report one (NVIDIA)."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from mujoco.egl import egl_ext as EGL          # bindings only; does not create a display
    proto = ctypes.CFUNCTYPE(EGL.EGLBoolean, EGL.EGLDeviceEXT, EGL.EGLint, ctypes.POINTER(ctypes.c_ssize_t))
    query = proto(EGL.eglGetProcAddress("eglQueryDeviceAttribEXT"))
    out = {}
    for i, dev in enumerate(EGL.eglQueryDevicesEXT(32)):
        v = ctypes.c_ssize_t(-1)
        if query(dev, EGL_CUDA_DEVICE_NV, ctypes.byref(v)) == EGL.EGL_TRUE and v.value >= 0:
            out[i] = int(v.value)
    return out


def egl_index_for_gpu(gpu: int) -> int:
    """EGL device index that renders on nvidia-smi / CUDA GPU `gpu`."""
    for egl_i, cuda_i in egl_cuda_table().items():
        if cuda_i == gpu:
            return egl_i
    raise RuntimeError(f"no EGL device reports CUDA device {gpu}")


if __name__ == "__main__":
    for e, c in sorted(egl_cuda_table().items()):
        print(f"EGL device {e} -> GPU {c}")
