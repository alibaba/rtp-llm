"""Opt-in SM120 decode mixer supplied by rtp-kernel.

The optional rtp_kernel.gr_mixer API uses the retained GR implementation
without changing the shared DeepGEMM dependency. The model runner must
serialize forwards/replays using this module, as it does for its other mutable
decode state. Concurrent model instances must own separate mixer instances.
"""

import logging

import torch


class DecodeGrMixer:
    @classmethod
    def create(cls, down, up, inject):
        tensors = [(down, (320, 10240)), (up, (10240, 320))]
        if inject is not None:
            tensors.append((inject, (4, 10240)))
        if any(
            tuple(t.shape) != shape
            or not t.is_cuda
            or t.device != down.device
            or t.dtype != torch.bfloat16
            or not t.is_contiguous()
            or t.requires_grad
            or t.data_ptr() % 16
            for t, shape in tensors
        ):
            return None
        if torch.version.hip is not None or torch.cuda.get_device_capability(
            down.device
        ) != (12, 0):
            return None
        try:
            import rtp_kernel.gr_mixer as gr_mixer
        except ImportError as exc:
            raise RuntimeError(
                "RTP_QWEN4_FUSED_GR=1 requires rtp_kernel.gr_mixer from an "
                "rtp-kernel CUDA13 wheel with SM120 GR support; install that "
                "wheel specified by the CUDA13 dependency requirements"
            ) from exc
        return cls(gr_mixer, up, inject is not None)

    def __init__(self, kernel, up, has_inject):
        self.kernel = kernel
        self.device = up.device
        self.has_inject = has_inject
        self.packed_up = kernel.prepare_up(up)
        # The original Tensor is also the loader's dictionary value. Its update
        # guard prevents a hot update from invalidating packed/captured weights.
        up._rtp_requires_restart_for_update = True
        self.workspaces = {}

    def supports(self, x):
        return (
            x.ndim == 2
            and 1 <= x.shape[0] <= 64
            and x.shape[1] == 10240
            and x.device == self.device
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and not x.requires_grad
            and x.data_ptr() % 16 == 0
        )

    def __call__(self, x, down, inject):
        rows = x.shape[0]
        workspace = self.workspaces.get(rows)
        if workspace is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"SM120 GR rows={rows} must be warmed before CUDA Graph capture"
                )
            # CudaGraphRunner warms the exact shape twice and synchronizes
            # before switching to its capture stream. Scratch is retained for
            # the graph lifetime; serialized replays reuse its epoch counters.
            workspace = self.kernel.GRWorkspace(rows, self.device, self.has_inject)
            self.workspaces[rows] = workspace
            logging.info(
                "SM120 GR workspace prepared: rows=%d write_gate=%s",
                rows,
                self.has_inject,
            )
        return self.kernel.fused_gr(
            x, down, self.packed_up, inject, workspace=workspace
        )
