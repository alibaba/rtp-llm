"""TP projection collectives for grouped E4M3 activations on Blackwell.

The input transition communicates FP8 values and their UE8M0 scales, then
consumes each rank's rows in a pipelined GEMM. The output transition reduces
FP8 GEMM partials into BF16 sequence-parallel rows. Resources are initialized
collectively before CUDA Graph capture and are reused by all projection layers.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm


@dataclass(frozen=True)
class Fp8Activation:
    """Group128 E4M3 rows with TMA-aligned packed UE8M0 scale storage."""

    values: torch.Tensor
    scale_wire: torch.Tensor

    def __post_init__(self):
        if (
            self.values.ndim != 2
            or self.values.dtype != torch.float8_e4m3fn
            or not self.values.is_contiguous()
        ):
            raise ValueError("FP8 activation values must be contiguous E4M3 [M,K]")
        m, k = self.values.shape
        if (
            k % 128
            or self.scale_wire.shape != ((k + 511) // 512, (m + 3) // 4 * 4)
            or self.scale_wire.dtype != torch.int32
            or not self.scale_wire.is_contiguous()
            or self.scale_wire.device != self.values.device
        ):
            raise ValueError("FP8 activation scales must be packed UE8M0 group128")

    @property
    def shape(self):
        return self.values.shape

    @property
    def device(self):
        return self.values.device

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def is_cuda(self):
        return self.values.is_cuda


class Fp8CollectiveProjection:
    def __init__(
        self, group, device, *, max_m: int, hidden_size: int,
        enable_ag: bool = True, enable_rs: bool = True,
    ):
        self.group = group
        self.device = torch.device(device)
        self.world_size = dist.get_world_size(group)
        self.max_m = int(max_m)
        self.hidden_size = int(hidden_size)
        self.enable_ag = enable_ag
        self.enable_rs = enable_rs
        if self.world_size < 2 or self.max_m % self.world_size:
            raise ValueError("FP8 collective capacity must be divisible by TP size")
        if self.hidden_size % 512:
            raise ValueError("FP8 collective hidden size must be 512-aligned")
        if enable_ag and not callable(getattr(symm, "_pipelined_multi_all_gather_and_consume", None)):
            raise RuntimeError("PyTorch FP8 AllGather consumer pipeline unavailable")
        if enable_ag:
            local_m = self.max_m // self.world_size
            groups = (self.hidden_size + 511) // 512
            aligned_local_m = (local_m + 3) // 4 * 4
            ag_bytes = local_m * self.hidden_size + groups * aligned_local_m * 4
            symm.get_symm_mem_workspace(group.group_name, min_size=ag_bytes)
        self.rs_workspace = None
        self.deep_gemm = None
        if enable_rs:
            import deep_gemm

            if not callable(getattr(deep_gemm, "fp8_gemm_rs_nt", None)):
                raise RuntimeError("DeepGEMM FP8 GEMM/RS API unavailable")
            if not hasattr(deep_gemm, "GemmRSBuffer"):
                raise RuntimeError("DeepGEMM GEMM/RS workspace unavailable")
            self.rs_workspace = deep_gemm.GemmRSBuffer(
                group, max_m=self.max_m, n=self.hidden_size, device=self.device
            )
            self.deep_gemm = deep_gemm

    def can_run_ag(self, local_rows: int) -> bool:
        return 0 < local_rows and local_rows * self.world_size <= self.max_m

    def can_run_rs(self, rows: int) -> bool:
        return 0 < rows <= self.max_m and rows % self.world_size == 0

    def all_gather_gemm(self, local_input: torch.Tensor | Fp8Activation, projection):
        """Gather FP8 values and scales, projecting each source rank's rows."""
        if not self.enable_ag:
            raise RuntimeError("FP8 AG/GEMM was not initialized")
        if (
            local_input.device != self.device
            or local_input.shape[1] != self.hidden_size
            or not self.can_run_ag(local_input.shape[0])
            or not projection.scale_ue8m0
            or projection.K != self.hidden_size
        ):
            raise ValueError(
                "FP8 AG/GEMM input or projection is incompatible: "
                f"input_shape={tuple(local_input.shape)} device={local_input.device} "
                f"expected_device={self.device} tp={self.world_size} max_m={self.max_m} "
                f"projection_k={projection.K} scale_ue8m0={projection.scale_ue8m0}"
            )
        local_m, k = local_input.shape
        if isinstance(local_input, Fp8Activation):
            values, scale_wire = local_input.values, local_input.scale_wire
        else:
            if (
                local_input.ndim != 2
                or local_input.dtype != torch.bfloat16
                or not local_input.is_contiguous()
            ):
                raise ValueError("FP8 AG/GEMM requires BF16 or prequantized E4M3")
            values, scales = projection.quantize_input(local_input)
            aligned_m = (local_m + 3) // 4 * 4
            scale_wire = scales.as_strided(
                ((k + 511) // 512, aligned_m), (aligned_m, 1)
            )
        values_wire = values.view(torch.uint8)
        gathered = torch.empty(
            (local_m * self.world_size, k), dtype=torch.uint8, device=self.device
        )
        gathered_scale = torch.empty(
            (self.world_size * scale_wire.shape[0], scale_wire.shape[1]),
            dtype=torch.int32,
            device=self.device,
        )
        output = torch.empty(
            (local_m * self.world_size, projection.N),
            dtype=torch.bfloat16,
            device=self.device,
        )

        def consume(pair, rank):
            rank_values, rank_scales = pair
            projection.forward_quantized(
                rank_values.view(torch.float8_e4m3fn),
                rank_scales.T[:local_m],
                out=output.narrow(0, rank * local_m, local_m),
            )

        symm._pipelined_multi_all_gather_and_consume(
            [values_wire, scale_wire],
            consume,
            [gathered, gathered_scale],
            self.group.group_name,
            ag_out_needed=False,
        )
        return output

    def gemm_reduce_scatter(self, values, scales, projection):
        """Fuse grouped FP8 GEMM with a BF16 ReduceScatter result."""
        if not self.enable_rs:
            raise RuntimeError("FP8 GEMM/RS was not initialized")
        m, k = values.shape
        if (
            values.dtype != torch.float8_e4m3fn
            or values.device != self.device
            or not self.can_run_rs(m)
            or k != projection.K
            or projection.N != self.hidden_size
            or not projection.scale_ue8m0
            or projection.bias is not None
        ):
            raise ValueError("FP8 GEMM/RS input or projection is incompatible")
        output = torch.empty(
            (m // self.world_size, self.hidden_size),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.deep_gemm.fp8_gemm_rs_nt(
            (values, scales),
            (projection.weight, projection.weight_scales),
            output,
            self.rs_workspace,
            compiled_dims="nk",
        )
        return output
