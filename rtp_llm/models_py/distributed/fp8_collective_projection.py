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
        decode_staging: bool = False,
    ):
        self.group = group
        self.device = torch.device(device)
        self.world_size = dist.get_world_size(group)
        self.max_m = int(max_m)
        self.hidden_size = int(hidden_size)
        self.enable_ag = enable_ag
        self.enable_rs = enable_rs
        self.decode_staging = decode_staging
        if self.world_size < 2 or self.max_m % self.world_size:
            raise ValueError("FP8 collective capacity must be divisible by TP size")
        if self.hidden_size % 512:
            raise ValueError("FP8 collective hidden size must be 512-aligned")
        if enable_ag and not decode_staging and not callable(getattr(symm, "_pipelined_multi_all_gather_and_consume", None)):
            raise RuntimeError("PyTorch FP8 AllGather consumer pipeline unavailable")
        self.custom_ag = None
        if enable_ag:
            if decode_staging:
                from rtp_llm.models_py.distributed.custom_all_gather import create_custom_all_gather

                self.custom_ag = create_custom_all_gather(
                    group, self.device, max_m=self.max_m, k=self.hidden_size,
                    fp8=True, staging_only=True,
                )
                if self.custom_ag is None:
                    raise RuntimeError("Decode FP8 staging AllGather is unavailable")
            else:
                local_m = self.max_m // self.world_size
                groups = (self.hidden_size + 511) // 512
                aligned_local_m = (local_m + 3) // 4 * 4
                ag_bytes = local_m * self.hidden_size + groups * aligned_local_m * 4
                symm.get_symm_mem_workspace(group.group_name, min_size=ag_bytes)
        self.rs_workspace = None
        self.deep_gemm = None
        self.push_rs = None
        if enable_rs:
            if decode_staging:
                from rtp_llm.models_py.distributed.push_reduce_scatter import create_push_reduce_scatter

                self.push_rs = create_push_reduce_scatter(
                    group, self.device, max_m=self.max_m, n=self.hidden_size
                )
                if self.push_rs is None:
                    raise RuntimeError("Decode BF16 push ReduceScatter is unavailable")
            else:
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

    def eligible_ag(self, hidden, attention_inputs, metadata) -> bool:
        if not self.enable_ag or attention_inputs is None:
            return False
        if getattr(attention_inputs, "is_mtp_draft_update", False):
            return False
        if self.decode_staging:
            if not getattr(metadata, "is_target_verify", False):
                return False
            if not self.can_run_ag(hidden.shape[0]):
                raise RuntimeError("Decode FP8 AllGather exceeds its initialized capacity")
            return True
        if not self.can_run_ag(hidden.shape[0]):
            return False
        return (
            (hidden.shape[0] >= 4096 or isinstance(hidden, Fp8Activation))
            and attention_inputs.is_prefill
            and not getattr(metadata, "is_target_verify", False)
            and not torch.cuda.is_current_stream_capturing()
        )

    def can_run_rs(self, rows: int) -> bool:
        return 0 < rows <= self.max_m and rows % self.world_size == 0

    def eligible_rs(self, rows, attention_inputs, metadata, producer_available) -> bool:
        if not self.enable_rs or not producer_available or attention_inputs is None:
            return False
        if getattr(attention_inputs, "is_mtp_draft_update", False):
            return False
        if self.decode_staging:
            if not getattr(metadata, "is_target_verify", False):
                return False
            if not self.can_run_rs(rows):
                raise RuntimeError("Decode FP8 ReduceScatter exceeds its initialized capacity")
            return True
        return (
            rows >= 512
            and self.can_run_rs(rows)
            and attention_inputs.is_prefill
            and not getattr(metadata, "is_target_verify", False)
            and not torch.cuda.is_current_stream_capturing()
        )

    def reduce_scatter_bf16(self, partial):
        if self.push_rs is None or partial.dtype != torch.bfloat16 or not self.can_run_rs(partial.shape[0]):
            raise ValueError("Decode push ReduceScatter requires configured BF16 [M,H] partials")
        if partial.shape[1] != self.hidden_size or partial.device != self.device:
            raise ValueError("Decode push ReduceScatter has incompatible shape or device")
        partial = partial.contiguous()
        if partial.data_ptr() % 16:
            partial = partial.clone()
        output = torch.empty(
            (partial.shape[0] // self.world_size, self.hidden_size),
            dtype=torch.bfloat16, device=self.device,
        )
        self.push_rs.reduce_scatter(partial, output)
        return output

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
        if getattr(self, "decode_staging", False):
            gathered_values, gathered_scales = self.custom_ag.all_gather_fp8(
                values, scale_wire, staging=True,
            )
            return projection.forward_quantized(gathered_values, gathered_scales)
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
        """Project grouped FP8 input and return BF16 sequence-parallel rows."""
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
        if self.decode_staging:
            partial = projection.forward_quantized(values, scales)
            return self.reduce_scatter_bf16(partial)
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
