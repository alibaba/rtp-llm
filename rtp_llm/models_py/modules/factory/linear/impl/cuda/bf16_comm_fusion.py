"""BF16 sequence-parallel projections with shape-selected collectives."""

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed.collective_torch import (
    STAGING_MAX_LOCAL_M,
    create_custom_all_gather,
    create_push_reduce_scatter,
    is_same_host_group,
)


class Bf16CollectiveProjection:
    """Own prefill workspaces; decode keeps its existing graph-local policy."""

    def __init__(self, group, device, *, max_m, hidden_size, output_projections):
        import torch.distributed._symmetric_memory as symm

        self.group = group
        self.device = torch.device(device)
        self.world_size = group.size()
        self.max_m = max_m
        self.hidden_size = hidden_size
        if self.world_size not in (2, 4, 8, 16) or max_m <= 0 or max_m % self.world_size:
            raise ValueError("BF16 collective capacity must be positive and divisible by TP size")
        same_host = is_same_host_group(group)
        workspace_bytes = max_m // self.world_size * hidden_size * 2
        if same_host:
            symm.get_symm_mem_workspace(group.group_name, min_size=workspace_bytes)
        self.custom_ag = create_custom_all_gather(
            group, device, max_m=min(max_m, 4095 * self.world_size),
            k=hidden_size, fp8=False, staging_only=False,
        )
        self.peer_access = same_host or self.custom_ag is not None
        if self.peer_access and not same_host:
            symm.get_symm_mem_workspace(group.group_name, min_size=workspace_bytes)
        self.push_rs = create_push_reduce_scatter(
            group, device, max_m=min(max_m, 512), n=hidden_size,
        )
        self.deep_gemm = None
        self.rs_workspace = None
        self.output_weights = {}
        if self.peer_access and self.world_size in (2, 4, 8):
            import deep_gemm

            if not callable(getattr(deep_gemm, "bf16_gemm_rs_nn", None)):
                raise RuntimeError("DeepGEMM BF16 GEMM/RS API unavailable")
            self.rs_workspace = deep_gemm.GemmRSBuffer(
                group, max_m=max_m, n=hidden_size, device=device,
            )
            self.deep_gemm = deep_gemm
            self.output_weights = {
                id(projection): projection.weight.t().contiguous()
                for projection in output_projections
            }

    def eligible(self, rows, attention_inputs, metadata, *, gathered=False):
        total = rows if gathered else rows * self.world_size
        return (
            attention_inputs is not None
            and attention_inputs.is_prefill
            and not getattr(metadata, "is_target_verify", False)
            and 0 < total <= self.max_m
            and total % self.world_size == 0
        )

    def all_gather_gemm(self, local_input, projection):
        rows = local_input.shape[0]
        if rows >= 4096 and self.peer_access:
            _, outputs = torch.ops.symm_mem.fused_all_gather_matmul(
                local_input, [projection.weight.t()], 0,
                self.group.group_name, return_A=False,
            )
            return outputs[0]
        if self.custom_ag is not None and rows < 4096:
            gathered = self.custom_ag.all_gather(
                local_input, staging=rows < STAGING_MAX_LOCAL_M,
            )
        else:
            gathered = local_input.new_empty(rows * self.world_size, self.hidden_size)
            dist.all_gather_into_tensor(gathered, local_input, group=self.group)
        return projection(gathered)

    def gemm_reduce_scatter(self, value, projection):
        rows = value.shape[0]
        output = value.new_empty(rows // self.world_size, self.hidden_size)
        if rows >= 512 and self.rs_workspace is not None:
            self.deep_gemm.bf16_gemm_rs_nn(
                value.contiguous(), self.output_weights[id(projection)],
                output, self.rs_workspace, compiled_dims="nk",
            )
        else:
            partial = projection(value).contiguous()
            if rows < 512 and self.push_rs is not None:
                self.push_rs.reduce_scatter(partial, output)
            else:
                dist.reduce_scatter_tensor(output, partial, group=self.group)
        return output
