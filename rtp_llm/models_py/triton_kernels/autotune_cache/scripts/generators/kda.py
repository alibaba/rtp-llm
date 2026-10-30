# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Adapted for rtp-llm


class KDACacheGenerator:
    """Run in-tree KDA kernels to populate a Triton autotune cache.

    Forward-only, prefill-only. Exercises Kimi Linear's chunk_kda path
    (use_gate_in_kernel=True, safe_gate=False, lower_bound=None,
    IS_VARLEN=True) and the standalone fused_kda_gate interface.

    cuLA prefill owns its tuning separately. fused_recurrent_kda (decode)
    is not autotuned, so it produces no winners to cache.
    """

    def __init__(self, head_dim: int, *, device):
        self.head_dim = head_dim
        self.device = device
        self._tensors: tuple | None = None

    @property
    def tensors(self) -> tuple:
        if self._tensors is None:
            self._tensors = self._prepare_tensors()
        return self._tensors

    def _prepare_tensors(self) -> tuple:
        import torch

        torch.manual_seed(42)
        dtype = torch.bfloat16
        # Varlen layout: [1, total_tokens, H, D] + cu_seqlens.
        # Matches the IS_VARLEN=True path used by Kimi Linear inference.
        N, T_per_seq, H, D = 2, 1504, 4, self.head_dim
        T_total = N * T_per_seq
        print(
            f"Generating cache with head_dim={D}, varlen "
            f"(N={N}, T_per_seq={T_per_seq}, total_tokens={T_total})"
        )

        q = torch.rand(1, T_total, H, D, dtype=dtype)
        k = torch.rand(1, T_total, H, D, dtype=dtype)
        v = torch.rand(1, T_total, H, D, dtype=dtype)
        g = torch.randn(1, T_total, H, D, dtype=torch.float32)
        A_log = torch.randn(H, dtype=torch.float)
        dt_bias = torch.randn(H * D, dtype=torch.float)
        beta = torch.randn(1, T_total, H, dtype=dtype).float().sigmoid()
        # Initial states: one per sequence, shape [N, H, D, D].
        h0 = torch.randn(N, H, D, D, dtype=torch.float32)
        cu_seqlens = torch.tensor(
            [i * T_per_seq for i in range(N + 1)], dtype=torch.int32
        )

        A_log = A_log.to(self.device)
        dt_bias = dt_bias.to(self.device)
        q = q.to(self.device)
        k = k.to(self.device)
        v = v.to(self.device)
        beta = beta.to(self.device)
        h0 = h0.to(self.device)
        g = g.to(self.device)
        cu_seqlens = cu_seqlens.to(self.device)
        return q, k, v, g, beta, h0, A_log, dt_bias, cu_seqlens

    def generate_kda(self) -> None:
        from rtp_llm.models_py.triton_kernels.kda import chunk_kda, fused_kda_gate

        q, k, v, g, beta, h0, A_log, dt_bias, cu_seqlens = self.tensors

        # Exercise Kimi Linear's production path: use_gate_in_kernel=True,
        # safe_gate=False (default), lower_bound=None, IS_VARLEN=True.
        chunk_kda(
            q=q.clone(),
            k=k.clone(),
            v=v.clone(),
            g=g.clone(),
            beta=beta.clone(),
            A_log=A_log.clone(),
            dt_bias=dt_bias.clone(),
            scale=None,
            initial_state=h0.clone(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens.clone(),
            use_gate_in_kernel=True,
        )

        # Also cover the standalone gate kernel without a lower bound.
        _ = fused_kda_gate(
            g=g.clone(),
            A_log=A_log.clone(),
            dt_bias=dt_bias.clone(),
        )
