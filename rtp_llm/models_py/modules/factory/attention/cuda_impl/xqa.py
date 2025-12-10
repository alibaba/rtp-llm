import logging
from typing import Any

import torch
import torch.nn.functional as F
from flashinfer.xqa import xqa

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHADecodeImplBase,
)
from rtp_llm.ops import FMHAType
from rtp_llm.ops.compute_ops import (  # XQAAttnOp,
    FusedRopeKVCacheDecodeOp,
    PyAttentionInputs,
)


class XQAImpl(FMHADecodeImplBase):

    def __init__(
        self, config: GptInitModelParameters, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            XQAWrapper(config, attn_inputs),
            FusedRopeKVCacheDecodeOp(config.gpt_init_params),
            attn_inputs,
        )

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.XQA

    def support_cuda_graph(self) -> bool:
        return True


class XQAWrapper:
    def __init__(
        self,
        config: GptInitModelParameters,
        attn_inputs: PyAttentionInputs,
    ):
        self.config = config
        self.attn_inputs = attn_inputs
        self.enable_xqa = config.py_env_configs.fmha_config.enable_xqa
        if not self.enable_xqa:
            raise ValueError("XQA is not enabled")

        # TODO: refactor support xqa static method and fmha type lately.
        assert (
            self.enable_xqa and not self.attn_inputs.is_prefill
        ), "XQA is not supported"
        group_size = self.config.head_num // self.config.head_num_kv
        input_type_supported = self.config.data_type in ["bf16", "fp16"]
        output_type_supported = self.config.data_type in ["bf16", "fp16", "fp8_e4m3"]
        kv_cache_type_supported = self.config.kv_cache_data_type in [
            "bf16",
            "fp16",
            "fp8",
        ]
        group_size_supported = group_size <= 16
        head_dim_supported = self.config.size_per_head in [64, 128, 256]
        page_size_supported = self.config.seq_size_per_block in [16, 32, 64, 128]
        assert (
            input_type_supported
            and output_type_supported
            and kv_cache_type_supported
            and group_size_supported
            and head_dim_supported
            and page_size_supported
        ), "XQA is not supported"

        # init workspace_buffer and semaphores
        self.workspace_buffer = torch.zeros(
            248 * 1024 * 1024, dtype=torch.uint8, device="cuda"
        )
        self.semaphores = torch.zeros(8 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    @staticmethod
    def support(attn_inputs: PyAttentionInputs) -> bool:
        try:
            return True
        except (ImportError, AttributeError):
            return False

    def prepare(self, attn_inputs: PyAttentionInputs):

        class XQAParams:
            pass

        params = XQAParams()
        params.page_table = attn_inputs.kv_cache_block_id_device
        params.seq_lens = attn_inputs.sequence_lengths
        params.batch_size = attn_inputs.sequence_lengths.size(0)
        params.max_seq_len = (
            attn_inputs.sequence_lengths.max().item() + 1
            if attn_inputs.sequence_lengths.numel() > 0
            else 0
        )  # for rope cache
        return params

    def init_spec_mask(self, q_4d: torch.Tensor):
        # init spec mask
        q_len_per_req = q_4d.shape[1]
        batch_size = q_4d.shape[0]
        if q_len_per_req > 1:
            num_packed_masks_per_token = (q_len_per_req + 31) // 32
            q_indices = torch.arange(
                q_len_per_req, device=q_4d.device, dtype=torch.int32
            ).unsqueeze(1)
            kv_indices = torch.arange(
                q_len_per_req, device=q_4d.device, dtype=torch.int32
            ).unsqueeze(0)
            causal_bool_mask = kv_indices <= q_indices

            padded_seq_len = num_packed_masks_per_token * 32
            if padded_seq_len > q_len_per_req:
                padding = torch.zeros(
                    q_len_per_req,
                    padded_seq_len - q_len_per_req,
                    device=q_4d.device,
                    dtype=torch.bool,
                )
                causal_bool_mask = torch.cat([causal_bool_mask, padding], dim=1)

            causal_bool_mask = causal_bool_mask.view(
                q_len_per_req, num_packed_masks_per_token, 32
            )
            bit_positions = torch.tensor(
                [1 << i for i in range(32)], device=q_4d.device, dtype=torch.int64
            )
            mask_uint32 = (
                (causal_bool_mask.to(torch.int64) * bit_positions)
                .sum(dim=-1)
                .to(torch.uint32)
            )
            mask_uint32 = (
                mask_uint32.unsqueeze(0)
                .expand(batch_size, q_len_per_req, num_packed_masks_per_token)
                .contiguous()
            )
            mask = mask_uint32.view(torch.uint16)
            return mask
        else:
            return None

    def forward(
        self,
        q: torch.Tensor,
        kv_cache,
        fmha_params,
    ) -> torch.Tensor:
        # [num_pages, num_kv_heads, page_size, head_dim] - HND 布局
        k_cache = kv_cache.k_cache_base[:, 0, ...]
        v_cache = kv_cache.k_cache_base[:, 1, ...]
        page_table = fmha_params.page_table
        seq_lens = fmha_params.seq_lens  # cpu device
        num_kv_heads = k_cache.shape[1]
        page_size = k_cache.shape[2]
        kv_layout = "HND"

        q_len_per_req = 1
        if q.dim() == 3:
            # [batch_size, num_heads, head_dim]
            q_4d = q.unsqueeze(1)  # [batch_size, 1, num_heads, head_dim]
        elif q.dim() == 4:
            # [batch_size, seq_len, num_heads, head_dim] - speculative decoding
            q_len_per_req = q.shape[1]
            q_4d = q
        else:
            raise ValueError(f"Unexpected q dimension: {q.dim()}, expected 3 or 4")

        if seq_lens.dim() == 1:
            new_seq_lens = seq_lens + 1
            seq_lens_4d = (
                new_seq_lens.unsqueeze(1).to(torch.uint32).to(q.device)
            )  # [batch_size] -> [batch_size, 1]
        else:
            new_seq_lens = seq_lens[:, 0] + 1
            seq_lens_4d = new_seq_lens.to(torch.uint32).to(q.device)

        output = torch.zeros_like(q_4d)
        enable_pdl = False
        try:
            compute_capability = torch.cuda.get_device_capability(q.device)
            enable_pdl = compute_capability[0] >= 9  # SM90+
        except Exception as e:
            logging.warning(f"[XQA] 无法获取 GPU 计算能力，禁用 PDL 优化: {e}")
            enable_pdl = False
        spec_mask = self.init_spec_mask(q_4d)

        # when nb_sub_seq_per_seq is -1, xqa will use the best config for the current gpu.
        # https://code.alibaba-inc.com/foundation_models/flashinfer/blob/main/best_config/NVIDIA_L20X_XQA_inbf16_cachefp8_outbf16_ps64_hd128_nq12_nkv1.json
        xqa(
            q_4d,
            k_cache,
            v_cache,
            page_table,
            seq_lens_4d,
            output,
            workspace_buffer=self.workspace_buffer,
            semaphores=self.semaphores,
            num_kv_heads=num_kv_heads,
            page_size=page_size,
            kv_layout=kv_layout,
            enable_pdl=enable_pdl,
            q_seq_len=q_len_per_req,
            mask=spec_mask,
            nb_sub_seq_per_seq=1,
            use_qgmma=True,
            sinks=None,
            q_scale=1.0,
            kv_scale=1.0,
            rcp_out_scale=1,
        )
        return output
