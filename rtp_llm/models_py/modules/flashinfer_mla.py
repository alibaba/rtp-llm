from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import flashinfer
import torch
import torch.nn.functional as F

# from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.ops import KVCache, PyAttentionInputs
from rtp_llm.utils.model_weight import W


def fill_flash_params(page_size: int, attention_inputs: PyAttentionInputs, device):
    prefix_lengths = None
    kv_cache_block_id = None

    bs = attention_inputs.input_lengths.size(0)
    if attention_inputs.kv_cache_block_id_host.size(0) > 0:
        kv_cache_block_id = attention_inputs.kv_cache_block_id_host.cpu().tolist()

    if attention_inputs.prefix_lengths.size(0) > 0:
        prefix_lengths = attention_inputs.prefix_lengths.cpu().tolist()

    input_lengths = attention_inputs.input_lengths.cpu().tolist()
    sequence_lengths = attention_inputs.sequence_lengths.cpu().tolist()

    if len(sequence_lengths) == 0:
        sequence_lengths.append(0)

    # offset = 0
    max_q_len = 1
    max_kv_len = 0
    accu_q_len = 0
    total_page_idx = 0

    batch_indice: list = []
    positions: list = []
    paged_kv_last_page_len: list = []
    kvlen: list = []
    page_indice: list = []
    page_indptr: list = [0]
    qo_indptr: list = [0]

    for i in range(bs):
        seq_len = 0
        if prefix_lengths:
            input_length = input_lengths[i]
            prefix_length = prefix_lengths[i]
            for j in range(input_length):
                batch_indice.append(i)
                positions.append(j + prefix_length)
                # offset += 1
            seq_len = input_length + prefix_length
            max_q_len = max(max_q_len, input_length)
            accu_q_len += input_length
        else:
            batch_indice.append(i)
            positions.append(sequence_lengths[i])
            seq_len = sequence_lengths[i] + 1
            accu_q_len += 1
        paged_kv_last_page_len.append((seq_len - 1) % page_size + 1)
        kvlen.append(seq_len)
        max_kv_len = max(max_kv_len, seq_len)

        page_num = (seq_len + page_size - 1) // page_size

        if kv_cache_block_id:
            for j in range(page_num):
                page_idx = kv_cache_block_id[i][j]
                page_indice.append(page_idx)
                total_page_idx += 1
        page_indptr.append(total_page_idx)
        qo_indptr.append(accu_q_len)

    params = SimpleNamespace()
    params.batch_indice: torch.Tensor = (
        torch.tensor(batch_indice).to(device).to(torch.int32)
    )
    params.positions: torch.Tensor = torch.tensor(positions).to(device).to(torch.int32)
    params.paged_kv_last_page_len: torch.Tensor = (
        torch.tensor(paged_kv_last_page_len).to(device).to(torch.int32)
    )
    params.kvlen: torch.Tensor = torch.tensor(kvlen).to(device).to(torch.int32)
    params.page_indice: torch.Tensor = (
        torch.tensor(page_indice).to(device).to(torch.int32)
    )
    params.page_indptr: torch.Tensor = (
        torch.tensor(page_indptr).to(device).to(torch.int32)
    )
    params.qo_indptr: torch.Tensor = torch.tensor(qo_indptr).to(device).to(torch.int32)

    return params


class MlaFlashInferPrefillOp(object):
    def __init__(
        self,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        page_size: int,
        softmax_extra_scale: float,
        use_mla: bool,
        weights: List[Dict[str, torch.Tensor]] | None,
    ):
        super().__init__()
        if weights is None:
            raise Exception(f"MlaAbsorbAttention need weights but got none")
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.scale = (self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5
        self.weights = weights
        self.token_per_block = page_size
        self.softmax_extra_scale = softmax_extra_scale
        self.use_mla = use_mla
        g_workspace_buffer = torch.empty(
            512 * 1024 * 1024,
            dtype=torch.int8,
            device=self.weights[0].get(W.mla_k_nope_w).device,
        )
        self.prefill_wrapper_ragged = (
            flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
                g_workspace_buffer, "NHD", backend="fa2"
            )
        )

    def support(self, attention_inputs: PyAttentionInputs):
        return self.use_mla and attention_inputs.is_prefill

    def prepare(self, attention_inputs: PyAttentionInputs):
        return fill_flash_params(
            self.token_per_block,
            attention_inputs,
            self.weights[0].get(W.mla_k_nope_w).device,
        )

    def forward(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        fmha_params: Any,
        layer_id: int,
    ) -> torch.Tensor:

        k_nope_weight = self.weights[layer_id].get(W.mla_k_nope_w, None)
        v_weight = self.weights[layer_id].get(W.mla_v_w, None)

        q_nope = q_nope.view(-1, self.num_heads, self.qk_nope_head_dim)
        q_pe = q_pe.view(-1, self.num_heads, self.qk_rope_head_dim)

        seq_len = q_nope.size(0)
        page_indptr = (
            torch.Tensor([0, seq_len])
            .to(fmha_params.qo_indptr.dtype)
            .to(fmha_params.qo_indptr.device)
        )

        k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)

        k_nope = F.linear(compressed_kv, k_nope_weight.transpose(0, 1), None)
        k_nope = k_nope.view(-1, self.num_heads, self.qk_nope_head_dim)
        value_states = F.linear(compressed_kv, v_weight.transpose(0, 1), None)
        value_states = value_states.view(-1, self.num_heads, self.qk_nope_head_dim)

        q = torch.cat([q_nope, q_pe], dim=-1)
        k = k_pe.new_empty(
            k_pe.size(0), self.num_heads, self.qk_rope_head_dim + self.qk_nope_head_dim
        )
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe

        self.prefill_wrapper_ragged.plan(
            fmha_params.qo_indptr,
            page_indptr,
            self.num_heads,
            self.num_heads,
            self.qk_rope_head_dim + self.qk_nope_head_dim,
            self.qk_nope_head_dim,
            sm_scale=(1.0 / (self.qk_rope_head_dim + self.qk_nope_head_dim) ** 0.5)
            * self.softmax_extra_scale,
            causal=True,
            q_data_type=q.dtype,
            kv_data_type=k.dtype,
        )

        attn_output, _ = self.prefill_wrapper_ragged.run_return_lse(q, k, value_states)
        attn_output = attn_output.view(-1, self.num_heads, self.qk_nope_head_dim)
        return attn_output


class MlaFlashInferDecodeOp(object):
    def __init__(
        self,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        token_per_block: int,
        softmax_extra_scale: float,
        use_mla: bool,
        weights: List[Dict[str, torch.Tensor]] | None = None,
    ):
        super().__init__()
        if weights is None:
            raise Exception(f"MlaAbsorbAttention need weights but got none")
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.scale = (self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5
        self.token_per_block = token_per_block
        self.softmax_extra_scale = softmax_extra_scale
        self.weights = weights
        self.use_mla = use_mla
        g_workspace_buffer = torch.empty(
            512 * 1024 * 1024,
            dtype=torch.int8,
            device=self.weights[0].get(W.mla_vc).device,
        )
        self.mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
            g_workspace_buffer, backend="fa2"
        )

    def support(self, attention_inputs: PyAttentionInputs):
        return self.use_mla and not attention_inputs.is_prefill

    def prepare(self, attention_inputs: PyAttentionInputs):
        return fill_flash_params(
            self.token_per_block, attention_inputs, self.weights[0].get(W.mla_vc).device
        )

    def forward(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
        fmha_params: Any,
        layer_id: int,
    ) -> torch.Tensor:

        k_weight = self.weights[layer_id].get(W.mla_kc, None)
        v_weight = self.weights[layer_id].get(W.mla_vc, None)

        compressed_kv = kv_cache.k_cache_base

        self.mla_wrapper.plan(
            fmha_params.qo_indptr,
            fmha_params.page_indptr,
            fmha_params.page_indice,
            fmha_params.kvlen,
            self.num_heads,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.token_per_block,
            False,  # causal
            self.scale,
            q_nope.dtype,
            compressed_kv.dtype,
        )

        q_nope = q_nope.view(-1, self.num_heads, self.qk_nope_head_dim)
        q_pe = q_pe.view(-1, self.num_heads, self.qk_rope_head_dim)

        q_nope = torch.bmm(q_nope.transpose(0, 1), k_weight)
        q_nope = q_nope.transpose(0, 1)
        fuse_q = torch.cat((q_nope, q_pe), dim=-1)
        q_nope, q_pe = torch.split(
            fuse_q, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        profiler_args = ()

        num_heads = q_nope.shape[1]
        page_size = self.mla_wrapper._page_size
        sm_scale = self.mla_wrapper._sm_scale * self.softmax_extra_scale

        attn_output = torch.empty_like(q_nope)
        self.mla_wrapper._cached_module.run.default(
            self.mla_wrapper._float_workspace_buffer,
            self.mla_wrapper._int_workspace_buffer,
            self.mla_wrapper._plan_info,
            q_nope,
            q_pe,
            compressed_kv,
            k_pe,
            self.mla_wrapper._kv_indices_buf,
            attn_output,
            None,
            1,
            num_heads,
            page_size,
            sm_scale,
            *profiler_args,
        )

        attn_output = attn_output.view(-1, self.num_heads, self.kv_lora_rank)
        attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), v_weight)
        attn_bmm_output = attn_bmm_output.transpose(0, 1)
        return attn_bmm_output