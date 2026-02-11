import logging
import os
from typing import Any, Dict, List, Optional

import torch

# import flashinfer
import triton
import triton.language as tl

# NVTX for profiling
try:
    if hasattr(torch.cuda, "nvtx"):
        nvtx = torch.cuda.nvtx
    else:
        raise ImportError("torch.cuda.nvtx not available")
except (ImportError, AttributeError):
    # Fallback if nvtx is not available
    class nvtx:
        @staticmethod
        def range_push(msg: str) -> None:
            pass

        @staticmethod
        def range_pop() -> None:
            pass


from flashinfer import (
    BatchMLAPagedAttentionWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)
from flashinfer.jit import gen_batch_mla_module, gen_batch_prefill_module
from flashinfer.utils import is_sm90a_supported

from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory
from rtp_llm.models_py.utils.arch import is_cuda
from rtp_llm.ops import AttentionConfigs
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs, rtp_llm_ops
from rtp_llm.utils.model_weight import W

g_workspace_buffer = None
warm_up_done = False


def warmup_flashinfer_python():
    global warm_up_done
    if warm_up_done:
        return
    warm_up_done = True
    modules = []
    for backend in ["fa2", "fa3"]:
        if backend == "fa3" and not is_sm90a_supported(torch.device("cuda")):
            continue
        modules.append(
            gen_batch_prefill_module(
                backend,
                torch.bfloat16,
                torch.bfloat16,
                torch.bfloat16,
                torch.int32,
                192,
                128,
                0,
                False,
                False,
                False,
            )
        )

    for backend in ["fa2", "fa3"]:
        if backend == "fa3" and not is_sm90a_supported(torch.device("cuda")):
            continue
        modules.append(
            gen_batch_mla_module(
                backend,
                torch.bfloat16,
                torch.bfloat16,
                torch.bfloat16,
                torch.int32,
                512,
                64,
                False,
            )
        )


def check_attention_inputs(attention_inputs: PyAttentionInputs) -> None:
    device = attention_inputs.input_lengths.device
    dtype = torch.int32

    default_tensors = {
        "prefix_lengths": torch.empty(0, dtype=dtype, device=device),
        "sequence_lengths": torch.empty(0, dtype=dtype, device=device),
        "kv_cache_block_id_host": torch.empty(0, dtype=dtype, device=device),
    }

    for attr_name, default_tensor in default_tensors.items():
        if getattr(attention_inputs, attr_name) is None:
            setattr(attention_inputs, attr_name, default_tensor)


# adapted from sglang/python/sglang/srt/layers/attention/utils.py
@triton.jit
def concat_and_cast_mha_k_kernel(
    k_ptr,
    k_nope_ptr,
    k_rope_ptr,
    head_cnt: tl.constexpr,
    k_stride0: tl.constexpr,
    k_stride1: tl.constexpr,
    nope_stride0: tl.constexpr,
    nope_stride1: tl.constexpr,
    rope_stride0: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    head_range = tl.arange(0, head_cnt)

    k_head_ptr = k_ptr + pid_loc * k_stride0 + head_range[:, None] * k_stride1

    nope_offs = tl.arange(0, nope_dim)

    src_nope_ptr = (
        k_nope_ptr
        + pid_loc * nope_stride0
        + head_range[:, None] * nope_stride1
        + nope_offs[None, :]
    )
    dst_nope_ptr = k_head_ptr + nope_offs[None, :]

    src_nope = tl.load(src_nope_ptr)
    tl.store(dst_nope_ptr, src_nope)

    rope_offs = tl.arange(0, rope_dim)
    src_rope_ptr = k_rope_ptr + pid_loc * rope_stride0 + rope_offs[None, :]
    dst_rope_ptr = k_head_ptr + nope_dim + rope_offs[None, :]
    src_rope = tl.load(src_rope_ptr)
    tl.store(dst_rope_ptr, src_rope)


def concat_and_cast_mha_k_triton(
    k: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
):
    nvtx.range_push("concat_and_cast_mha_k_triton:start")
    # The source data type will be implicitly converted to the target data type.
    assert (
        len(k.shape) == 3 and len(k_nope.shape) == 3 and len(k_rope.shape) == 3
    ), f"shape should be 3d, but got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"
    assert (
        k.shape[0] == k_nope.shape[0] and k.shape[0] == k_rope.shape[0]
    ), f"invalid shape, got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"
    assert (
        k.shape[1] == k_nope.shape[1] and 1 == k_rope.shape[1]
    ), f"invalid shape, got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"
    assert (
        k.shape[-1] == k_nope.shape[-1] + k_rope.shape[-1]
    ), f"invalid shape, got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"

    nvtx.range_push("concat_and_cast_mha_k_triton:prepare_grid")
    nope_dim = k_nope.shape[-1]
    rope_dim = k_rope.shape[-1]
    grid = (k.shape[0],)
    nvtx.range_pop()

    nvtx.range_push("concat_and_cast_mha_k_kernel:launch")
    concat_and_cast_mha_k_kernel[grid](
        k,
        k_nope,
        k_rope,
        k.shape[1],
        k.stride(0),
        k.stride(1),
        k_nope.stride(0),
        k_nope.stride(1),
        k_rope.stride(0),
        nope_dim,
        rope_dim,
    )
    nvtx.range_pop()
    nvtx.range_pop()


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
        quant_config: Optional[object] = None,
    ):
        super().__init__()

        if weights is None:
            raise Exception(f"MlaAbsorbAttention need weights but got none")
        self.quant_config = quant_config
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.scale = (self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5
        self.weights = weights
        self.token_per_block = page_size
        self.softmax_extra_scale = softmax_extra_scale
        self.use_mla = use_mla
        global g_workspace_buffer
        if g_workspace_buffer is None:
            g_workspace_buffer = torch.empty(
                512 * 1024 * 1024,
                dtype=torch.int8,
                device=self.weights[0].get(W.mla_k_nope_w).device,
            )

        self.prefill_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            g_workspace_buffer,
            "NHD",
            backend="auto",
            use_cuda_graph=False,
        )

    def plan(self, mla_params: Any):
        self.prefill_wrapper.plan(
            mla_params.qo_indptr_d,
            mla_params.prefill_ragged_kv_len_indptr_d,
            self.num_heads,
            self.num_heads,
            self.qk_rope_head_dim + self.qk_nope_head_dim,
            self.qk_nope_head_dim,
            sm_scale=(1.0 / (self.qk_rope_head_dim + self.qk_nope_head_dim) ** 0.5)
            * self.softmax_extra_scale,
            causal=True,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        self.reuse_cache_page_indice = mla_params.reuse_cache_page_indice_d
        self.qo_indptr = mla_params.qo_indptr_d
        self.batch_reuse_info_vec = mla_params.batch_reuse_info_vec_d

    def _reuse_kv_cache_indexed_batched(
        self,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """使用融合 CUDA kernel 的优化版本"""

        reuse_cache_page_indice = self.reuse_cache_page_indice
        num_blocks = 0
        if reuse_cache_page_indice is not None:
            num_blocks = reuse_cache_page_indice.size(0)

        if num_blocks == 0:
            return compressed_kv, k_pe

        compressed_kv_dim = compressed_kv.size(1)
        qo_indptr = self.qo_indptr
        batch_reuse_info = self.batch_reuse_info_vec

        # 计算总长度
        total_reuse_len = num_blocks * self.token_per_block
        if total_reuse_len == 0:
            return compressed_kv, k_pe

        total_final_len = compressed_kv.size(0) + total_reuse_len

        # 创建输出 tensor
        final_compressed_kv = torch.empty(
            (total_final_len, compressed_kv_dim),
            dtype=compressed_kv.dtype,
            device=compressed_kv.device,
        )
        final_k_pe = torch.empty(
            (total_final_len, k_pe.size(1)),
            dtype=k_pe.dtype,
            device=k_pe.device,
        )

        # 调用融合 kernel
        k_pe = k_pe.contiguous()
        rtp_llm_ops.reuse_kv_cache_indexed_batched(
            final_compressed_kv,
            final_k_pe,
            compressed_kv,
            k_pe,
            kv_cache.kv_cache_base,
            reuse_cache_page_indice,
            batch_reuse_info,
            qo_indptr,
            self.token_per_block,
        )
        return final_compressed_kv, final_k_pe

    def _concat_and_cast_mha_k(self, k_nope, k_pe):
        nvtx.range_push("_concat_and_cast_mha_k:start")
        # Temporary for DeepSeek V3/R1 only, but can generalize if needed
        nvtx.range_push("_concat_and_cast_mha_k:prepare_shape")
        k_shape = (
            k_nope.shape[0],
            self.num_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
        )
        nvtx.range_pop()

        if (
            is_cuda()
            and (self.num_heads == 128)
            and (self.qk_nope_head_dim == 128)
            and (self.qk_rope_head_dim == 64)
        ):
            nvtx.range_push("_concat_and_cast_mha_k:mla_k_merge_path")
            k = k_nope.new_empty(*k_shape)
            rtp_llm_ops.mla_k_merge(k, k_nope, k_pe)
            nvtx.range_pop()
        elif is_cuda():
            nvtx.range_push("_concat_and_cast_mha_k:triton_path")
            attn_dtype = k_nope.dtype
            k = k_nope.new_empty(*k_shape, dtype=attn_dtype)
            concat_and_cast_mha_k_triton(k, k_nope, k_pe)
            nvtx.range_pop()
        else:
            nvtx.range_push("_concat_and_cast_mha_k:fallback_path")
            k = k_nope.new_empty(*k_shape)
            k[..., : self.qk_nope_head_dim] = k_nope
            k[..., self.qk_nope_head_dim :] = k_pe
            nvtx.range_pop()
        nvtx.range_pop()
        return k

    def _copy_inputs_to_cpu(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_id: int,
    ) -> Dict[str, Any]:
        """Copy inputs to CPU early to avoid copy failure when CUDA error occurs"""
        try:
            inputs_cpu = {
                "q": q.cpu().clone() if q is not None else None,
                "q_shape": list(q.shape) if q is not None else None,
                "compressed_kv": (
                    compressed_kv.cpu().clone() if compressed_kv is not None else None
                ),
                "compressed_kv_shape": (
                    list(compressed_kv.shape) if compressed_kv is not None else None
                ),
                "k_pe": k_pe.cpu().clone() if k_pe is not None else None,
                "k_pe_shape": list(k_pe.shape) if k_pe is not None else None,
                "layer_id": layer_id,
                "num_heads": self.num_heads,
                "qk_nope_head_dim": self.qk_nope_head_dim,
                "qk_rope_head_dim": self.qk_rope_head_dim,
                "kv_lora_rank": self.kv_lora_rank,
                "token_per_block": self.token_per_block,
                "softmax_extra_scale": self.softmax_extra_scale,
            }

            # Add KV cache information if available
            if kv_cache is not None:
                try:
                    inputs_cpu["kv_cache"] = {
                        "kv_cache_base_shape": (
                            list(kv_cache.kv_cache_base.shape)
                            if kv_cache.kv_cache_base is not None
                            else None
                        ),
                        "kv_cache_base_dtype": (
                            str(kv_cache.kv_cache_base.dtype)
                            if kv_cache.kv_cache_base is not None
                            else None
                        ),
                    }
                except Exception:
                    inputs_cpu["kv_cache"] = {
                        "error": "Failed to extract KV cache info"
                    }

            # Add plan-related information if available
            try:
                if hasattr(self, "qo_indptr") and self.qo_indptr is not None:
                    inputs_cpu["qo_indptr"] = (
                        self.qo_indptr.cpu().clone()
                        if isinstance(self.qo_indptr, torch.Tensor)
                        else self.qo_indptr
                    )
                if (
                    hasattr(self, "reuse_cache_page_indice")
                    and self.reuse_cache_page_indice is not None
                ):
                    inputs_cpu["reuse_cache_page_indice"] = (
                        self.reuse_cache_page_indice.cpu().clone()
                        if isinstance(self.reuse_cache_page_indice, torch.Tensor)
                        else self.reuse_cache_page_indice
                    )
                if (
                    hasattr(self, "batch_reuse_info_vec")
                    and self.batch_reuse_info_vec is not None
                ):
                    inputs_cpu["batch_reuse_info_vec"] = (
                        self.batch_reuse_info_vec.cpu().clone()
                        if isinstance(self.batch_reuse_info_vec, torch.Tensor)
                        else self.batch_reuse_info_vec
                    )
            except Exception:
                pass

            return inputs_cpu
        except Exception as e:
            logging.warning(f"[FlashInferMLA Debug] Failed to copy inputs to CPU: {e}")
            return {}

    def _dump_inputs_for_debug_from_cpu(
        self,
        inputs_cpu: Dict[str, Any],
        error_msg: str,
    ) -> None:
        """Dump inputs for debugging CUDA/TVM errors using pre-copied CPU data"""
        import time

        # Log immediately to ensure we see the dump attempt even if it fails
        logging.error(
            "[FlashInferMLA Debug] CUDA/TVM error detected, starting dump process..."
        )
        logging.error(f"[FlashInferMLA Debug] Error message: {error_msg}")

        timestamp = int(time.time())
        dump_dir = os.getenv("RTP_LLM_DEBUG_DUMP_DIR", "./rtp_llm_debug")
        os.makedirs(dump_dir, exist_ok=True)

        dump_file = os.path.join(
            dump_dir, f"flashinfer_mla_prefill_inputs_{timestamp}.pt"
        )

        try:
            # Use pre-copied CPU data directly
            dump_data = {
                "error_msg": error_msg,
                "layer_id": inputs_cpu.get("layer_id"),
                "num_heads": inputs_cpu.get("num_heads"),
                "qk_nope_head_dim": inputs_cpu.get("qk_nope_head_dim"),
                "qk_rope_head_dim": inputs_cpu.get("qk_rope_head_dim"),
                "kv_lora_rank": inputs_cpu.get("kv_lora_rank"),
                "token_per_block": inputs_cpu.get("token_per_block"),
                "softmax_extra_scale": inputs_cpu.get("softmax_extra_scale"),
                "q": inputs_cpu.get("q"),
                "q_shape": inputs_cpu.get("q_shape"),
                "k": inputs_cpu.get("k"),
                "k_shape": (
                    list(inputs_cpu["k"].shape)
                    if inputs_cpu.get("k") is not None
                    else None
                ),
                "value_states": inputs_cpu.get("value_states"),
                "value_states_shape": (
                    list(inputs_cpu["value_states"].shape)
                    if inputs_cpu.get("value_states") is not None
                    else None
                ),
                "compressed_kv": inputs_cpu.get(
                    "compressed_kv_after_reuse", inputs_cpu.get("compressed_kv")
                ),
                "compressed_kv_shape": (
                    list(inputs_cpu["compressed_kv_after_reuse"].shape)
                    if inputs_cpu.get("compressed_kv_after_reuse") is not None
                    else inputs_cpu.get("compressed_kv_shape")
                ),
                "k_pe": inputs_cpu.get("k_pe_after_reuse", inputs_cpu.get("k_pe")),
                "k_pe_shape": (
                    list(inputs_cpu["k_pe_after_reuse"].shape)
                    if inputs_cpu.get("k_pe_after_reuse") is not None
                    else inputs_cpu.get("k_pe_shape")
                ),
                "k_nope": inputs_cpu.get("k_nope"),
                "k_nope_shape": (
                    list(inputs_cpu["k_nope"].shape)
                    if inputs_cpu.get("k_nope") is not None
                    else None
                ),
                "kv_cache": inputs_cpu.get("kv_cache"),
                "qo_indptr": inputs_cpu.get("qo_indptr"),
                "reuse_cache_page_indice": inputs_cpu.get("reuse_cache_page_indice"),
                "batch_reuse_info_vec": inputs_cpu.get("batch_reuse_info_vec"),
            }

            # Save to file
            torch.save(dump_data, dump_file)
            logging.error(
                f"[FlashInferMLA Debug] CUDA/TVM error detected. Inputs dumped to: {dump_file}"
            )
            logging.error(f"[FlashInferMLA Debug] Error message: {error_msg}")
            if inputs_cpu.get("q") is not None:
                logging.error(
                    f"[FlashInferMLA Debug] q shape: {inputs_cpu.get('q_shape')}"
                )
            if inputs_cpu.get("k") is not None:
                logging.error(
                    f"[FlashInferMLA Debug] k shape: {list(inputs_cpu['k'].shape)}"
                )
            if inputs_cpu.get("value_states") is not None:
                logging.error(
                    f"[FlashInferMLA Debug] value_states shape: {list(inputs_cpu['value_states'].shape)}"
                )
            if inputs_cpu.get("compressed_kv_after_reuse") is not None:
                logging.error(
                    f"[FlashInferMLA Debug] compressed_kv shape: {list(inputs_cpu['compressed_kv_after_reuse'].shape)}"
                )
        except Exception as dump_error:
            logging.error(f"[FlashInferMLA Debug] Failed to dump inputs: {dump_error}")

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_id: int,
    ) -> torch.Tensor:
        nvtx.range_push(f"MlaFlashInferPrefillOp.forward:layer_{layer_id}")

        nvtx.range_push("forward:reuse_kv_cache")
        compressed_kv, k_pe = self._reuse_kv_cache_indexed_batched(
            compressed_kv, k_pe, kv_cache
        )
        nvtx.range_pop()

        nvtx.range_push("forward:prepare_k_pe")
        k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)
        nvtx.range_pop()

        nvtx.range_push("forward:create_linear_proj")
        self.k_nope_proj = LinearFactory.create_linear_from_weights(
            self.weights[layer_id],
            W.mla_k_nope_w,
            W.mla_k_nope_s,
            None,
            self.quant_config,
        )

        self.v_proj = LinearFactory.create_linear_from_weights(
            self.weights[layer_id], W.mla_v_w, W.mla_v_s, None, self.quant_config
        )
        nvtx.range_pop()

        nvtx.range_push("forward:k_nope_proj")
        k_nope = self.k_nope_proj(compressed_kv)
        nvtx.range_pop()

        nvtx.range_push("forward:v_proj")
        value_states = self.v_proj(compressed_kv)
        nvtx.range_pop()

        nvtx.range_push("forward:reshape_k_nope_value_states")
        k_nope = k_nope.view(-1, self.num_heads, self.qk_nope_head_dim)
        value_states = value_states.view(-1, self.num_heads, self.qk_nope_head_dim)
        nvtx.range_pop()

        nvtx.range_push("forward:concat_and_cast_mha_k")
        k = self._concat_and_cast_mha_k(k_nope, k_pe)
        nvtx.range_pop()

        # Copy intermediate tensors to CPU before the risky operation
        nvtx.range_push("forward:copy_intermediate_to_cpu")
        # inputs_cpu.update(
        #     {
        #         "k": k.cpu().clone(),
        #         "k_nope": k_nope.cpu().clone(),
        #         "value_states": value_states.cpu().clone(),
        #         "compressed_kv_after_reuse": compressed_kv.cpu().clone(),
        #         "k_pe_after_reuse": k_pe.cpu().clone(),
        #     }
        # )
        nvtx.range_pop()

        # TODO: add TRT prefill support
        nvtx.range_push("forward:prefill_wrapper_run")
        try:
            attn_output = self.prefill_wrapper.run(q, k, value_states)
            nvtx.range_push("forward:reshape_attn_output")
            attn_output = attn_output.view(-1, self.num_heads, self.qk_nope_head_dim)
            nvtx.range_pop()
            nvtx.range_pop()
            nvtx.range_pop()
            return attn_output
        except RuntimeError as e:
            # nvtx.range_pop()
            # error_msg = str(e)
            # self._dump_inputs_for_debug_from_cpu(inputs_cpu, error_msg)
            # nvtx.range_pop()
            raise  # Re-raise the exception after dumping


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
        max_bs: int = 0,
        max_context_len: int = 0,
        num_tokens: int = 0,
        is_cuda_graph: bool = False,
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
        self.use_cuda_graph = is_cuda_graph
        global g_workspace_buffer
        self.kv_indices_d = torch.empty(
            ((max_context_len + self.token_per_block - 1) // self.token_per_block)
            * max_bs,
            dtype=torch.int32,
            device="cuda",
        )
        self.qo_indptr_h = torch.arange(0, max_bs + 1, dtype=torch.int32, device="cpu")
        self.kv_indptr_h = torch.zeros((max_bs + 1,), dtype=torch.int32, device="cpu")
        self.kv_len_arr_h = torch.ones((max_bs,), dtype=torch.int32, device="cpu")

        if g_workspace_buffer is None:
            g_workspace_buffer = torch.empty(
                512 * 1024 * 1024,
                dtype=torch.int8,
                device=self.weights[0].get(W.mla_vc).device,
            )

        self.mla_wrapper = BatchMLAPagedAttentionWrapper(
            g_workspace_buffer,
            backend="auto",
            use_cuda_graph=is_cuda_graph,
            qo_indptr=self.qo_indptr_h,
            kv_indptr=self.kv_indptr_h,
            kv_indices=self.kv_indices_d,
            kv_len_arr=self.kv_len_arr_h,
        )

    def plan(self, fmha_params: Any):
        if self.use_cuda_graph and self.kv_indices_d.size(
            0
        ) < fmha_params.page_indice_d.size(0):
            logging.error(
                f"kv_indices_d.size(0): {self.kv_indices_d.size(0)}, fmha_params.page_indice_d.size(0): {fmha_params.page_indice_d.size(0)}"
            )
            raise ValueError(
                f"kv_indices_d.size(0) < fmha_params.page_indice_d.size(0)"
            )

        self.mla_wrapper.plan(
            fmha_params.qo_indptr_h,
            fmha_params.decode_page_indptr_h,
            fmha_params.page_indice_d,
            fmha_params.kvlen_h,
            self.num_heads,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.token_per_block,
            True,  # causal
            self.scale * self.softmax_extra_scale,
            torch.bfloat16,
            torch.bfloat16,
        )

    def forward(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_id: int,
    ) -> torch.Tensor:

        k_weight = self.weights[layer_id].get(W.mla_kc, None)
        v_weight = self.weights[layer_id].get(W.mla_vc, None)

        compressed_kv = kv_cache.kv_cache_base

        q_nope = q_nope.view(-1, self.num_heads, self.qk_nope_head_dim)
        q_pe = q_pe.view(-1, self.num_heads, self.qk_rope_head_dim)

        q_nope = torch.bmm(q_nope.transpose(0, 1), k_weight)
        q_nope = q_nope.transpose(0, 1)

        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        attn_output = torch.empty_like(q_nope)
        self.mla_wrapper.run(q_nope, q_pe, compressed_kv, k_pe, attn_output)

        attn_output = attn_output.view(-1, self.num_heads, self.kv_lora_rank)
        attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), v_weight)
        attn_bmm_output = attn_bmm_output.transpose(0, 1)

        return attn_bmm_output
