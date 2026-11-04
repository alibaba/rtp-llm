from __future__ import annotations

import logging
import os
from collections import Counter

import torch


def _flag(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default)
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


def cmp_requested() -> bool:
    enabled = _flag("RTP_LLM_HY4_CMP", "1")
    if _flag("RTP_LLM_HY4_CMP_REQUIRE") and not enabled:
        raise RuntimeError("HY4 CMP REQUIRE needs RTP_LLM_HY4_CMP=1")
    return enabled


def _multi_token_decode(inputs) -> bool:
    return bool(getattr(inputs, "is_target_verify", False)) or bool(
        getattr(inputs, "is_draft_extend", False))


def implementation_capabilities() -> dict:
    return {
        "version": "production-attention-v2",
        "validation": "NOT_RUN",
        "phase": "decode, flattened target_verify/draft_extend (M<=256)",
        "heads": [8, 64],
        "weight_format": "MXFP8 group32",
        "kv_cache": ["BF16", "native fp8_ds_mla 656-byte slots"],
        "fp8_cache_attention": "native SparseMlaFp8Op; BF16 query/output, FP8 cache",
        "fused_kernels": ["qkv_norm_quant", "output_bmm_gate_quant",
                          "indexer_wk_norm_rope_cache", "indexer_q_rope_quant_weighted_scale", "qb_rope_direct_pe"],
        "native_operations": ["absorbed query BMM", "Indexer Q projection", "sink attention", "O projection"],
        "absorbed_query_pdl": False,
        "raw_headgate": "exact-TF32 eligible weights; native high-precision fallback",
        "pdl_edges": ["output_bmm_gate_quant -> native O projection", "Indexer Q tail -> native score"],
        "multistream": ["gate", "Indexer", "Q expansion"],
        "graph_replay_count_source": "GPU trace; Python counters count dispatch only",
    }


class Hy4Cmp:
    def __init__(self, attention):
        self.attention = attention
        self.require = _flag("RTP_LLM_HY4_CMP_REQUIRE")
        self.pdl = _flag("RTP_LLM_HY4_CMP_PDL", "1")
        self.multistream = _flag("RTP_LLM_HY4_CMP_MULTISTREAM", "1")
        self.counts = Counter()
        self._workspaces = {}
        self._streams = {}
        self._events = {}
        self._poisoned = False

    def clone_for_cuda_graph(self):
        clone = type(self)(self.attention)
        clone.require = self.require
        clone.pdl = self.pdl
        clone.multistream = self.multistream
        return clone

    def _unsupported(self, hidden, fmha, cache):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import SparseMlaImpl, SparseMlaFp8Op
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.mxfp8_linear import CudaMxfp8Linear
        from rtp_llm.utils.model_weight import W

        a = self.attention
        if type(fmha) is not SparseMlaImpl:
            return "requires SparseMlaImpl"
        if (hidden.ndim != 2 or hidden.shape[1] != 6144
                or not 1 <= hidden.shape[0] <= 256
                or hidden.dtype != torch.bfloat16 or not hidden.is_cuda or not hidden.is_contiguous()):
            return "requires CUDA BF16 [1..256,6144]"
        if torch.cuda.get_device_capability(hidden.device) != (10, 3):
            return "requires SM103"
        if (a.q_lora_rank, a.kv_lora_rank, a.qk_nope_head_dim,
                a.qk_rope_head_dim, a.v_head_dim) != (2048, 512, 192, 64, 256):
            return "unsupported MLA geometry"
        if not fmha._fuse_qk_rope_cat_cache_mla or fmha._is_neox_style:
            return "QB requires fused non-NeoX cache write"
        if a.num_heads not in (8, 64) or a.gating_type != "elementwise":
            return "requires H8/H64 elementwise gate"
        if a.attn_sink is None or not fmha.is_sparse():
            return "requires sparse sink attention"
        if cache is None:
            return "requires real LayerKVCache"
        storage = cache.kv_cache_base
        if not storage.is_cuda or storage.device != hidden.device:
            return f"MLA cache device={storage.device}, CMP requires {hidden.device}"
        if storage.ndim != 3 or storage.shape[0] < 1 or storage.shape[1] != 64:
            return f"MLA cache shape={tuple(storage.shape)}, CMP requires [blocks>0,64,slot]"
        if not storage.is_contiguous():
            return f"MLA cache stride={storage.stride()}, CMP requires contiguous storage"
        if fmha._kv_cache_type == "fp8_ds_mla":
            if (not isinstance(fmha.fmha_impl, SparseMlaFp8Op)
                    or storage.dtype not in (torch.uint8, torch.float8_e4m3fn)
                    or storage.shape[2] != 656):
                return "requires native FP8 MLA operator and 656-byte cache slots"
        elif (fmha._kv_cache_type != "auto" or isinstance(fmha.fmha_impl, SparseMlaFp8Op)
                or storage.dtype != torch.bfloat16 or storage.shape[2] != 576):
            return "requires native BF16 MLA operator and 576-element cache slots"
        if getattr(fmha, "cp_params", None) is not None:
            return "context parallel unsupported"
        if getattr(fmha, "pinned_mla_groups", {}):
            return "pinned MLA cache unsupported"
        reason = self._request_layout_reason(hidden, fmha)
        if reason is not None:
            return reason
        if not all(isinstance(op, CudaMxfp8Linear) for op in
                   (a.fused_qkv_a_proj, a.q_b_proj, a.o_proj)):
            return "requires native MXFP8 projections"
        if (a.fused_qkv_a_proj.K, a.fused_qkv_a_proj.N,
                a.q_b_proj.K, a.q_b_proj.N, a.o_proj.K, a.o_proj.N) != (
                    6144, 2624, 2048, a.num_heads * 256,
                    a.num_heads * 256, 6144):
            return "unsupported local projection geometry"
        for norm, width in ((a.q_a_layernorm, 2048), (a.kv_a_layernorm, 512)):
            weight = norm.weight.data
            if (weight.dtype != torch.bfloat16 or weight.device != hidden.device
                    or tuple(weight.shape) != (width,) or not weight.is_contiguous()):
                return "requires contiguous BF16 Q/KV norm weights"
        v_weight = fmha.weights[a.layer_idx][W.mla_vc]
        if (v_weight.dtype != torch.bfloat16 or v_weight.device != hidden.device
                or tuple(v_weight.shape) != (a.num_heads, 512, 256)
                or not v_weight.is_contiguous()):
            return "requires contiguous BF16 [H,512,256] absorbed V weight"
        if a.indexer is not None and not a.reuse_topk_indices:
            ix = a.indexer
            if (ix.use_hadamard or ix.is_neox_style or ix.index_n_heads != 32
                    or ix.index_head_dim != 128 or ix.rope_head_dim != 64
                    or ix.blocksize != 64 or ix.scale_fmt != "ue8m0"
                    or ix.k_norm.variance_epsilon != 1e-6):
                return "unsupported fused Indexer geometry/epsilon/scale format"
            if not ix._fuse_logits_head_gate:
                return "requires native high-precision fused head-gate reference"
            if not all(isinstance(op, CudaMxfp8Linear) for op in (ix.wk, ix.wq_b)):
                return "requires MXFP8 Indexer Q/K weights"
            if ix.wk.bias is not None or ix.wq_b.bias is not None:
                return "Indexer projection bias unsupported"
            if (ix.wk.K, ix.wk.N, ix.wq_b.K, ix.wq_b.N) != (6144,128,2048,4096):
                return "unsupported Indexer projection geometry"
            for tensor in (ix.k_norm.weight, ix.k_norm.beta):
                if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != (128,):
                    return "requires BF16 Indexer norm weight/bias"
            metadata = getattr(fmha.fmha_params, "schedule_metadata", None)
            if self.pdl and (not isinstance(metadata, torch.Tensor) or metadata.numel() == 0):
                return "score PDL requires request-prepared metadata"
        return None

    @staticmethod
    def _request_layout_reason(hidden, fmha):
        inputs, params = fmha.attn_inputs, fmha.fmha_params
        rows = hidden.shape[0]
        multi = _multi_token_decode(inputs)
        if bool(getattr(inputs, "is_target_verify", False)) and bool(getattr(inputs, "is_draft_extend", False)):
            return "ambiguous MTP phase"
        total = int(getattr(params, "multi_token_decode_total_tokens", 0))
        if multi:
            if not bool(inputs.is_prefill) or total != rows:
                return "MTP requires native flattened prefill-shaped decode metadata"
            table = getattr(inputs, "kv_cache_kernel_block_id_device", None)
            if not isinstance(table, torch.Tensor) or table.numel() == 0:
                table = getattr(inputs, "kv_cache_block_id_device", None)
            if (not isinstance(table, torch.Tensor) or table.ndim != 2
                    or table.shape[0] < 1 or rows % table.shape[0]
                    or table.device != hidden.device or table.dtype != torch.int32):
                return "MTP requires uniform flattened rows per physical request table"
            if inputs.input_lengths.numel() != table.shape[0] or inputs.prefix_lengths.numel() != table.shape[0]:
                return "MTP request length metadata size mismatch"
            width = rows // table.shape[0]
            if inputs.input_lengths.is_cuda or inputs.prefix_lengths.is_cuda:
                for tensor in (inputs.input_lengths, inputs.prefix_lengths):
                    if (tensor.device != hidden.device or tensor.dtype != torch.int32
                            or tensor.ndim != 1 or not tensor.is_contiguous()):
                        return "MTP CUDA request length metadata layout mismatch"
                # fill_params builds this host prefix sum before graph capture.
                # Validate every request width without reading CUDA lengths.
                indptr = getattr(params, "qo_indptr_h", None)
                if (not isinstance(indptr, torch.Tensor) or indptr.is_cuda
                        or indptr.dtype != torch.int32
                        or indptr.shape != (table.shape[0] + 1,)
                        or indptr.tolist() != list(range(0, rows + 1, width))):
                    return "MTP requires native host proof of uniform query widths"
            elif any(length != width for length in inputs.input_lengths.tolist()):
                return "ragged MTP query widths unsupported by native paged Indexer mapping"
        elif total:
            return "multi-token metadata supplied to single-token decode"
        for owner in (params, fmha.rope_params):
            for name, dtype in (("positions_d", torch.int32), ("slot_mapping", torch.int64)):
                tensor = getattr(owner, name, None)
                if (not isinstance(tensor, torch.Tensor) or tensor.shape != (rows,)
                        or tensor.dtype != dtype or tensor.device != hidden.device
                        or not tensor.is_contiguous()):
                    return "native per-token position/slot metadata layout mismatch"
        lengths = getattr(params, "expanded_seq_lens", None)
        if (not isinstance(lengths, torch.Tensor) or lengths.shape != (rows,)
                or lengths.dtype != torch.int32 or lengths.device != hidden.device):
            return "native per-token causal lengths layout mismatch"
        return None

    def _workspace(self, hidden, fused, fmha):
        from .hy4_cmp_kernels.ops import output_bmm_gate, qkv_norm, qb

        key = (hidden.device, hidden.shape[0],
               bool(getattr(fmha.attn_inputs, "is_target_verify", False)),
               bool(getattr(fmha.attn_inputs, "is_draft_extend", False)),
               fmha.attn_inputs.input_lengths.numel())
        if key not in self._workspaces:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("HY4 CMP bucket must warm up before capture")
            qkv_norm.prepare_extension()
            output_bmm_gate.prepare_extension()
            qb.prepare_extension()
            self.attention.q_b_proj._packed_weight_scale()
            self.attention.o_proj._packed_weight_scale()
            self._workspaces[key] = {
                "qkv": qkv_norm.allocate_outputs(fused, need_q_bf16=True),
                "qb": qb.allocate_outputs(hidden.shape[0], self.attention.num_heads, device=hidden.device),
                "kv_write_query": torch.zeros((hidden.shape[0],1,256), dtype=hidden.dtype, device=hidden.device),
                "output": output_bmm_gate.allocate_outputs(
                    hidden.shape[0], self.attention.num_heads, device=hidden.device),
            }
        if not self._streams:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("HY4 CMP streams must warm up before capture")
            for name in ("gate", "indexer", "key", "head_gate"):
                self._streams[name] = torch.cuda.Stream(device=hidden.device)
                self._events[name] = torch.cuda.Event()
            self._events["input"] = torch.cuda.Event()
            self._events["qnorm"] = torch.cuda.Event()
        return self._workspaces[key]

    def _prepare_indexer(self, hidden, x8, xs, fmha, cache, workspace):
        from .hy4_cmp_kernels.ops import head_gate_precompute, indexer_k_fused, indexer_tail

        ix = self.attention.indexer
        if "indexer_k_plan" in workspace:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Indexer bucket must warm up before capture")
        indexer_tail.prepare_extension()
        indexer_k_fused.prepare_extension()
        ix.wq_b._packed_weight_scale()
        wk_scale = ix.wk._packed_weight_scale()
        key_out = indexer_k_fused.allocate_outputs(hidden.shape[0], device=hidden.device)
        workspace["indexer_k_out"] = key_out
        workspace["indexer_k_plan"] = indexer_k_fused.prepare_plan(
            x8, xs, ix.wk.weight, wk_scale, ix.k_norm.weight, ix.k_norm.beta,
            fmha.fmha_params.positions_d, ix.cos_sin_cache, fmha.fmha_params.slot_mapping,
            ix.indexer_op._kv_cache_blocks(cache), out=key_out)
        query = torch.empty((hidden.shape[0],32,128), dtype=hidden.dtype, device=hidden.device)
        workspace["indexer_q_out"] = indexer_tail.allocate_outputs(query)
        workspace["weighted_gate"] = torch.empty((hidden.shape[0],32,1), dtype=torch.float32, device=hidden.device)
        workspace["head_gate_weight"] = ix.weights_proj.weight.contiguous()
        workspace["head_gate_plan"] = head_gate_precompute.prepare_plan(
            hidden, workspace["head_gate_weight"], raw_dot_policy="exact_tf32")

    def _indexer_key(self, x8, xs, fmha, cache, workspace):
        from .hy4_cmp_kernels.ops import indexer_k_fused

        ix = self.attention.indexer
        for tensor in (fmha.fmha_params.positions_d, fmha.fmha_params.slot_mapping,
                       cache.kv_scale_base):
            tensor.record_stream(torch.cuda.current_stream())
        return indexer_k_fused.run(
            x8, xs, ix.wk.weight, ix.wk._packed_weight_scale(), ix.k_norm.weight, ix.k_norm.beta,
            fmha.fmha_params.positions_d, ix.cos_sin_cache, fmha.fmha_params.slot_mapping,
            ix.indexer_op._kv_cache_blocks(cache), out=workspace["indexer_k_out"],
            plan=workspace["indexer_k_plan"], pdl=self.pdl)

    def _indexer_query_score(self, normalized, fmha, cache, workspace):
        from .hy4_cmp_kernels.ops import indexer_tail
        import deep_gemm

        ix = self.attention.indexer
        fmha.fmha_params.positions_d.record_stream(torch.cuda.current_stream())
        query = ix.wq_b(normalized.q_fp8, input_scales=normalized.q_scale).view(-1,32,128)
        torch.cuda.current_stream().wait_event(self._events["head_gate"])
        torch.cuda.current_stream().wait_event(self._events["key"])
        tail = indexer_tail.indexer_q_tail(
            query, fmha.fmha_params.positions_d, ix.cos_sin_cache,
            out=workspace["indexer_q_out"], raw_gate=workspace["head_gate_plan"].output,
            weighted_gate_out=workspace["weighted_gate"],
            gate_scale=ix.softmax_scale*ix.weights_scale, pdl=self.pdl, release="early")
        if deep_gemm.get_pdl():
            raise RuntimeError("score requires exclusive native PDL scope")
        try:
            deep_gemm.set_pdl(self.pdl)
            return ix._compute_topk(tail.q_fp8, workspace["weighted_gate"], cache,
                                    fmha.fmha_params, fmha.attn_inputs, None)
        finally:
            deep_gemm.set_pdl(False)

    def __call__(self, hidden_states, fmha_impl, kv_cache=None,
                 prev_topk_indices=None, return_topk=False,
                 x_fp8=None, x_scale=None):
        a = self.attention
        if self._poisoned:
            raise RuntimeError("HY4 CMP instance cannot be reused after a failed decode")
        kwargs = dict(hidden_states=hidden_states, fmha_impl=fmha_impl,
                      kv_cache=kv_cache, prev_topk_indices=prev_topk_indices,
                      return_topk=return_topk, x_fp8=x_fp8, x_scale=x_scale)
        if (bool(getattr(fmha_impl.attn_inputs, "is_prefill", False))
                and not _multi_token_decode(fmha_impl.attn_inputs)):
            self.counts["native_prefill"] += 1
            if self.counts["native_prefill"] == 1:
                logging.info("HY4 CMP layer=%s native prefill dispatch", a.layer_idx)
            return a(**kwargs)
        reason = self._unsupported(hidden_states, fmha_impl, kv_cache)
        if reason is not None:
            self.counts["fallback:" + reason] += 1
            if self.require:
                raise RuntimeError(f"HY4 CMP layer {a.layer_idx}: {reason}")
            logging.warning("HY4 CMP layer %s native decode fallback: %s", a.layer_idx, reason)
            return a(**kwargs)
        try:
            return self._run_decode(hidden_states, fmha_impl, kv_cache,
                                    prev_topk_indices, return_topk, x_fp8, x_scale)
        except BaseException:
            self._poisoned = True
            self.counts["failed_decode"] += 1
            raise

    def _run_decode(self, hidden_states, fmha_impl, kv_cache,
                    prev_topk_indices, return_topk, x_fp8, x_scale):
        a = self.attention
        from .hy4_cmp_kernels.ops import output_bmm_gate, qkv_norm
        from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
        from rtp_llm.models_py.kernels.cuda.mxfp8_ops import mxfp8_quant_act_packed
        from rtp_llm.utils.model_weight import W
        import deep_gemm

        if not all(callable(getattr(deep_gemm, name, None))
                   for name in ("get_pdl", "set_pdl")):
            raise RuntimeError("HY4 CMP requires DeepGEMM PDL API")
        if deep_gemm.get_pdl():
            raise RuntimeError("HY4 CMP requires exclusive native DeepGEMM PDL scope")
        if (x_fp8 is None) != (x_scale is None):
            raise ValueError("HY4 CMP requires both prequantized input and scale")
        if x_fp8 is None:
            x_fp8, x_scale = mxfp8_quant_act_packed(hidden_states)
        fused = a.fused_qkv_a_proj(x_fp8, input_scales=x_scale)
        workspace = self._workspace(hidden_states, fused, fmha_impl)
        full_indexer = a.indexer is not None and not a.reuse_topk_indices
        if full_indexer:
            self._prepare_indexer(hidden_states, x_fp8, x_scale, fmha_impl, kv_cache, workspace)
        main = torch.cuda.current_stream(hidden_states.device)
        gate_stream = self._streams["gate"] if self.multistream else main
        indexer_stream = self._streams["indexer"] if self.multistream else main
        self._events["input"].record(main)
        with torch.cuda.stream(gate_stream):
            gate_stream.wait_event(self._events["input"])
            hidden_states.record_stream(gate_stream)
            gate = a.gate_proj(hidden_states)
            self._events["gate"].record()
        if full_indexer:
            key_stream = self._streams["key"] if self.multistream else main
            with torch.cuda.stream(key_stream):
                key_stream.wait_event(self._events["input"])
                x_fp8.record_stream(key_stream)
                x_scale.record_stream(key_stream)
                self._indexer_key(x_fp8, x_scale, fmha_impl, kv_cache, workspace)
                self._events["key"].record()
        normalized = qkv_norm.qkv_norm_quant(
            fused, a.q_a_layernorm.weight.data, a.kv_a_layernorm.weight.data,
            out=workspace["qkv"], q_eps=a.q_a_layernorm.variance_epsilon,
            kv_eps=a.kv_a_layernorm.variance_epsilon)
        self._events["qnorm"].record(main)
        if full_indexer:
            from .hy4_cmp_kernels.ops import head_gate_precompute
            head_stream = self._streams["head_gate"] if self.multistream else main
            with torch.cuda.stream(head_stream):
                head_stream.wait_event(self._events["qnorm"])
                hidden_states.record_stream(head_stream)
                head_gate_precompute.precompute(
                    hidden_states, workspace["head_gate_weight"],
                    out=workspace["head_gate_plan"].output, plan=workspace["head_gate_plan"])
                self._events["head_gate"].record()
        with torch.cuda.stream(indexer_stream):
            indexer_stream.wait_event(self._events["qnorm"])
            hidden_states.record_stream(indexer_stream)
            x_fp8.record_stream(indexer_stream)
            x_scale.record_stream(indexer_stream)
            normalized.q_bf16.record_stream(indexer_stream)
            normalized.q_fp8.record_stream(indexer_stream)
            normalized.q_scale.record_stream(indexer_stream)
            if full_indexer:
                topk = self._indexer_query_score(normalized, fmha_impl, kv_cache, workspace)
            else:
                topk = a._run_sparse_indexer(
                    hidden_states, normalized.q_bf16, None, kv_cache, fmha_impl,
                    x_fp8, x_scale, normalized.q_fp8, normalized.q_scale,
                    prev_topk_indices)
            self._events["indexer"].record()
        from .hy4_cmp_kernels.ops import qb
        if "qb_plan" not in workspace:
            workspace["qb_plan"] = qb.prepare_plan(
                normalized.q_fp8, normalized.q_scale, a.q_b_proj.weight,
                a.q_b_proj._packed_weight_scale(), fmha_impl.rope_params.positions_d,
                fmha_impl._cos_sin_cache, workspace["qb"], heads=a.num_heads)
        qb.qb_rope(normalized.q_fp8, normalized.q_scale, a.q_b_proj.weight,
                   a.q_b_proj._packed_weight_scale(), fmha_impl.rope_params.positions_d,
                   fmha_impl._cos_sin_cache, out=workspace["qb"], plan=workspace["qb_plan"], pdl=self.pdl)
        q = workspace["qb"].q_bf16
        torch.bmm(q[..., :192].transpose(0,1), fmha_impl.weights[a.layer_idx][W.mla_kc],
                  out=workspace["qb"].transformed[..., :512].transpose(0,1))
        main.wait_event(self._events["indexer"])
        if topk is not None:
            topk.record_stream(main)
        raw = fmha_impl.forward(
            q.view(-1, a.num_heads, a.q_head_dim), normalized.kv_bf16,
            fused[:, a.q_lora_rank + a.kv_lora_rank:], kv_cache, a.layer_idx,
            topk, attn_sink=a.attn_sink, return_absorbed_output=True,
            prepared_query=workspace["qb"].transformed,
            kv_write_query=workspace["kv_write_query"])
        main.wait_event(self._events["gate"])
        gate.record_stream(main)
        v_weight = fmha_impl.weights[a.layer_idx][W.mla_vc]
        if "output_plan" not in workspace:
            workspace["output_plan"] = output_bmm_gate.prepare_plan(
                raw, v_weight, gate, out=workspace["output"], tuning="epilogue4")
        quantized = output_bmm_gate.output_bmm_gate(
            raw, v_weight, gate, out=workspace["output"],
            plan=workspace["output_plan"], pdl=self.pdl)
        try:
            deep_gemm.set_pdl(self.pdl)
            output = a.o_proj(quantized.q_fp8, input_scales=quantized.q_scale)
        finally:
            deep_gemm.set_pdl(False)
        if a.parallelism_config.get_attn_tp_size() > 1:
            output = all_reduce(output, group=Group.TP)
        phase = "capture_dispatch" if torch.cuda.is_current_stream_capturing() else "eager_dispatch"
        self.counts[phase] += 1
        if _multi_token_decode(fmha_impl.attn_inputs):
            self.counts["target_verify_dispatch" if fmha_impl.attn_inputs.is_target_verify
                        else "draft_extend_dispatch"] += 1
        if self.counts[phase] == 1:
            logging.info("HY4 CMP actual dispatch layer=%s heads=%s phase=%s pdl=%s kernels=%s",
                         a.layer_idx, a.num_heads, phase, self.pdl,
                         implementation_capabilities()["fused_kernels"])
        return (output, topk) if return_topk else output
