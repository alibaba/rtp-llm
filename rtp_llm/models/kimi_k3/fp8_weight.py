"""K3-only load-time block FP8 policy; native MoE remains outside this manifest."""

import logging
from typing import Dict

import torch

from rtp_llm.model_loader.per_block_fp8_quant_weight import (
    LoadQuantPerBlockFp8Weight,
    per_block_cast_to_fp8,
)
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.utils.model_weight import W, transpose_slice_k, transpose_slice_v


class KimiK3LoadFp8Weight(LoadQuantPerBlockFp8Weight):
    # Deliberately instantiated by the K3 manifest, never selected globally.
    w8a8_weight_list: Dict[str, str] = {
        W.linear_attn_qkvg_fa_beta_w: W.linear_attn_qkvg_fa_beta_s,
        W.linear_attn_f_b_w: W.linear_attn_f_b_s,
        W.linear_attn_out_w: W.linear_attn_out_s,
        W.mla_fusedqkrope_w: W.mla_fusedqkrope_s,
        W.mla_q_b_w: W.mla_q_b_s,
        W.mla_kv_b_w: W.mla_kv_b_s,
        W.attn_gate_w: W.attn_gate_s,
        W.attn_o_w: W.attn_o_s,
    }

    @classmethod
    def support(cls, quant_config, src_weight_info):
        return False

    def __init__(self, src_weight_info, quant_config, *, derive_mla=False, layer_id=-1):
        if src_weight_info.name not in self.w8a8_weight_list:
            raise ValueError(
                f"not a K3 FP8 attention projection: {src_weight_info.name}"
            )
        super().__init__(src_weight_info, quant_config, name=src_weight_info.name)
        self.layer_id = layer_id
        self.tp_ranges = None
        self.source = src_weight_info
        self.derive_mla = derive_mla
        if derive_mla:
            for name in (W.mla_kc, W.mla_vc):
                self.sub_weights[name] = AtomicWeight(name, [])

    def _load_raw_tensor(self, tensor_source, layer_id, device, load_config):
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
            is_deep_gemm_e8m0_used,
        )

        if load_config.merge_lora:
            raise ValueError("K3 online FP8 does not support load-time LoRA merging")
        raw = self.source._load_raw_tensor(tensor_source, layer_id, device, load_config)
        # K3 dense source weights are [in,out]; quantized storage is [out,in].
        matrix = raw[self.source.name].T.contiguous()
        self.use_ue8m0 = is_deep_gemm_e8m0_used()
        weight, scale = self._quantize_matrix(matrix, use_ue8m0=self.use_ue8m0)
        return {self.kernel.name: weight, self.scale.name: scale}

    @staticmethod
    def _quantize_matrix(matrix, *, use_ue8m0):
        # Every 128x128 block is independent. Bound the existing quantizer's
        # FP32 scratch instead of materializing several TP-global FP32 copies.
        # Each weight element is still quantized exactly once with its final scale.
        rows, columns = matrix.shape
        weight = torch.empty_like(matrix, dtype=torch.float8_e4m3fn)
        scale = torch.empty(
            ((rows + 127) // 128, (columns + 127) // 128),
            dtype=torch.float32,
            device=matrix.device,
        )
        for begin in range(0, rows, 1024):
            end = min(begin + 1024, rows)
            block_weight, block_scale = per_block_cast_to_fp8(
                matrix[begin:end], 128, use_ue8m0=use_ue8m0
            )
            weight[begin:end].copy_(block_weight)
            scale[begin // 128 : (end + 127) // 128].copy_(block_scale)
            del block_weight, block_scale
        return weight, scale

    def _split(self, tensor, load_config):
        weight, scale = tensor[self.kernel.name], tensor[self.scale.name]
        tp, rank = load_config.tp_size, load_config.tp_rank
        if not 0 <= rank < tp:
            raise ValueError(f"invalid K3 FP8 TP rank: {rank}/{tp}")
        if tp not in (1, 2, 4, 8, 16):
            raise ValueError(f"K3 FP8 supports TP1/2/4/8/16, got {tp}")
        name = self.kernel.name
        if name == W.linear_attn_qkvg_fa_beta_w:
            cfg = self.source.config
            widths = [cfg.linear_num_key_heads * cfg.linear_key_head_dim] * 2
            widths += [cfg.linear_num_value_heads * cfg.linear_value_head_dim] * 2
            rows, blocks, offset = [], [], 0
            self.tp_ranges = {"axis": 0, "sharded": [], "replicated": []}
            for width in widths:
                local = width // tp
                if width % tp or local % 128:
                    raise ValueError("KDA head shards must align to FP8 blocks")
                begin = offset + rank * local
                self.tp_ranges["sharded"].append((begin, begin + local))
                rows.append(weight[begin : begin + local])
                blocks.append(scale[begin // 128 : (begin + local) // 128])
                offset += width
            self.tp_ranges["replicated"].append((offset, weight.shape[0]))
            rows.append(weight[offset:])
            blocks.append(scale[offset // 128 :])
            weight, scale = torch.cat(rows), torch.cat(blocks)
        elif name != W.mla_fusedqkrope_w:
            axis = 1 if name in (W.attn_o_w, W.linear_attn_out_w) else 0
            width = weight.shape[axis]
            if width % tp or (width // tp) % 128:
                raise ValueError(f"unaligned K3 FP8 shard: {name}, {width=}, {tp=}")
            local = width // tp
            self.tp_ranges = {
                "axis": axis,
                "sharded": [(rank * local, (rank + 1) * local)],
            }
            weight = weight.narrow(axis, rank * local, local)
            scale = scale.narrow(axis, rank * local // 128, local // 128)
        else:
            self.tp_ranges = {"axis": 0, "replicated": [(0, weight.shape[0])]}

        # Row slices can be contiguous views of a full TP-global allocation.
        # Tail cropping can likewise retain the quantizer's temporary padding.
        # Own exactly the local logical storage before releasing raw_tensors.
        def compact(value):
            value = value.contiguous()
            if (
                value.storage_offset() != 0
                or value.untyped_storage().nbytes()
                != value.numel() * value.element_size()
            ):
                value = value.clone(memory_format=torch.contiguous_format)
            return value

        return {name: compact(weight), self.scale.name: compact(scale)}

    def _postprocess(self, tensor, device, load_config):
        from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
            _transform_scale_ue8m0,
        )

        weight, scale = tensor[self.kernel.name], tensor[self.scale.name]
        result = {}
        if self.derive_mla:
            cfg = self.source.config
            dense = (
                weight.float()
                * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[
                    : weight.shape[0], : weight.shape[1]
                ]
            )
            dense = dense.to(torch.bfloat16)
            heads = cfg.head_num // load_config.tp_size
            args = (heads, cfg.nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank)
            result[W.mla_kc] = transpose_slice_k([dense], *args).contiguous()
            result[W.mla_vc] = transpose_slice_v([dense], *args).contiguous()
        if self.use_ue8m0:
            scale = _transform_scale_ue8m0(scale, mn=weight.shape[0])
        else:
            # Match the existing Linear's non-UE8M0 storage contract.
            weight = weight.reshape(weight.shape[1], weight.shape[0])
            scale = scale.reshape(scale.shape[1], scale.shape[0])
        result[self.kernel.name] = weight
        result[self.scale.name] = scale
        logging.info(
            "K3_FP8_WEIGHT layer=%d name=%s logical_shape=%s scale_shape=%s "
            "scale_dtype=%s TP=%d rank=%d derived_mla=%s weight_padding=0 tp_ranges=%s",
            self.layer_id,
            self.kernel.name,
            tuple(tensor[self.kernel.name].shape),
            tuple(scale.shape),
            scale.dtype,
            load_config.tp_size,
            load_config.tp_rank,
            self.derive_mla,
            self.tp_ranges,
        )
        return result
