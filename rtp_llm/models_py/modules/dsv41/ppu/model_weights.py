"""PPU descriptors and views over the framework's actual ModelWeights.

The public V4.1 descriptor builders remain authoritative for names, ordering,
norms and mHC. Only layer splitting/preprocessing is PPU-local: dense FP8 and
UE8M0 bytes stay unpacked until the resident BF16 consumer is bound; routed
experts are already EP-local when MoeAtomicWeight's loader returns them.
Engram host tables deliberately remain outside the generic device loader.
"""

from collections import Counter
import logging

import torch

from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.deepseek_v41 import DeepSeekV41Weight
from rtp_llm.models_py.modules.dsv41.ppu.linear import PPUResidentBF16Linear, PPUPTPCLinear
from rtp_llm.models_py.modules.dsv41.ppu.quant_conversion import audit_loaded_tensor, convert_dense_ptpc
from rtp_llm.models_py.modules.dsv41.ppu.quant_mode import MXFP4_FP8_PTPC, quant_mode
from rtp_llm.models_py.modules.dsv41.ppu.weight import V41Geometry
from rtp_llm.utils.model_weight import CkptWeightInfo, W


class _StorageTensorSource(TensorSource):
    """Preserve byte storage when public raw loaders request a quant dtype."""

    def __init__(self, source):
        self.source = source

    def load_tensor(self, name, data_type=None):
        tensors = self.source.load_tensor(name, None)
        if data_type in (
            torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.int8, torch.uint8
        ):
            if any(t.element_size() != 1 for t in tensors):
                raise TypeError(f"{name} must retain byte-sized checkpoint storage")
            return [t.view(data_type) for t in tensors]
        return [t.to(data_type) for t in tensors]

    def has_tensor(self, name):
        return self.source.has_tensor(name)

    def get_database(self):
        return self.source.get_database()


class PPUSharedAtomicWeight(AtomicWeight):
    """Reuse AtomicWeight.load/raw merge, without CUDA TP/layout transforms."""

    @classmethod
    def create(cls, weight_info, quant_config=None):
        # Called by the existing ModelWeightInfo.to_quant_weight_info. This
        # descriptor already declares both storage dtypes; no CUDA quant clone.
        if not isinstance(weight_info, cls):
            raise TypeError("PPU descriptor create requires its own descriptor")
        return weight_info

    def _split(self, tensor, load_config):
        if load_config.tp_size != 1:
            raise ValueError("shared PPU layer weights require Attention TP1")
        return tensor

    def _load_raw_tensor(self, tensor_source, layer_id, device, load_config):
        return super()._load_raw_tensor(
            _StorageTensorSource(tensor_source), layer_id, device, load_config
        )

    def _postprocess(self, tensor, device, load_config):
        return tensor


class PPUSharedMoeWeight(MoeAtomicWeight):
    """Public expert selection/stacking, with no second EP/TP split or shuffle."""

    @classmethod
    def create(cls, weight_info, quant_config=None):
        if not isinstance(weight_info, cls):
            raise TypeError("PPU expert descriptor create requires its own descriptor")
        return weight_info

    def _split(self, tensor, load_config):
        if load_config.tp_size != 1 or load_config.ffn_tp_size != 1:
            raise ValueError("shared PPU experts require Attention/FFN TP1")
        return tensor

    def _load_raw_tensor(self, tensor_source, layer_id, device, load_config):
        result = super()._load_raw_tensor(
            _StorageTensorSource(tensor_source), layer_id, device, load_config
        )
        # Report each selected logical expert after the production stack/split.
        # The optional receipt is independent of native backend packing.
        import os

        if os.environ.get("RTP_V41_PPU_WEIGHT_AUDIT_DIR"):
            if len(self.weights) != 1:
                raise ValueError("expert audit expects native separate w1/w2/w3 descriptors")
            selected = load_config.get_selected_experts(layer_id, self.config.expert_num)
            value = result[self.name]
            if value.shape[0] != len(selected):
                raise ValueError("expert audit selection differs from actual stacked buffer")
            for local, expert in enumerate(selected):
                name = self.weights[0].name.format(i=layer_id, i_1=layer_id + 1, expert_id=expert)
                audit_loaded_tensor(name, value[local], load_config.ep_rank)
        return result

    def _postprocess(self, tensor, device, load_config):
        return tensor


class PPUPTPCWeight(PPUSharedAtomicWeight):
    """One real loader operation produces both payload and channel scales."""

    def __init__(self, original, scale_name):
        super().__init__(
            original.name, original.weights, original.process_fun,
            data_type=torch.float8_e4m3fn,
        )
        self.scale_name = scale_name

    def get_ckpt_tensor_names(self):
        names = super().get_ckpt_tensor_names()
        return names + [n.removesuffix(".weight") + ".scale" for n in names]

    def get_tensor_names(self, layer_id, load_config):
        names = super().get_tensor_names(layer_id, load_config)
        return names | {n.removesuffix(".weight") + ".scale" for n in names}

    def _load_raw_tensor(self, tensor_source, layer_id, device, load_config):
        values, scales = [], []
        for item in self.weights:
            name = item.tensor_name(layer_id)
            weight = item.merge_fun(tensor_source.load_tensor(name, None))
            scale = item.merge_fun(tensor_source.load_tensor(
                name.removesuffix(".weight") + ".scale", None
            ))
            value, scale = convert_dense_ptpc(weight, scale)
            observer = getattr(tensor_source, "observe_conversion", None)
            if observer is not None:
                observer(name, value, scale)
            # Merge bytes so concatenation also works on CPU torch versions
            # without a float8 cat implementation. Scale rows use the same order.
            values.append(value.view(torch.uint8).to(device))
            scales.append(scale.to(device))
            audit_loaded_tensor(name, values[-1].view(torch.float8_e4m3fn), getattr(load_config, "ep_rank", 0))
            audit_loaded_tensor(name.removesuffix(".weight") + ".scale", scales[-1], getattr(load_config, "ep_rank", 0))
        return {
            self.name: self.process_fun(values).view(torch.float8_e4m3fn),
            self.scale_name: self.process_fun(scales),
        }

    def _merge_lora(self, *args, **kwargs):
        raise NotImplementedError("Flash1017 PTPC does not support merged LoRA")


# Existing W keys, not a second weight namespace. The checkpoint stores two
# shared w13 constituents; their descriptor concatenates [gate=w1, up=w3]
# and their scale rows in precisely that same order, once.
_DENSE_PAIRS = {
    W.v4_attn_wq_a_w: W.v4_attn_wq_a_s,
    W.v4_attn_wq_b_w: W.v4_attn_wq_b_s,
    W.v4_attn_wkv_w: W.v4_attn_wkv_s,
    W.v4_attn_wo_a_w: W.v4_attn_wo_a_s,
    W.v4_attn_wo_b_w: W.v4_attn_wo_b_s,
    W.v4_indexer_wq_b_w: W.v4_indexer_wq_b_s,
    W.v4_shared_w13_w: W.v4_shared_w13_s,
    W.v4_shared_w2_w: W.v4_shared_w2_s,
    W.v4_dspark_main_proj_w: W.v4_dspark_main_proj_s,
}


def _ppu_descriptors(weights, mode=None):
    mode = quant_mode() if mode is None else mode
    result = []
    for original in weights:
        if isinstance(original, MoeAtomicWeight):
            result.append(
                PPUSharedMoeWeight(
                    original.name,
                    original.weights,
                    original.process_fun,
                    data_type=(
                        torch.uint8 if mode == MXFP4_FP8_PTPC or original.name.endswith(".scale")
                        else original.data_type
                    ),
                    config=original.config,
                )
            )
            continue
        dtype = original.data_type
        scale_key = _DENSE_PAIRS.get(original.name)
        # weights_proj is stored BF16, not an FP8 projection.
        if original.name == W.v4_indexer_weights_proj_w:
            scale_key = None
        if scale_key is not None and mode == MXFP4_FP8_PTPC:
            result.append(PPUPTPCWeight(original, scale_key))
            continue
        if scale_key is not None:
            dtype = torch.float8_e4m3fn
        result.append(
            PPUSharedAtomicWeight(
                original.name, original.weights, original.process_fun, data_type=dtype
            )
        )
        if scale_key is not None:
            scales = [
                CkptWeightInfo(
                    item.name.removesuffix(".weight") + ".scale", item.merge_fun
                )
                for item in original.weights
            ]
            result.append(
                PPUSharedAtomicWeight(
                    scale_key,
                    scales,
                    original.process_fun,
                    data_type=torch.uint8,
                )
            )
    return result


class DeepSeekV41PPUWeight(DeepSeekV41Weight):
    """Keep the public per-layer graph, including all native packed experts."""

    @classmethod
    def framework_base_available(cls):
        return True

    def __init__(self, *args, **kwargs):
        self.ppu_quant_mode = quant_mode()
        super().__init__(*args, **kwargs)

    def _get_hf_layer_weight_info(self, layer_id):
        return _ppu_descriptors(super()._get_hf_layer_weight_info(layer_id), self.ppu_quant_mode)


class SharedWeightViews:
    """Single-owner conversions; tensors remain keyed by W in ModelWeights.

    Repeated consumers of the same key receive the identical resident object.
    No checkpoint path or whole-model loader is involved in these views.
    """

    def __init__(self, weights, config, *, ep_rank=0, ep_size=1, num_physical_experts=None):
        if not isinstance(weights, ModelWeights):
            raise TypeError("shared PPU requires framework ModelWeights")
        self.weights = weights
        self.quant_mode = quant_mode()
        logging.info("V4.1 PPU loaded weight view quant_mode=%s ep_rank=%s", self.quant_mode, ep_rank)
        self.config = dict(config)
        self.config["_ppu_quant_mode"] = self.quant_mode
        if self.config.get("engram_layer_ids") and "engram_token_map" not in self.config:
            from pathlib import Path

            from rtp_llm.models_py.modules.dsv41.ppu.engram import load_token_map

            self.config["engram_token_map"] = load_token_map(
                str(Path(self.config["checkpoint_path"]) / "tokenizer.json")
            )
        self.geometry = V41Geometry.from_config(self.config)
        self.ep_rank, self.ep_size = int(ep_rank), int(ep_size)
        from rtp_llm.models_py.modules.dsv41.ppu.eplb import physical_expert_geometry

        logical = self.geometry.n_routed_experts
        self.physical_experts, self.local_experts = physical_expert_geometry(
            logical, logical if num_physical_experts is None else num_physical_experts,
            self.ep_size,
        )
        if not 0 <= self.ep_rank < self.ep_size:
            raise ValueError("invalid PPU EP rank")
        if len(weights.weights) != self.geometry.num_layers:
            raise ValueError("ModelWeights layer count differs from checkpoint config")
        self._linears = {}
        self._resident = {}
        self._routed = {}
        self.conversions = Counter()

    def eplb_maps(self, layer_id):
        row = self.weights.weights[layer_id]
        mapping, counts = row.get(W.log2phy), row.get(W.logic_expert_cnt)
        if (mapping is None) != (counts is None):
            raise ValueError("framework loader must provide both EPLB map tensors")
        if mapping is None and self.physical_experts != self.geometry.n_routed_experts:
            raise ValueError("redundant experts require the framework's loaded slot maps")
        return mapping, counts

    def eplb_raw_tensors(self, layer_id):
        # ModelWeights retains w1/w3 before the once-only w13 concatenation.
        # They remain valid consumers/diagnostic views after a live shuffle.
        return tuple(
            (f"model_weights.{sub}.{part}", self.tensor(layer_id, getattr(W, f"v4_routed_{sub}_{part}")))
            for sub in ("w1", "w3", "w2") for part in ("w", "s")
        )

    def tensor(self, layer_id, key):
        source = self.weights.global_weights if layer_id is None else self.weights.weights[layer_id]
        return source[key]

    def linear(self, layer_id, key, scale_key=None, *, output_dtype=torch.bfloat16):
        scale_key = _DENSE_PAIRS.get(key) if scale_key is None else scale_key
        if key == W.v4_indexer_weights_proj_w:
            scale_key = None
        identity = (layer_id, key, scale_key, output_dtype)
        if self.quant_mode == MXFP4_FP8_PTPC and scale_key is not None:
            if output_dtype != torch.bfloat16:
                raise ValueError("PTPC projections require BF16 GEMM output")
            if identity not in self._linears:
                self._linears[identity] = PPUPTPCLinear(
                    self.tensor(layer_id, key), self.tensor(layer_id, scale_key)
                )
                op = self._linears[identity]
                label = f"dense:{self.geometry.num_layers}:L{layer_id}:{key}"
                audit_loaded_tensor(label + ":weight", op.weight, self.ep_rank, layout=True)
                audit_loaded_tensor(label + ":scale", op.scale, self.ep_rank, layout=True)
                self.conversions[f"L{layer_id}:{key}:ptpc"] += 1
            return self._linears[identity]
        if identity not in self._linears:
            resident_key = (layer_id, key, scale_key)
            if resident_key not in self._resident:
                value = self.tensor(layer_id, key)
                scale = self.tensor(layer_id, scale_key) if scale_key is not None else None
                if scale is not None and value.dtype in (torch.int8, torch.uint8):
                    value = value.view(torch.float8_e4m3fn)
                self._resident[resident_key] = PPUResidentBF16Linear.from_quantized(
                    value, scale, "fp8_32x32_ue8m0" if scale is not None else "bf16"
                ).weight
                self.conversions[f"L{layer_id}:{key}:resident"] += 1
            self._linears[identity] = PPUResidentBF16Linear(
                self._resident[resident_key], output_dtype=output_dtype
            )
        return self._linears[identity]

    def routed(self, layer_id):
        if layer_id not in self._routed:
            g = self.geometry
            values, scales = [], []
            for sub in ("w1", "w3", "w2"):
                value = self.tensor(layer_id, getattr(W, f"v4_routed_{sub}_w"))
                scale = self.tensor(layer_id, getattr(W, f"v4_routed_{sub}_s"))
                n, k = (
                    (g.hidden_size, g.moe_inter_dim)
                    if sub == "w2"
                    else (g.moe_inter_dim, g.hidden_size)
                )
                if value.dtype not in (torch.int8, torch.uint8) or tuple(value.shape) != (
                    self.local_experts, n, k // 2
                ):
                    raise ValueError(f"L{layer_id} routed {sub} is not EP-local packed FP4")
                if tuple(scale.shape) != (self.local_experts, n, k // 32):
                    raise ValueError(f"L{layer_id} routed {sub} scale is not group-32")
                if scale.element_size() != 1:
                    raise TypeError("routed scales must retain UE8M0 bytes")
                values.append(value)
                scales.append(scale.view(torch.uint8))
            # PPU kernel preprocessing remains owned/cached by PPUMoE.
            self._routed[layer_id] = (
                (torch.cat(values[:2], dim=1), torch.cat(scales[:2], dim=1)),
                (values[2], scales[2]),
            )
            self.conversions[f"L{layer_id}:routed_gate_up_concat"] += 1
        return self._routed[layer_id]
