"""Complete target checkpoint tree for Qwen4's split NVFP4/BF16 recipe.

This owner is deliberately not registered as a runnable model until its native
runtime integration is verified. It is also not a new internal weight naming
system: the child hierarchy follows the actual checkpoint module hierarchy.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from safetensors import safe_open

from rtp_llm.models_py.downstream_heads import DecoderHandlerOwner, needs_generation_projection
from rtp_llm.models_py.layers.embedding import ParallelLMHead, output_vocab_head_kwargs
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.weight_mapper import WeightsMapper
from rtp_llm.models_py.new_models.qwen4_exp.attention_weights import Qwen4AttentionWeights
from rtp_llm.models_py.new_models.qwen4_exp.checkpoint_validation import validate_checkpoint_directory
from rtp_llm.models_py.new_models.qwen4_exp.gdn_weights import Qwen4GDNWeights
from rtp_llm.models_py.new_models.qwen4_exp.hyper_connection import Qwen4TensorWeight, Qwen4HyperConnection, Qwen4ProjectionWeight
from rtp_llm.models_py.new_models.qwen4_exp.dense_weights import Qwen4ShardedProjection
from rtp_llm.models_py.new_models.qwen4_exp.expert_factory import make_qwen4_experts
from rtp_llm.models_py.new_models.qwen4_exp.ple_layer import Qwen4PLELayer
from rtp_llm.models_py.new_models.qwen4_exp.qsa_indexer import Qwen4QSAIndexer


class Qwen4CheckpointPlan:
    """Header-only construction facts, checked before allocating model storage."""
    def __init__(self, directory, model_config=None):
        self.model_config = model_config
        self.root = Path(directory)
        self.source_config = validate_checkpoint_directory(self.root)
        self.text = self.source_config.get("text_config", self.source_config)
        quant = self.source_config.get("quantization_config", {})
        self.source_kind = "bf16" if not quant else ("nvfp4" if quant.get("quant_method") == "modelopt" else "fp8")
        self.source_fp8 = self.source_kind == "fp8"
        def option(name, default):
            return model_config.get(name, default) if isinstance(model_config, dict) else getattr(model_config, name, default)
        self.dense_storage = option("_qwen4_dense_storage", "fp8" if self.source_fp8 else "source_bf16")
        self.vocab_storage = option("_qwen4_vocab_storage", self.dense_storage)
        self.ple_prefetch_mb = option("qwen4_ple_prefetch_mb", 256)
        if self.dense_storage not in ("fp8", "source_bf16") or self.vocab_storage not in ("fp8", "source_bf16"):
            raise ValueError("Invalid Qwen4 dense/vocabulary storage recipe")
        self.dense_fp8 = self.source_fp8 and self.dense_storage == "fp8"
        self.vocab_fp8 = self.source_fp8 and self.vocab_storage == "fp8"
        index = self.root / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
        else:
            # Ordinary BF16 HF releases need not have an index when stored in
            # one file. Quantized releases retain main's strict manifest check.
            weight_map = {}
            for path in sorted(self.root.glob("*.safetensors")):
                with safe_open(str(path), framework="pt", device="cpu") as shard:
                    for name in shard.keys():
                        if name in weight_map:
                            raise ValueError(f"Duplicate Qwen4 checkpoint tensor: {name}")
                        weight_map[name] = path.name
            if not weight_map:
                raise ValueError("Qwen4 checkpoint contains no safetensors weights")
        self.shapes = {}
        self.dtypes = {}
        for filename in set(weight_map.values()):
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".safetensors":
                raise ValueError("Qwen4 checkpoint shards must be relative safetensors")
            with safe_open(str(self.root / relative), framework="pt", device="cpu") as shard:
                for name in shard.keys():
                    if weight_map.get(name) != filename:
                        raise ValueError(f"Qwen4 shard/index mismatch: {name}")
                    view = shard.get_slice(name)
                    self.shapes[name] = tuple(view.get_shape())
                    self.dtypes[name] = view.get_dtype()
        if set(weight_map) != set(self.shapes):
            raise ValueError("Qwen4 index contains tensors absent from shards")
        for name, dtype in self.dtypes.items():
            host_fp8 = self.source_fp8 and ".ple_embedding.ngram_embedding.shard_" in name
            if ".mlp.experts." not in name and not host_fp8 and name.endswith(".weight") and dtype not in ("BF16", "F16", "F32", "F64"):
                raise ValueError(f"Qwen4 NVFP4 source requires floating BF16-compatible non-expert weights: {name} ({dtype})")
        anchor = "layers.0.attn_hyper_connection.hc_norm.weight"
        prefixes = [name[:-len(anchor)] for name in self.shapes if name.endswith(anchor) and not name.startswith("mtp.")]
        if len(prefixes) != 1:
            raise ValueError("Qwen4 target prefix must be uniquely anchored by HC weights")
        self.prefix = prefixes[0]
        self.num_layers = self.positive("num_hidden_layers")
        self.hidden_size = self.positive("hidden_size")
        self.hc_mult = self.positive("hc_count")
        self.vocab_size = self.positive("vocab_size")
        self.layer_types = []
        for i in range(self.num_layers):
            base = self.prefix + f"layers.{i}."
            linear = any(name.startswith(base+"linear_attn.") for name in self.shapes)
            full = any(name.startswith(base+"self_attn.") for name in self.shapes)
            if linear == full:
                raise ValueError(f"Qwen4 layer {i} needs exactly one attention subtree")
            kind = "linear_attention" if linear else "full_attention"
            declared = self.text.get("layer_types")
            if declared is not None and (len(declared) != self.num_layers or declared[i] != kind):
                raise ValueError("Qwen4 layer types disagree with checkpoint")
            self.layer_types.append(kind)
        self.ple_layers = tuple(int(i)-1 for i in self.text.get("ple_layer_ids", ()))
        if len(set(self.ple_layers)) != len(self.ple_layers) or any(not 0 <= i < self.num_layers for i in self.ple_layers):
            raise ValueError("Qwen4 PLE layer IDs must be unique one-based layer indices")

    def positive(self, name):
        value = self.text.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"Qwen4 {name} must be a positive integer")
        return value

    def lowrank(self, relative):
        name = self.prefix+relative+".input_mix_weight_down.weight"
        shape = self.shapes.get(name)
        if shape is None or len(shape) != 2 or shape[1] != self.hidden_size*self.hc_mult or shape[0] <= 0:
            raise ValueError(f"Qwen4 HC projection shape is invalid: {name}")
        return shape[0]


class _Qwen4VocabStorage:
    def process_weights_after_loading(self):
        if self._storage_ready:
            return
        # Main quantizes vocabulary matrices even for dimensions not aligned to
        # 16; only ordinary projections use the alignment eligibility check.
        if self.fp8_storage:
            values = self.weight.float()
            scale = values.abs().amax().clamp_min(1e-12).reshape(1) / 448.0
            self.weight.data = (values / scale).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
            self.storage_scale = scale
            self.weight._rtp_requires_restart_for_update = True
            self.storage_scale._rtp_requires_restart_for_update = True
        self._storage_ready = True


class Qwen4VocabWeight(_Qwen4VocabStorage, Qwen4ProjectionWeight):
    def __init__(self, vocab_size, hidden_size, *, tp_size, tp_rank, embedding, dtype, fp8_storage=False):
        axis = 1 if embedding else 0
        shape = [vocab_size, hidden_size]
        if shape[axis] % tp_size or not 0 <= tp_rank < tp_size:
            raise ValueError("Qwen4 vocabulary shard dimension must divide TP")
        self.full_shape = tuple(shape)
        self.axis, self.tp_size, self.tp_rank = axis, tp_size, tp_rank
        shape[axis] //= tp_size
        super().__init__(tuple(shape), dtype=dtype, fp8_storage=fp8_storage)

    def prepare_weight_updates(self, weights):
        from rtp_llm.models_py.runtime_weight_update import prepare_leaf_weight_updates
        return prepare_leaf_weight_updates(self, weights)

    def load_weights(self, weights):
        for name, tensor in weights.items():
            if name != "weight" or tuple(tensor.shape) != self.full_shape:
                raise ValueError("Invalid Qwen4 vocabulary tensor")
            count = self.full_shape[self.axis] // self.tp_size
            local = tensor.narrow(self.axis, self.tp_rank*count, count)
            super().load_weights({"weight":local})


class Qwen4PrunedLMHead(_Qwen4VocabStorage, ParallelLMHead):
    """Select the engine's token IDs before applying Qwen4's storage recipe."""

    supports_independent_weight_updates = False

    def __init__(self, *args, fp8_storage=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fp8_storage = fp8_storage
        self.register_buffer("storage_scale", None)
        self._storage_ready = False


class Qwen4SharedExpertWeights(RtpModule):
    def __init__(self, hidden, intermediate, dtype, *, tp_size=1, tp_rank=0, fp8_storage=False):
        super().__init__()
        self.fp8_storage = fp8_storage
        self._storage_ready = False
        self.register_buffer("up_storage_scale", None)
        for name in ("gate_proj", "up_proj"):
            setattr(self, name, Qwen4ShardedProjection((intermediate, hidden), axis=0,
                tp_size=tp_size, tp_rank=tp_rank, dtype=dtype))
        self.down_proj = Qwen4ShardedProjection((hidden, intermediate), axis=1,
            tp_size=tp_size, tp_rank=tp_rank, dtype=dtype, fp8_storage=fp8_storage)

    def process_weights_after_loading(self):
        if self._storage_ready:
            return
        # Main's merged shared FFN has one amax domain for gate + up, not two
        # independently quantized scalar domains.
        weight = torch.cat((self.gate_proj.weight, self.up_proj.weight), dim=0)
        if self.fp8_storage and all(size % 16 == 0 for size in weight.shape):
            values = weight.float()
            scale = values.abs().amax().clamp_min(1e-12).reshape(1) / 448.0
            weight = (values / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            self.up_storage_scale = scale
        width = self.gate_proj.weight.shape[0]
        self.gate_proj.weight.data, self.up_proj.weight.data = weight[:width], weight[width:]
        self.down_proj.process_weights_after_loading()
        self._storage_ready = True


class Qwen4MoeWeights(RtpModule):
    def __init__(self, plan, layer_idx, load):
        super().__init__()
        h = plan.hidden_size
        experts = plan.positive("num_experts")
        intermediate = plan.positive("moe_intermediate_size")
        self.gate = Qwen4ProjectionWeight((experts,h), dtype=load.compute_dtype, fp8_storage=plan.dense_fp8)
        self.experts = make_qwen4_experts(plan, layer_idx, load)
        base = plan.prefix+f"layers.{layer_idx}.mlp."
        shared_shape = plan.shapes.get(base+"shared_expert.gate_proj.weight")
        self.shared_expert = None if shared_shape is None else Qwen4SharedExpertWeights(h, shared_shape[0], load.compute_dtype,
            tp_size=load.ffn_tp_size, tp_rank=load.ffn_tp_rank, fp8_storage=plan.dense_fp8)
        self.shared_expert_gate = None
        if base+"shared_expert_gate.weight" in plan.shapes:
            self.shared_expert_gate = Qwen4ProjectionWeight((1,h),dtype=load.compute_dtype, fp8_storage=plan.dense_fp8)


class Qwen4DecoderWeights(RtpModule):
    def __init__(self, plan, layer_idx, load):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = plan.layer_types[layer_idx]
        h, text, dtype = plan.hidden_size, plan.text, load.compute_dtype
        eps = float(text.get("rms_norm_eps",1e-6))
        for name in ("attn_hyper_connection", "mlp_hyper_connection"):
            setattr(self,name,Qwen4HyperConnection(h,plan.hc_mult,
                plan.lowrank(f"layers.{layer_idx}.{name}"),norm_eps=eps,dtype=dtype,fp8_storage=plan.dense_fp8,
                role=getattr(load.parallelism_config, "role_type", "PDFUSION")))
        if self.layer_type == "linear_attention":
            fields = ("linear_key_head_dim", "linear_value_head_dim", "linear_num_key_heads", "linear_num_value_heads", "linear_conv_kernel_dim")
            la = SimpleNamespace(**{name:plan.positive(name) for name in fields})
            self.linear_attn = Qwen4GDNWeights(la,h,eps,load.attn_tp_size,load.attn_tp_rank,dtype,None,f"layers.{layer_idx}.linear_attn",fp8_storage=plan.dense_fp8)
        else:
            qsa = None
            if int(text.get("indexer_head_dim",0)) > 0:
                # Checkpoint-only dictionaries have no native attention config;
                # target and draft services have their own finalized RoPE style.
                attn_config = getattr(plan.model_config, "attn_config", None)
                qsa = Qwen4QSAIndexer(h,n_heads=plan.positive("indexer_n_heads"),
                    head_dim=plan.positive("indexer_head_dim"), token_budget=plan.positive("indexer_budget"),
                    compress_ratio=plan.positive("indexer_compress_ratio"),kv_heads=plan.positive("indexer_kv_heads"),norm_eps=eps,dtype=dtype,fp8_storage=plan.dense_fp8,
                    rope_config=attn_config.rope_config if attn_config is not None else None)
            self.self_attn = Qwen4AttentionWeights(h,plan.positive("num_attention_heads"),plan.positive("num_key_value_heads"),
                plan.positive("head_dim"),tp_size=load.attn_tp_size,tp_rank=load.attn_tp_rank,norm_eps=eps,dtype=dtype,qsa_indexer=qsa,fp8_storage=plan.dense_fp8)
        self.mlp = Qwen4MoeWeights(plan,layer_idx,load)
        self.ple = None
        if layer_idx in plan.ple_layers:
            ngram = plan.positive("ngram_size")
            heads = int(text.get("heads_per_ngram",8))
            total = plan.positive("ple_embed_dim")
            if total % ((ngram-1)*heads):
                raise ValueError("Qwen4 PLE total width must divide ngram head count")
            eos = text.get("eos_token_id", plan.source_config.get("eos_token_id"))
            if eos is None and plan.model_config is not None and not isinstance(plan.model_config, dict):
                from rtp_llm.config.qwen4_exp_config import ple_eos_token_id
                eos = ple_eos_token_id(plan.model_config)
            if type(eos) is not int or not 0 <= eos < plan.vocab_size:
                raise ValueError("Qwen4 PLE requires a resolved eos_token_id in the vocabulary")
            self.ple = Qwen4PLELayer(hidden_size=h,hc_mult=plan.hc_mult,ngram_size=ngram,
                heads_per_ngram=heads,shard_count=plan.positive("split_ngram_parts"),
                embedding_dim=total//((ngram-1)*heads),conv_kernel_size=plan.positive("ple_conv_kernel_size"),
                eos_token_id=eos,tp_size=load.attn_tp_size,tp_rank=load.attn_tp_rank,norm_eps=eps,dtype=dtype,
                source_fp8=plan.source_fp8,fp8_storage=plan.dense_fp8,preserve_fp8_tables=plan.source_fp8,
                prefetch_budget_mb=plan.ple_prefetch_mb)


class Qwen4TargetWeights(DecoderHandlerOwner, RtpModule):
    @classmethod
    def from_ft_checkpoint_header(cls, model_config, load_config, checkpoint):
        from rtp_llm.models_py.new_models.qwen4_exp.rank_local_ft import rank_local_plan
        return cls(model_config, load_config, checkpoint_plan=rank_local_plan(model_config, load_config, checkpoint))

    def load_ft_checkpoint(self, checkpoint):
        from rtp_llm.models_py.new_models.qwen4_exp.rank_local_ft import load_rank_local
        load_rank_local(self, checkpoint)

    def __init__(self, model_config, load_config, *, checkpoint_plan=None):
        super().__init__()
        directory = model_config.get("ckpt_path") if isinstance(model_config,dict) else model_config.ckpt_path
        self.checkpoint_plan = plan = checkpoint_plan or Qwen4CheckpointPlan(directory, model_config)
        self.WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={plan.prefix:""})
        self._init_decoder_handlers(load_config, plan.hidden_size)
        def option(name, default):
            return model_config.get(name, default) if isinstance(model_config, dict) else getattr(model_config, name, default)
        input_vocab = option("input_vocab_size", 0) or plan.text.get("input_vocab_size") or plan.vocab_size
        if type(input_vocab) is not int or input_vocab <= 0:
            raise ValueError("Qwen4 input vocabulary must have a positive size")
        self.embed_tokens = Qwen4VocabWeight(input_vocab,plan.hidden_size,tp_size=load_config.attn_tp_size,
            tp_rank=load_config.attn_tp_rank,embedding=True,dtype=load_config.compute_dtype,fp8_storage=plan.vocab_fp8)
        head_options = output_vocab_head_kwargs(model_config)
        if not needs_generation_projection(model_config):
            self.lm_head = None
        elif head_options:
            self.lm_head = Qwen4PrunedLMHead(
                plan.vocab_size, plan.hidden_size,
                tp_size=load_config.lm_head_tp_size, tp_rank=load_config.lm_head_tp_rank,
                params_dtype=torch.float32 if option("enable_fp32_lm_head", False) else load_config.compute_dtype,
                fp8_storage=plan.vocab_fp8, **head_options,
            )
        else:
            self.lm_head = Qwen4VocabWeight(plan.vocab_size,plan.hidden_size,tp_size=load_config.lm_head_tp_size,
                tp_rank=load_config.lm_head_tp_rank,embedding=False,dtype=load_config.compute_dtype,fp8_storage=plan.vocab_fp8)
        tied = option("tie_word_embeddings", plan.text.get("tie_word_embeddings", plan.source_config.get("tie_word_embeddings", False)))
        self._untied_vocab = not tied
        # Unlike the output head, Qwen4's input embedding shards hidden
        # columns. Derive the mirror from its full HF source, never its shard.
        self._head_from_embedding = self.lm_head is not None and tied and not (
            "lm_head.weight" in plan.shapes or plan.prefix + "lm_head.weight" in plan.shapes
        )
        self.layers = nn.ModuleList([Qwen4DecoderWeights(plan,i,load_config) for i in range(plan.num_layers)])
        self.hyper_connection_mixer = Qwen4HyperConnection(plan.hidden_size,plan.hc_mult,
            plan.lowrank("hyper_connection_mixer"),with_inject=False,
            norm_eps=float(plan.text.get("rms_norm_eps",1e-6)),dtype=load_config.compute_dtype,fp8_storage=plan.dense_fp8,
            role=getattr(load_config.parallelism_config, "role_type", "PDFUSION"))

    def load_weights(self, weights):
        items = weights.items() if isinstance(weights,dict) else weights
        def mapped_weights():
            for name, tensor in self.WEIGHTS_MAPPER.apply(self._capture_handler_inputs(items)):
                yield name, tensor
                if self._head_from_embedding and name == "embed_tokens.weight":
                    if tensor.shape[0] < self.checkpoint_plan.vocab_size:
                        raise ValueError("Qwen4 tied embedding does not cover the output vocabulary")
                    yield "lm_head.weight", tensor[:self.checkpoint_plan.vocab_size]
        return super().load_weights(mapped_weights())

    def checkpoint_weight_name_filter(self):
        ple_filters = [layer.ple.checkpoint_name_filter(f"layers.{i}.ple.")
            for i,layer in enumerate(self.layers) if layer.ple is not None]
        prefix = self.checkpoint_plan.prefix
        def keep(name):
            if name in self._handler_checkpoint_names:
                return True
            if self.lm_head is None and self.WEIGHTS_MAPPER.map_name(name) == "lm_head.weight":
                return False
            if name.startswith(prefix):
                name = name[len(prefix):]
            elif name != "lm_head.weight":
                return False
            return all(predicate(name) for predicate in ple_filters)
        return keep

    def close(self):
        for layer in self.layers:
            if layer.ple is not None:
                layer.ple.close()
