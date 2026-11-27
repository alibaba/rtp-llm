"""Qwen4 full-attention parameters, independent of native runtime imports.

The source Q projection interleaves Q and gate rows per head. KV replication
follows attention TP, while the QSA indexer is replicated on every rank.
"""
import torch

from rtp_llm.models_py.module_base import RtpModule, copy_weight_
from rtp_llm.models_py.new_models.qwen4_exp.hyper_connection import Qwen4TensorWeight
from rtp_llm.models_py.new_models.qwen4_exp.hyper_connection import Qwen4ProjectionWeight
from rtp_llm.models_py.new_models.qwen4_exp.dense_weights import Qwen4BlockProjection, Qwen4ShardedProjection


class _PlainOutputProjection(Qwen4ShardedProjection):
    def prepare_weight_updates(self, weights):
        from rtp_llm.models_py.runtime_weight_update import prepare_leaf_weight_updates
        return prepare_leaf_weight_updates(self, weights)


class Qwen4AttentionWeights(RtpModule):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, *,
                 tp_size=1, tp_rank=0, norm_eps=1e-6, dtype=torch.bfloat16,
                 qsa_indexer=None, fp8_storage=False):
        super().__init__()
        dimensions = (hidden_size, num_heads, num_kv_heads, head_dim, tp_size)
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("Qwen4 attention dimensions must be positive integers")
        if not 0 <= tp_rank < tp_size or num_heads % tp_size:
            raise ValueError("Qwen4 attention Q heads must divide attention TP")
        if num_heads % num_kv_heads:
            raise ValueError("Qwen4 Q heads must be divisible by KV heads")
        if num_kv_heads >= tp_size:
            if num_kv_heads % tp_size:
                raise ValueError("Qwen4 KV heads must divide attention TP")
            self.local_kv_heads = num_kv_heads // tp_size
            self.kv_head_start = tp_rank * self.local_kv_heads
        else:
            if tp_size % num_kv_heads:
                raise ValueError("Qwen4 replicated KV heads require divisible TP")
            self.local_kv_heads = 1
            self.kv_head_start = tp_rank // (tp_size // num_kv_heads)
        self.hidden_size = hidden_size
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, head_dim
        self.tp_size, self.tp_rank = tp_size, tp_rank
        self.local_heads = num_heads // tp_size
        self.q_size = self.local_heads * head_dim
        self.kv_size = self.local_kv_heads * head_dim
        self.norm_eps = norm_eps
        projection = Qwen4BlockProjection if fp8_storage else Qwen4TensorWeight
        self.qkv_proj = projection((self.q_size + 2*self.kv_size, hidden_size), dtype=dtype)
        self.gate = Qwen4ProjectionWeight((self.q_size, hidden_size), dtype=dtype, fp8_storage=fp8_storage)
        self.o_proj = (
            projection((hidden_size, self.q_size), dtype=dtype) if fp8_storage
            else _PlainOutputProjection((hidden_size, num_heads * head_dim), axis=1,
                                        tp_size=tp_size, tp_rank=tp_rank, dtype=dtype)
        )
        self.q_norm = Qwen4TensorWeight((head_dim,), dtype=dtype)
        self.k_norm = Qwen4TensorWeight((head_dim,), dtype=dtype)
        self.qsa_indexer = qsa_indexer
        self._qkv_parts = set()
        object.__setattr__(self, "_runtime", None)

    def _load_qkv_part(self, part, tensor):
        if part in self._qkv_parts:
            raise RuntimeError(f"Duplicate Qwen4 attention projection {part}")
        self._prepare_child_weight_loading(self.qkv_proj)
        if part == "q":
            expected = (2*self.num_heads*self.head_dim, self.hidden_size)
            if tuple(tensor.shape) != expected:
                raise ValueError(f"Qwen4 interleaved Q/gate shape must be {expected}")
            q, gate = tensor.reshape(self.num_heads, 2*self.head_dim, self.hidden_size).split(self.head_dim, dim=1)
            start = self.tp_rank*self.local_heads
            q = q[start:start+self.local_heads].reshape(self.q_size, self.hidden_size)
            gate = gate[start:start+self.local_heads].reshape(self.q_size, self.hidden_size)
            self._load_child_weights(self.gate, {"weight": gate})
            offset, local = 0, q
        else:
            expected = (self.num_kv_heads*self.head_dim, self.hidden_size)
            if tuple(tensor.shape) != expected:
                raise ValueError(f"Qwen4 {part.upper()} shape must be {expected}")
            start = self.kv_head_start*self.head_dim
            local = tensor[start:start+self.kv_size]
            offset = self.q_size + (self.kv_size if part == "v" else 0)
        copy_weight_(self.qkv_proj.weight.data.narrow(0, offset, local.shape[0]), local, part)
        self._qkv_parts.add(part)
        if self._qkv_parts == {"q", "k", "v"}:
            self.qkv_proj._mark_weight_loaded("weight")
        self._finalize_child_weight_loading(self.qkv_proj)

    def load_weights(self, weights):
        for name, tensor in weights.items():
            # Released checkpoints nest these under self_attn.indexer; older
            # module-local fixtures used the indexer names directly.
            name = name.removeprefix("indexer.")
            if name in ("q_proj.weight", "k_proj.weight", "v_proj.weight"):
                self._load_qkv_part(name[0], tensor)
            elif name == "o_proj.weight":
                if isinstance(self.o_proj, _PlainOutputProjection):
                    self._load_child_weights(self.o_proj, {"weight": tensor})
                    continue
                expected = (self.hidden_size, self.num_heads*self.head_dim)
                if tuple(tensor.shape) != expected:
                    raise ValueError(f"Qwen4 output projection shape must be {expected}")
                self._load_child_weights(self.o_proj, {"weight": tensor[:,self.tp_rank*self.q_size:(self.tp_rank+1)*self.q_size]})
            elif name in ("q_norm.weight", "k_norm.weight"):
                self._load_child_weights(getattr(self, name.split(".")[0]), {"weight": tensor})
            elif name.split(".")[0] in ("index_qk_proj", "q_layernorm", "k_layernorm") and self.qsa_indexer is not None:
                self._load_child_weights(self.qsa_indexer, {name: tensor})
            else:
                raise RuntimeError(f"Unsupported Qwen4 attention tensor {name}")

    def bind_runtime(self, parallelism_config, *, layer_idx=0, is_mtp_draft=False,
                     rope_config=None, hw_kernel_config=None):
        """Bind existing main attention/QSA transaction execution to owned tensors."""
        if self._runtime is None:
            from rtp_llm.models_py.new_models.qwen4_exp.runtime_bindings import bind_attention
            runtime = bind_attention(self, parallelism_config, layer_idx=layer_idx,
                is_mtp_draft=is_mtp_draft, rope_config=rope_config,
                hw_kernel_config=hw_kernel_config)
            object.__setattr__(self, "_runtime", runtime)
        return self._runtime

    def forward(self, *args, **kwargs):
        if self._runtime is None:
            raise RuntimeError("Qwen4 attention runtime is not bound")
        return self._runtime(*args, **kwargs)
