"""Direct V4.1 parameter ownership with current-main attention execution."""

import torch
from torch import nn

from rtp_llm.models_py.module_base import RtpModule, copy_weight_
from rtp_llm.models_py.modules.dsv41.attn_type import (
    CSA_KV,
    CSA_STATE,
    HCA_KV,
    HCA_STATE,
    INDEXER_KV,
    INDEXER_STATE,
    SWA_KV,
)
from rtp_llm.models_py.modules.dsv41.bounded_replay import (
    enabled as bounded_replay_enabled,
)
from rtp_llm.models_py.modules.dsv41.rope import precompute_freqs_cis
from rtp_llm.models_py.new_models.deepseek_v41.attention_sources import (
    OwnedV41AttentionSources,
)
from rtp_llm.models_py.new_models.deepseek_v41.linear_owner import Dsv41MXFP8Linear
from rtp_llm.models_py.read_spec import (
    ReadSpec,
    make_sharded_read_spec,
    unwrap_checkpoint_tensor,
)
from rtp_llm.models_py.weight_mapper import WeightsMapper


class _GroupedOutputWeights(Dsv41MXFP8Linear):
    """Checkpoint owner; the attention module prepares both grouped WO-A kernels."""

    def process_weights_after_loading(self):
        self.validate_weights_loaded()
        self.validate_runtime_device(self.weight.device)
        # No standalone linear is invoked for this projection. Constructing one
        # would retain a third packed scale alongside the stacked/group kernels.


class OwnedV41Attention(RtpModule):
    """Main runtime methods, checkpoint-native Parameters, no ModelWeights."""

    def __init__(self, args, layer_id, shared_attention):
        RtpModule.__init__(self)
        self.layer_id, self.dim = layer_id, args.dim
        self.q_lora_rank, self.o_lora_rank = args.q_lora_rank, args.o_lora_rank
        self.head_dim, self.rope_head_dim = args.head_dim, args.rope_head_dim
        self.window_size, self.eps = args.window_size, args.norm_eps
        self.compress_ratio = args.compress_ratios[layer_id]
        self.tp_size, self.tp_rank = args.tp_size, args.tp_rank
        if (
            not 0 <= self.tp_rank < self.tp_size
            or args.n_heads % self.tp_size
            or args.o_groups % self.tp_size
        ):
            raise ValueError(
                "V4.1 attention heads/groups must divide a valid TP partition"
            )
        self.n_heads, self.n_groups = (
            args.n_heads // self.tp_size,
            args.o_groups // self.tp_size,
        )
        self.softmax_scale = args.head_dim**-0.5
        self.index_topk, self.index_n_heads, self.index_head_dim = (
            args.index_topk,
            args.index_n_heads,
            args.index_head_dim,
        )
        self.v41_config = args.v41_config
        self.skip_post_q_norm = True
        self._shared_attention = shared_attention
        shared_attention.setdefault("layers", {})[layer_id] = self
        kv_sources = args.v41_config.get("kv_source_layer_ids", [])
        self.swa_bounded_replay = (
            bool(kv_sources) and bounded_replay_enabled() and layer_id > max(kv_sources)
        )
        self.wq_a = Dsv41MXFP8Linear(args.dim, args.q_lora_rank)
        self.wq_b = Dsv41MXFP8Linear(
            args.q_lora_rank,
            args.n_heads * args.head_dim,
            tp_size=args.tp_size,
            tp_rank=args.tp_rank,
            partition="column",
        )
        self.wkv = Dsv41MXFP8Linear(args.dim, args.head_dim)
        self.wo_a = _GroupedOutputWeights(
            args.n_heads * args.head_dim // args.o_groups,
            args.o_groups * args.o_lora_rank,
            tp_size=args.tp_size,
            tp_rank=args.tp_rank,
            partition="column",
        )
        self.wo_b = Dsv41MXFP8Linear(
            args.o_groups * args.o_lora_rank,
            args.dim,
            tp_size=args.tp_size,
            tp_rank=args.tp_rank,
            partition="row",
        )
        self.q_norm = nn.Parameter(
            torch.empty(args.q_lora_rank, dtype=torch.bfloat16), requires_grad=False
        )
        self.kv_norm = nn.Parameter(
            torch.empty(args.head_dim, dtype=torch.bfloat16), requires_grad=False
        )
        self.attn_sink = nn.Parameter(
            torch.empty(self.n_heads, dtype=torch.float32), requires_grad=False
        )
        self.sources = OwnedV41AttentionSources(
            layer_id=layer_id,
            dim=args.dim,
            head_dim=args.head_dim,
            q_lora_rank=args.q_lora_rank,
            index_n_heads=args.index_n_heads,
            index_head_dim=args.index_head_dim,
            compress_ratio=self.compress_ratio,
            v41_config=args.v41_config,
        )
        self.is_kv_source, self.is_index_source = (
            self.sources.is_kv_source,
            self.sources.is_index_source,
        )
        self.kv_source_layer_id, self.index_source_layer_id = (
            self.sources.kv_source_layer_id,
            self.sources.index_source_layer_id,
        )
        # Runtime source views below are properties, so migration/rebinding
        # cannot leave an alias pointing at an old Parameter object.
        self.compressor = self.indexer = None
        self.wq_a_wkv = None
        self._wo_a_groups_v41 = []
        self.register_buffer("_wo_a_stk_w", None, persistent=False)
        self.register_buffer("_wo_a_stk_s", None, persistent=False)
        self._initialize_runtime_metadata(args)
        self.WEIGHTS_MAPPER = WeightsMapper(
            exact_mapping={"q_norm.weight": "q_norm", "kv_norm.weight": "kv_norm"},
            prefix_mapping={
                "compressor.": "sources.compressor.",
                "indexer.": "sources.indexer.",
            },
        )

    @property
    def wo_a_w(self):
        return self.wo_a.weight

    @property
    def wo_a_s(self):
        return self.wo_a.scale

    @property
    def global_wkv(self):
        return self.sources.compressor.wkv.weight

    @property
    def global_norm(self):
        return self.sources.compressor.norm.weight

    @property
    def global_wgate(self):
        return (
            self.sources.compressor.wgate.weight if self.compress_ratio == 2 else None
        )

    @property
    def index_wk(self):
        return self.sources.indexer.wk.weight

    @property
    def index_k_norm(self):
        return self.sources.indexer.k_norm.weight

    @property
    def index_wq(self):
        return self.sources.indexer.wq_b

    @property
    def index_weights(self):
        return self.sources.indexer.weights_proj.weight

    def _initialize_runtime_metadata(self, args):
        compressed = bool(self.compress_ratio)
        self._rope_base = args.compress_rope_theta if compressed else args.rope_theta
        self._rope_o_seq_len = args.original_seq_len if compressed else 0
        self._rope_factor, self._rope_beta_fast, self._rope_beta_slow = (
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        )
        self._rope_dim, self._rope_max_seq_len = args.rope_head_dim, args.max_seq_len
        self.freqs_cis = precompute_freqs_cis(
            args.rope_head_dim,
            args.max_seq_len,
            self._rope_o_seq_len,
            self._rope_base,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        )
        self._fp8_decode_op = self._cp_ctx = self._prefill_meta_shared = (
            self._kv_cache
        ) = self._block_tables_by_type = None
        self._pool_spec = {
            SWA_KV: (torch.uint8, 528),
            CSA_KV: (torch.uint8, 288),
            HCA_KV: (torch.uint8, 288),
            INDEXER_KV: (torch.uint8, 68),
            CSA_STATE: (torch.float32, 2 * args.head_dim),
            HCA_STATE: (torch.float32, 2 * args.head_dim),
            INDEXER_STATE: (torch.float32, 4 * args.index_head_dim),
        }

    def checkpoint_read_spec(self, name, shape):
        if name == "attn_sink":
            if tuple(shape) != (self.n_heads * self.tp_size,):
                raise ValueError(
                    "V4.1 attention sink must have one value per global head"
                )
            return make_sharded_read_spec(
                name, tuple(shape), 0, self.tp_size, self.tp_rank
            )
        return self.mapped_child_read_spec(name, shape)

    def load_weights(self, weights):
        items = weights.items() if isinstance(weights, dict) else weights

        def mapped():
            for name, value in self.WEIGHTS_MAPPER.apply(items):
                if name == "attn_sink":
                    if name in getattr(self, "_rtp_loaded_weight_names", ()):
                        raise RuntimeError("Duplicate V4.1 attention sink")
                    tensor, spec = unwrap_checkpoint_tensor(value)
                    if spec is None:
                        tensor, _ = unwrap_checkpoint_tensor(
                            self.checkpoint_read_spec(name, tensor.shape).materialize(
                                tensor
                            )
                        )
                    else:
                        expected = self.checkpoint_read_spec(name, spec.source_shape)
                        if (
                            spec.slices != expected.slices
                            or spec.shard_rank != expected.shard_rank
                        ):
                            raise ValueError(
                                "V4.1 attention sink ReadSpec belongs to another TP shard"
                            )
                    copy_weight_(self.attn_sink, tensor, "V41.attn_sink")
                    self._mark_weight_loaded(name)
                else:
                    yield name, value

        RtpModule.load_weights(self, mapped())

    def validate_runtime_device(self, device):
        self.wkv.validate_runtime_device(device)

    def bind_resident_parameters(self, bank, prefix):
        self._resident_qkv = bank, tuple(prefix + n for n in (
            "wq_a.weight", "wq_a.scale", "wkv.weight", "wkv.scale"
        ))

    @torch.no_grad()
    def _prepare_qkv(self, merge):
        # Keep the four checkpoint Parameters (and their identity). A native
        # prefill projection reads their shared allocation, not another copy.
        q, kv = self.wq_a, self.wkv
        resident = getattr(self, "_resident_qkv", None)
        if resident is None:
            merged = merge(q.weight, q.scale, kv.weight, kv.scale)
        else:
            from rtp_llm.models_py.modules.dsv41.utils import merge_v41_qkv_weights
            bank, keys = resident
            linear = merge_v41_qkv_weights(bank, *keys)
            qw, qs, kw, ks = keys
            merged = None if linear is None else (linear, (bank[qw], bank[kw]), (bank[qs], bank[ks]))
        if merged is None:
            object.__setattr__(self, "wq_a_wkv", None)
            return
        linear, weights, scales = merged
        for projection, weight, scale in zip((q, kv), weights, scales):
            projection.weight.set_(weight)
            projection.scale.set_(scale)
            object.__setattr__(projection, "_executor", None)
        # Derived executor; never register a second checkpoint owner or let
        # Module._apply convert its packed scales independently of the owners.
        object.__setattr__(self, "wq_a_wkv", linear)

    def _apply(self, fn, recurse=True):
        object.__setattr__(self, "wq_a_wkv", None)
        return super()._apply(fn, recurse=recurse)

    def process_weights_after_loading(self):
        self.validate_weights_loaded()
        self.validate_runtime_device(self.wkv.weight.device)
        from rtp_llm.models_py.modules.dsv41.fp8.attention import _prepare_wo_a_stacked
        from rtp_llm.models_py.modules.dsv41.utils import (
            _v4_fp8_linear,
            merge_v41_qkv_tensors,
        )

        self._prepare_qkv(merge_v41_qkv_tensors)
        self._wo_a_stk_w, self._wo_a_stk_s = _prepare_wo_a_stacked(
            self.wo_a.weight,
            self.wo_a.scale,
            self.n_groups,
            self.o_lora_rank,
            self.n_heads * self.head_dim // self.n_groups,
        )
        self._wo_a_groups_v41 = [
            _v4_fp8_linear(
                self.wo_a.weight[g * self.o_lora_rank : (g + 1) * self.o_lora_rank],
                self.wo_a.scale[
                    g * self.o_lora_rank // 32 : (g + 1) * self.o_lora_rank // 32
                ],
            )
            for g in range(self.n_groups)
        ]


class OwnedV41CommitAttention(RtpModule):
    """Draft PREFILL owns only WKV/norm and uses V4.1's 528-byte writer ABI."""

    def __init__(self, args, layer_id, shared_attention):
        RtpModule.__init__(self)
        if args.compress_ratios[layer_id] != 0:
            raise ValueError("V4.1 commit-only attention requires SWA")
        self.layer_id, self.dim = layer_id, args.dim
        self.tp_size, self.tp_rank = args.tp_size, args.tp_rank
        self.n_heads, self.n_groups = (
            args.n_heads // args.tp_size,
            args.o_groups // args.tp_size,
        )
        self.head_dim, self.rope_head_dim = args.head_dim, args.rope_head_dim
        self.q_lora_rank, self.o_lora_rank = args.q_lora_rank, args.o_lora_rank
        self.window_size, self.compress_ratio, self.eps = (
            args.window_size,
            0,
            args.norm_eps,
        )
        self.softmax_scale = args.head_dim**-0.5
        self.wkv = Dsv41MXFP8Linear(args.dim, args.head_dim)
        self.kv_norm = nn.Parameter(
            torch.empty(args.head_dim, dtype=torch.bfloat16), requires_grad=False
        )
        self.wq_a = self.wq_b = self.wo_a = self.wo_b = self.compressor = (
            self.indexer
        ) = None
        OwnedV41Attention._initialize_runtime_metadata(self, args)
        self.WEIGHTS_MAPPER = WeightsMapper(exact_mapping={"kv_norm.weight": "kv_norm"})

    def checkpoint_read_spec(self, name, shape):
        return self.mapped_child_read_spec(name, shape)

    def load_weights(self, weights):
        items = weights.items() if isinstance(weights, dict) else weights
        RtpModule.load_weights(self, self.WEIGHTS_MAPPER.apply(items))

    def validate_runtime_device(self, device):
        self.wkv.validate_runtime_device(device)
