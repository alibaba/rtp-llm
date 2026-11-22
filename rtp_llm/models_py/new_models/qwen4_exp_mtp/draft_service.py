"""One-layer direct Qwen4 draft for released BF16, split and mixed recipes."""

from dataclasses import replace

import torch
from torch import nn

from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.qwen4_exp.hyper_connection import Qwen4HyperConnection
from rtp_llm.models_py.new_models.qwen4_exp.target_weights import (
    Qwen4CheckpointPlan,
    Qwen4DecoderWeights,
    Qwen4VocabWeight,
)
from rtp_llm.models_py.new_models.qwen4_exp_mtp.input_owner import Qwen4MTPInputOwner
from rtp_llm.config.qwen4_mtp_vocab import MTPVocabPolicy, mtp_vocab_policy
from rtp_llm.models_py.new_models.qwen4_exp_mtp.vocabulary import (
    BorrowedVocab,
    EMBED_SCALE,
    HEAD_SCALE,
    Qwen4MTPVocabLoan,
    VocabLeaf,
)
from rtp_llm.models_py.weight_mapper import WeightsMapper


class _DraftPlan(Qwen4CheckpointPlan):
    def __init__(self, directory, config):
        super().__init__(directory, config)
        spec = self.text.get("mtp") or {}
        if (
            spec.get("hybrid") is not True
            or spec.get("num_hidden_layers") != 1
            or spec.get("layer_types") != ["full_attention"]
            or self.text.get("mtp_use_dedicated_embeddings", False)
            or spec.get("mtp_use_hidden_state_from_layer") is not None
        ):
            raise ValueError(
                "Qwen4 MTP requires one full-attention draft layer using final target HC states"
            )
        self.prefix, self.num_layers = "mtp.", 1
        self.layer_types, self.ple_layers = ("full_attention",), ()
        self.positive("indexer_head_dim")
        for name in (
            "layers.0.attn_hyper_connection.hc_norm.weight",
            "layers.0.self_attn.indexer.index_qk_proj.weight",
            "layers.0.self_attn.indexer.q_layernorm.weight",
            "layers.0.self_attn.indexer.k_layernorm.weight",
        ):
            if self.prefix + name not in self.shapes:
                raise ValueError(f"Missing Qwen4 MTP anchor: {name}")


class Qwen4MTPDraftWeights(RtpModule):
    """Load only mtp.* plus the source vocabulary when it isn't a target loan."""

    runtime_parameter_paths = {
        "embedding": "embed_tokens.weight",
        "lm_head": "lm_head.weight",
    }
    WEIGHTS_MAPPER = WeightsMapper(
        exact_mapping={
            "mtp." + name + ".weight": "mtp_input_projection." + name + ".weight"
            for name in (
                "fc_embedding",
                "fc_hidden",
                "pre_fc_norm_embedding",
                "pre_fc_norm_hidden",
            )
        },
        prefix_mapping={
            "mtp.layers.0.self_attn.indexer.": "layers.0.self_attn.",
            "mtp.layers.": "layers.",
            "mtp.hyper_connection_mixer.": "hyper_connection_mixer.",
            "model.language_model.embed_tokens.": "embed_tokens.",
        },
    )

    @staticmethod
    def capture_runtime_loan(target, target_config, draft_config, exports, device):
        policy = mtp_vocab_policy(target_config, draft_config)
        if policy is not MTPVocabPolicy.BORROW_SCALED_FP8:
            if policy is not MTPVocabPolicy.BORROW:
                raise ValueError("Source-BF16 draft must not borrow target vocabulary")
            return None  # Ordinary floating tensors use the generic parameter binder.
        source = {}
        for name, leaf, scale_key in (
            ("embedding", target.embed_tokens, EMBED_SCALE),
            ("lm_head", target.lm_head, HEAD_SCALE),
        ):
            if not isinstance(leaf, Qwen4VocabWeight):
                raise ValueError("Qwen4 FP8 vocabulary requires a Qwen4 target owner")
            source[name], source[scale_key] = leaf.weight, leaf.storage_scale
        if set(exports) != set(source) or any(
            exports[k] is not v for k, v in source.items()
        ):
            raise ValueError(
                "Expected both live target vocabulary weights and their own scales"
            )
        return Qwen4MTPVocabLoan.capture(
            target,
            target_config,
            draft_config,
            embedding=VocabLeaf("embed_tokens.weight", "embed_tokens.storage_scale"),
            head=VocabLeaf("lm_head.weight", "lm_head.storage_scale"),
            embedding_shape=target.embed_tokens.weight.shape,
            head_shape=target.lm_head.weight.shape,
            device=device,
        )

    def __init__(self, config, load):
        super().__init__()
        directory = (
            config["ckpt_path"] if isinstance(config, dict) else config.ckpt_path
        )
        self.checkpoint_plan = plan = _DraftPlan(directory, config)
        if load.compute_dtype != torch.bfloat16:
            raise ValueError("Qwen4 draft requires BF16 compute")
        self._vocab_loan = loan = load.prepared_runtime_loan
        if loan is not None:
            if not isinstance(loan, Qwen4MTPVocabLoan) or not plan.vocab_fp8:
                raise ValueError("Invalid scaled loan for Qwen4 draft vocabulary")
            loan.validate_binding()
        for name, embedding, size, rank in (
            ("embed_tokens", True, load.attn_tp_size, load.attn_tp_rank),
            ("lm_head", False, load.lm_head_tp_size, load.lm_head_tp_rank),
        ):
            if loan is None:
                leaf = Qwen4VocabWeight(
                    plan.vocab_size,
                    plan.hidden_size,
                    tp_size=size,
                    tp_rank=rank,
                    embedding=embedding,
                    dtype=load.compute_dtype,
                    fp8_storage=plan.vocab_fp8,
                )
            else:
                src = getattr(loan.target, name)
                if (src.full_shape, src.axis, src.tp_size, src.tp_rank) != (
                    (plan.vocab_size, plan.hidden_size),
                    int(embedding),
                    size,
                    rank,
                ):
                    raise ValueError("Qwen4 scaled vocabulary shard layout differs")
                local_shape = [plan.vocab_size, plan.hidden_size]
                axis = int(embedding)
                if local_shape[axis] % size:
                    raise ValueError("Qwen4 vocabulary shard dimension must divide TP")
                local_shape[axis] //= size
                if tuple(src.weight.shape) != tuple(local_shape):
                    raise ValueError("Qwen4 scaled vocabulary shard shape differs")
                leaf = BorrowedVocab(loan.embedding if embedding else loan.head)
            setattr(self, name, leaf)
        eps = float(plan.text.get("rms_norm_eps", 1e-6))
        self.mtp_input_projection = Qwen4MTPInputOwner(
            plan.hidden_size,
            plan.hc_mult,
            norm_eps=eps,
            dtype=load.compute_dtype,
            fp8_storage=plan.dense_fp8,
            role=getattr(load.parallelism_config, "role_type", "PDFUSION"),
        )
        self.layers = nn.ModuleList([Qwen4DecoderWeights(plan, 0, load)])
        self.hyper_connection_mixer = Qwen4HyperConnection(
            plan.hidden_size,
            plan.hc_mult,
            plan.lowrank("hyper_connection_mixer"),
            with_inject=False,
            fp8_storage=plan.dense_fp8,
            norm_eps=eps,
            dtype=load.compute_dtype,
            role=getattr(load.parallelism_config, "role_type", "PDFUSION"),
        )
        self.config = config
        self._loans = frozenset(
            self.runtime_parameter_paths.values() if loan is not None else ()
        )

    def prepare_borrowed_parameters(self, bindings):
        if self._vocab_loan is not None or self.checkpoint_plan.vocab_fp8:
            raise ValueError(
                "Scaled draft vocabulary cannot use dense parameter bindings"
            )
        if set(bindings) != set(self.runtime_parameter_paths.values()):
            raise ValueError(
                "Qwen4 must borrow both runtime vocabulary matrices or neither"
            )
        owners = [b.owner for b in bindings.values()]
        if owners[0] is not owners[1]:
            raise ValueError("Qwen4 draft vocabulary requires one target owner")
        # The caller already validated the Qwen4 model config/quant recipe.
        # Check rank, axis, and full shape too: same-shaped local shards of an
        # incorrect TP rank would otherwise pass the generic shape validation.
        for name, binding in bindings.items():
            dst = self.get_submodule(name.rsplit(".", 1)[0])
            src = binding.owner.get_submodule(binding.source_path.rsplit(".", 1)[0])
            if not isinstance(src, Qwen4VocabWeight) or (
                src.axis,
                src.tp_size,
                src.tp_rank,
                src.full_shape,
            ) != (dst.axis, dst.tp_size, dst.tp_rank, dst.full_shape):
                raise ValueError(
                    f"Qwen4 draft/target vocabulary shard layout differs for {name}"
                )
        self._loans = frozenset(bindings)

    def checkpoint_weight_name_filter(self):
        return lambda name: (
            name.startswith("mtp.")
            or (
                name in ("model.language_model.embed_tokens.weight", "lm_head.weight")
                and self.WEIGHTS_MAPPER.map_name(name) not in self._loans
            )
        )

    def load_weights(self, weights):
        items = weights.items() if isinstance(weights, dict) else weights
        return super().load_weights(self.WEIGHTS_MAPPER.apply(items))

    def runtime_weight_view(self):
        if self._vocab_loan is not None:
            self._vocab_loan.validate_binding()
        exports = {
            "embedding": self.embed_tokens.weight,
            "lm_head": self.lm_head.weight,
        }
        if self.embed_tokens.storage_scale is not None:
            exports[EMBED_SCALE] = self.embed_tokens.storage_scale
        if self.lm_head.storage_scale is not None:
            exports[HEAD_SCALE] = self.lm_head.storage_scale
        return exports


def _bind_draft(owner, config, load):
    from rtp_llm.models_py.model_desc.module_base import GptModelBase
    from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpModel
    from rtp_llm.models_py.model_desc.qwen4_exp_runtime.mtp_runtime import (
        Qwen4ExpMTPRuntime,
    )
    from rtp_llm.models_py.modules import Embedding, MultimodalEmbeddingInjector
    from rtp_llm.models_py.modules.qwen4_exp.cpu_target_status import (
        prepare_cpu_target_status,
    )
    from rtp_llm.models_py.modules.qwen4_exp.tp_reduce import (
        install_target_graph_all_reduce,
        prepare_target_reduce,
    )
    from rtp_llm.models_py.new_models.qwen4_exp.runtime_bindings import bind_decoder

    pc = load.parallelism_config

    class DirectDraft(Qwen4ExpMTPRuntime, Qwen4ExpModel):
        def __init__(self):
            GptModelBase.__init__(
                self,
                config,
                pc,
                None,
                load.max_generate_batch_size,
                load.fmha_config,
                load.py_hw_kernel_config,
            )
            self.layers = nn.ModuleList(
                [bind_decoder(owner.layers[0], config, load, is_mtp_draft=True)]
            )
            if owner.embed_tokens.storage_scale is None:
                self.embed_tokens = Embedding(config, pc, owner.embed_tokens.weight)
            else:
                from rtp_llm.models_py.modules.base.common.fp8_embedding import (
                    Fp8VocabEmbedding,
                )

                self.embed_tokens = Fp8VocabEmbedding(
                    owner.embed_tokens.weight,
                    pc.get_attn_tp_size(),
                    owner.embed_tokens.weight.device,
                    weight_scale=owner.embed_tokens.storage_scale,
                )
                self.embed_tokens.prepare_byte_gather(
                    role=getattr(pc, "role_type", "PDFUSION")
                )
            self.multimodal_embedding_injector = MultimodalEmbeddingInjector()
            self.mtp_input_projection = owner.mtp_input_projection
            self.hc_mult = config.hc_mult
            self.hyper_connection_mixer = owner.hyper_connection_mixer.runtime
            self.ple_layers = nn.ModuleDict()
            self._ple_target_transaction = None
            self._ple_graph_target_transactions = {}
            self._capture_mtp_target_hidden = True
            self._cpu_target_status = prepare_cpu_target_status(config, pc)
            self._target_reduce = prepare_target_reduce(config, pc)
            if self._target_reduce is not None:
                install_target_graph_all_reduce(
                    self.layers[0].self_attn, self.layers[0].mlp, self._target_reduce
                )

    return DirectDraft()


class Qwen4ExpMTPForCausalLM(Qwen4MTPDraftWeights):
    """Direct Qwen4 draft service using the original target/draft cache runtime."""

    def __init__(self, config, load):
        from rtp_llm.config.qwen4_exp_config import (
            freeze_runtime_config,
            resolve_qwen4_parallelism,
            validate_qwen4_runtime,
        )

        if load.parallelism_config is None or not getattr(config, "is_mtp", False):
            raise ValueError("Qwen4 MTP requires resolved draft config and parallelism")
        pc = resolve_qwen4_parallelism(config, load.parallelism_config)
        validate_qwen4_runtime(config, pc)
        freeze_runtime_config(config, pc)
        load = replace(
            load,
            parallelism_config=pc,
            ffn_tp_size=pc.get_ffn_tp_size(),
            ffn_tp_rank=pc.get_ffn_tp_rank(),
        )
        if (
            config.num_layers != 1
            or config.enable_qwen4_ple
            or not config.enable_qwen4_qsa
        ):
            raise ValueError(
                "Qwen4 MTP draft requires one QSA layer without target PLE"
            )
        super().__init__(config, load)
        self.load_config = load
        object.__setattr__(self, "_runtime", None)

    def process_weights_after_loading(self):
        if self._vocab_loan is not None:
            self._vocab_loan.validate_binding()
        if self._runtime is None:
            for module in reversed(tuple(self.modules())):
                if module is not self:
                    hook = getattr(module, "process_weights_after_loading", None)
                    if hook is not None:
                        hook()
            runtime = _bind_draft(self, self.config, self.load_config)
            loans = {
                id(b.parameter())
                for b in self.__dict__.get(
                    "_rtp_borrowed_parameter_bindings", {}
                ).values()
            }
            for parameter in self.parameters():
                # Borrowed live target parameters must remain untouched.
                if id(parameter) not in loans:
                    parameter._rtp_requires_restart_for_update = True
            object.__setattr__(self, "_runtime", runtime)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            runtime = self.__dict__.get("_runtime")
            if runtime is None:
                raise
            return getattr(runtime, name)

    def forward(self, *args, **kwargs):
        if self._runtime is None:
            raise RuntimeError("Qwen4 MTP weights/runtime are not finalized")
        if self._vocab_loan is not None:
            self._vocab_loan.validate_binding()
        return self._runtime(*args, **kwargs)

    @property
    def kv_cache(self):
        runtime = self.__dict__.get("_runtime")
        return None if runtime is None else runtime.kv_cache

    @kv_cache.setter
    def kv_cache(self, value):
        if self._runtime is None:
            raise RuntimeError("Qwen4 draft runtime is not finalized")
        self._runtime.kv_cache = value

    def close(self):
        if self._runtime is not None:
            self._runtime.close()
