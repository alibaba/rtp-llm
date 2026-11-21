"""Check that the weight reader and the existing V4 runtime use the same ranks."""

from rtp_llm.models_py.expert_placement import expert_placement_for_layer
from rtp_llm.models_py.layers.embedding import ParallelLMHead


def make_lm_head(args, dtype):
    # The legacy head bypasses splitting only when head TP, DP and EP are
    # all one. Under CP, head TP can differ from attention TP.
    return ParallelLMHead(
        args.vocab_size,
        args.dim,
        tp_size=args.lm_head_tp_size,
        tp_rank=args.lm_head_tp_rank,
        params_dtype=dtype,
        tp_pad8=max(args.lm_head_tp_size, args.dp_size, args.ep_size) > 1,
    )


def validate_v4_static_placement(args, model_config, load_config):
    # No routed FFN exists on a commit-only DSpARK worker.
    if args.commit_only:
        return
    if args.n_physical_experts not in (None, args.n_routed_experts):
        # Matches _CudaFp8Fp4Strategy.check_conditions. Reject before any
        # attention/expert parameter is allocated or checkpoint file staged.
        raise ValueError("V4 FP8_FP4 executor does not support redundant experts")
    if not getattr(model_config, "phy2log_path", None):
        return
    for layer in range(args.n_layers):
        placement = expert_placement_for_layer(
            model_config,
            layer_index=layer,
            logical_expert_count=args.n_routed_experts,
            ep_size=load_config.ep_size,
            ep_rank=load_config.ep_rank,
            parallelism_config=load_config.parallelism_config,
        )
        # The FP8_FP4 gate sends logical IDs directly to the executor, whose
        # packed slots are rank-contiguous. Allowing a nonidentity file here
        # can silently execute the wrong expert, even if every rank loads the
        # correct number of tensors and the placement adds no redundant slots.
        if not placement.is_identity:
            raise ValueError(
                f"V4 FP8_FP4 layer {layer} requires identity phy2log placement"
            )


def validate_load_groups(args, load_config):
    # CP reuses the physical TP group. The native runtime resolves attention
    # to TP=1 in that case: slicing the Q/O heads by physical TP here would
    # silently produce different checkpoint weights than the forward assumes.
    # Likewise the EP rank used by FST routing must own the same experts as
    # the executor, or a load can silently omit parameters.
    for label, runtime, reader in (
        (
            "attention",
            (args.tp_size, args.tp_rank),
            (load_config.attn_tp_size, load_config.attn_tp_rank),
        ),
        (
            "expert",
            (args.ep_size, args.ep_rank),
            (load_config.ep_size, load_config.ep_rank),
        ),
    ):
        if runtime != reader:
            raise ValueError(
                f"V4 {label} rank mismatch: runtime={runtime}, loader={reader}"
            )
