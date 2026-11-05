"""Give CP2 target and MTP draft separate traces around the existing oracles."""

import os
from pathlib import Path

from rtp_llm.models_py.modules.dsv4 import _record_tensor as recorder
from rtp_llm.test.smoke import deepseek_v4_tail_reuse_4layer_mtp_probe as mtp_probe


_draft_private_cache = None


def before_forward_layers(ctx):
    mtp_probe.before_forward_layers(ctx)
    mode = mtp_probe._mode(ctx)
    if mode == "target":
        recorder._DBG_CASE = "four_layer_cp2"
    elif mode == "draft":
        recorder._DBG_CASE = "mtp_draft_prefill_cp2"
        # The shared tail oracle captures resumed inputs only. Private A also
        # needs its original target pre-HC activation before draft fusion.
        if (
            os.environ.get("T10_PRIVATE_FORK_REFERENCE") == "1"
            and os.environ.get("T07_CACHE_PROBE_ROLE") == "prefill"
            and int(ctx.kwargs["attn_inputs"].prefix_lengths.reshape(-1)[0]) == 0
        ):
            model = ctx.kwargs["prepare_hidden_fn"].__self__
            inputs = model._cur_inputs.input_hiddens
            flat = inputs.reshape(-1, inputs.size(-1))
            rows = int(ctx.args[2].numel())
            if flat.size(0) < rows:
                raise AssertionError("cold draft pre-HC input is incomplete")
            mtp_probe._pending.draft_prefill_input = (
                flat,
                flat[:rows].detach().clone(),
                rows,
            )
            mtp_probe._pending.draft_prefill_runtime = mtp_probe._remember_runtime(
                ctx.args[0]
            )


def after_forward_layers(ctx):
    global _draft_private_cache
    if (
        mtp_probe._mode(ctx) != "draft"
        or os.environ.get("T10_PRIVATE_FORK_REFERENCE") != "1"
        or os.environ.get("T07_CACHE_PROBE_ROLE") != "prefill"
    ):
        mtp_probe.after_forward_layers(ctx)
        return
    (Path(os.environ["T07_CACHE_PROBE_DIR"]) / "mtp_draft_prefill").mkdir(
        parents=True, exist_ok=True
    )
    probe = mtp_probe.target_probe
    target_cache = probe._private_fork_cache
    original = probe.after_forward_layers

    def private_draft(inner_ctx):
        # Shared MTP code handles source maps, pre-HC restoration and output
        # directory isolation. Enable its existing private-cache algorithm
        # solely inside the CP2 draft call; this is a same-input draft oracle.
        previous = os.environ.get("T10_PRIVATE_FORK_REFERENCE")
        os.environ["T10_PRIVATE_FORK_REFERENCE"] = "1"
        try:
            return original(inner_ctx)
        finally:
            if previous is None:
                os.environ.pop("T10_PRIVATE_FORK_REFERENCE", None)
            else:
                os.environ["T10_PRIVATE_FORK_REFERENCE"] = previous

    probe._private_fork_cache = _draft_private_cache
    probe.after_forward_layers = private_draft
    try:
        mtp_probe.after_forward_layers(ctx)
    finally:
        _draft_private_cache = probe._private_fork_cache
        probe._private_fork_cache = target_cache
        probe.after_forward_layers = original


def before_forward_decode(ctx):
    mtp_probe.before_forward_decode(ctx)


def after_forward_decode(ctx):
    mtp_probe.after_forward_decode(ctx)
