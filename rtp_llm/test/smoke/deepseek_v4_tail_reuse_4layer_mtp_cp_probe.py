"""Give CP2 target and MTP draft separate traces around the existing oracles."""

from rtp_llm.models_py.modules.dsv4 import _record_tensor as recorder
from rtp_llm.test.smoke import deepseek_v4_tail_reuse_4layer_mtp_probe as mtp_probe


def before_forward_layers(ctx):
    mtp_probe.before_forward_layers(ctx)
    mode = mtp_probe._mode(ctx)
    if mode == "target":
        recorder._DBG_CASE = "four_layer_cp2"
    elif mode == "draft":
        recorder._DBG_CASE = "mtp_draft_prefill_cp2"


def after_forward_layers(ctx):
    mtp_probe.after_forward_layers(ctx)


def before_forward_decode(ctx):
    mtp_probe.before_forward_decode(ctx)


def after_forward_decode(ctx):
    mtp_probe.after_forward_decode(ctx)
