"""Opt-in SM residency margin for Blackwell MegaMoE FP8 launches."""

import os
from contextlib import contextmanager

import torch


@contextmanager
def configure_mega_moe_fp8_num_sms(deep_gemm, device):
    """Use the same two-SM margin for FP8 buffer allocation and launch.

    Set MEGA_MOE_FP8_RESERVE_SM=1 before starting the prefill process.
    Unset or 0 preserves the existing launch configuration.
    """
    if os.environ.get("MEGA_MOE_FP8_RESERVE_SM", "0") != "1":
        yield
        return

    properties = torch.cuda.get_device_properties(device)
    if properties.major < 10:
        yield
        return

    current_num_sms = deep_gemm.get_num_sms()
    # Use physical SMs to avoid compounding an outer context's reservation.
    # Never increase that context's budget; two-CTA clusters need an even grid.
    target_num_sms = min(max(2, properties.multi_processor_count - 2), current_num_sms)
    target_num_sms -= target_num_sms % 2
    if target_num_sms == current_num_sms:
        yield
        return

    deep_gemm.set_num_sms(target_num_sms)
    try:
        yield
    finally:
        deep_gemm.set_num_sms(current_num_sms)
