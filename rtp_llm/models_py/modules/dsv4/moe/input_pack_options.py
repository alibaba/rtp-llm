"""Shared validation and route-masking rules for MegaMoE input packers."""

import torch


def validate_pack_options(
    tokens,
    valid_token_count=None,
    valid_token_mask=None,
    shared_input=None,
    shared_out=None,
):
    """Validate metadata before writes; keep standalone kernel loading supported."""
    if valid_token_count is not None and (
        not isinstance(valid_token_count, int)
        or isinstance(valid_token_count, bool)
        or not 0 <= valid_token_count <= tokens
    ):
        raise ValueError("valid_token_count is outside the local token shard")
    if valid_token_mask is not None and (
        valid_token_mask.ndim != 1 or valid_token_mask.numel() != tokens
    ):
        raise ValueError("valid_token_mask must contain one entry per local token")
    if (shared_input is None) != (shared_out is None):
        raise ValueError("shared_input and shared_out must be supplied together")
    if shared_input is not None and (
        shared_input.ndim != 2
        or shared_out.ndim != 2
        or shared_input.shape[0] != tokens
        or shared_out.shape[0] < tokens
        or shared_input.shape[1] != shared_out.shape[1]
        or shared_input.device != shared_out.device
        or shared_input.dtype != shared_out.dtype
    ):
        raise ValueError("shared input/output shape, dtype or device mismatch")


def mask_pack_routes(indices, weights, valid_token_count=None, valid_token_mask=None):
    # Packers validate metadata once, before masking or writing any output.
    valid = None
    if valid_token_mask is not None:
        valid = valid_token_mask.to(device=indices.device, dtype=torch.bool)
    if valid_token_count is not None:
        prefix = (
            torch.arange(indices.shape[0], device=indices.device) < valid_token_count
        )
        valid = prefix if valid is None else valid & prefix
    if valid is not None:
        indices = torch.where(valid[:, None], indices, 0)
        weights = torch.where(valid[:, None], weights, 0)
    return indices, weights
