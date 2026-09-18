"""Upstream parity for the GDN (linear-attention) path on a tiny K != V config.

Executes the upstream ``Qwen4ExpTextGatedDeltaNet`` (transformers main) against
our ``Qwen3NextGatedDeltaNet`` with identical weights and compares outputs.

Set ``QWEN4_EXP_UPSTREAM_SRC`` to transformers' ``modeling_qwen4_exp.py``; the
test skips when unset (CI-safe). The upstream class is exec'd rather than
imported because upstream main's config framework cannot be loaded by the
released transformers; only the two leaf ``nn.Module`` definitions are taken and
their runtime dependencies are injected from ``transformers.integrations.fla``.
"""

import ast
import math
import os
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

_UPSTREAM_SRC = os.environ.get("QWEN4_EXP_UPSTREAM_SRC", "")

_HIDDEN = 32
_SEQ = 16
_NUM_K = 2
_NUM_V = 3  # deliberately K != V
_HEAD_K = 4
_HEAD_V = 4
_CONV_K = 4
_EPS = 1e-6


def _torch_causal_conv1d_fn(x, weight, bias=None, activation=None, **kwargs):
    """Torch stand-in for the kernelized depthwise causal conv1d.

    ``x`` is ``[B, C, T]``, ``weight`` is ``[C, K]`` (depthwise). The upstream
    file already sliced ``conv1d.weight`` to ``[C, K]``.
    """
    channels = x.shape[1]
    kernel = weight.shape[-1]
    padded = F.pad(x, (kernel - 1, 0))
    out = F.conv1d(padded, weight.unsqueeze(1), bias, groups=channels)
    if activation in ("silu", "swish"):
        out = F.silu(out)
    elif activation == "gelu":
        out = F.gelu(out)
    return out


def _kwarg_filtered(fn):
    """Drop kwargs the vendored transformers release does not accept yet."""
    import inspect

    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return fn
    allowed = set(parameters)

    def wrapper(*args, **kwargs):
        return fn(*args, **{k: v for k, v in kwargs.items() if k in allowed})

    return wrapper


def _fla_helpers():
    """The GDN reference helpers, wherever this transformers release keeps them."""
    names = (
        "apply_mask_to_padding_states",
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
    )
    for module_name in (
        "transformers.integrations.fla",
        "transformers.models.qwen3_next.modeling_qwen3_next",
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
    ):
        try:
            module = __import__(module_name, fromlist=list(names) + ["causal_conv1d_fn"])
        except ModuleNotFoundError:
            continue
        if all(hasattr(module, name) for name in names):
            helpers = {
                name: _kwarg_filtered(getattr(module, name)) for name in names
            }
            conv = getattr(module, "causal_conv1d_fn", None)
            helpers["causal_conv1d_fn"] = conv if conv is not None else _torch_causal_conv1d_fn
            return helpers
    raise unittest.SkipTest("transformers release lacks the GDN reference helpers")


def _annotation_only_names(body) -> set:
    """Names referenced solely by annotations, which exec evaluates eagerly."""
    import builtins

    names = set()
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        annotations = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotations.append(node.returns)
            arguments = node.args
            annotations += [a.annotation for a in arguments.args]
            annotations += [a.annotation for a in arguments.kwonlyargs]
            annotations += [a.annotation for a in arguments.posonlyargs]
            if arguments.vararg is not None:
                annotations.append(arguments.vararg.annotation)
            if arguments.kwarg is not None:
                annotations.append(arguments.kwarg.annotation)
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
        for annotation in annotations:
            if annotation is None:
                continue
            for sub in ast.walk(annotation):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
    return {
        name
        for name in names
        if name not in ("torch", "nn", "F", "math") and not hasattr(builtins, name)
    }


class _AnnotationPlaceholder:
    """Stand-in for upstream annotation-only types; supports ``X[...]``."""

    def __class_getitem__(cls, item):
        return cls


def _exec_upstream_gdn():
    helpers = _fla_helpers()

    with open(_UPSTREAM_SRC, "r") as handle:
        tree = ast.parse(handle.read())
    wanted = {"Qwen4ExpTextRMSNormGated", "Qwen4ExpTextGatedDeltaNet"}
    body = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in wanted:
            node.decorator_list = []  # drop @use_kernelized_func / @use_kernels
            body.append(node)
    assert len(body) == 2, f"expected 2 upstream classes, got {[n.name for n in body]}"
    namespace = {
        name: type(name, (_AnnotationPlaceholder,), {})
        for name in _annotation_only_names(body)
    }
    namespace.update({"torch": torch, "nn": nn, "F": F, "math": math, **helpers})
    module = ast.Module(body=body, type_ignores=[])
    for node in ast.walk(module):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = []
    exec(compile(module, "<upstream>", "exec"), namespace)
    return namespace


def _tiny_gdn_config():
    return SimpleNamespace(
        hidden_size=_HIDDEN,
        linear_num_key_heads=_NUM_K,
        linear_num_value_heads=_NUM_V,
        linear_key_head_dim=_HEAD_K,
        linear_value_head_dim=_HEAD_V,
        linear_conv_kernel_dim=_CONV_K,
        hidden_act="silu",
        output_gate_type="sigmoid",
        rms_norm_eps=_EPS,
        layer_types=["linear_attention"],
    )


def _seed_upstream_gdn(config, seed=0):
    namespace = _exec_upstream_gdn()
    torch.manual_seed(seed)
    gdn = namespace["Qwen4ExpTextGatedDeltaNet"](config, 0)
    return gdn


@unittest.skipUnless(
    os.path.isfile(_UPSTREAM_SRC), "QWEN4_EXP_UPSTREAM_SRC not set to a file"
)
class GdnUpstreamParityTest(unittest.TestCase):
    def _weights_from_upstream(self, upstream) -> dict:
        from rtp_llm.utils.model_weight import W, merge_ba_transpose_reorder

        return {
            W.linear_attn_qkvz_w: torch.cat(
                [upstream.in_proj_qkv.weight, upstream.in_proj_z.weight], dim=0
            ).T.contiguous(),
            W.linear_attn_ba_w: torch.cat(
                [upstream.in_proj_b.weight, upstream.in_proj_a.weight], dim=0
            ).T.contiguous(),
            W.linear_attn_conv1d_w: upstream.conv1d.weight.detach().clone(),
            W.linear_attn_dt_b: upstream.dt_bias.detach().clone(),
            W.linear_attn_alog: upstream.A_log.detach().clone(),
            W.linear_attn_norm_w: upstream.norm.weight.detach().clone(),
            W.linear_attn_out_w: upstream.out_proj.weight.detach().T.contiguous(),
        }

    def _our_gdn(self, upstream):
        from rtp_llm.config.model_config import ModelConfig
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextGatedDeltaNet
        from rtp_llm.ops import ParallelismConfig

        config = ModelConfig()
        config.hidden_size = _HIDDEN
        config.layernorm_eps = _EPS
        linear_attn_config = config.linear_attention_config
        linear_attn_config.linear_num_key_heads = _NUM_K
        linear_attn_config.linear_num_value_heads = _NUM_V
        linear_attn_config.linear_key_head_dim = _HEAD_K
        linear_attn_config.linear_value_head_dim = _HEAD_V
        linear_attn_config.linear_conv_kernel_dim = _CONV_K
        parallelism = ParallelismConfig()
        parallelism.tp_size = 1
        parallelism.dp_size = 1
        parallelism.ep_size = 1
        parallelism.world_size = 1
        return Qwen3NextGatedDeltaNet(
            linear_attn_config,
            parallelism,
            self._weights_from_upstream(upstream),
            _EPS,
        )

    def test_tiny_kgneqv_prefill_matches_upstream(self):
        upstream = _seed_upstream_gdn(_tiny_gdn_config()).to(torch.bfloat16).cuda()
        ours = self._our_gdn(upstream).to(torch.bfloat16).cuda()
        ours = ours.eval()
        for param in ours.parameters():
            param.requires_grad_(False)

        torch.manual_seed(1234)
        hidden = torch.randn(1, _SEQ, _HIDDEN, device="cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            expected = upstream(hidden)

            from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextMetadata
            from rtp_llm.models_py.triton_kernels.causal_conv1d.causal_conv1d import (
                prepare_causal_conv1d_metadata,
            )

            cu_seqlens = torch.tensor([0, _SEQ], dtype=torch.int32, device="cuda")
            attn_inputs = SimpleNamespace(
                is_prefill=True,
                is_target_verify=False,
                cu_seqlens_device=cu_seqlens,
                prefix_lengths=None,
                kv_cache_kernel_block_id_device=None,
                seq_size_per_block=None,
            )
            attn_meta = Qwen3NextMetadata(
                prefill_conv1d_meta=prepare_causal_conv1d_metadata(
                    cu_seqlens, torch.device("cuda")
                )
            )
            actual = ours(
                hidden_states=hidden,
                fmha_impl=None,
                kv_cache=None,
                attention_inputs=attn_inputs,
                attn_meta=attn_meta,
            )

        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        diff = (actual.float() - expected.float()).abs().max().item()
        scale = expected.float().abs().max().item()
        self.assertLessEqual(
            diff,
            max(1e-3, 0.05 * scale),
            f"GDN parity mismatch: max|diff|={diff} (upstream max={scale})",
        )


if __name__ == "__main__":
    unittest.main()
