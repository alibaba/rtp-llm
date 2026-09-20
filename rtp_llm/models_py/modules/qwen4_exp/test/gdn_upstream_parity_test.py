"""Upstream parity for the GDN (linear-attention) path on a tiny K != V config.

Executes the upstream ``Qwen4ExpTextGatedDeltaNet`` (transformers main) against
our ``Qwen3NextGatedDeltaNet`` with identical weights and compares outputs.

Set ``QWEN4_EXP_UPSTREAM_SRC`` to transformers' ``modeling_qwen4_exp.py``; the
test skips when unset (CI-safe). The upstream class is exec'd rather than
imported because upstream main's config framework cannot be loaded by the
released transformers. The upstream file defines its own GQA-capable reference
kernels (``torch_chunk_gated_delta_rule`` etc.); those are exec'd together with
the classes. The vendored release's helpers are only a fallback for upstream
files that import them instead of defining them -- note the vendored 5.2.0
chunk rule assumes K == V heads and cannot run this test's GQA config.
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
_NUM_V = 6  # GQA ratio 3, mirroring the real config (16 k-heads / 48 v-heads)
_HEAD_K = 4
_HEAD_V = 4
_CONV_K = 4
_EPS = 1e-6

_UPSTREAM_CLASSES = ("Qwen4ExpTextRMSNormGated", "Qwen4ExpTextGatedDeltaNet")
_UPSTREAM_FUNCS = (
    "l2norm",
    "apply_mask_to_padding_states",
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
)


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
    """Fallback GDN helpers from whichever module this transformers release keeps them in.

    Only used for upstream files that import the kernels instead of defining
    them; the vendored release's ``torch_chunk_gated_delta_rule`` does not
    support K != V heads.
    """
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
            module = __import__(
                module_name, fromlist=list(names) + ["causal_conv1d_fn"]
            )
        except ModuleNotFoundError:
            continue
        if all(hasattr(module, name) for name in names):
            helpers = {name: _kwarg_filtered(getattr(module, name)) for name in names}
            conv = getattr(module, "causal_conv1d_fn", None)
            helpers["causal_conv1d_fn"] = (
                conv if conv is not None else _torch_causal_conv1d_fn
            )
            return helpers
    return {}


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


def _is_torchdynamo_exporting() -> bool:
    """Upstream helper absent from the vendored release; never exporting here."""
    try:
        return torch.compiler.is_exporting()
    except AttributeError:
        return False


def _upstream_activation_fn():
    from transformers.activations import ACT2FN

    return ACT2FN


def _exec_upstream_gdn():
    with open(_UPSTREAM_SRC, "r") as handle:
        tree = ast.parse(handle.read())
    body = []
    found = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in _UPSTREAM_CLASSES:
            node.decorator_list = (
                []
            )  # drop @use_kernelized_func / @use_accelerate_hooks
            body.append(node)
            found.add(node.name)
        elif isinstance(node, ast.FunctionDef) and node.name in _UPSTREAM_FUNCS:
            node.decorator_list = []  # drop @use_kernel_func_from_hub_with_fallback
            body.append(node)
            found.add(node.name)
    assert (
        set(_UPSTREAM_CLASSES) <= found
    ), f"upstream file lacks classes: {set(_UPSTREAM_CLASSES) - found}"
    missing = [name for name in _UPSTREAM_FUNCS if name not in found]
    namespace = {
        name: type(name, (_AnnotationPlaceholder,), {})
        for name in _annotation_only_names(body)
    }
    namespace.update(
        {
            "torch": torch,
            "nn": nn,
            "F": F,
            "math": math,
            "ACT2FN": _upstream_activation_fn(),
            "is_torchdynamo_exporting": _is_torchdynamo_exporting,
        }
    )
    if missing:
        fallback = _fla_helpers()
        for name in missing:
            if name not in fallback:
                raise unittest.SkipTest(
                    f"upstream defines no {name} and this transformers release "
                    "has no GDN helper fallback"
                )
            namespace[name] = fallback[name]
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
        from rtp_llm.utils.model_weight import (
            W,
            merge_ba_transpose_reorder,
            merge_qkvz_transpose_reorder,
        )

        # Use the loader's mapping functions so the checkpoint -> runtime
        # layout chain is itself under test ([q|k|v|z] and [b|a] row order).
        return {
            W.linear_attn_qkvz_w: merge_qkvz_transpose_reorder(
                [upstream.in_proj_qkv.weight, upstream.in_proj_z.weight]
            ).contiguous(),
            W.linear_attn_ba_w: merge_ba_transpose_reorder(
                [upstream.in_proj_b.weight, upstream.in_proj_a.weight]
            ).contiguous(),
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
            # Qwen4Exp-serving semantics: gate the output norm with the config's
            # output_gate_type ("sigmoid"), not the Qwen3-Next default silu.
            norm_activation="sigmoid",
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
            # The conv1d kernel always reads a per-sequence prefix-length vector
            # (engine passes zeros for an empty cache); None breaks its compile.
            prefix_lengths = torch.zeros(1, dtype=torch.int32, device="cuda")
            attn_inputs = SimpleNamespace(
                is_prefill=True,
                is_target_verify=False,
                cu_seqlens_device=cu_seqlens,
                input_lengths=torch.tensor([_SEQ], dtype=torch.int32),
                prefix_lengths=prefix_lengths,
                prefix_lengths_device=prefix_lengths,
                kv_cache_kernel_block_id_device=None,
                seq_size_per_block=None,
                cache_store_inputs=None,
                cache_store_writer=None,
            )
            attn_meta = Qwen3NextMetadata(
                prefill_conv1d_meta=prepare_causal_conv1d_metadata(
                    cu_seqlens, torch.device("cuda")
                )
            )
            # The runtime consumes flattened [T, H] tokens; upstream keeps [B, T, H].
            actual = ours(
                hidden_states=hidden.squeeze(0),
                fmha_impl=None,
                kv_cache=None,
                attention_inputs=attn_inputs,
                attn_meta=attn_meta,
            )

        self.assertEqual(
            tuple(actual.shape), tuple(expected.shape[1:]), "shape mismatch"
        )
        expected = expected.squeeze(0)
        diff = (actual.float() - expected.float()).abs().max().item()
        scale = expected.float().abs().max().item()
        self.assertLessEqual(
            diff,
            max(1e-3, 0.05 * scale),
            f"GDN parity mismatch: max|diff|={diff} (upstream max={scale})",
        )

    def test_decode_step_matches_upstream(self):
        """Single-token decode: our recurrent kernel vs the upstream torch rule.

        The prefill test already covers conv/gating/projection; this isolates the
        recurrent GQA kernel (a different kernel) with an explicit initial state,
        comparing both the output and the updated state. Upstream keeps the
        recurrent state K-first [B, H, K, V]; ours is V-first [B, H, V, K].
        """
        namespace = _exec_upstream_gdn()
        torch.manual_seed(0)
        upstream = (
            namespace["Qwen4ExpTextGatedDeltaNet"](_tiny_gdn_config(), 0)
            .to(torch.bfloat16)
            .cuda()
        )
        ours = self._our_gdn(upstream).to(torch.bfloat16).cuda().eval()

        from rtp_llm.models_py.triton_kernels.fla.fused_recurrent import (
            fused_recurrent_gated_delta_rule,
        )

        torch.manual_seed(7)
        batch, seq = 1, 1
        q = torch.randn(
            batch, seq, _NUM_K, _HEAD_K, device="cuda", dtype=torch.bfloat16
        )
        k = torch.randn(
            batch, seq, _NUM_K, _HEAD_K, device="cuda", dtype=torch.bfloat16
        )
        v = torch.randn(
            batch, seq, _NUM_V, _HEAD_V, device="cuda", dtype=torch.bfloat16
        )
        g = -torch.rand(batch, seq, _NUM_V, device="cuda", dtype=torch.bfloat16)
        beta = torch.rand(
            batch, seq, _NUM_V, device="cuda", dtype=torch.bfloat16
        ).sigmoid()
        initial_v_first = (
            torch.randn(
                batch, _NUM_V, _HEAD_V, _HEAD_K, device="cuda", dtype=torch.bfloat16
            )
            * 0.1
        )

        ratio = _NUM_V // _NUM_K
        with torch.no_grad():
            expected, expected_state = namespace["torch_recurrent_gated_delta_rule"](
                q.repeat_interleave(ratio, dim=2),
                k.repeat_interleave(ratio, dim=2),
                v,
                g=g,
                beta=beta,
                initial_state=initial_v_first.transpose(-1, -2).contiguous(),
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            actual, actual_state = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=None,
                initial_state=initial_v_first,
                inplace_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )

        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertEqual(tuple(actual_state.shape), tuple(expected_state.shape))
        out_diff = (actual.float() - expected.float()).abs().max().item()
        out_scale = expected.float().abs().max().item()
        state_diff = (
            (
                actual_state.float()
                - expected_state.transpose(-1, -2).contiguous().float()
            )
            .abs()
            .max()
            .item()
        )
        state_scale = expected_state.float().abs().max().item()
        self.assertLessEqual(
            out_diff,
            max(1e-3, 0.05 * out_scale),
            f"decode output mismatch: max|diff|={out_diff} (upstream max={out_scale})",
        )
        self.assertLessEqual(
            state_diff,
            max(1e-3, 0.05 * state_scale),
            f"decode state mismatch: max|diff|={state_diff} "
            f"(upstream max={state_scale})",
        )

    def test_stagewise_localization(self):
        """Diagnostic: report the first GDN stage that diverges from upstream.

        Replays both forwards stage by stage and logs max|diff| after each one:
        input projections -> conv1d -> g/beta gating -> chunk rule -> gated RMS
        norm -> output projection.
        """
        import logging

        namespace = _exec_upstream_gdn()
        torch.manual_seed(0)
        upstream = (
            namespace["Qwen4ExpTextGatedDeltaNet"](_tiny_gdn_config(), 0)
            .to(torch.bfloat16)
            .cuda()
        )
        ours = self._our_gdn(upstream).to(torch.bfloat16).cuda().eval()

        torch.manual_seed(1234)
        hidden = torch.randn(1, _SEQ, _HIDDEN, device="cuda", dtype=torch.bfloat16)

        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextMetadata
        from rtp_llm.models_py.triton_kernels.causal_conv1d.causal_conv1d import (
            prepare_causal_conv1d_metadata,
        )
        from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule
        from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating

        cu_seqlens = torch.tensor([0, _SEQ], dtype=torch.int32, device="cuda")
        prefix_lengths = torch.zeros(1, dtype=torch.int32, device="cuda")
        attn_inputs = SimpleNamespace(
            is_prefill=True,
            is_target_verify=False,
            cu_seqlens_device=cu_seqlens,
            input_lengths=torch.tensor([_SEQ], dtype=torch.int32),
            prefix_lengths=prefix_lengths,
            prefix_lengths_device=prefix_lengths,
            kv_cache_kernel_block_id_device=None,
            seq_size_per_block=None,
            cache_store_inputs=None,
            cache_store_writer=None,
        )
        attn_meta = Qwen3NextMetadata(
            prefill_conv1d_meta=prepare_causal_conv1d_metadata(
                cu_seqlens, torch.device("cuda")
            )
        )

        def report(tag, actual, expected):
            diff = (actual.float() - expected.float()).abs().max().item()
            scale = expected.float().abs().max().item()
            logging.warning(
                "[gdn-stage] %-22s max|diff|=%.6g ref_max=%.6g",
                tag,
                diff,
                scale,
            )

        hk, hv, dk, dv = _NUM_K, _NUM_V, _HEAD_K, _HEAD_V
        up = upstream
        with torch.no_grad():
            # ---- upstream stage by stage (mirrors its forward, no cache) ----
            up_qkv = up.in_proj_qkv(hidden)
            up_z = up.in_proj_z(hidden)
            up_b = up.in_proj_b(hidden)
            up_a = up.in_proj_a(hidden)
            up_conv = namespace["causal_conv1d_fn"](
                up_qkv.transpose(1, 2),
                up.conv1d.weight.squeeze(1),
                up.conv1d.bias,
                activation=up.activation,
            )[:, :, -_SEQ:]
            up_beta = up_b.sigmoid()
            up_g = -up.A_log.float().exp() * F.softplus(up_a.float() + up.dt_bias)
            up_q, up_k, up_v = torch.split(
                up_conv.transpose(1, 2),
                [hk * dk, hk * dk, hv * dv],
                dim=-1,
            )
            up_q = up_q.reshape(1, _SEQ, -1, dk)
            up_k = up_k.reshape(1, _SEQ, -1, dk)
            up_v = up_v.reshape(1, _SEQ, -1, dv)
            ratio = hv // hk
            up_q = up_q.repeat_interleave(ratio, dim=2)
            up_k = up_k.repeat_interleave(ratio, dim=2)
            up_core, _ = namespace["torch_chunk_gated_delta_rule"](
                up_q,
                up_k,
                up_v,
                g=up_g,
                beta=up_beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
            up_norm = up.norm(up_core.reshape(-1, dv), up_z.reshape(-1, dv)).reshape(
                1, _SEQ, -1
            )
            up_out = up.out_proj(up_norm).squeeze(0)

            # ---- ours stage by stage (mirrors production prefill, no cache) ----
            p_qkvz, p_ba = ours._input_project(hidden.squeeze(0))
            o_qkv, o_z, o_b, o_a = ours.fix_query_key_value_ordering(p_qkvz, p_ba)
            gdn = ours.prefill_gdn
            o_conv = gdn._conv1d(
                o_qkv, None, 1, attn_inputs, attn_meta.get_prefill_conv1d_meta()
            )
            o_g, o_beta = fused_gdn_gating(gdn.alog, o_a, o_b, gdn.dt_bias)
            o_q, o_k, o_v = torch.split(o_conv, [hk * dk, hk * dk, hv * dv], dim=-1)
            o_q = o_q.view(1, _SEQ, hk, dk)
            o_k = o_k.view(1, _SEQ, hk, dk)
            o_v = o_v.view(1, _SEQ, hv, dv)
            o_core, _, _ = chunk_gated_delta_rule(
                o_q,
                o_k,
                o_v,
                o_g,
                o_beta,
                initial_state=None,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
            o_core = o_core.squeeze(0)
            o_norm = ours.norm(o_core.reshape(-1, hv * dv), o_z)
            o_out = ours.out_proj(o_norm)

            report("in_proj_qkv", o_qkv, up_qkv.squeeze(0))
            report("in_proj_z", o_z, up_z.squeeze(0))
            report(
                "in_proj_ba",
                p_ba,
                torch.cat([up_b, up_a], dim=-1).squeeze(0),
            )
            report("conv1d", o_conv, up_conv.squeeze(0).transpose(0, 1))
            report("g", o_g, up_g.squeeze(0))
            report("beta", o_beta, up_beta.squeeze(0))
            report("chunk_rule", o_core, up_core.squeeze(0))
            report("gated_norm", o_norm, up_norm.squeeze(0))
            report("out_proj", o_out, up_out)

        out_diff = (o_out.float() - up_out.float()).abs().max().item()
        scale = up_out.float().abs().max().item()
        self.assertLessEqual(
            out_diff,
            max(1e-3, 0.05 * scale),
            "stagewise replay diverged at the final out_proj "
            "(see [gdn-stage] lines above for the first diverging stage)",
        )


if __name__ == "__main__":
    unittest.main()
