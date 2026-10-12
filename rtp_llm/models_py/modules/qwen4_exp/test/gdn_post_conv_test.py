import os
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rtp_llm.models_py.modules.qwen4_exp.gdn_post_conv_triton import try_gdn_post_conv
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated
from rtp_llm.models_py.triton_kernels.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)
from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating
from rtp_llm.models_py.triton_kernels.fla.utils import get_abs_err, get_err_ratio


def assert_close(prefix, ref, actual, ratio):
    # The generic FLA helper can downgrade errors in CI; this gate is always strict.
    assert torch.isfinite(ref).all() and torch.isfinite(actual).all(), prefix
    absolute = get_abs_err(ref, actual)
    error = get_err_ratio(ref, actual)
    assert absolute <= 1e-6 or error < ratio, f"{prefix}: abs={absolute}, ratio={error}"


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None, "NVIDIA CUDA is required"
)
class GdnPostConvTest(unittest.TestCase):
    def test_matches_gating_recurrence_norm_and_each_cached_token(self):
        torch.manual_seed(390)
        for batch, steps, h, hv in (
            (1, 2, 1, 2),
            (3, 2, 1, 2),
            (1, 3, 1, 2),
            (2, 4, 1, 2),
            (8, 4, 1, 2),
            (1, 2, 2, 6),
            (8, 4, 2, 6),
        ):
            for state_dtype in (torch.bfloat16, torch.float32):
                for activation in ("sigmoid", "silu"):
                    with self.subTest(
                        batch=batch,
                        steps=steps,
                        dtype=state_dtype,
                        activation=activation,
                    ):
                        for layout in ("fused", "split"):
                            with patch.dict(
                                os.environ,
                                {"RTP_LLM_QWEN4_GDN_POST_CONV_LAYOUT": layout},
                            ):
                                self._run_case(
                                    batch, steps, state_dtype, activation, h, hv
                                )

    def test_single_token_falls_back_without_touching_state(self):
        batch, heads, value_heads, dim = 3, 2, 6, 128
        qkv = torch.randn(
            batch, 1, heads * 2 + value_heads, dim, device="cuda", dtype=torch.bfloat16
        )
        q, k, v = torch.split(qkv, [heads, heads, value_heads], dim=2)
        a = torch.randn(batch, value_heads, device="cuda", dtype=torch.bfloat16)
        b = torch.randn_like(a)
        a_log = torch.randn(value_heads, device="cuda")
        dt = torch.randn_like(a_log)
        z = torch.randn(batch, value_heads * dim, device="cuda", dtype=torch.bfloat16)
        weight = torch.ones(dim, device="cuda", dtype=torch.bfloat16)
        state = torch.randn(1 + batch * 3, value_heads, dim, dim, device="cuda")
        before = state.clone()
        block_map = torch.arange(
            1, state.shape[0], device="cuda", dtype=torch.int32
        ).reshape(batch, 3)
        lengths = torch.full((batch,), 129, device="cuda", dtype=torch.int32)
        for layout in ("fused", "split"):
            with self.subTest(layout=layout), patch.dict(
                os.environ, {"RTP_LLM_QWEN4_GDN_POST_CONV_LAYOUT": layout}
            ):
                actual = try_gdn_post_conv(
                    q,
                    k,
                    v,
                    a,
                    b,
                    a_log,
                    dt,
                    z,
                    weight,
                    state,
                    block_map,
                    lengths,
                    page_size=128,
                    eps=1e-6,
                    activation="sigmoid",
                )
                self.assertIsNone(actual)
                self.assertTrue(torch.equal(state, before))

    def test_model_norm_route_matches_legacy_and_preserves_hooks(self):
        from rtp_llm.models_py.model_desc.qwen3_next import (
            Qwen3NextGatedDeltaNet,
            Qwen3NextGatedDeltaNetDecode,
        )
        from rtp_llm.models_py.modules.qwen4_exp import gdn_post_conv_triton

        torch.manual_seed(398)
        for steps, hook_kind in (
            (1, None),
            (3, None),
            (3, "module"),
            (3, "instance"),
            (3, "class"),
            (3, "global"),
        ):
            with self.subTest(steps=steps, hook=hook_kind):
                batch, heads, value_heads, dim = 3, 2, 6, 128
                qkv = torch.randn(
                    batch,
                    steps,
                    heads * 2 + value_heads,
                    dim,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                q, k, v = torch.split(qkv, [heads, heads, value_heads], dim=2)
                mixed = qkv.reshape(batch * steps, -1)
                a = torch.randn(
                    batch * steps, value_heads, device="cuda", dtype=torch.bfloat16
                )
                b = torch.randn_like(a)
                a_log = torch.randn(value_heads, device="cuda")
                dt = torch.randn_like(a_log)
                z = torch.randn(
                    batch * steps,
                    value_heads * dim,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
                initial = (
                    torch.randn(
                        1 + batch * (steps + 2), value_heads, dim, dim, device="cuda"
                    )
                    * 0.1
                )
                state = initial.clone()
                block_map = torch.arange(
                    1, initial.shape[0], device="cuda", dtype=torch.int32
                ).reshape(batch, steps + 2)
                lengths = torch.full((batch,), 129, device="cuda", dtype=torch.int32)
                norm = RmsNormGated(
                    weight, group_size=dim, eps=1e-6, activation="sigmoid"
                )
                inputs = SimpleNamespace(
                    is_prefill=steps > 1,
                    is_target_verify=steps > 1,
                    sequence_lengths_plus_1_device=lengths,
                )
                meta = SimpleNamespace(
                    is_target_verify=steps > 1, is_cp_linear_attn=False
                )
                cache = SimpleNamespace(kv_cache_base=state, seq_size_per_block=128)

                def recurrence(*args):
                    g, beta = fused_gdn_gating(a_log, a, b, dt)
                    output, _ = fused_recurrent_gated_delta_rule(
                        q,
                        k,
                        v,
                        g.view(batch, steps, value_heads),
                        beta.view(batch, steps, value_heads),
                        initial_state=state,
                        block_map=block_map,
                        seq_size_per_block=128,
                        sequence_lengths=lengths,
                        use_qk_l2norm_in_kernel=True,
                    )
                    return output.reshape(batch * steps, value_heads, dim)

                decoder = SimpleNamespace(
                    _is_target_verify=lambda *_: steps > 1,
                    _conv1d=lambda value, *_: value,
                    _get_bs_from_attenion_input=lambda *_: (batch, steps),
                    _get_ssm_states=lambda *_: state,
                    _get_fla_block_map=lambda *_: block_map,
                    _fla=recurrence,
                    head_k_dim=dim,
                    head_v_dim=dim,
                    local_num_k_heads=heads,
                    local_num_v_heads=value_heads,
                    alog=a_log,
                    dt_bias=dt,
                )
                owner = SimpleNamespace(
                    _qwen4_fused_post_conv=True,
                    _input_project=lambda *_: (None, None),
                    fix_query_key_value_ordering=lambda *_: (mixed, z, b, a),
                    decode_gdn=partial(Qwen3NextGatedDeltaNetDecode.forward, decoder),
                    norm=norm,
                    out_proj=torch.nn.Identity(),
                    parallelism_config=SimpleNamespace(get_attn_tp_size=lambda: 1),
                    local_num_v_heads=value_heads,
                    head_v_dim=dim,
                )
                calls = []
                original = gdn_post_conv_triton.try_gdn_post_conv

                def traced(*args, **kwargs):
                    result = original(*args, **kwargs)
                    calls.append(result is not None)
                    return result

                handles = []
                patcher = None
                hooked_calls = []
                if hook_kind == "module":
                    handles.append(
                        norm.register_forward_hook(lambda *args: hooked_calls.append(1))
                    )
                elif hook_kind == "global":
                    handles.append(
                        torch.nn.modules.module.register_module_forward_hook(
                            lambda module, args, result: (
                                hooked_calls.append(1) if module is norm else None
                            )
                        )
                    )
                elif hook_kind == "instance":
                    original_forward = norm.forward

                    def replaced(*args, **kwargs):
                        hooked_calls.append(1)
                        return original_forward(*args, **kwargs) + 0.25

                    patcher = patch.object(norm, "forward", replaced)
                elif hook_kind == "class":
                    original_forward = RmsNormGated.forward

                    def replaced(instance, *args, **kwargs):
                        hooked_calls.append(1)
                        return original_forward(instance, *args, **kwargs) + 0.25

                    patcher = patch.object(RmsNormGated, "forward", replaced)
                try:
                    if patcher is not None:
                        patcher.start()
                    values, states = [], []
                    with patch.object(
                        gdn_post_conv_triton, "try_gdn_post_conv", traced
                    ):
                        for enabled in ("0", "1"):
                            state.copy_(initial)
                            with patch.dict(
                                os.environ,
                                {
                                    "RTP_LLM_QWEN4_FUSED_GDN_POST_CONV": enabled,
                                    "RTP_LLM_QWEN4_GDN_POST_CONV_LAYOUT": "split",
                                },
                            ):
                                values.append(
                                    Qwen3NextGatedDeltaNet.forward(
                                        owner, mixed, None, cache, inputs, meta
                                    ).clone()
                                )
                            states.append(state.clone())
                    assert_close(
                        "model norm route output",
                        values[0].float(),
                        values[1].float(),
                        0.005,
                    )
                    assert_close("model norm route cache", states[0], states[1], 0.005)
                    self.assertEqual(calls, [steps > 1] if hook_kind is None else [])
                    if hook_kind is not None:
                        self.assertEqual(len(hooked_calls), 2)
                finally:
                    if patcher is not None:
                        patcher.stop()
                    for handle in handles:
                        handle.remove()

    def test_shared_norm_weight_at_real_tp8_head_counts(self):
        torch.manual_seed(391)
        for layout in ("fused", "split"):
            for steps in (2, 4):
                with self.subTest(layout=layout, steps=steps), patch.dict(
                    os.environ, {"RTP_LLM_QWEN4_GDN_POST_CONV_LAYOUT": layout}
                ):
                    self._run_case(
                        8, steps, torch.float32, "sigmoid", 2, 6, shared_weight=True
                    )

    def _run_case(
        self, batch, steps, state_dtype, activation, h, hv, shared_weight=False
    ):
        dim = 128
        # Non-contiguous QKV heads and BA row strides mirror the post-projection views.
        qkv = torch.randn(
            batch, steps, 2 * h + hv, dim, device="cuda", dtype=torch.bfloat16
        )
        q, k, v = torch.split(qkv, [h, h, hv], dim=2)
        ba = torch.randn(batch * steps, 2 * hv, device="cuda", dtype=torch.bfloat16)
        b, a = ba[:, :hv], ba[:, hv:]
        a_log = torch.randn(hv, device="cuda")
        dt = torch.randn(hv, device="cuda")
        z = torch.randn(batch * steps, hv * dim, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(
            dim if shared_weight else hv * dim, device="cuda", dtype=torch.bfloat16
        )
        pool_size = 1 + batch * (steps + 2)
        initial = (
            torch.randn(pool_size, hv, dim, dim, device="cuda", dtype=state_dtype) * 0.1
        )
        block_map = torch.arange(
            1, pool_size, device="cuda", dtype=torch.int32
        ).reshape(batch, steps + 2)
        lengths = torch.tensor(
            ([2, 127, 128, 129, 130, 256, 257, 0])[:batch],
            device="cuda",
            dtype=torch.int32,
        )
        norm = RmsNormGated(weight, group_size=dim, eps=1e-6, activation=activation)
        old_state, new_state = initial.clone(), initial.clone()

        def reference():
            g, beta = fused_gdn_gating(a_log, a, b, dt)
            out, _ = fused_recurrent_gated_delta_rule(
                q,
                k,
                v,
                g.view(batch, steps, hv),
                beta.view(batch, steps, hv),
                initial_state=old_state,
                block_map=block_map,
                seq_size_per_block=128,
                sequence_lengths=lengths,
                use_qk_l2norm_in_kernel=True,
            )
            return norm(out.reshape(batch * steps, hv * dim), z)

        def candidate():
            result = try_gdn_post_conv(
                q,
                k,
                v,
                a,
                b,
                a_log,
                dt,
                z,
                weight,
                new_state,
                block_map,
                lengths,
                page_size=128,
                eps=1e-6,
                activation=activation,
            )
            self.assertIsNotNone(result)
            return result

        actual, expected = candidate(), reference()
        valid = batch - 1 if batch == 8 else batch
        assert_close(
            "post-conv output",
            expected[: valid * steps].float(),
            actual[: valid * steps].float(),
            0.005,
        )
        assert_close("post-conv cache", old_state.float(), new_state.float(), 0.005)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = candidate()
        for i in range(2):
            qkv.copy_(torch.randn_like(qkv))
            ba.copy_(torch.randn_like(ba))
            z.copy_(torch.randn_like(z))
            old_state.copy_(initial)
            new_state.copy_(initial)
            if i == 1:
                block_map.copy_(block_map.flip(0))
                lengths[:valid].copy_(lengths[:valid].roll(1))
            graph.replay()
            expected = reference()
            assert_close(
                "post-conv replay output",
                expected[: valid * steps].float(),
                actual[: valid * steps].float(),
                0.005,
            )
            assert_close(
                "post-conv replay cache", old_state.float(), new_state.float(), 0.005
            )


if __name__ == "__main__":
    unittest.main()
