import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch

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
class GdnCompositionTest(unittest.TestCase):
    def test_composed_decode_preserves_state_and_norm_hooks(self):
        from rtp_llm.models_py.model_desc.qwen3_next import (
            Qwen3NextGatedDeltaNet,
            Qwen3NextGatedDeltaNetDecode,
        )

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
                    _input_project=lambda *_: (None, None),
                    fix_query_key_value_ordering=lambda *_: (mixed, z, b, a),
                    decode_gdn=partial(Qwen3NextGatedDeltaNetDecode.forward, decoder),
                    norm=norm,
                    out_proj=torch.nn.Identity(),
                    parallelism_config=SimpleNamespace(get_attn_tp_size=lambda: 1),
                    local_num_v_heads=value_heads,
                    head_v_dim=dim,
                )
                owner._output_project = partial(
                    Qwen3NextGatedDeltaNet._output_project, owner
                )
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
                # Explicit conv/FLA/norm composition is the reference for the
                # model wiring; it does not invoke either production forward.
                expected = norm(
                    recurrence().reshape(batch * steps, value_heads * dim), z
                ).clone()
                expected_state = state.clone()
                if hook_kind in ("instance", "class"):
                    expected += 0.25
                hooked_calls.clear()
                try:
                    if patcher is not None:
                        patcher.start()
                    state.copy_(initial)
                    actual = Qwen3NextGatedDeltaNet.forward(
                        owner, mixed, None, cache, inputs, meta
                    )
                    assert_close(
                        "composed output", expected.float(), actual.float(), 0.005
                    )
                    assert_close("composed cache", expected_state, state, 0.005)
                    if hook_kind is not None:
                        self.assertEqual(len(hooked_calls), 1)
                finally:
                    if patcher is not None:
                        patcher.stop()
                    for handle in handles:
                        handle.remove()


if __name__ == "__main__":
    unittest.main()
