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


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None, "NVIDIA CUDA is required"
)
class GdnDecodeConvGateSM90Test(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("The paged Conv1D/gating fusion is qualified on SM90")
        torch.manual_seed(159)

    @staticmethod
    def make_pair(tp=8):
        from rtp_llm.config.model_config import ModelConfig
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextGatedDeltaNet
        from rtp_llm.ops import DataType, ParallelismConfig
        from rtp_llm.utils.model_weight import W

        config = ModelConfig()
        config.hidden_size = 2560
        linear = config.linear_attention_config
        linear.linear_num_key_heads = 16
        linear.linear_num_value_heads = 48
        linear.linear_key_head_dim = linear.linear_value_head_dim = 128
        linear.linear_conv_kernel_dim = 4
        linear.ssm_state_dtype = DataType.TYPE_FP32
        linear.conv_state_dtype = DataType.TYPE_BF16
        parallelism = ParallelismConfig()
        parallelism.tp_size = parallelism.world_size = tp
        parallelism.dp_size = parallelism.ep_size = 1
        assert parallelism.get_attn_tp_size() == tp

        def weight(*shape, scale=1.0, dtype=torch.bfloat16):
            return torch.randn(*shape, device="cuda", dtype=dtype) * scale

        weights = {
            W.linear_attn_qkvz_w: weight(2560, 16384 // tp, scale=2560**-0.5),
            W.linear_attn_ba_w: weight(2560, 96 // tp, scale=2560**-0.5),
            W.linear_attn_conv1d_w: weight(10240 // tp, 1, 4, scale=0.5),
            W.linear_attn_dt_b: weight(48 // tp, scale=0.2, dtype=torch.float32),
            W.linear_attn_alog: weight(48 // tp, scale=0.3, dtype=torch.float32),
            W.linear_attn_norm_w: 1 + weight(128, scale=0.1),
            W.linear_attn_out_w: weight(6144 // tp, 2560, scale=(6144 // tp) ** -0.5),
        }
        layers = []
        for _ in range(2):
            layer = Qwen3NextGatedDeltaNet(
                linear, parallelism, dict(weights), 1e-6, norm_activation="sigmoid"
            ).eval()
            # The check ends at the rank-local output, before TP reduction.
            layer.tp_all_reduce = lambda x, group: x
            layers.append(layer)
        for layer in layers:
            assert tuple(layer.in_proj_fused.weight.shape) == (16480 // tp, 2560)
            assert callable(layer.decode_gdn._decode_conv_gate)
        # Isolate Conv1D/gating here; the default packed recurrence has its
        # own paged-state and Graph coverage in gdn_sm90_candidates_test.
        for layer in layers:
            layer.decode_gdn._packed_decode = None
        layers[0].decode_gdn._decode_conv_gate = None
        return tuple(layers)

    @staticmethod
    def make_decode_case(layer, batch, tokens=1):
        converter = layer.decode_gdn.linear_cache_converter
        pages = batch * 4 + 2
        packed = torch.full(
            (pages, converter.block_size_bytes + 128),
            -91,
            dtype=torch.int8,
            device="cuda",
        )
        for view in (
            converter.get_ssm_state_tensor(packed),
            converter.get_conv_state_tensor(packed),
        ):
            view.copy_(torch.randn_like(view) * 0.05)
        mapping = torch.zeros(batch, 7, dtype=torch.int32, device="cuda")[:, :4]
        mapping.copy_(
            torch.randperm(pages - 2, device="cuda", dtype=torch.int32).view(batch, 4)
            + 1
        )
        lengths = torch.tensor(
            [65 if i % 2 == 0 else 19 for i in range(batch)],
            dtype=torch.int32,
            device="cuda",
        )
        verify = tokens > 1
        inputs = SimpleNamespace(
            is_prefill=verify,
            is_target_verify=verify,
            is_cuda_graph=False,
            kv_cache_kernel_block_id_device=mapping,
            sequence_lengths_plus_1_device=lengths,
            prefix_lengths=torch.ones(batch, dtype=torch.int32),
        )
        meta = SimpleNamespace(is_target_verify=verify, is_cp_linear_attn=False)
        hidden = torch.randn(batch * tokens, 2560, dtype=torch.bfloat16, device="cuda")
        return hidden, packed, inputs, meta

    @staticmethod
    def run_decode(layer, hidden, packed, inputs, meta):
        cache = SimpleNamespace(kv_cache_base=packed, seq_size_per_block=64)
        return layer(hidden, None, cache, inputs, meta)

    def assert_exact_decode(self, expected, actual, ref_state, state, initial):
        self.assertTrue(torch.isfinite(expected).all().item())
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(ref_state, state))
        self.assertTrue(torch.equal(state[0], initial[0]))
        self.assertTrue(torch.equal(state[-1], initial[-1]))

    @torch.inference_mode()
    def test_real_decode_bitwise_and_graph_page_replay(self):
        self.check_decode_bitwise_and_graph_page_replay(8)

    @torch.inference_mode()
    def test_tp1_decode_bitwise_and_graph_page_replay(self):
        self.check_decode_bitwise_and_graph_page_replay(1)

    @torch.inference_mode()
    def test_tp4_decode_bitwise_and_graph_page_replay(self):
        self.check_decode_bitwise_and_graph_page_replay(4)

    def check_decode_bitwise_and_graph_page_replay(self, tp):
        old, new = self.make_pair(tp)
        with patch.object(
            new.decode_gdn,
            "_conv1d",
            side_effect=AssertionError("Eligible call fell back"),
        ):
            for batch in (1, 16, 32, 64):
                with self.subTest(batch=batch):
                    hidden, initial, inputs, meta = self.make_decode_case(new, batch)
                    original, fused = initial.clone(), initial.clone()
                    expected = self.run_decode(old, hidden, original, inputs, meta)
                    actual = self.run_decode(new, hidden, fused, inputs, meta)
                    self.assert_exact_decode(expected, actual, original, fused, initial)

            hidden, initial, inputs, meta = self.make_decode_case(new, 17)
            original, fused = initial.clone(), initial.clone()
            inputs.is_cuda_graph = True
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                self.run_decode(new, hidden, fused, inputs, meta)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=side):
                actual = self.run_decode(new, hidden, fused, inputs, meta)
            address = actual.data_ptr()
            for step in range(2):
                hidden.mul_(1.125)
                mapping = inputs.kv_cache_kernel_block_id_device
                mapping[:-1].copy_(mapping[:-1].roll(1, dims=0))
                lengths = inputs.sequence_lengths_plus_1_device
                lengths[:-1].add_(step)
                mapping[-1].zero_()
                lengths[-1] = 0
                original.copy_(initial)
                fused.copy_(initial)
                expected = self.run_decode(old, hidden, original, inputs, meta)
                graph.replay()
                self.assertEqual(actual.data_ptr(), address)
                self.assert_exact_decode(expected, actual, original, fused, initial)

    @torch.inference_mode()
    def test_unpadded_decode_bitwise_across_context_boundary(self):
        old, new = self.make_pair()
        cases = ((1, 8192), (1, 32768), (16, 2048), (32, 2048), (64, 2048))
        with patch.object(
            new.decode_gdn,
            "_conv1d",
            side_effect=AssertionError("Unpadded S=1 decode fell back"),
        ):
            for batch, prefix in cases:
                with self.subTest(batch=batch, prefix=prefix):
                    hidden, initial, inputs, meta = self.make_decode_case(new, batch)
                    for projected in new._input_project(hidden):
                        self.assertEqual(projected.stride(), (2060, 1))

                    page = prefix // 64
                    short_map = inputs.kv_cache_kernel_block_id_device
                    long_map = short_map.new_zeros((batch, page + 4))[:, : page + 1]
                    # Only the final prefix page and its successor are visited.
                    long_map[:, page - 1 : page + 1] = short_map[:, :2]
                    inputs.kv_cache_kernel_block_id_device = long_map
                    lengths = inputs.sequence_lengths_plus_1_device
                    lengths.fill_(prefix + 1)
                    original, fused = initial.clone(), initial.clone()

                    # Carry the actual committed state across the boundary and
                    # two later tokens rather than resetting it each step.
                    for _ in range(3):
                        expected = self.run_decode(old, hidden, original, inputs, meta)
                        actual = self.run_decode(new, hidden, fused, inputs, meta)
                        self.assert_exact_decode(
                            expected, actual, original, fused, initial
                        )
                        hidden.normal_()
                        lengths.add_(1)

    @torch.inference_mode()
    def test_unpadded_decode_graph_refreshes_paged_state(self):
        self.check_unpadded_decode_graph(
            8,
            (
                (1, (8192, 32768)),
                (16, (2048, 2048)),
                (32, (2048, 2048)),
                (64, (2048, 2048)),
            ),
        )

    @torch.inference_mode()
    def test_tp2_decode_graph_refreshes_paged_state(self):
        self.check_unpadded_decode_graph(
            2, tuple((b, (2048, 2048)) for b in (16, 32, 64))
        )

    @torch.inference_mode()
    def test_tp1_decode_graph_refreshes_paged_state(self):
        self.check_unpadded_decode_graph(
            1,
            (
                (1, (8192, 32768)),
                (16, (2048, 2048)),
                (32, (2048, 2048)),
                (64, (2048, 2048)),
            ),
        )

    @torch.inference_mode()
    def test_tp4_decode_graph_refreshes_paged_state(self):
        self.check_unpadded_decode_graph(
            4,
            (
                (1, (8192, 32768)),
                (16, (2048, 2048)),
                (32, (2048, 2048)),
                (64, (2048, 2048)),
            ),
        )

    def check_unpadded_decode_graph(self, tp, cases):
        old, new = self.make_pair(tp=tp)
        with patch.object(
            new.decode_gdn,
            "_conv1d",
            side_effect=AssertionError("Unpadded S=1 Graph fell back"),
        ):
            for batch, prefixes in cases:
                with self.subTest(batch=batch):
                    hidden, initial, inputs, meta = self.make_decode_case(new, batch)
                    for projected in new._input_project(hidden):
                        self.assertEqual(projected.stride(), (16480 // tp, 1))

                    physical_pairs = inputs.kv_cache_kernel_block_id_device
                    columns = max(prefixes) // 64 + 1
                    table = physical_pairs.new_zeros((batch, columns + 3))[:, :columns]
                    table[:, :2].copy_(physical_pairs[:, :2])
                    lengths = inputs.sequence_lengths_plus_1_device
                    lengths.fill_(65)
                    inputs.kv_cache_kernel_block_id_device = table
                    # The reference uses the full, same live page table in
                    # eager mode, independently of the Graph FLA table view.
                    reference_inputs = SimpleNamespace(**vars(inputs))
                    inputs.is_cuda_graph = True
                    original, fused = initial.clone(), initial.clone()

                    side = torch.cuda.Stream()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        self.run_decode(new, hidden, fused, inputs, meta)
                    torch.cuda.current_stream().wait_stream(side)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=side):
                        actual = self.run_decode(new, hidden, fused, inputs, meta)
                    address = actual.data_ptr()
                    fused.copy_(initial)

                    for epoch, prefix in enumerate(prefixes):
                        page = prefix // 64
                        table.zero_()
                        table[:, page - 1 : page + 1].copy_(
                            physical_pairs[:, 2 * epoch : 2 * epoch + 2]
                        )
                        lengths.fill_(prefix + 1)
                        # New live request/page pair, then advance three
                        # tokens without resetting the recurrent state.
                        for step in range(3):
                            with self.subTest(prefix=prefix, epoch=epoch, step=step):
                                hidden.normal_()
                                expected = self.run_decode(
                                    old, hidden, original, reference_inputs, meta
                                )
                                graph.replay()
                                self.assertEqual(actual.data_ptr(), address)
                                self.assert_exact_decode(
                                    expected, actual, original, fused, initial
                                )
                                lengths.add_(1)

    @torch.inference_mode()
    def test_target_verify_uses_original_multi_token_path(self):
        for tp in (1, 2, 4, 8):
            with self.subTest(tp=tp):
                old, new = self.make_pair(tp)
                hidden, initial, inputs, meta = self.make_decode_case(new, 3, tokens=2)
                original, fused = initial.clone(), initial.clone()
                expected = self.run_decode(old, hidden, original, inputs, meta)
                with patch.object(
                    new.decode_gdn,
                    "_decode_conv_gate",
                    side_effect=AssertionError("S>1 invoked the S=1 fusion"),
                ):
                    actual = self.run_decode(new, hidden, fused, inputs, meta)
                self.assert_exact_decode(expected, actual, original, fused, initial)

    @torch.inference_mode()
    def test_conv_gate_rejects_unsupported_inputs_without_state_writes(self):
        from rtp_llm.models_py.modules.qwen4_exp.gdn_decode_conv_gate_triton import (
            try_decode_conv_gate,
        )

        for tp in (1, 2, 4, 8):
            with self.subTest(tp=tp):
                batch, dim, heads = 2, 10240 // tp, 48 // tp

                def bf16(*shape):
                    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

                args = dict(
                    x=bf16(batch, dim),
                    weight=bf16(dim, 4),
                    state=bf16(5, 3, dim),
                    block_map=torch.tensor(
                        [[1, 2], [3, 4]], device="cuda", dtype=torch.int32
                    ),
                    lengths=torch.full((batch,), 65, device="cuda", dtype=torch.int32),
                    a=bf16(batch, heads),
                    b=bf16(batch, heads),
                    a_log=torch.zeros(heads, device="cuda"),
                    dt_bias=torch.zeros(heads, device="cuda"),
                    page=64,
                )
                self.assertIsNotNone(try_decode_conv_gate(**args))
                snapshot = args["state"].clone()
                # These are structural checks only: no host read of live
                # lengths/page IDs is introduced into the production wrapper.
                rejected = (
                    dict(page=32),
                    dict(x=bf16(0, dim)),
                    dict(x=bf16(129, dim)),
                    dict(x=bf16(batch, 640), a=bf16(batch, 3)),
                    dict(x=bf16(batch, dim * 2)[:, ::2]),
                    dict(weight=args["weight"].t().contiguous().t()),
                    dict(state=args["state"].float()),
                    dict(a=bf16(batch, heads * 2)[:, ::2]),
                    dict(a_log=args["a_log"].half()),
                    dict(lengths=args["lengths"].long()),
                    dict(
                        lengths=torch.ones(batch * 2, device="cuda", dtype=torch.int32)[
                            ::2
                        ]
                    ),
                    dict(block_map=None),
                    dict(block_map=args["block_map"][:, ::2]),
                )
                for override in rejected:
                    with self.subTest(rejected=tuple(override)):
                        candidate = {**args, **override}
                        state_before = candidate["state"].clone()
                        self.assertIsNone(try_decode_conv_gate(**candidate))
                        self.assertTrue(torch.equal(candidate["state"], state_before))
                        self.assertTrue(torch.equal(args["state"], snapshot))
                for capability in ((8, 0), (10, 0), (12, 0)):
                    with patch.object(
                        torch.cuda, "get_device_capability", return_value=capability
                    ):
                        self.assertIsNone(try_decode_conv_gate(**args))
                    self.assertTrue(torch.equal(args["state"], snapshot))


if __name__ == "__main__":
    unittest.main()
