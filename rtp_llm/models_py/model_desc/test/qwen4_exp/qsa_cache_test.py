import json
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main, skipUnless
from unittest.mock import Mock, patch

import torch
from torch import nn

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_ROPE_POS_TAG,
)
from rtp_llm.models_py.model_desc.module_base import AttnImplFactory
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_cache_views import (
    _same_tensor_metadata,
    _validate_required_blocks,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_selection import (
    select_qsa_paged_tokens,
)
from rtp_llm.models_py.model_desc.test.qwen4_exp.fixtures import _bare_model
from rtp_llm.models_py.model_desc.test.qwen4_exp.qsa_fixtures import QSARuntimeFixture
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import (
    apply_partial_rope,
    build_interleaved_mrope,
)
from rtp_llm.ops import HybridAttentionType, RopeStyle
from rtp_llm.ops.compute_ops import PyCacheStoreInputs
from rtp_llm.utils import hot_hook_runtime


class Qwen4ExpQSACacheTest(QSARuntimeFixture, TestCase):
    @skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_eager_decode_combines_main_geometry_rope_and_side_verdict(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa import fmha_validation

        self.D = self.KV_TOKENS_PER_BLOCK = self.STATE_TOKENS_PER_BLOCK = 128
        self.indexer.k_norm_gamma = torch.randn(128, device="cuda").bfloat16()
        indexer = self._decode_indexer()
        rope = self._base_rope_config(3)
        rope.dim = 64
        q = torch.randn(1, 2, 128, device="cuda", dtype=torch.bfloat16)
        raw = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16)
        item = torch.Tensor.item
        for corruption in (None, "main", "axes", "logical", "side"):
            with self.subTest(corruption=corruption):
                kv = torch.zeros(7, 32 * 128 * 2, device="cuda", dtype=torch.uint8)
                state = torch.zeros(8, 8 * 128, device="cuda")
                context = self._single_request_context(
                    kv,
                    state,
                    is_prefill=False,
                    input_length=1,
                    sequence_length=3,
                    position_ids=[[3, 3, 3]],
                )
                # Use the existing host metadata ABI; no GPU length readback is
                # needed to construct the request's logical position.
                for name in ("input_lengths", "prefix_lengths", "sequence_lengths"):
                    value = getattr(context.main_inputs, name).cpu()
                    for inputs in (
                        context.main_inputs,
                        context.indexer_kv_inputs,
                        context.indexer_state_inputs,
                    ):
                        setattr(inputs, name, value)
                main = torch.zeros(3, 32768, device="cuda", dtype=torch.bfloat16)
                context.main_cache.kv_cache_base = main
                table = torch.tensor([[1, 2]], device="cuda", dtype=torch.int32)
                context.main_inputs.kv_cache_kernel_block_id_device = table
                owner = SimpleNamespace(
                    attn_inputs=context.main_inputs,
                    attn_configs=SimpleNamespace(kernel_tokens_per_block=128),
                    head_dim=128,
                    kv_head_num=1,
                    is_prefill=False,
                    is_target_verify=False,
                    _selected_indices=None,
                    _is_draft_graph=lambda: False,
                    _is_prefix_reuse_prefill=lambda: False,
                    _paged_kv_scale_args=lambda cache: None,
                )
                owner._validate_main_paged_geometry = lambda *args, **kwargs: fmha_validation._validate_main_paged_geometry(
                    owner, *args, **kwargs
                )
                if corruption == "main":
                    table[0, 0] = 100
                elif corruption == "axes":
                    context.main_inputs.combo_position_ids[2] = 4
                elif corruption == "logical":
                    context.main_inputs.combo_position_ids.fill_(4)
                elif corruption == "side":
                    context.indexer_kv_inputs.kv_cache_kernel_block_id_device.zero_()
                reads = []

                def counted(tensor, *args, **kwargs):
                    if tensor.is_cuda:
                        reads.append(tensor.numel())
                    return item(tensor, *args, **kwargs)

                with patch.object(torch.Tensor, "item", counted):
                    fmha_validation.validate_qsa_before_side_write(
                        owner, context, indexer, raw
                    )
                    self.assertEqual(reads, [])
                    if corruption is None:
                        context.select_decode_tokens(
                            q, raw, indexer=indexer, rope_config=rope
                        )
                        context.finalize_side_cache()
                    else:
                        with self.assertRaises(
                            ValueError
                            if corruption in ("axes", "logical")
                            else RuntimeError
                        ):
                            context.select_decode_tokens(
                                q, raw, indexer=indexer, rope_config=rope
                            )
                self.assertIsNone(context._prewrite_checks)
                if corruption is None:
                    self.assertEqual(reads, [1])
                else:
                    torch.testing.assert_close(kv, torch.zeros_like(kv), atol=0, rtol=0)
                    torch.testing.assert_close(
                        state, torch.zeros_like(state), atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        main, torch.zeros_like(main), atol=0, rtol=0
                    )

    def test_step_metadata_cache_reuses_views_and_never_survives_next_forward(self):
        from types import SimpleNamespace

        from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context import (
            Qwen4ExpQSARuntimeContext,
        )

        values = torch.tensor([1, 2], dtype=torch.int32)
        owner = SimpleNamespace(metadata_cache={})
        host_values = Qwen4ExpQSARuntimeContext.host_values
        first = host_values(owner, values)
        self.assertIs(host_values(owner, values.view_as(values)), first)
        values.add_(1)
        next_step = SimpleNamespace(metadata_cache={})
        self.assertEqual(host_values(next_step, values), [2, 3])

    def test_batched_state_preflight_reads_fresh_values_before_any_write(self):
        context = self._draft_incremental_context(
            self.kv_base,
            self.state_base,
            prefixes=[8],
            lengths=[16],
            is_mtp_draft=False,
        )
        context = replace(context, metadata_cache={})
        options = dict(
            indexer=self._decode_indexer(),
            token_count=16,
            device=self.kv_base.device,
            rope_config=self.rope_config,
            draft=False,
        )
        expected = context._draft_incremental_prefill_geometry(**options)
        actual = context._draft_incremental_prefill_geometry(**options)
        self.assertIs(actual.cu_seqlens, expected.cu_seqlens)
        self.assertIs(actual.visible_lengths, expected.visible_lengths)
        self.assertIs(actual.compressed_lengths, expected.compressed_lengths)
        for name in (
            "visible_lengths",
            "compressed_lengths",
            "current_cos",
            "current_sin",
        ):
            torch.testing.assert_close(
                getattr(actual, name), getattr(expected, name), rtol=0, atol=0
            )
        context.indexer_state_inputs.kv_cache_block_id_device[0, 2] = 0
        kv_before, state_before = self.kv_base.clone(), self.state_base.clone()
        with self.assertRaisesRegex(RuntimeError, "incremental tail resolves"):
            context._draft_incremental_prefill_geometry(**options)
        torch.testing.assert_close(self.kv_base, kv_before, rtol=0, atol=0)
        torch.testing.assert_close(self.state_base, state_before, rtol=0, atol=0)

    @torch.inference_mode()
    def test_prefill_forward_shares_geometry_and_keeps_layer_pool_validation(self):
        context = self._draft_incremental_context(
            self.kv_base,
            self.state_base,
            prefixes=[8],
            lengths=[16],
            kv_table=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
            state_table=torch.tensor([[4, 5, 6, 7]], dtype=torch.int32),
            is_mtp_draft=False,
        )
        indexer = self._decode_indexer()
        options = dict(
            indexer=indexer,
            token_count=16,
            device=self.kv_base.device,
            rope_config=self.rope_config,
            draft=False,
        )
        other_kv = SimpleNamespace(
            tag=INDEXER_KV_TAG,
            kv_cache_base=self.kv_base.clone(),
            seq_size_per_block=self.KV_TOKENS_PER_BLOCK,
        )
        layer_caches = [
            (context.main_cache, context.indexer_kv_cache, context.indexer_state_cache),
            (context.main_cache, other_kv, context.indexer_state_cache),
        ]
        contexts, plans = [], []

        class PlanLayer(nn.Module):
            layer_type = HybridAttentionType.NONE
            self_attn = SimpleNamespace(qsa_indexer=indexer)

            def forward(self, hidden, fmha_impl, *, qsa_runtime, **kwargs):
                contexts.append(qsa_runtime)
                plans.append(qsa_runtime._draft_incremental_prefill_geometry(**options))
                return hidden

        model = _bare_model(layer_caches)
        nn.Module.__init__(model)
        model.kv_cache.layers = layer_caches
        model.config = SimpleNamespace()
        model.ple_layers = {}
        model.parallelism_config = None
        model.weight = None
        model.fmha_config = None
        model.py_hw_kernel_config = None
        model.layers = nn.ModuleList(PlanLayer() for _ in layer_caches)
        model.word_embedding = lambda inputs: torch.zeros(inputs.input_ids.numel(), 4)
        model._initial_hyper_states = lambda hidden: hidden
        model._build_attn_meta = Mock(return_value=None)
        model.hyper_connection_mixer = lambda hidden: (hidden, None, None)

        inputs = SimpleNamespace(
            input_ids=torch.zeros(16, dtype=torch.int32),
            attention_inputs={
                "full": context.main_inputs,
                INDEXER_KV_TAG: context.indexer_kv_inputs,
                context.indexer_state_cache.tag: context.indexer_state_inputs,
            },
        )
        # FMHA owns its own wrapper around the current request tensors.
        fmha_inputs = SimpleNamespace(**vars(context.main_inputs))
        fmha = SimpleNamespace(attn_inputs=fmha_inputs)
        with patch.object(
            AttnImplFactory,
            "get_fmha_impl",
            return_value=fmha,
        ) as prepare:
            model(inputs)
        prepare.assert_called_once()

        context, second = contexts
        first, plan = plans
        self.assertIs(context.main_inputs, fmha_inputs)
        self.assertIs(second.main_inputs, fmha_inputs)
        self.assertIs(plan.visible_lengths, first.visible_lengths)
        self.assertIs(plan.current_cos, first.current_cos)
        self.assertIs(plan.rope_sin, first.rope_sin)
        self.assertEqual(plan.kv_pool.data_ptr(), other_kv.kv_cache_base.data_ptr())
        self.assertNotEqual(plan.kv_pool.data_ptr(), first.kv_pool.data_ptr())
        small_pool = replace(
            second,
            indexer_state_cache=SimpleNamespace(
                tag="indexer_state",
                kv_cache_base=self.state_base[:5],
                seq_size_per_block=self.STATE_TOKENS_PER_BLOCK,
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "incremental tail resolves"):
            small_pool._draft_incremental_prefill_geometry(**options)
        recovered = context._draft_incremental_prefill_geometry(**options)
        self.assertIs(recovered.visible_lengths, first.visible_lengths)
        for tensor, delta, error, message in (
            (
                fmha_inputs.prefix_lengths,
                self.KV_TOKENS_PER_BLOCK,
                RuntimeError,
                "cu_kv_seqlens do not match",
            ),
            (fmha_inputs.combo_position_ids, 1, ValueError, "logical cache positions"),
        ):
            self.assertTrue(tensor.is_inference())
            original = tensor.clone()

            def change_metadata(_layer, _args):
                tensor.add_(delta)

            try:
                with model.layers[1].register_forward_pre_hook(change_metadata):
                    with self.assertRaisesRegex(error, message):
                        model(inputs, fmha)
            finally:
                tensor.copy_(original)
        with tempfile.TemporaryDirectory() as tmp:
            hook_file = Path(tmp) / "hooks.py"
            hook_file.write_text(
                "def change_positions(ctx):\n"
                "    if ctx.args[1] == 1:\n"
                "        ctx.result.main_inputs.combo_position_ids.add_(1)\n"
            )
            config_file = Path(tmp) / "config.json"
            target = (
                f"{type(model).__module__}.{type(model).__name__}._qsa_runtime_context"
            )
            config_file.write_text(
                json.dumps(
                    {
                        "hook_file": str(hook_file),
                        "function_hooks": [
                            {"target": target, "after": "change_positions"}
                        ],
                    }
                )
            )
            positions = fmha_inputs.combo_position_ids.clone()
            try:
                self.assertTrue(hot_hook_runtime.install_from_config(str(config_file)))
                with self.assertRaisesRegex(ValueError, "logical cache positions"):
                    model(inputs, fmha)
            finally:
                hot_hook_runtime.reset_for_test()
                fmha_inputs.combo_position_ids.copy_(positions)
        context.main_inputs.cu_seqlens_device[1] -= 1
        with self.assertRaisesRegex(
            RuntimeError, "cu_seqlens do not match input_lengths"
        ):
            context._draft_incremental_prefill_geometry(**options)
        torch.testing.assert_close(
            self.kv_base, torch.zeros_like(self.kv_base), rtol=0, atol=0
        )
        torch.testing.assert_close(
            self.state_base, torch.zeros_like(self.state_base), rtol=0, atol=0
        )

    def test_draft_incremental_geometry_reuses_eager_cache_and_rejects_graph(self):
        context = replace(
            self._draft_incremental_context(
                self.kv_base,
                self.state_base,
                prefixes=[7],
                lengths=[3],
            ),
            metadata_cache={},
        )
        options = dict(
            indexer=self._decode_indexer(),
            token_count=3,
            device=self.kv_base.device,
            rope_config=self._base_rope_config(3),
            draft=True,
        )
        first = context._draft_incremental_prefill_geometry(**options)
        again = context._draft_incremental_prefill_geometry(**options)
        self.assertIs(again.visible_lengths, first.visible_lengths)
        context.main_inputs.is_cuda_graph = True
        with self.assertRaisesRegex(RuntimeError, "does not support CUDA Graph"):
            context._draft_incremental_prefill_geometry(**options)

    def check_prefix_reuse_rope_across_layers(self, device, prefix, length):
        self.D = self.KV_TOKENS_PER_BLOCK = self.STATE_TOKENS_PER_BLOCK = 128
        self.indexer.k_norm_gamma = torch.ones(self.D, device=device)
        indexer = self._decode_indexer()
        entries = self.KV_TOKENS_PER_BLOCK // self.RATIO
        pages = (
            prefix + length + self.KV_TOKENS_PER_BLOCK - 1
        ) // self.KV_TOKENS_PER_BLOCK
        table = torch.arange(1, pages + 1, device=device, dtype=torch.int32)[None]
        kv_pool = torch.zeros(
            pages + 1, entries * self.D * 2, device=device, dtype=torch.uint8
        )
        state_pool = torch.zeros(pages + 1, 2 * self.RATIO * self.D, device=device)
        mrope = SimpleNamespace(
            **(
                vars(self.rope_config)
                | dict(dim=64, mrope_dim1=12, mrope_dim2=10, mrope_dim3=10)
            )
        )
        for config, sidecar in (
            (self._base_rope_config(1), False),
            (self._base_rope_config(3), False),
            (mrope, False),
            (mrope, True),
        ):
            with self.subTest(
                style=config.style, factor=config.index_factor, sidecar=sidecar
            ):
                context = self._draft_incremental_context(
                    kv_pool,
                    state_pool,
                    prefixes=[prefix],
                    lengths=[length],
                    kv_table=table,
                    state_table=table,
                    position_factor=config.index_factor,
                    is_mtp_draft=False,
                )
                context.main_inputs.prefix_lengths_device = (
                    context.main_inputs.prefix_lengths
                )
                if sidecar:
                    context = replace(
                        context,
                        position_state_cache=SimpleNamespace(
                            tag=INDEXER_ROPE_POS_TAG,
                            seq_size_per_block=self.STATE_TOKENS_PER_BLOCK,
                            kv_cache_base=torch.zeros(
                                pages + 1,
                                2 * self.RATIO * 3,
                                dtype=torch.int32,
                                device=device,
                            ),
                        ),
                        position_state_inputs=context.indexer_state_inputs,
                    )

                def prepare(context, metadata, config=config):
                    return replace(
                        context, metadata_cache=metadata
                    )._draft_incremental_prefill_geometry(
                        indexer=indexer,
                        token_count=length,
                        device=device,
                        rope_config=config,
                        draft=False,
                    )

                def assert_rope_equal(actual, expected):
                    for name in ("current_cos", "current_sin", "rope_cos", "rope_sin"):
                        torch.testing.assert_close(
                            getattr(actual, name),
                            getattr(expected, name),
                            rtol=0,
                            atol=0,
                        )

                expected = prepare(context, None)
                for metadata, count in ((None, 24), ({}, 2)):
                    with patch.object(
                        torch, "cos", wraps=torch.cos
                    ) as cos, patch.object(torch, "sin", wraps=torch.sin) as sin:
                        plans = [prepare(context, metadata) for _ in range(12)]
                    self.assertEqual((cos.call_count, sin.call_count), (count, count))
                    for plan in plans:
                        assert_rope_equal(plan, expected)
                        if metadata is not None:
                            self.assertIs(plan.current_cos, plans[0].current_cos)
                            self.assertIs(plan.rope_cos, plans[0].rope_cos)

                changed_config = SimpleNamespace(
                    **(vars(config) | dict(base=2000, scale=2.0))
                )
                adjusted = prepare(context, metadata, changed_config)
                assert_rope_equal(adjusted, prepare(context, None, changed_config))
                self.assertFalse(
                    torch.equal(adjusted.current_cos, expected.current_cos)
                )

                positions = context.main_inputs.combo_position_ids
                if sidecar:
                    positions[1::3].add_(31)
                    changed = prepare(context, {})
                    assert_rope_equal(changed, prepare(context, None))
                    self.assertFalse(
                        torch.equal(changed.current_cos, expected.current_cos)
                    )
                    self.assertFalse(torch.equal(changed.rope_sin, expected.rope_sin))

                    # A group starting before this chunk reads layer-local history.
                    context.main_inputs.prefix_lengths_device = (
                        context.main_inputs.prefix_lengths - 1
                    )
                    cache = context.position_state_cache
                    other_layer = replace(
                        context,
                        position_state_cache=SimpleNamespace(
                            tag=cache.tag,
                            seq_size_per_block=cache.seq_size_per_block,
                            kv_cache_base=torch.full_like(cache.kv_cache_base, 47),
                        ),
                    )
                    historical_metadata = {}
                    first = prepare(context, historical_metadata)
                    second = prepare(other_layer, historical_metadata)
                    assert_rope_equal(first, prepare(context, None))
                    assert_rope_equal(second, prepare(other_layer, None))
                    self.assertFalse(torch.equal(first.rope_cos[0], second.rope_cos[0]))
                else:
                    positions[0] += 1
                    for _ in range(2):
                        with self.assertRaisesRegex(
                            ValueError, "three axes are equal|logical cache positions"
                        ):
                            prepare(context, metadata)

    def test_prefix_reuse_rope_across_layers_cpu(self):
        self.check_prefix_reuse_rope_across_layers(torch.device("cpu"), 128, 9)

    @skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_prefix_reuse_rope_across_layers_cuda(self):
        self.check_prefix_reuse_rope_across_layers(torch.device("cuda:0"), 6144, 2048)

    def test_prefill_checks_share_success_without_sharing_layer_pools(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa.validation import DeviceChecks

        context = replace(
            self._draft_incremental_context(
                self.kv_base,
                self.state_base,
                prefixes=[8],
                lengths=[16],
                is_mtp_draft=False,
            ),
            metadata_cache={},
        )
        options = dict(
            indexer=self._decode_indexer(),
            token_count=16,
            device=self.kv_base.device,
            rope_config=self.rope_config,
            draft=False,
        )

        def plan(owner):
            checks = DeviceChecks()
            # The model preflight is covered separately; exercise the planner
            # with the scope marker it supplies before any persistent write.
            checks.prefill_metadata_readonly = True
            object.__setattr__(owner, "_prewrite_checks", checks)
            return owner._draft_incremental_prefill_geometry(**options), checks

        expected, first = plan(context)
        self.assertTrue(first._checks)
        # New host tensor storage needs fresh geometry even when its values
        # match the verified QSA tables and cumulative lengths.
        other_inputs = SimpleNamespace(**vars(context.main_inputs))
        other_inputs.input_lengths = context.main_inputs.input_lengths.clone()
        other = replace(
            context,
            main_inputs=other_inputs,
            indexer_kv_cache=SimpleNamespace(
                **{
                    **vars(context.indexer_kv_cache),
                    "kv_cache_base": self.kv_base.clone(),
                }
            ),
            indexer_state_cache=SimpleNamespace(
                **{
                    **vars(context.indexer_state_cache),
                    "kv_cache_base": self.state_base.clone(),
                }
            ),
        )
        with patch.object(torch, "zeros_like", wraps=torch.zeros_like) as allocate_cu:
            actual, second = plan(other)
        allocate_cu.assert_not_called()
        self.assertIsNot(actual.visible_lengths, expected.visible_lengths)
        # This fixture has no position pool. Its separate logical RoPE check
        # must remain even after the six read-only CU/page flags are accepted.
        self.assertEqual(len(first._checks) - len(second._checks), 6)
        self.assertEqual(len(second._checks), 1)
        self.assertIs(second._checks[0][1], ValueError)
        self.assertNotEqual(
            actual.state_pool.data_ptr(), expected.state_pool.data_ptr()
        )
        torch.testing.assert_close(actual.visible_lengths, expected.visible_lengths)
        torch.testing.assert_close(actual.cu_seqlens, expected.cu_seqlens)
        other.main_inputs.combo_position_ids.add_(1)
        with self.assertRaisesRegex(ValueError, "logical cache positions"):
            plan(other)
        other.main_inputs.combo_position_ids.sub_(1)
        smaller_kv = replace(
            other,
            indexer_kv_cache=SimpleNamespace(
                **{
                    **vars(other.indexer_kv_cache),
                    "kv_cache_base": self.kv_base[:1].clone(),
                }
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "out-of-range physical block"):
            plan(smaller_kv)
        smaller = replace(
            other,
            indexer_state_cache=SimpleNamespace(
                **{
                    **vars(other.indexer_state_cache),
                    "kv_cache_base": self.state_base[:6].clone(),
                }
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "incremental tail resolves"):
            plan(smaller)
        context.main_inputs.cu_seqlens_device[0] = 1
        with self.assertRaisesRegex(RuntimeError, "cu_seqlens do not match"):
            plan(context)
        context.main_inputs.cu_seqlens_device[0] = 0
        context.main_inputs.cu_kv_seqlens_device[0] = 1
        with self.assertRaisesRegex(RuntimeError, "cu_kv_seqlens do not match"):
            plan(context)
        context.main_inputs.cu_kv_seqlens_device[0] = 0
        context.indexer_state_inputs.kv_cache_block_id_device[0, 2] = 0
        with self.assertRaisesRegex(RuntimeError, "incremental tail resolves"):
            plan(context)

    def test_metadata_alias_fast_path_preserves_distinct_tensor_validation(self):
        values = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        alias = values.view_as(values)
        with patch.object(torch, "equal", wraps=torch.equal) as equal:
            self.assertTrue(_same_tensor_metadata(values, alias))
            equal.assert_not_called()

            self.assertTrue(_same_tensor_metadata(values, values.clone()))
            self.assertFalse(_same_tensor_metadata(values, values.T))
            self.assertFalse(_same_tensor_metadata(values, values + 1))
            self.assertEqual(equal.call_count, 3)

    def test_required_block_validation_preserves_error_cases(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                table = torch.tensor([[1, 2]], dtype=torch.int32, device=device)
                required = torch.tensor([2], dtype=torch.int32, device=device)
                _validate_required_blocks(
                    table, required, pool_blocks=3, tag=INDEXER_KV_TAG
                )
                _validate_required_blocks(
                    table[:, :0],
                    torch.zeros_like(required),
                    pool_blocks=3,
                    tag=INDEXER_KV_TAG,
                )
                for count, message in (
                    (-1, "negative required-column count"),
                    (3, "does not cover"),
                ):
                    with self.assertRaisesRegex(RuntimeError, message):
                        _validate_required_blocks(
                            table,
                            torch.tensor([count], dtype=torch.int32, device=device),
                            pool_blocks=3,
                            tag=INDEXER_KV_TAG,
                        )
                with self.assertRaisesRegex(RuntimeError, "unallocated"):
                    _validate_required_blocks(
                        torch.tensor([[1, 0]], dtype=torch.int32, device=device),
                        required,
                        pool_blocks=3,
                        tag=INDEXER_KV_TAG,
                    )
                table = torch.tensor([[1, 3]], dtype=torch.int32, device=device)
                _validate_required_blocks(
                    table,
                    torch.tensor([1], dtype=torch.int32, device=device),
                    pool_blocks=3,
                    tag=INDEXER_KV_TAG,
                )
                with self.assertRaisesRegex(RuntimeError, "out-of-range"):
                    _validate_required_blocks(
                        table,
                        required,
                        pool_blocks=3,
                        tag=INDEXER_KV_TAG,
                    )

    def test_token_selection_rejects_invalid_lengths(self):
        logits = torch.zeros(1, 1)
        for length, message in (
            (-1, "non-negative"),
            (8, "do not cover every completed block"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                select_qsa_paged_tokens(
                    logits,
                    torch.tensor([length], dtype=torch.int32),
                    compress_ratio=4,
                    token_budget=4,
                )

    @skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_required_block_validation_handles_strides_and_fallback(self):
        from rtp_llm.models_py.kernels.cuda.qsa.qsa_validation_triton import (
            is_supported,
            required_block_flags,
        )

        table = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.int32, device="cuda").T
        required = torch.tensor([2, 0, 2, 0, 2, 0], dtype=torch.int32, device="cuda")[
            ::2
        ]
        self.assertTrue(is_supported(table, required))
        self.assertEqual(required_block_flags(table, required, pool_blocks=4).item(), 0)
        _validate_required_blocks(table, required, pool_blocks=4, tag=INDEXER_KV_TAG)
        table[1, 1] = 4
        self.assertEqual(
            required_block_flags(table, required, pool_blocks=4).item(),
            1 << 2,
        )
        with self.assertRaisesRegex(RuntimeError, "out-of-range"):
            _validate_required_blocks(
                table, required, pool_blocks=4, tag=INDEXER_KV_TAG
            )

        wide_table = torch.ones(1, 4097, dtype=torch.int32, device="cuda")
        self.assertFalse(is_supported(wide_table, required[:1]))
        _validate_required_blocks(
            wide_table,
            required[:1],
            pool_blocks=4,
            tag=INDEXER_KV_TAG,
        )

    def test_ragged_prefill_writes_each_tag_local_pool(self):
        first = torch.randn(5, self.D, dtype=torch.bfloat16)
        second = torch.randn(8, self.D, dtype=torch.bfloat16)
        raw_keys = torch.cat([first, second])

        lengths, cos, sin, result = self._context().write_prefill_indexer_cache(
            raw_keys, indexer=self.indexer, rope_config=self.rope_config
        )

        self.assertEqual(lengths, self.lengths)
        self.assertEqual(result["num_state_writes"], 13)
        self.assertEqual(result["num_kv_writes"], 3)
        kv_pool = self.kv_base.view(torch.bfloat16).view(7, 2, self.D)
        state_pool = self.state_base.view(8, 2 * self.RATIO, self.D)
        torch.testing.assert_close(
            kv_pool[3, 0], self._expected(first[:4], cos[0, 0], sin[0, 0], 0)
        )
        torch.testing.assert_close(
            kv_pool[1, 0], self._expected(second[:4], cos[1, 0], sin[1, 0], 0)
        )
        torch.testing.assert_close(
            kv_pool[1, 1], self._expected(second[4:8], cos[1, 4], sin[1, 4], 4)
        )
        torch.testing.assert_close(state_pool[6, 4], first[4].float())
        torch.testing.assert_close(state_pool[2, 7], second[7].float())

    def test_eager_prefill_fake_row_preserves_both_pools_and_undo(self):
        self.kv_base.random_(0, 256)
        self.state_base.normal_()
        context = self._context()
        for inputs in (
            context.main_inputs,
            context.indexer_kv_inputs,
            context.indexer_state_inputs,
        ):
            inputs.is_s_padded = True
        self.kv_table[1].zero_()
        self.state_table[1].zero_()
        before = self.kv_base.clone(), self.state_base.clone()
        raw = torch.randn(self.total_tokens, self.D, dtype=torch.bfloat16)
        _, _, _, result = context.write_prefill_indexer_cache(
            raw, indexer=self.indexer, rope_config=self.rope_config
        )
        self.assertEqual(result["num_state_writes"], self.lengths[0])
        self.assertEqual(result["num_kv_writes"], self.lengths[0] // self.RATIO)
        self.assertTrue(bool((result["state_slots"][self.lengths[0] :] == -1).all()))
        torch.testing.assert_close(self.kv_base[0], before[0][0], rtol=0, atol=0)
        torch.testing.assert_close(self.state_base[0], before[1][0], rtol=0, atol=0)
        context.rollback_side_cache()
        torch.testing.assert_close(self.kv_base, before[0], rtol=0, atol=0)
        torch.testing.assert_close(self.state_base, before[1], rtol=0, atol=0)

    def test_base_rope_prefill_factor_one_and_three_match(self):
        raw_keys = torch.randn(self.total_tokens, self.D, dtype=torch.bfloat16)
        scalar_positions = []
        for seq_len in self.lengths:
            scalar_positions.extend(range(seq_len))

        factor_one = self._context(
            kv_base=torch.zeros_like(self.kv_base),
            state_base=torch.zeros_like(self.state_base),
        )
        self._set_position_ids(factor_one, scalar_positions)
        _, cos_one, sin_one, _ = factor_one.write_prefill_indexer_cache(
            raw_keys,
            indexer=self.indexer,
            rope_config=self._base_rope_config(1),
        )

        factor_three = self._context(
            kv_base=torch.zeros_like(self.kv_base),
            state_base=torch.zeros_like(self.state_base),
        )
        positions_three = (
            torch.tensor(scalar_positions, dtype=torch.int32).unsqueeze(1).expand(-1, 3)
        )
        self._set_position_ids(factor_three, positions_three)
        _, cos_three, sin_three, _ = factor_three.write_prefill_indexer_cache(
            raw_keys,
            indexer=self.indexer,
            rope_config=self._base_rope_config(3),
        )

        torch.testing.assert_close(cos_one, cos_three)
        torch.testing.assert_close(sin_one, sin_three)

    def test_base_rope_prefill_rejects_non_logical_position_before_write(self):
        raw_keys = torch.randn(self.total_tokens, self.D, dtype=torch.bfloat16)
        scalar_positions = []
        for seq_len in self.lengths:
            scalar_positions.extend(range(seq_len))
        scalar_positions[4] += 1
        for style in (RopeStyle.Base, "Base", 1):
            with self.subTest(style=style):
                context = self._context()
                self._set_position_ids(context, scalar_positions)
                with patch(
                    "rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context."
                    "write_indexer_cache"
                ) as writer:
                    with self.assertRaisesRegex(
                        RuntimeError, "logical cache positions"
                    ):
                        context.write_prefill_indexer_cache(
                            raw_keys,
                            indexer=self.indexer,
                            rope_config=self._base_rope_config(1, style=style),
                        )

                writer.assert_not_called()

    def test_non_neox_indexer_rope_is_rejected_before_prefill_write(self):
        raw_keys = torch.randn(self.total_tokens, self.D, dtype=torch.bfloat16)
        scalar_positions = []
        for seq_len in self.lengths:
            scalar_positions.extend(range(seq_len))
        context = self._context()
        self._set_position_ids(context, scalar_positions)

        with patch(
            "rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context.write_indexer_cache"
        ) as writer:
            with self.assertRaisesRegex(RuntimeError, "indexer_is_neox_style=true"):
                context.write_prefill_indexer_cache(
                    raw_keys,
                    indexer=self.indexer,
                    rope_config=self._base_rope_config(1, indexer_is_neox_style=False),
                )

        writer.assert_not_called()

    def test_bad_2d_pool_row_is_rejected_before_mutation(self):
        bad_kv = torch.zeros(7, self.kv_base.shape[1] - 1, dtype=torch.uint8)
        raw_keys = torch.randn(self.total_tokens, self.D, dtype=torch.bfloat16)

        with self.assertRaisesRegex(RuntimeError, "row has"):
            self._context(kv_base=bad_kv).write_prefill_indexer_cache(
                raw_keys, indexer=self.indexer, rope_config=self.rope_config
            )

        self.assertEqual(int(torch.count_nonzero(bad_kv)), 0)
        self.assertEqual(int(torch.count_nonzero(self.state_base)), 0)

    def test_unsupported_or_inconsistent_side_metadata_is_rejected(self):
        raw_keys = torch.randn(self.total_tokens, self.D, dtype=torch.bfloat16)
        cases = (
            ("prefill", {"is_prefill": False}),
            ("prefix", {"prefix_lengths": torch.tensor([1, 0], dtype=torch.int32)}),
            ("target-verify mode", {"is_target_verify": True}),
            ("CUDA Graph", {"is_cuda_graph": True}),
            ("PD", {"cache_store_inputs": object()}),
        )
        for expected, overrides in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(RuntimeError, expected):
                    self._context(**overrides).write_prefill_indexer_cache(
                        raw_keys, indexer=self.indexer, rope_config=self.rope_config
                    )
        self.assertEqual(int(torch.count_nonzero(self.kv_base)), 0)
        self.assertEqual(int(torch.count_nonzero(self.state_base)), 0)

    def test_ordinary_prefill_still_rejects_nonzero_prefix_reuse(self):
        context = self._draft_incremental_context(
            self.kv_base,
            self.state_base,
            prefixes=[7],
            lengths=[1],
            is_mtp_draft=False,
        )
        with patch(
            "rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context.write_indexer_cache"
        ) as writer:
            with self.assertRaisesRegex(RuntimeError, "prefix reuse"):
                context.write_prefill_indexer_cache(
                    torch.randn(1, self.D, dtype=torch.bfloat16),
                    indexer=self.indexer,
                    rope_config=self._base_rope_config(3),
                )
        writer.assert_not_called()

    @skipUnless(torch.cuda.is_available(), "CUDA is required for paged metadata")
    def test_mtp_prefix_reuse_uses_cache_positions_for_shifted_native_rows(self):
        from rtp_llm.models_py.model_desc.qwen4_exp_mtp import (
            normalize_mtp_prefill_positions,
        )
        from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import build_base_rope

        device = torch.device("cuda", torch.cuda.current_device())
        indexer = self._decode_indexer()
        indexer.k_norm_gamma = indexer.k_norm_gamma.to(device)
        config = self._base_rope_config(3)
        for length, published in (
            (length, published)
            for length in (1, 3, 4, 8)
            for published in (False, True)
        ):
            with self.subTest(length=length, published=published):
                context = self._draft_incremental_context(
                    torch.zeros(16, 2 * self.D * 2, dtype=torch.uint8, device=device),
                    torch.zeros(
                        16, 2 * self.RATIO * self.D, dtype=torch.float32, device=device
                    ),
                    prefixes=[8],
                    lengths=[length],
                    is_mtp_draft=True,
                    position_overrides=list(range(9, 8 + length)) + [8 + length - 1],
                )
                shifted = context.main_inputs.combo_position_ids.clone()
                groups = {
                    "full": context.main_inputs,
                    "indexer_kv": context.indexer_kv_inputs,
                    "indexer_state": context.indexer_state_inputs,
                }
                publication = PyCacheStoreInputs()
                shared_positions = shifted.clone()
                for value in groups.values():
                    value.combo_position_ids = shared_positions
                    value.prefix_lengths_device = torch.tensor(
                        [8], dtype=torch.int32, device=device
                    )
                    value.cache_store_inputs = publication if published else None
                    value.cache_store_writer = SimpleNamespace(write=lambda *_: None)
                inputs = SimpleNamespace(
                    input_ids=torch.zeros(length, dtype=torch.int32, device=device),
                    combo_position_ids=context.main_inputs.combo_position_ids,
                    attention_inputs=groups,
                )
                normalize_mtp_prefill_positions(inputs, 3)
                plan = context._draft_incremental_prefill_geometry(
                    indexer=indexer,
                    token_count=length,
                    device=device,
                    rope_config=config,
                    draft=False,
                    draft_prefix_reuse=True,
                )
                positions = torch.arange(
                    8, 8 + length, dtype=torch.int32, device=device
                )
                expected_cos, expected_sin = build_base_rope(
                    positions[:, None].expand(-1, 3).reshape(-1),
                    config,
                    token_count=length,
                    dtype=torch.bfloat16,
                    device=device,
                )
                torch.testing.assert_close(plan.current_cos, expected_cos)
                torch.testing.assert_close(plan.current_sin, expected_sin)
                if length == 3:
                    for value in groups.values():
                        value.combo_position_ids.copy_(shifted)
                        value.cache_store_inputs = None
                    for rope, draft_prefix_reuse in (
                        (config, False),
                        (self.rope_config, True),
                    ):
                        with self.assertRaisesRegex(
                            ValueError,
                            "positions must equal the logical cache positions",
                        ):
                            context._draft_incremental_prefill_geometry(
                                indexer=indexer,
                                token_count=length,
                                device=device,
                                rope_config=rope,
                                draft=False,
                                draft_prefix_reuse=draft_prefix_reuse,
                            )

    def test_decode_completes_prefill_tail_then_scores_same_projection(self):
        prefill_raw = torch.randn(7, self.D, dtype=torch.bfloat16)
        decode_raw = torch.randn(1, self.D, dtype=torch.bfloat16)
        decode_q = torch.randn(1, 2, self.D, dtype=torch.bfloat16)
        indexer = self._decode_indexer()
        prefill = self._single_request_context(
            self.kv_base,
            self.state_base,
            is_prefill=True,
            input_length=7,
            sequence_length=0,
            position_ids=[[pos, pos, pos] for pos in range(7)],
        )
        _, prefill_cos, prefill_sin, result = prefill.write_prefill_indexer_cache(
            prefill_raw, indexer=indexer, rope_config=self.rope_config
        )
        self.assertEqual(result["num_kv_writes"], 1)

        decode = self._single_request_context(
            self.kv_base,
            self.state_base,
            is_prefill=False,
            input_length=7,
            sequence_length=7,
            position_ids=[[7, 7, 7]],
        )
        observed = {}

        def _score(
            q,
            weight,
            pool,
            table,
            lengths,
            *,
            block_size,
            max_ctx_len,
            validate_block_table,
        ):
            self.assertFalse(validate_block_table)
            observed.update(
                q=q,
                weight=weight,
                pool=pool,
                table=table,
                lengths=lengths,
                block_size=block_size,
                max_ctx_len=max_ctx_len,
            )
            return torch.tensor([[0.25, 3.0]], dtype=torch.float32)

        with patch(
            "rtp_llm.models_py.kernels.cuda.qsa.indexer_paged_score."
            "qsa_paged_indexer_score",
            side_effect=_score,
        ):
            selected = decode.select_decode_tokens(
                decode_q,
                decode_raw,
                indexer=indexer,
                rope_config=self.rope_config,
            )

        kv_pool = self.kv_base.view(torch.bfloat16).view(7, 2, self.D)
        state_pool = self.state_base.view(8, 2 * self.RATIO, self.D)
        expected_second = self._expected(
            torch.cat([prefill_raw[4:], decode_raw]),
            prefill_cos[0, 4],
            prefill_sin[0, 4],
            4,
        )
        torch.testing.assert_close(kv_pool[3, 1], expected_second)
        torch.testing.assert_close(state_pool[6, 7], decode_raw[0].float())
        self.assertEqual(observed["q"].shape, (1, 1, 2, self.D))
        torch.testing.assert_close(
            observed["weight"], torch.full((1, 2), 1.0 / self.D**0.5)
        )
        torch.testing.assert_close(
            observed["lengths"], torch.tensor([[2]], dtype=torch.int32)
        )
        self.assertEqual(observed["block_size"], 2)
        self.assertEqual(observed["max_ctx_len"], 2)
        torch.testing.assert_close(
            selected,
            torch.tensor(
                [[4, 5, 6, 7, 0, 1, 2, 3, -1, -1, -1]],
                dtype=torch.int32,
            ),
        )

    def test_decode_base_rope_factor_one_and_three_rotate_q_identically(self):
        q = torch.randn(1, 2, self.D, dtype=torch.bfloat16)
        raw = torch.randn(1, self.D, dtype=torch.bfloat16)
        rotated_queries = []

        def _score(
            q,
            weight,
            pool,
            table,
            lengths,
            *,
            block_size,
            max_ctx_len,
            validate_block_table,
        ):
            self.assertFalse(validate_block_table)
            rotated_queries.append(q.clone())
            return torch.zeros((1, 1), dtype=torch.float32)

        with patch(
            "rtp_llm.models_py.kernels.cuda.qsa.indexer_paged_score."
            "qsa_paged_indexer_score",
            side_effect=_score,
        ):
            for index_factor, position_ids in ((1, [[3]]), (3, [[3, 3, 3]])):
                context = self._single_request_context(
                    torch.zeros_like(self.kv_base),
                    torch.zeros_like(self.state_base),
                    is_prefill=False,
                    input_length=3,
                    sequence_length=3,
                    position_ids=position_ids,
                )
                context.select_decode_tokens(
                    q,
                    raw,
                    indexer=self._decode_indexer(),
                    rope_config=self._base_rope_config(index_factor),
                )

        self.assertEqual(len(rotated_queries), 2)
        torch.testing.assert_close(rotated_queries[0], rotated_queries[1])

    def test_decode_base_rope_rejects_non_logical_position_before_write(self):
        q = torch.randn(1, 2, self.D, dtype=torch.bfloat16)
        raw = torch.randn(1, self.D, dtype=torch.bfloat16)
        context = self._single_request_context(
            self.kv_base,
            self.state_base,
            is_prefill=False,
            input_length=3,
            sequence_length=3,
            position_ids=[[4]],
        )

        with patch(
            "rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context.write_indexer_cache"
        ) as writer:
            with self.assertRaisesRegex(ValueError, "logical cache positions"):
                context.select_decode_tokens(
                    q,
                    raw,
                    indexer=self._decode_indexer(),
                    rope_config=self._base_rope_config(1),
                )

        writer.assert_not_called()

    def test_decode_rejects_compressed_length_beyond_tag_local_table(self):
        q = torch.randn(1, 2, self.D, dtype=torch.bfloat16)
        raw = torch.randn(1, self.D, dtype=torch.bfloat16)
        decode = self._single_request_context(
            self.kv_base,
            self.state_base,
            is_prefill=False,
            input_length=11,
            sequence_length=11,
            position_ids=[[11, 11, 11]],
        )

        with self.assertRaisesRegex(RuntimeError, "compressed decode length"):
            decode.select_decode_tokens(
                q,
                raw,
                indexer=self._decode_indexer(),
                rope_config=self.rope_config,
            )

        self.assertEqual(int(torch.count_nonzero(self.kv_base)), 0)
        self.assertEqual(int(torch.count_nonzero(self.state_base)), 0)

    def test_decode_rejects_unallocated_required_kv_block_before_side_write(self):
        q = torch.randn(1, 2, self.D, dtype=torch.bfloat16)
        raw = torch.randn(1, self.D, dtype=torch.bfloat16)
        self.kv_base.random_(0, 256)
        self.state_base.normal_()
        kv_before = self.kv_base.clone()
        state_before = self.state_base.clone()
        decode = self._single_request_context(
            self.kv_base,
            self.state_base,
            is_prefill=False,
            input_length=7,
            sequence_length=7,
            position_ids=[[7, 7, 7]],
        )
        decode.indexer_kv_inputs.kv_cache_kernel_block_id_device.zero_()

        with self.assertRaisesRegex(RuntimeError, "unallocated"):
            decode.select_decode_tokens(
                q,
                raw,
                indexer=self._decode_indexer(),
                rope_config=self.rope_config,
            )

        self.assertTrue(torch.equal(self.kv_base, kv_before))
        self.assertTrue(torch.equal(self.state_base, state_before))

    @skipUnless(torch.cuda.is_available(), "CUDA is required for the paged scorer")
    def test_cuda_decode_selection_matches_direct_paged_score_math(self):
        device = torch.device("cuda")
        kv_base = torch.zeros(7, 2 * self.D * 2, dtype=torch.uint8, device=device)
        state_base = torch.zeros(
            8, 2 * self.RATIO * self.D, dtype=torch.float32, device=device
        )
        indexer = self._decode_indexer()
        indexer.k_norm_gamma = indexer.k_norm_gamma.to(device)
        prefill_raw = torch.randn(7, self.D, dtype=torch.bfloat16, device=device)
        decode_raw = torch.randn(1, self.D, dtype=torch.bfloat16, device=device)
        decode_q = torch.randn(1, 2, self.D, dtype=torch.bfloat16, device=device)
        self._single_request_context(
            kv_base,
            state_base,
            is_prefill=True,
            input_length=7,
            sequence_length=0,
            position_ids=[[pos, pos, pos] for pos in range(7)],
        ).write_prefill_indexer_cache(
            prefill_raw, indexer=indexer, rope_config=self.rope_config
        )
        decode = self._single_request_context(
            kv_base,
            state_base,
            is_prefill=False,
            input_length=7,
            sequence_length=7,
            position_ids=[[7, 7, 7]],
        )

        selected = decode.select_decode_tokens(
            decode_q,
            decode_raw,
            indexer=indexer,
            rope_config=self.rope_config,
        )

        positions = torch.arange(8, dtype=torch.int32, device=device)
        positions = positions.unsqueeze(1).expand(-1, 3)
        cos, sin = build_interleaved_mrope(
            positions.reshape(-1),
            self.rope_config,
            token_count=8,
            dtype=torch.bfloat16,
            device=device,
        )
        q = apply_partial_rope(decode_q, cos[7].view(1, 1, -1), sin[7].view(1, 1, -1))
        keys = kv_base.view(torch.bfloat16).view(7, 2, self.D)[3]
        logits = torch.relu(q.float() @ keys.float().T).sum(dim=1) / self.D**0.5
        chosen = logits[0].topk(2).indices.tolist()
        expected = []
        for block in chosen:
            expected.extend(range(block * self.RATIO, (block + 1) * self.RATIO))
        expected.extend([-1] * (8 + self.RATIO - 1 - len(expected)))
        torch.testing.assert_close(
            selected, torch.tensor([expected], dtype=torch.int32, device=device)
        )

    def test_paged_topk_expands_blocks_and_appends_partial_tail(self):
        block_logits = torch.tensor(
            [
                [0.2, 1.5, 0.7, float("-inf")],
                [3.0, 20.0, float("-inf"), float("-inf")],
            ],
            dtype=torch.float32,
        )
        token_lengths = torch.tensor([14, 5], dtype=torch.int32)

        selected = select_qsa_paged_tokens(
            block_logits,
            token_lengths,
            compress_ratio=self.RATIO,
            token_budget=8,
        )

        expected = torch.tensor(
            [
                [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1],
                [0, 1, 2, 3, -1, -1, -1, -1, 4, -1, -1],
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(selected, expected)

        # The runtime has already checked both length and table capacity. Its
        # decode path must avoid a per-layer device-to-host scalar read here.
        with patch.object(torch.Tensor, "item", side_effect=AssertionError("sync")):
            prevalidated = select_qsa_paged_tokens(
                block_logits,
                token_lengths,
                compress_ratio=self.RATIO,
                token_budget=8,
                validate_lengths=False,
            )
        torch.testing.assert_close(prevalidated, expected)


if __name__ == "__main__":
    main()
