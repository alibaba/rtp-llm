import os
import weakref
from types import SimpleNamespace
from typing import Sequence
from unittest import TestCase, main, skipUnless
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.factory.attention import attn_factory
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
    flashinfer_mla_wrapper,
    flashmla_dense_prefill,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
    MlaFlashMLAPrefillImpl,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_dense_prefill import (
    FlashMLADeviceParams,
    MlaFlashMLAPrefillOp,
    build_flashmla_device_params,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_forward_plan import (
    FlashMLAForwardRoute,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_page_rr_cache import (
    MlaPageRRCacheAdapter,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import MlaImplBase
from rtp_llm.models_py.modules.factory.linear.quantized_activation import (
    QuantizedActivation,
)
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, ParallelismConfig
from rtp_llm.ops.compute_ops import rtp_llm_ops

_TEST_TMPDIR = os.environ.get("TEST_TMPDIR")
if _TEST_TMPDIR:
    os.environ.setdefault("DG_JIT_CACHE_DIR", os.path.join(_TEST_TMPDIR, "deep_gemm"))

CUDA_AVAILABLE = torch.cuda.is_available()
GIB = 1024**3


class FlashMlaWorkspaceLifetimeTest(TestCase):
    def test_release_drops_scratch_but_keeps_plan_and_consumed_output(self) -> None:
        op = object.__new__(MlaFlashMLAPrefillOp)
        op._forward_workspace = SimpleNamespace(
            packed_kv=torch.empty(128), attention_output=torch.arange(16)
        )
        op._fp8_prefix_rope = torch.empty(64)
        plan = object()
        op._forward_plan = plan
        consumed_output = op._forward_workspace.attention_output.clone()
        refs = [
            weakref.ref(op._forward_workspace.packed_kv),
            weakref.ref(op._forward_workspace.attention_output),
            weakref.ref(op._fp8_prefix_rope),
        ]
        wrapper = object.__new__(MlaFlashMLAPrefillImpl)
        wrapper.fmha_impl = op
        with patch("torch.cuda.empty_cache") as flush:
            wrapper.release_forward_workspace()
            wrapper.release_forward_workspace()
        flush.assert_not_called()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertIs(op._forward_plan, plan)
        torch.testing.assert_close(consumed_output, torch.arange(16))


class FlashMlaPageRRPrefixCopyTest(TestCase):
    def setUp(self) -> None:
        self.op = object.__new__(MlaFlashMLAPrefillOp)
        self.op.kv_lora_rank = 512
        self.op.qk_rope_head_dim = 64
        self.op.q_lens = [1, 2]
        self.op.batch_reuse_info_host = ((0, 2, 0, 0), (0, 1, 0, 0))
        self.op.has_reuse_cache = True
        self.op._forward_plan = SimpleNamespace(route=FlashMLAForwardRoute.FULL)
        self.latent = torch.tensor([3, 5, 6], dtype=torch.bfloat16)[:, None].repeat(
            1, 512
        )
        self.quantized = QuantizedActivation(
            torch.zeros((3, 512), dtype=torch.float8_e4m3fn),
            torch.zeros((1, 4), dtype=torch.int32),
            self.latent,
        )
        self.k_pe = self.latent[:, :64] * 10
        prefix = torch.tensor([1, 2, 4], dtype=torch.bfloat16)[:, None]
        self.prefix = torch.cat((prefix.repeat(1, 512), prefix.repeat(1, 64) * 10), 1)

    def test_materialized_prefix_interleaves_retained_bf16_query_rows(self) -> None:
        latent, rope = self.op._gather_reused_kv(
            self.quantized, self.k_pe, None, self.prefix
        )
        expected = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.bfloat16)[:, None]
        torch.testing.assert_close(latent, expected.repeat(1, 512), rtol=0, atol=0)
        torch.testing.assert_close(rope, expected.repeat(1, 64) * 10, rtol=0, atol=0)

    def test_page_rr_chunk_copies_latent_and_rope_without_aggregate_offsets(
        self,
    ) -> None:
        self.op.num_heads = 2
        self.op.qk_nope_head_dim = 128
        self.op.v_head_dim = 128
        compressed = torch.empty((3, 512), dtype=torch.bfloat16)
        packed = torch.full((3, 2 * 320), -1, dtype=torch.bfloat16)

        self.op._copy_page_rr_prefix_chunk(self.prefix, compressed, packed)

        torch.testing.assert_close(compressed, self.prefix[:, :512], rtol=0, atol=0)
        rope = packed.view(3, 2, 320)[:, :, 128:192]
        torch.testing.assert_close(
            rope,
            self.prefix[:, 512:].unsqueeze(1).expand_as(rope),
            rtol=0,
            atol=0,
        )


class FlashMlaDensePrefillConfigForwardingTest(TestCase):
    def test_physical_page_owners_do_not_follow_kernel_subpages(self):
        configs = AttentionConfigs()
        configs.head_num = 8
        configs.kv_lora_rank = 512
        configs.rope_head_dim = 64
        configs.nope_head_dim = configs.v_head_dim = 128
        configs.tokens_per_block = 1024
        configs.kernel_tokens_per_block = 128
        configs.use_mla = True
        positions = torch.tensor([0, 127, 128, 381, 1023, 1024, 8192, 8321])
        # Physical IDs [2, 5] expanded exactly as the allocator's kernel view.
        table = torch.tensor([list(range(16, 24)) + list(range(40, 48))])
        inputs = SimpleNamespace(
            is_prefill=True,
            cache_store_inputs=None,
            kv_cache_kernel_block_id_device=table,
        )
        for sharded, rank, expected in (
            (True, 0, [2048, 2175, 2176, 2429, 3071, -1, 5120, 5249]),
            (True, 1, [-1, -1, -1, -1, -1, 2048, -1, -1]),
            (True, 7, [-1] * 8),
            (False, 0, [2048, 2175, 2176, 2429, 3071, 5120]),
        ):
            with self.subTest(sharded=sharded, rank=rank):
                parallel = ParallelismConfig()
                parallel.tp_size, parallel.tp_rank = 8, rank
                parallel.prefill_cp_config.kv_cache_sharded = sharded
                with (
                    patch.object(
                        flashmla_dense_prefill,
                        "MlaFlashMLAPrefillOp",
                        return_value=SimpleNamespace(),
                    ),
                    patch.object(
                        MlaFlashMLAPrefillImpl, "create_params", return_value=None
                    ),
                    patch("torch.full", return_value=torch.tensor(1.0)),
                ):
                    impl = MlaFlashMLAPrefillImpl(
                        configs,
                        inputs,
                        [],
                        torch.empty(0),
                        parallelism_config=parallel,
                    )
                query_positions = positions if sharded else positions[:6]
                impl.fmha_params = SimpleNamespace(
                    positions_d=query_positions,
                    batch_indice_d=torch.zeros_like(query_positions),
                )
                with (
                    patch("torch.cuda.current_stream", return_value=None),
                    patch.object(torch.Tensor, "record_stream", return_value=None),
                ):
                    slots = impl._device_slot_mapping()
                self.assertEqual(slots.tolist(), expected)

    def test_swa_maps_both_pages_on_each_rank_while_full_keeps_page_owners(self):
        configs = AttentionConfigs()
        configs.head_num = 8
        configs.kv_lora_rank = 512
        configs.rope_head_dim = 64
        configs.nope_head_dim = configs.v_head_dim = 128
        configs.tokens_per_block = 128
        configs.kernel_tokens_per_block = 128
        configs.use_mla = True
        positions = torch.tensor([0, 127, 128, 255], dtype=torch.int64)
        inputs = SimpleNamespace(
            is_prefill=True,
            cache_store_inputs=None,
            kv_cache_kernel_block_id_device=torch.tensor([[11, 22]]),
        )
        for shards in (8, 16):
            for rank in range(shards):
                for sliding_window in (0, 128):
                    for fp8_compute in (False, True):
                        with self.subTest(
                            shards=shards,
                            rank=rank,
                            window=sliding_window,
                            fp8=fp8_compute,
                        ):
                            parallel = ParallelismConfig()
                            parallel.tp_size, parallel.tp_rank = shards, rank
                            parallel.prefill_cp_config.kv_cache_sharded = True
                            configs.sliding_window = sliding_window
                            configs.kv_cache_dtype = (
                                KvCacheDataType.FP8
                                if fp8_compute
                                else KvCacheDataType.BASE
                            )
                            configs.mla_fp8_compute = fp8_compute
                            configs.mla_fp8_q_scale = 0.5
                            configs.mla_fp8_kv_scale = 0.25
                            backend_args = {}

                            def backend(*args, **kwargs):
                                backend_args.update(kwargs)
                                return SimpleNamespace()

                            # The kernel backend and CUDA-only scalar allocation
                            # are external; wrapper construction and mapping stay real.
                            with (
                                patch.object(
                                    flashmla_dense_prefill,
                                    "MlaFlashMLAPrefillOp",
                                    backend,
                                ),
                                patch.object(
                                    MlaFlashMLAPrefillImpl,
                                    "create_params",
                                    return_value=None,
                                ),
                                patch("torch.full", return_value=torch.tensor(0.25)),
                            ):
                                impl = MlaFlashMLAPrefillImpl(
                                    configs,
                                    inputs,
                                    [],
                                    torch.empty(0),
                                    parallelism_config=parallel,
                                )
                            impl.fmha_params = SimpleNamespace(
                                positions_d=positions,
                                batch_indice_d=torch.zeros_like(positions),
                            )
                            with (
                                patch("torch.cuda.current_stream", return_value=None),
                                patch.object(
                                    torch.Tensor, "record_stream", return_value=None
                                ),
                            ):
                                slots = impl._device_slot_mapping()
                            expected = [1408, 1535, 2816, 2943]
                            if sliding_window == 0:
                                expected = (
                                    [1408, 1535, -1, -1]
                                    if rank == 0
                                    else (
                                        [-1, -1, 1408, 1535]
                                        if rank == 1
                                        else [-1, -1, -1, -1]
                                    )
                                )
                            self.assertEqual(slots.tolist(), expected)
                            if sliding_window == 0:
                                self.assertIs(
                                    backend_args["page_rr_cache_adapter"],
                                    impl.page_rr_cache_adapter,
                                )
                            else:
                                self.assertIsNone(backend_args["page_rr_cache_adapter"])
                            self.assertEqual(backend_args["kernel_page_tokens"], 128)
                            self.assertEqual(
                                backend_args["prefix_chunk_alignment_tokens"],
                                128,
                            )
                            self.assertEqual(backend_args["fp8_compute"], fp8_compute)
                            self.assertEqual(backend_args["q_scale"], 0.5)
                            self.assertEqual(backend_args["kv_scale"], 0.25)
                            self.assertEqual(
                                impl.kv_cache_write_op.kv_cache_type,
                                "fp8" if fp8_compute else "auto",
                            )
                            self.assertTrue(parallel.kv_page_rr_enabled())

    def test_wrapper_forwards_expanded_kv_budget(self) -> None:
        configs = AttentionConfigs()
        configs.head_num = 96
        configs.kv_lora_rank = 512
        configs.rope_head_dim = 64
        configs.nope_head_dim = 128
        configs.v_head_dim = 128
        configs.tokens_per_block = 8192
        configs.kernel_tokens_per_block = 4096
        configs.softmax_extra_scale = 1.0
        configs.use_mla = True
        configs.mla_prefill_expanded_kv_budget_gib = 5.0
        captured: dict[str, object] = {}

        def make_op(*args: object, **kwargs: object) -> object:
            captured["expanded_kv_budget_gib"] = float(kwargs["expanded_kv_budget_gib"])
            captured["kernel_page_tokens"] = int(kwargs["kernel_page_tokens"])
            captured["prefix_chunk_alignment_tokens"] = int(
                kwargs["prefix_chunk_alignment_tokens"]
            )
            return object()

        with (
            patch.object(
                flashmla_dense_prefill,
                "MlaFlashMLAPrefillOp",
                side_effect=make_op,
            ),
            patch.object(
                flashinfer_mla_wrapper,
                "NewMlaRotaryEmbeddingOp",
                return_value=object(),
            ),
            patch.object(
                flashinfer_mla_wrapper,
                "MlaKVCacheWriteOp",
                return_value=object(),
            ),
            patch.object(
                flashinfer_mla_wrapper.MlaFlashInferImplBase,
                "__init__",
                return_value=None,
            ),
        ):
            MlaFlashMLAPrefillImpl(
                configs,
                SimpleNamespace(),
                [],
                torch.empty(0),
            )

        self.assertEqual(captured["expanded_kv_budget_gib"], 5.0)
        self.assertEqual(captured["kernel_page_tokens"], 4096)
        self.assertEqual(captured["prefix_chunk_alignment_tokens"], 8192)

    def test_wrapper_does_not_expand_prefill_cp_config(self) -> None:
        configs = AttentionConfigs()
        configs.head_num = 96
        configs.kv_lora_rank = 512
        configs.rope_head_dim = 64
        configs.nope_head_dim = 128
        configs.v_head_dim = 128
        configs.tokens_per_block = 1024
        configs.kernel_tokens_per_block = 128
        configs.softmax_extra_scale = 1.0
        configs.use_mla = True
        configs.mla_prefill_expanded_kv_budget_gib = 5.0
        parallelism = SimpleNamespace(
            tp_size=8,
            tp_rank=5,
            kv_page_rr_enabled=lambda: False,
            prefill_cp_config=SimpleNamespace(kv_cache_sharded=True),
        )
        captured: dict[str, object] = {}

        def make_op(*args: object, **kwargs: object) -> object:
            captured.update(kwargs)
            return object()

        with patch.object(
            flashmla_dense_prefill,
            "MlaFlashMLAPrefillOp",
            side_effect=make_op,
        ), patch.object(
            flashinfer_mla_wrapper,
            "NewMlaRotaryEmbeddingOp",
            return_value=object(),
        ), patch.object(
            flashinfer_mla_wrapper,
            "MlaKVCacheWriteOp",
            return_value=object(),
        ), patch.object(
            flashinfer_mla_wrapper.MlaFlashInferImplBase,
            "__init__",
            return_value=None,
        ):
            impl = MlaFlashMLAPrefillImpl(
                configs,
                SimpleNamespace(),
                [],
                torch.empty(0),
                parallelism_config=parallelism,
            )

        self.assertIsNone(captured["page_rr_cache_adapter"])
        self.assertEqual(captured["kernel_page_tokens"], 128)
        self.assertEqual(captured["prefix_chunk_alignment_tokens"], 1024)
        self.assertIsNone(impl.page_rr_cache_adapter)

    def test_factory_skips_mla_impl_without_page_rr_prefill_capability(self) -> None:
        class UnsupportedImpl(MlaImplBase):
            @staticmethod
            def support(attn_configs: object, attn_inputs: object) -> bool:
                return True

            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

        class SupportedImpl(UnsupportedImpl):
            @classmethod
            def support_page_rr_prefill(cls) -> bool:
                return True

        weight = SimpleNamespace(
            weights=[],
            get_global_weight=lambda _name: torch.empty(0),
        )
        attn_inputs = SimpleNamespace(
            is_prefill=True,
            is_target_verify=False,
            is_mtp_draft_update=False,
            input_lengths_host=torch.tensor([1], dtype=torch.int32),
            prefix_lengths_host=torch.tensor([0], dtype=torch.int32),
        )
        configs = SimpleNamespace(
            indexer_topk=128,
            is_sparse=False,
            mla_fp8_compute=False,
        )
        parallelism = SimpleNamespace(
            kv_page_rr_enabled=lambda: True,
            prefill_cp_config=SimpleNamespace(is_enabled=lambda: False),
        )

        with patch.object(
            attn_factory,
            "PREFILL_MLA_IMPS",
            [UnsupportedImpl, SupportedImpl],
        ):
            impl = attn_factory.get_mla_impl(
                configs,
                weight,
                attn_inputs,
                parallelism_config=parallelism,
            )

        self.assertIsInstance(impl, SupportedImpl)


def _indptr(lengths: list[int]) -> torch.Tensor:
    values = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    return torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            values.cumsum(0, dtype=torch.int32),
        )
    )


def _padding_offset(lengths: list[int]) -> torch.Tensor:
    max_length = max(lengths)
    offsets = [
        batch * max_length - sum(lengths[:batch])
        for batch, length in enumerate(lengths)
        for _ in range(length)
    ]
    return torch.tensor(offsets, dtype=torch.int32, device="cuda")


def _attention_inputs(
    q_lens: list[int],
    prefix_lens: list[int],
    block_tables: list[torch.Tensor],
    current_group: int,
) -> SimpleNamespace:
    if len(q_lens) != len(prefix_lens):
        raise ValueError("q_lens and prefix_lens must have the same batch size")
    return SimpleNamespace(
        is_prefill=True,
        total_tokens=sum(q_lens),
        input_lengths_host=torch.tensor(q_lens, dtype=torch.int32),
        prefix_lengths_host=torch.tensor(prefix_lens, dtype=torch.int32),
        input_lengths=torch.tensor(q_lens, dtype=torch.int32, device="cuda"),
        prefix_lengths=torch.tensor(prefix_lens, dtype=torch.int32, device="cuda"),
        cu_seqlens=_indptr(q_lens),
        cu_kv_seqlens=_indptr(
            [q_len + prefix_len for q_len, prefix_len in zip(q_lens, prefix_lens)]
        ),
        padding_offset=_padding_offset(q_lens),
        kv_cache_kernel_block_id_device_by_group=block_tables,
        kv_cache_kernel_block_id_device=block_tables[current_group],
    )


def _assert_cuda_i32(test: TestCase, tensor: torch.Tensor) -> None:
    test.assertTrue(tensor.is_cuda)
    test.assertEqual(tensor.dtype, torch.int32)


@skipUnless(CUDA_AVAILABLE, "requires CUDA")
class FlashMlaDensePrefillParamsTest(TestCase):
    page_size = 128

    def _read_page_rr_prefix(
        self,
        *,
        raw_dtype: torch.dtype,
        mla_fp8_compute: bool,
        kv_scale: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        canonical = (
            torch.arange(2 * 576, dtype=torch.float32, device="cuda")
            .remainder(9)
            .sub(4)
            .reshape(2, 576)
            .to(raw_dtype)
        )
        adapter = SimpleNamespace(
            page_tokens=self.page_size,
            read_prefix_chunk=Mock(return_value=canonical),
        )
        op = object.__new__(MlaFlashMLAPrefillOp)
        op.page_rr_cache_adapter = adapter
        op._direct_attn_inputs = SimpleNamespace(
            kv_cache_kernel_block_id_device=torch.ones(
                (1, 1), dtype=torch.int32, device="cuda"
            )
        )
        op.kv_lora_rank = 512
        op.qk_rope_head_dim = 64
        op.fp8_compute = mla_fp8_compute
        op.kv_scale = kv_scale
        kv_cache = SimpleNamespace(
            kv_cache_base=torch.empty(
                (1, self.page_size, 576), dtype=raw_dtype, device="cuda"
            )
        )
        actual = op._read_page_rr_prefix(
            kv_cache,
            SimpleNamespace(),
        )
        adapter.read_prefix_chunk.assert_called_once()
        return canonical, actual

    def test_page_rr_fp8_prefix_is_dequantized_with_fixed_kv_scale(self) -> None:
        raw, actual = self._read_page_rr_prefix(
            raw_dtype=torch.float8_e4m3fn,
            mla_fp8_compute=True,
            kv_scale=0.5,
        )

        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(
            actual,
            raw.to(torch.bfloat16) * 0.5,
            rtol=0,
            atol=0,
        )

    def test_page_rr_fp8_conversion_is_scoped_to_each_chunk(self) -> None:
        raw_chunks = (
            torch.full(
                (3, 576),
                2,
                dtype=torch.float8_e4m3fn,
                device="cuda",
            ),
            torch.full(
                (1, 576),
                6,
                dtype=torch.float8_e4m3fn,
                device="cuda",
            ),
        )
        adapter = SimpleNamespace(
            read_prefix_chunk=Mock(side_effect=raw_chunks),
        )
        op = object.__new__(MlaFlashMLAPrefillOp)
        op.page_rr_cache_adapter = adapter
        op._direct_attn_inputs = SimpleNamespace(
            kv_cache_kernel_block_id_device=torch.ones(
                (1, 1), dtype=torch.int32, device="cuda"
            )
        )
        op.kv_lora_rank = 512
        op.qk_rope_head_dim = 64
        op.fp8_compute = True
        op.kv_scale = 0.25
        kv_cache = SimpleNamespace(
            kv_cache_base=torch.empty(
                (1, self.page_size, 576),
                dtype=torch.float8_e4m3fn,
                device="cuda",
            )
        )

        actual = tuple(
            op._read_page_rr_prefix(kv_cache, descriptor)
            for descriptor in (SimpleNamespace(), SimpleNamespace())
        )

        self.assertEqual(adapter.read_prefix_chunk.call_count, 2)
        self.assertEqual([chunk.shape[0] for chunk in actual], [3, 1])
        for restored, raw in zip(actual, raw_chunks, strict=True):
            self.assertEqual(restored.dtype, torch.bfloat16)
            torch.testing.assert_close(
                restored,
                raw.to(torch.bfloat16) * 0.25,
                rtol=0,
                atol=0,
            )

    def test_page_rr_bf16_prefix_is_forwarded_without_copy(self) -> None:
        raw, actual = self._read_page_rr_prefix(
            raw_dtype=torch.bfloat16,
            mla_fp8_compute=False,
        )

        self.assertIs(actual, raw)

    def test_page_rr_prefix_cache_dtype_must_match_fp8_mode(self) -> None:
        for raw_dtype, mla_fp8_compute in (
            (torch.bfloat16, True),
            (torch.float8_e4m3fn, False),
        ):
            with self.subTest(
                raw_dtype=raw_dtype, mla_fp8_compute=mla_fp8_compute
            ), self.assertRaisesRegex(RuntimeError, "raw cache"):
                self._read_page_rr_prefix(
                    raw_dtype=raw_dtype,
                    mla_fp8_compute=mla_fp8_compute,
                )

    def _make_unplanned_op(
        self,
        *,
        expanded_kv_budget_gib: float = 5.0,
    ) -> MlaFlashMLAPrefillOp:
        op = object.__new__(MlaFlashMLAPrefillOp)
        op.num_heads = 12
        op.kv_lora_rank = 512
        op.qk_rope_head_dim = 64
        op.qk_nope_head_dim = 128
        op.v_head_dim = 128
        op.kernel_page_tokens = self.page_size
        op.prefix_chunk_alignment_tokens = self.page_size
        op.expanded_kv_budget_gib = expanded_kv_budget_gib
        op.page_rr_cache_adapter = None
        op.flash_mla_cuda = SimpleNamespace(dense_prefill_fwd=lambda *args: None)
        op.fp8_compute = False
        op._prefix_producer = None
        op._forward_plan = None
        op._prefix_runtime_launches = ()
        op._forward_workspace = None
        return op

    def test_expanded_kv_cost_includes_fp8_attention_copy(self) -> None:
        op = self._make_unplanned_op()
        bf16_bytes = 12 * (128 + 64 + 128) * torch.bfloat16.itemsize
        self.assertEqual(op._expanded_kv_bytes_per_token(), bf16_bytes)

        op.fp8_compute = True
        fp8_bytes = 12 * (128 + 64 + 128) * torch.float8_e4m3fn.itemsize
        self.assertEqual(
            op._expanded_kv_bytes_per_token(),
            bf16_bytes + fp8_bytes,
        )

    def _make_plan_params(
        self,
        *,
        q_lens: Sequence[int],
        reuse_lens: Sequence[int],
    ) -> FlashMLADeviceParams:
        self.assertEqual(len(q_lens), len(reuse_lens))
        q_lens = list(q_lens)
        reuse_lens = list(reuse_lens)
        max_blocks = max(
            (q_len + reuse_len + self.page_size - 1) // self.page_size
            for q_len, reuse_len in zip(q_lens, reuse_lens, strict=True)
        )
        block_table = torch.arange(
            len(q_lens) * max_blocks, dtype=torch.int32, device="cuda"
        ).view(len(q_lens), max_blocks)
        return build_flashmla_device_params(
            _attention_inputs(q_lens, reuse_lens, [block_table], current_group=0),
            self.page_size,
        )

    def test_plan_builds_full_route_once_without_prefix_metadata(self) -> None:
        op = self._make_unplanned_op(expanded_kv_budget_gib=0)
        params = self._make_plan_params(q_lens=(128,), reuse_lens=(1024,))
        with patch.object(
            flashmla_dense_prefill,
            "plan_flashmla_forward",
            wraps=flashmla_dense_prefill.plan_flashmla_forward,
        ) as planner:
            op.plan(params)

        self.assertEqual(planner.call_count, 1)
        self.assertIs(op._forward_plan.route, FlashMLAForwardRoute.FULL)
        self.assertEqual(op._prefix_runtime_launches, ())
        self.assertIsNone(op._forward_workspace)

    def test_plan_materializes_contiguous_prefix_launch_in_one_storage(self) -> None:
        op = self._make_unplanned_op(
            expanded_kv_budget_gib=256 * 12 * (128 + 64 + 128) * 2 / GIB
        )
        params = self._make_plan_params(q_lens=(2, 3), reuse_lens=(128, 128))
        with patch.object(
            op,
            "_materialize_prefix_runtime_launches",
            wraps=op._materialize_prefix_runtime_launches,
        ) as materializer:
            op.plan(params)

        self.assertEqual(materializer.call_count, 1)
        self.assertEqual(len(op._prefix_runtime_launches), 1)
        launch = op._prefix_runtime_launches[0]
        self.assertEqual(launch.qo_indptr.cpu().tolist(), [0, 2, 5])
        self.assertEqual(launch.kv_indptr.cpu().tolist(), [0, 128, 256])
        self.assertEqual(launch.gather_qo_indptr.cpu().tolist(), [0, 0, 0])
        self.assertEqual(
            launch.batch_reuse_info.cpu().tolist(),
            [[0, 128, 0, 1], [1, 128, 2, 1]],
        )
        self.assertEqual(launch.destination_starts.cpu().tolist(), [0, 2])
        self.assertEqual(launch.q_range, (0, 5))

    def test_physical_chunk_alignment_keeps_kernel_page_table_offsets(self) -> None:
        expanded_bytes_per_token = 12 * (128 + 64 + 128) * 2
        op = self._make_unplanned_op(
            expanded_kv_budget_gib=2048 * expanded_bytes_per_token / GIB
        )
        op.kernel_page_tokens = 128
        op.prefix_chunk_alignment_tokens = 1024
        params = self._make_plan_params(q_lens=(1,), reuse_lens=(3072,))

        op.plan(params)

        self.assertEqual(len(op._prefix_runtime_launches), 2)
        first, second = op._prefix_runtime_launches
        self.assertEqual(first.spec.slices[0].prefix_len, 2048)
        self.assertEqual(second.spec.slices[0].prefix_start, 2048)
        self.assertEqual(first.batch_reuse_info.cpu().tolist(), [[0, 2048, 0, 16]])
        self.assertEqual(second.batch_reuse_info.cpu().tolist(), [[0, 1024, 16, 8]])

    def test_page_rr_descriptors_reuse_the_common_prefix_launches(self) -> None:
        expanded_bytes_per_token = 12 * (128 + 64 + 128) * 2
        op = self._make_unplanned_op(
            expanded_kv_budget_gib=256 * expanded_bytes_per_token / GIB
        )
        op.page_rr_cache_adapter = MlaPageRRCacheAdapter(128, 8, 0)
        params = self._make_plan_params(q_lens=(1,), reuse_lens=(512,))

        op.plan(params)

        self.assertEqual(len(op._prefix_runtime_launches), 2)
        descriptors = [
            launch.page_rr_descriptor for launch in op._prefix_runtime_launches
        ]
        self.assertEqual(
            [descriptor.prefix_starts for descriptor in descriptors],
            [(0,), (256,)],
        )
        self.assertEqual(
            [descriptor.prefix_lens for descriptor in descriptors],
            [(256,), (256,)],
        )

    def test_production_shape_has_no_aggregate_page_rr_descriptor(self) -> None:
        op = self._make_unplanned_op(expanded_kv_budget_gib=4.0)
        op.num_heads = 6
        op.kernel_page_tokens = 128
        op.prefix_chunk_alignment_tokens = 1024
        op.page_rr_cache_adapter = MlaPageRRCacheAdapter(
            page_tokens=1024,
            kernel_page_tokens=128,
            shard_size=16,
            shard_rank=0,
        )
        prefix_len = 1_032_192
        params = self._make_plan_params(
            q_lens=(2048,) * 32,
            reuse_lens=(prefix_len,) * 32,
        )

        op.plan(params)

        descriptors = [
            launch.page_rr_descriptor for launch in op._prefix_runtime_launches
        ]
        descriptor_tokens = [descriptor.total_tokens for descriptor in descriptors]
        self.assertEqual(op._forward_plan.capacity_tokens, 1_118_208)
        self.assertEqual(len(descriptors), 30)
        self.assertIsNone(op._page_rr_full_descriptor)
        self.assertEqual(sum(descriptor_tokens), 32 * prefix_len)
        self.assertEqual(max(descriptor_tokens), 1_118_208)
        self.assertNotIn(32 * prefix_len, descriptor_tokens)

    def test_fp8_production_shapes_budget_bf16_and_fp8_kv_together(self) -> None:
        prefix_len = 1_032_192
        params = self._make_plan_params(
            q_lens=(2048,) * 32,
            reuse_lens=(prefix_len,) * 32,
        )
        cases = (
            (6, 4.0, 5760, 745_472),
            (6, 6.0, 5760, 1_118_208),
            (12, 6.0, 11520, 559_104),
        )
        for num_heads, budget_gib, bytes_per_token, capacity_tokens in cases:
            with self.subTest(num_heads=num_heads, budget_gib=budget_gib):
                op = self._make_unplanned_op(expanded_kv_budget_gib=budget_gib)
                op.num_heads = num_heads
                op.fp8_compute = True
                op.kernel_page_tokens = 128
                op.prefix_chunk_alignment_tokens = 1024

                op.plan(params)

                self.assertEqual(op._expanded_kv_bytes_per_token(), bytes_per_token)
                self.assertEqual(op._forward_plan.capacity_tokens, capacity_tokens)
                self.assertLessEqual(
                    op._forward_plan.max_expanded_kv_tokens,
                    capacity_tokens,
                )

    def test_plan_materializes_b1_then_noncontiguous_launches_once(self) -> None:
        op = self._make_unplanned_op(
            expanded_kv_budget_gib=256 * 12 * (128 + 64 + 128) * 2 / GIB
        )
        params = self._make_plan_params(q_lens=(2, 3, 1), reuse_lens=(384, 0, 128))
        with patch.object(
            flashmla_dense_prefill,
            "plan_flashmla_forward",
            wraps=flashmla_dense_prefill.plan_flashmla_forward,
        ) as planner:
            op.plan(params)

        self.assertEqual(planner.call_count, 1)
        self.assertTrue(op._forward_plan.requires_fp32_accumulator)
        self.assertEqual(len(op._prefix_runtime_launches), 2)
        b1, noncontiguous = op._prefix_runtime_launches
        self.assertEqual(b1.qo_indptr.cpu().tolist(), [0, 2])
        self.assertEqual(b1.kv_indptr.cpu().tolist(), [0, 256])
        self.assertEqual(b1.gather_qo_indptr.cpu().tolist(), [0, 0])
        self.assertEqual(b1.batch_reuse_info.cpu().tolist(), [[0, 256, 0, 2]])
        self.assertEqual(b1.destination_starts.cpu().tolist(), [0])
        self.assertEqual(b1.q_range, (0, 2))
        self.assertEqual(noncontiguous.qo_indptr.cpu().tolist(), [0, 2, 3])
        self.assertEqual(noncontiguous.kv_indptr.cpu().tolist(), [0, 128, 256])
        self.assertEqual(noncontiguous.gather_qo_indptr.cpu().tolist(), [0, 0, 0])
        self.assertEqual(
            noncontiguous.batch_reuse_info.cpu().tolist(),
            [[0, 128, 2, 1], [1, 128, 8, 1]],
        )
        self.assertEqual(noncontiguous.destination_starts.cpu().tolist(), [0, 5])
        self.assertIsNone(noncontiguous.q_range)

    def test_fixed_q4_uses_row_stride_block_table(self) -> None:
        block_table = torch.tensor(
            [[11, 12, 13, 14], [21, 22, 23, 24]],
            dtype=torch.int32,
            device="cuda",
        )
        attn_inputs = _attention_inputs(
            q_lens=[4, 4],
            prefix_lens=[130, 5],
            block_tables=[block_table],
            current_group=0,
        )

        params = build_flashmla_device_params(attn_inputs, self.page_size)

        self.assertEqual(list(params.q_lens_host), [4, 4])
        self.assertEqual(list(params.prefix_lens_host), [130, 5])
        self.assertEqual(list(params.kv_lens_host), [134, 9])
        self.assertIs(params.attn_inputs, attn_inputs)
        self.assertIsNone(params.slot_mapping)

        expected = {
            "qo_indptr_d": [0, 4, 8],
            "prefill_ragged_kv_len_indptr_d": [0, 134, 143],
            "positions_d": [130, 131, 132, 133, 5, 6, 7, 8],
            "batch_indice_d": [0, 0, 0, 0, 1, 1, 1, 1],
            # Column 2 is an offset into the fully flattened page table.  The
            # second request therefore starts at the row stride (4), not at
            # the first request's live-page count (2).
            "batch_reuse_info_vec_d": [[0, 130, 0, 2], [1, 5, 4, 1]],
        }
        for name, values in expected.items():
            actual = getattr(params, name)
            _assert_cuda_i32(self, actual)
            torch.testing.assert_close(
                actual.cpu(), torch.tensor(values, dtype=torch.int32), rtol=0, atol=0
            )

        self.assertEqual(
            params.qo_indptr_d.data_ptr(), attn_inputs.cu_seqlens.data_ptr()
        )
        self.assertEqual(
            params.prefill_ragged_kv_len_indptr_d.data_ptr(),
            attn_inputs.cu_kv_seqlens.data_ptr(),
        )

    def test_ragged_q_positions_and_prefix_pages(self) -> None:
        block_table = torch.arange(100, 115, dtype=torch.int32, device="cuda").reshape(
            3, 5
        )
        attn_inputs = _attention_inputs(
            q_lens=[2, 5, 1],
            prefix_lens=[0, 128, 257],
            block_tables=[block_table],
            current_group=0,
        )

        params = build_flashmla_device_params(attn_inputs, self.page_size)

        expected = {
            "qo_indptr_d": [0, 2, 7, 8],
            "prefill_ragged_kv_len_indptr_d": [0, 2, 135, 393],
            "positions_d": [0, 1, 128, 129, 130, 131, 132, 257],
            "batch_indice_d": [0, 0, 1, 1, 1, 1, 1, 2],
            "batch_reuse_info_vec_d": [
                [0, 0, 0, 0],
                [1, 128, 5, 1],
                [2, 257, 10, 3],
            ],
        }
        self.assertEqual(list(params.q_lens_host), [2, 5, 1])
        self.assertEqual(list(params.prefix_lens_host), [0, 128, 257])
        self.assertEqual(list(params.kv_lens_host), [2, 133, 258])
        for name, values in expected.items():
            actual = getattr(params, name)
            _assert_cuda_i32(self, actual)
            torch.testing.assert_close(
                actual.cpu(), torch.tensor(values, dtype=torch.int32), rtol=0, atol=0
            )

    def test_consecutive_plans_do_not_overwrite_prior_metadata(self) -> None:
        group_zero = torch.tensor(
            [[10, 11, 12], [20, 21, 22]],
            dtype=torch.int32,
            device="cuda",
        )
        group_one = torch.tensor(
            [[110, 111, 112], [120, 121, 122]],
            dtype=torch.int32,
            device="cuda",
        )
        groups = [group_zero, group_one]
        first_inputs = _attention_inputs([4, 4], [129, 1], groups, current_group=0)
        first = build_flashmla_device_params(first_inputs, self.page_size)
        first_snapshot = {
            name: getattr(first, name).clone()
            for name in (
                "qo_indptr_d",
                "prefill_ragged_kv_len_indptr_d",
                "positions_d",
                "batch_indice_d",
                "batch_reuse_info_vec_d",
            )
        }

        # A subsequent planner invocation models the next forward selecting a
        # different HybridCache group.  It must produce fresh metadata rather
        # than update storage still owned by the earlier forward in place.
        second_inputs = _attention_inputs([1, 3], [260, 64], groups, current_group=1)
        second = build_flashmla_device_params(second_inputs, self.page_size)

        self.assertIsNot(first, second)
        for name, snapshot in first_snapshot.items():
            torch.testing.assert_close(getattr(first, name), snapshot, rtol=0, atol=0)
        torch.testing.assert_close(
            second.batch_reuse_info_vec_d.cpu(),
            torch.tensor([[0, 260, 0, 3], [1, 64, 3, 1]], dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            second.positions_d.cpu(),
            torch.tensor([260, 64, 65, 66], dtype=torch.int32),
            rtol=0,
            atol=0,
        )

    def test_slot_mapping_reads_live_hybrid_group_alias(self) -> None:
        initial_group = torch.tensor(
            [[10, 11, 12], [20, 21, 22]],
            dtype=torch.int32,
            device="cuda",
        )
        live_group = torch.tensor(
            [[110, 111, 112], [120, 121, 122]],
            dtype=torch.int32,
            device="cuda",
        )
        attn_inputs = _attention_inputs(
            [4, 4], [130, 5], [initial_group, live_group], current_group=0
        )
        params = build_flashmla_device_params(attn_inputs, self.page_size)

        impl = object.__new__(MlaFlashMLAPrefillImpl)
        impl.fmha_params = params
        impl.attn_inputs = attn_inputs
        impl.seq_size_per_block = self.page_size

        # K3 selects the physical HybridCache group immediately before each
        # layer.  Cache write must use that live alias, not the group that was
        # visible when the per-forward plan was first built.
        attn_inputs.kv_cache_kernel_block_id_device = live_group
        slot_mapping = impl._device_slot_mapping()

        self.assertIsNotNone(slot_mapping)
        assert slot_mapping is not None
        torch.testing.assert_close(
            slot_mapping.cpu(),
            torch.tensor(
                [
                    111 * 128 + 2,
                    111 * 128 + 3,
                    111 * 128 + 4,
                    111 * 128 + 5,
                    120 * 128 + 5,
                    120 * 128 + 6,
                    120 * 128 + 7,
                    120 * 128 + 8,
                ],
                dtype=torch.int64,
            ),
            rtol=0,
            atol=0,
        )

    def test_reuse_gather_reads_live_hybrid_group_alias(self) -> None:
        initial_group = torch.tensor(
            [[10, 11, 12], [20, 21, 22]],
            dtype=torch.int32,
            device="cuda",
        )
        live_group = torch.tensor(
            [[110, 111, 112], [120, 121, 122]],
            dtype=torch.int32,
            device="cuda",
        )
        attn_inputs = _attention_inputs(
            [4, 4], [130, 5], [initial_group, live_group], current_group=0
        )
        params = build_flashmla_device_params(attn_inputs, self.page_size)

        op = self._make_unplanned_op(expanded_kv_budget_gib=0)
        op.plan(params)

        # Model-layer dispatch switches this alias after the per-forward plan.
        # Both cache write and reused-KV gather must observe the same live group.
        attn_inputs.kv_cache_kernel_block_id_device = live_group
        compressed_kv = torch.empty((8, 512), dtype=torch.bfloat16, device="cuda")
        k_pe = torch.empty((8, 1, 64), dtype=torch.bfloat16, device="cuda")
        kv_cache = SimpleNamespace(
            kv_cache_base=torch.empty(1, dtype=torch.uint8, device="cuda")
        )
        captured: dict[str, torch.Tensor] = {}

        def fake_reuse_gather(
            final_compressed_kv: torch.Tensor,
            final_k_pe: torch.Tensor,
            suffix_compressed_kv: torch.Tensor,
            suffix_k_pe: torch.Tensor,
            kv_cache_base: torch.Tensor,
            page_indices: torch.Tensor,
            batch_reuse_info: torch.Tensor,
            qo_indptr: torch.Tensor,
            page_size: int,
        ) -> None:
            captured["page_indices"] = page_indices
            captured["batch_reuse_info"] = batch_reuse_info
            captured["qo_indptr"] = qo_indptr
            self.assertEqual(page_size, self.page_size)

        with patch.object(
            rtp_llm_ops,
            "reuse_kv_cache_indexed_batched",
            side_effect=fake_reuse_gather,
        ):
            gathered_compressed_kv, gathered_k_pe = op._gather_reused_kv(
                compressed_kv, k_pe, kv_cache
            )

        self.assertEqual(captured["page_indices"].data_ptr(), live_group.data_ptr())
        self.assertNotEqual(
            captured["page_indices"].data_ptr(), initial_group.data_ptr()
        )
        self.assertIs(captured["batch_reuse_info"], params.batch_reuse_info_vec_d)
        self.assertIs(captured["qo_indptr"], params.qo_indptr_d)
        self.assertEqual(tuple(gathered_compressed_kv.shape), (143, 512))
        self.assertEqual(tuple(gathered_k_pe.shape), (143, 64))

    def test_cacheless_prefill_does_not_require_block_table(self) -> None:
        empty_table = torch.empty((1, 0), dtype=torch.int32, device="cuda")
        attn_inputs = _attention_inputs(
            q_lens=[4],
            prefix_lens=[0],
            block_tables=[empty_table],
            current_group=0,
        )

        params = build_flashmla_device_params(attn_inputs, self.page_size)

        self.assertFalse(params.has_reuse_cache)

    def test_rejects_query_write_past_block_table(self) -> None:
        block_table = torch.tensor([[17]], dtype=torch.int32, device="cuda")
        attn_inputs = _attention_inputs(
            q_lens=[4],
            prefix_lens=[127],
            block_tables=[block_table],
            current_group=0,
        )

        params = build_flashmla_device_params(attn_inputs, self.page_size)
        impl = object.__new__(MlaFlashMLAPrefillImpl)
        impl.page_rr_cache_adapter = None
        impl.seq_size_per_block = self.page_size
        with self.assertRaisesRegex(RuntimeError, "query write exceeds"):
            impl._validate_direct_cache_capacity(params, block_table)

    def test_page_rr_validates_rank_local_block_table_width(self) -> None:
        rank_zero_table = torch.tensor([[17, 18]], dtype=torch.int32, device="cuda")
        rank_seven_table = torch.tensor([[27]], dtype=torch.int32, device="cuda")
        # Nine global pages: rank 0 owns pages 0 and 8, while rank 7 owns only
        # page 7.  The cache adapter must validate those local widths rather
        # than requiring nine columns on every rank.
        for rank, table in ((0, rank_zero_table), (7, rank_seven_table)):
            with self.subTest(rank=rank):
                attn_inputs = _attention_inputs(
                    q_lens=[1],
                    prefix_lens=[8 * self.page_size],
                    block_tables=[table],
                    current_group=0,
                )
                params = build_flashmla_device_params(attn_inputs, self.page_size)
                adapter = MlaPageRRCacheAdapter(
                    page_tokens=self.page_size,
                    shard_size=8,
                    shard_rank=rank,
                )
                adapter.validate_block_table_capacity(table, params.kv_lens_host)
                self.assertIs(params.attn_inputs.kv_cache_kernel_block_id_device, table)
                self.assertEqual(params.batch_reuse_info_host[0], (0, 1024, 0, 8))

        too_narrow = _attention_inputs(
            q_lens=[1],
            prefix_lens=[8 * self.page_size],
            block_tables=[rank_zero_table[:, :1]],
            current_group=0,
        )
        with self.assertRaisesRegex(RuntimeError, "rank-local block table"):
            MlaPageRRCacheAdapter(
                page_tokens=self.page_size,
                shard_size=8,
                shard_rank=0,
            ).validate_block_table_capacity(
                too_narrow.kv_cache_kernel_block_id_device,
                (8 * self.page_size + 1,),
            )

    def test_page_rr_slot_mapping_writes_only_owner_pages(self) -> None:
        block_table = torch.tensor([[11, 12]], dtype=torch.int32, device="cuda")
        attn_inputs = _attention_inputs(
            q_lens=[4],
            prefix_lens=[127],
            block_tables=[block_table],
            current_group=0,
        )
        params = build_flashmla_device_params(attn_inputs, self.page_size)
        impl = object.__new__(MlaFlashMLAPrefillImpl)
        impl.fmha_params = params
        impl.attn_inputs = attn_inputs
        impl.seq_size_per_block = self.page_size
        impl.page_rr_cache_adapter = MlaPageRRCacheAdapter(
            page_tokens=self.page_size,
            shard_size=8,
            shard_rank=0,
        )

        slot_mapping = impl._device_slot_mapping()

        self.assertIsNotNone(slot_mapping)
        assert slot_mapping is not None
        torch.testing.assert_close(
            slot_mapping.cpu(),
            torch.tensor([11 * 128 + 127, -1, -1, -1], dtype=torch.int64),
            rtol=0,
            atol=0,
        )

    def test_page_rr_unused_tail_rank_accepts_empty_local_block_table(self) -> None:
        block_table = torch.empty((1, 0), dtype=torch.int32, device="cuda")
        attn_inputs = _attention_inputs(
            q_lens=[4],
            prefix_lens=[1],
            block_tables=[block_table],
            current_group=0,
        )
        params = build_flashmla_device_params(attn_inputs, self.page_size)
        impl = object.__new__(MlaFlashMLAPrefillImpl)
        impl.fmha_params = params
        impl.attn_inputs = attn_inputs
        impl.seq_size_per_block = self.page_size
        impl.page_rr_cache_adapter = MlaPageRRCacheAdapter(
            page_tokens=self.page_size,
            shard_size=8,
            shard_rank=7,
        )

        slot_mapping = impl._device_slot_mapping()

        self.assertIsNotNone(slot_mapping)
        assert slot_mapping is not None
        torch.testing.assert_close(
            slot_mapping.cpu(),
            torch.full((4,), -1, dtype=torch.int64),
            rtol=0,
            atol=0,
        )

        with self.assertRaisesRegex(RuntimeError, "rank-local block table"):
            MlaPageRRCacheAdapter(
                page_tokens=self.page_size,
                shard_size=8,
                shard_rank=0,
            ).validate_block_table_capacity(
                block_table,
                (5,),
            )


if __name__ == "__main__":
    main()
