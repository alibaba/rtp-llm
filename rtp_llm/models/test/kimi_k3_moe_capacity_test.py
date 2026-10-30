import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3, KimiK3Mtp
from rtp_llm.models.kimi_k3.kimi_k31 import KimiK31, KimiK31Mtp
from rtp_llm.ops import CPRotateMethod, KvCacheDataType, ParallelismConfig, RoleType

_PLAN_NAME = "rtp_llm.models_py.modules.factory.attention.mla_prefill_plan"
_PLAN_PATH = (
    Path(__file__).resolve().parents[2]
    / "models_py/modules/factory/attention/mla_prefill_plan.py"
)
_SPEC = importlib.util.spec_from_file_location(_PLAN_NAME, _PLAN_PATH)
_PLAN = importlib.util.module_from_spec(_SPEC)
with patch.dict(sys.modules, {_PLAN_NAME: _PLAN}):
    _SPEC.loader.exec_module(_PLAN)
Bf16MlaWorkspaceBudget = _PLAN.Bf16MlaWorkspaceBudget


class KimiK3MoeCapacityTest(unittest.TestCase):
    def model(
        self, owner=KimiK3, role=RoleType.PREFILL, tp=8, batch=16, width=1, chunk=128
    ):
        model = object.__new__(owner)
        model.model_config = NS(
            max_seq_len=4096,
            gen_num_per_cycle=width - 1,
            mm_model_config=NS(is_multimodal=False),
            moe_prefill_max_tokens_per_rank=999999,
            k3_runtime_config=NS(dsa=None),
            prefill_kv_workspace_budget_bytes=6 * 1024**3,
            _prefill_kv_workspace_budget_explicit=False,
            _paired_target_is_multimodal=False,
            attn_config=NS(
                is_sparse=False,
                kv_cache_dtype=KvCacheDataType.FP8,
                head_num=96,
                kv_lora_rank=512,
                nope_head_dim=128,
                rope_head_dim=64,
                v_head_dim=128,
                tokens_per_block=128,
            ),
        )
        model.runtime_config = NS(
            fifo_scheduler_config=NS(
                prefill_chunk_size=chunk,
                max_context_batch_size=3,
                max_batch_tokens_size=12288,
                cp_force_single_prefill=True,
                force_single_prefill=False,
            ),
            warm_up=False,
            warm_up_with_loss=False,
            use_batch_decode_scheduler=False,
            batch_decode_scheduler_config=NS(
                batch_decode_scheduler_warmup_type=0,
                batch_decode_scheduler_batch_size=batch,
            ),
        )
        model.parallelism_config = ParallelismConfig()
        model.parallelism_config.role_type = role
        model.parallelism_config.tp_size = tp
        cp = model.parallelism_config.prefill_cp_config
        cp.method = CPRotateMethod.DISABLED
        cp.preserve_tp_layout = False
        model.hw_kernel_config = NS(
            enable_cuda_graph=False,
            decode_capture_batch_sizes=[],
            generation_prefill_capture_token_buckets=[],
        )
        model.kv_cache_config = NS(multi_task_prompt_tokens={})
        model.max_generate_batch_size = batch
        model.weight = None
        model.fmha_config = None
        model.moe_config = None
        return model

    def capacity(self, model):
        self.assertFalse(model.parallelism_config.prefill_cp_config.is_enabled())
        target, draft = Mock(), Mock()
        with (
            patch.dict(
                sys.modules,
                {
                    "rtp_llm.models_py.model_desc.kimi_k3": NS(KimiK3Model=target),
                    "rtp_llm.models_py.model_desc.kimi_k3_mtp": NS(
                        KimiK3MtpModel=draft
                    ),
                    _PLAN_NAME: _PLAN,
                },
            ),
            patch("torch.cuda.empty_cache"),
        ):
            result = model._create_python_model()
            ctor = draft if isinstance(model, KimiK3Mtp) else target
            self.assertIs(result, ctor.return_value)
            self.assertIs(model.py_model, result)
            self.assertIsNone(ctor.call_args.kwargs["prefill_staging_tokens"])
            return ctor.call_args.kwargs["max_moe_tokens"]

    def test_query_cp_target_and_private_native_draft_allocate_their_own_rows(self):
        target = self.model(role=RoleType.PREFILL, batch=2, width=4, chunk=0)
        draft = self.model(KimiK3Mtp, role=RoleType.PREFILL, batch=2, width=4, chunk=0)
        for model in (target, draft):
            model.model_config.max_seq_len = 32
            model.runtime_config.fifo_scheduler_config.max_batch_tokens_size = 1
            model.runtime_config.fifo_scheduler_config.max_context_batch_size = 2
            model.runtime_config.fifo_scheduler_config.cp_force_single_prefill = False
        target.model_config.k3_runtime_config.dsa = NS()
        cp = target.parallelism_config.prefill_cp_config
        cp.method = CPRotateMethod.ALL_GATHER
        cp.prefill_cp_size = 8
        cp.preserve_tp_layout = True
        cp.kv_cache_sharded = True
        target_ctor, draft_ctor = Mock(), Mock()
        with (
            patch.dict(
                sys.modules,
                {
                    "rtp_llm.models_py.model_desc.kimi_k3": NS(KimiK3Model=target_ctor),
                    "rtp_llm.models_py.model_desc.kimi_k3_mtp": NS(
                        KimiK3MtpModel=draft_ctor
                    ),
                },
            ),
            patch("torch.cuda.empty_cache"),
        ):
            target._create_python_model()
            draft._create_python_model()
        self.assertEqual(target_ctor.call_args.kwargs["max_moe_tokens"], 64)
        self.assertEqual(draft_ctor.call_args.kwargs["max_moe_tokens"], 32)

    def test_query_cp_capacity_covers_unchunked_prefill_warmup_batch(self):
        model = self.model(batch=1, chunk=0)
        model.model_config.max_seq_len = 131090
        model.model_config.attn_config.is_sparse = True
        model.model_config.k3_runtime_config.dsa = NS()
        model.runtime_config.warm_up = True
        scheduler = model.runtime_config.fifo_scheduler_config
        scheduler.max_context_batch_size = 2
        cp = model.parallelism_config.prefill_cp_config
        cp.method = CPRotateMethod.ALL_GATHER
        cp.prefill_cp_size = 8
        cp.preserve_tp_layout = True
        cp.kv_cache_sharded = True
        ctor = Mock()
        with (
            patch.dict(
                sys.modules,
                {"rtp_llm.models_py.model_desc.kimi_k3": NS(KimiK3Model=ctor)},
            ),
            patch("torch.cuda.empty_cache"),
        ):
            model._create_python_model()

        # Each of the two 131089-token sequences pads to 131104 tokens at TP=8.
        warmup_rows = 2 * 131104
        self.assertGreaterEqual(
            ctor.call_args.kwargs["prefill_staging_tokens"], warmup_rows
        )
        self.assertGreaterEqual(ctor.call_args.kwargs["max_moe_tokens"], warmup_rows)

    def test_chunked_query_cp_staging_covers_serving_and_warmup_rows(self):
        for force_single, warmup, expected_rows in (
            (True, False, 128),
            (True, True, 176),
            (False, False, 368),
        ):
            with self.subTest(force_single=force_single, warmup=warmup):
                model = self.model(tp=8, batch=16, chunk=128)
                model.model_config.k3_runtime_config.dsa = NS()
                model.model_config.attn_config.is_sparse = True
                model.runtime_config.warm_up = warmup
                scheduler = model.runtime_config.fifo_scheduler_config
                scheduler.cp_force_single_prefill = force_single
                cp = model.parallelism_config.prefill_cp_config
                cp.method = CPRotateMethod.ALL_GATHER
                cp.prefill_cp_size = 8
                cp.preserve_tp_layout = True
                cp.kv_cache_sharded = True

                ctor = Mock()
                with (
                    patch.dict(
                        sys.modules,
                        {"rtp_llm.models_py.model_desc.kimi_k3": NS(KimiK3Model=ctor)},
                    ),
                    patch("torch.cuda.empty_cache"),
                ):
                    model._create_python_model()

                # Serving has one or sixteen rows, and warmup can have three.
                # Each extra row can introduce up to 15 padding tokens at TP=8.
                self.assertEqual(
                    ctor.call_args.kwargs["prefill_staging_tokens"], expected_rows
                )
                self.assertGreaterEqual(
                    ctor.call_args.kwargs["max_moe_tokens"], expected_rows
                )

    def test_chunk_budget_is_not_clamped_to_fifo_admission_budget(self):
        model = self.model(chunk=8192)
        model.runtime_config.fifo_scheduler_config.max_batch_tokens_size = 7000
        self.assertEqual(self.capacity(model), 8192)
        self.assertEqual(model.model_config.moe_prefill_max_tokens_per_rank, 999999)

    def test_target_and_draft_cover_all_padded_decode_and_capture_rows(self):
        for owner in (KimiK3, KimiK3Mtp):
            for role in (RoleType.PREFILL, RoleType.DECODE, RoleType.PDFUSION):
                for tp, batch, width, bucket in ((8, 63, 3, 63), (4, 100, 4, 99)):
                    with self.subTest(owner=owner, role=role, tp=tp, width=width):
                        model = self.model(owner, role, tp, batch, width, chunk=1)
                        model.hw_kernel_config.enable_cuda_graph = True
                        model.hw_kernel_config.decode_capture_batch_sizes = [1, bucket]
                        # Independently enumerate valid request padding.
                        physical_rows = []
                        for requests in range(1, max(batch, bucket) + 1):
                            for q in (1, width):
                                padded = requests
                                while padded * q % tp:
                                    padded += 1
                                physical_rows.append(padded * q)
                        self.assertEqual(self.capacity(model), max(physical_rows))

    def test_graph_buckets_do_not_resize_serving_capacity(self):
        model = self.model(role=RoleType.DECODE, batch=9, width=5)
        model.hw_kernel_config.decode_capture_batch_sizes = [1024]
        for enabled in (False, True):
            model.hw_kernel_config.enable_cuda_graph = enabled
            self.assertEqual(self.capacity(model), 80)

    def test_fixed_batch_expands_decode_workspace(self):
        model = self.model(role=RoleType.DECODE, tp=8, batch=8, width=3)
        self.assertEqual(self.capacity(model), 24)
        runtime = model.runtime_config
        runtime.use_batch_decode_scheduler = True
        runtime.batch_decode_scheduler_config.batch_decode_scheduler_batch_size = 9

        # Nine requests pad to 16 so verify rows divide evenly across TP8.
        self.assertEqual(self.capacity(model), 16 * 3)
        self.assertEqual(model.max_generate_batch_size, 9)
        self.assertEqual(runtime.max_generate_batch_size, 9)

    def test_prefill_warmup_and_generation_graph_bounds(self):
        model = self.model(width=3)
        model.runtime_config.warm_up = True
        model.runtime_config.warm_up_with_loss = True
        self.assertEqual(self.capacity(model), 12288)
        model.model_config.mm_model_config.is_multimodal = True
        self.assertEqual(self.capacity(model), 128)

        model = self.model(role=RoleType.PDFUSION)
        model.hw_kernel_config.enable_cuda_graph = True
        model.hw_kernel_config.generation_prefill_capture_token_buckets = [256, 1024]
        self.assertEqual(self.capacity(model), 1024)

    def test_unchunked_single_prompt_and_full_warmup(self):
        model = self.model(chunk=0)
        model.runtime_config.fifo_scheduler_config.max_batch_tokens_size = 1024
        self.assertEqual(self.capacity(model), 4096)
        model.runtime_config.warm_up = True
        self.assertEqual(self.capacity(model), 12288)

        model = self.model(role=RoleType.DECODE, chunk=0)
        model.runtime_config.warm_up = True
        model.runtime_config.use_batch_decode_scheduler = True
        batch_scheduler = model.runtime_config.batch_decode_scheduler_config
        batch_scheduler.batch_decode_scheduler_warmup_type = 1
        self.assertEqual(self.capacity(model), 12288)

    def test_native_draft_warmup_follows_target_multimodality(self):
        for target_cls, draft_cls, alias_count in (
            (KimiK3, KimiK3Mtp, 2),
            (KimiK31, KimiK31Mtp, 0),
        ):
            with self.subTest(draft=draft_cls.__name__):
                target = self.model(target_cls, chunk=0)
                draft = self.model(draft_cls, chunk=0)
                draft.runtime_config = target.runtime_config
                draft.runtime_config.fifo_scheduler_config.max_batch_tokens_size = 1024
                draft.runtime_config.warm_up = True
                for model in (target, draft):
                    config = model.model_config
                    config.vocab_size = 1
                    config.hidden_size = 1
                    config.data_type = "bf16"
                    config.enable_fp32_lm_head = False
                    config.mm_related_params = NS(special_token_ids={})

                for is_multimodal, expected_rows in ((False, 12288), (True, 4096)):
                    target.model_config.mm_model_config.is_multimodal = is_multimodal
                    aliases = draft_cls.speculative_weight_alias_names(
                        target, draft.model_config
                    )
                    self.assertEqual(len(aliases), alias_count)
                    draft._weight_alias_owner = target if aliases else None
                    self.assertEqual(self.capacity(draft), expected_rows)
                    self.assertFalse(draft.model_config.mm_model_config.is_multimodal)

    def test_unchunked_batch_budget_is_not_clamped_to_context_request_count(self):
        model = self.model(chunk=0)
        model.runtime_config.fifo_scheduler_config.max_batch_tokens_size = 32768
        self.assertEqual(self.capacity(model), 32768)

    def test_tokenized_system_prompt_extends_decode_capacity(self):
        model = self.model(KimiK3Mtp, RoleType.DECODE)
        model.kv_cache_config.multi_task_prompt_tokens = {1: [0] * 129}
        self.assertEqual(self.capacity(model), 136)

    def test_direct_construction_retains_explicit_prefill_budget(self):
        model = self.model()
        model.runtime_config = None
        self.assertEqual(self.capacity(model), 1000000)

    def test_default_bf16_workspace_covers_execution_rows_and_prefix(self):
        for owner in (KimiK3, KimiK3Mtp):
            with self.subTest(owner=owner):
                model = self.model(owner, chunk=0)
                model.model_config.max_seq_len = 1000000
                model.model_config.attn_config.kv_cache_dtype = KvCacheDataType.BASE
                with patch("torch.cuda.get_device_capability", return_value=(10, 0)):
                    rows = self.capacity(model)
                budget = model.model_config.prefill_kv_workspace_budget_bytes
                self.assertGreater(budget, 6 * 1024**3)
                workspace = Bf16MlaWorkspaceBudget(budget, 8, 512, 64, rows)
                for prefix in (0, 128 * 4096):
                    workspace.plan_capacity((rows,), (prefix,), 128, 12 * 320 * 2)

    def test_explicit_bf16_workspace_is_checked_before_model_creation(self):
        model = self.model(chunk=0)
        config = model.model_config
        config.max_seq_len = 1000000
        config.attn_config.kv_cache_dtype = KvCacheDataType.BASE
        config._prefill_kv_workspace_budget_explicit = True
        with patch("torch.cuda.get_device_capability", return_value=(10, 0)):
            with self.assertRaisesRegex(ValueError, "cannot cover .* execution tokens"):
                self.capacity(model)
            config.prefill_kv_workspace_budget_bytes = 12 * 1024**3
            self.capacity(model)
        self.assertEqual(config.prefill_kv_workspace_budget_bytes, 12 * 1024**3)

    def test_other_mla_budget_paths_keep_their_existing_limits(self):
        for cache, sparse, capability, budget in (
            (KvCacheDataType.FP8, False, 10, 0),
            (KvCacheDataType.FP8, True, 10, 1),
            (KvCacheDataType.BASE, True, 10, 1),
            (KvCacheDataType.BASE, False, 9, 1),
            (KvCacheDataType.BASE, False, 10, 0),
        ):
            with self.subTest(
                cache=cache, sparse=sparse, capability=capability, budget=budget
            ):
                model = self.model()
                config = model.model_config
                config.attn_config.kv_cache_dtype = cache
                config.attn_config.is_sparse = sparse
                config.prefill_kv_workspace_budget_bytes = budget
                config._prefill_kv_workspace_budget_explicit = True
                with patch(
                    "torch.cuda.get_device_capability", return_value=(capability, 0)
                ):
                    self.capacity(model)
                self.assertEqual(config.prefill_kv_workspace_budget_bytes, budget)

    def test_dense_fp8_workspace_requires_one_expanded_cache_page(self):
        model = self.model()
        config = model.model_config
        # 12 heads per rank, each with 320 BF16+FP8 K/V elements and an
        # extra 128 BF16 NoPE elements from the unpacked KV-up projection.
        page_bytes = 128 * 12 * (320 * 3 + 128 * 2)
        config._prefill_kv_workspace_budget_explicit = True
        config.prefill_kv_workspace_budget_bytes = page_bytes - 1
        with self.assertRaisesRegex(
            ValueError, f"requires at least {page_bytes} bytes"
        ):
            self.capacity(model)
        config.prefill_kv_workspace_budget_bytes = page_bytes
        self.capacity(model)
        self.assertEqual(config.prefill_kv_workspace_budget_bytes, page_bytes)


if __name__ == "__main__":
    unittest.main()
