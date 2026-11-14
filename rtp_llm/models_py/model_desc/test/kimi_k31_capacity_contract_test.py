"""Keep Query-CP row transport capacity consistent with non-chunk admission."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3
from rtp_llm.models_py.distributed.zigzag_row_exchange import validate_plan
from rtp_llm.models_py.modules.hybrid.natural_zigzag_plan import NaturalZigzagPlan
from rtp_llm.ops import CPRotateMethod, ParallelismConfig, RoleType, RuntimeConfig


class KimiK31CapacityContractTest(unittest.TestCase):
    def test_nonchunk_query_cp_capacity_covers_admissible_request_padding(self):
        runtime = RuntimeConfig()
        runtime.warm_up = False
        runtime.max_generate_batch_size = 32
        scheduler = runtime.fifo_scheduler_config
        scheduler.prefill_chunk_size = 0
        scheduler.max_batch_tokens_size = 262144
        scheduler.max_context_batch_size = 1
        scheduler.cp_force_single_prefill = False
        parallel = ParallelismConfig()
        parallel.tp_size = 8
        parallel.role_type = RoleType.PREFILL
        parallel.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
        parallel.prefill_cp_config.prefill_cp_size = 8
        parallel.prefill_cp_config.preserve_tp_layout = True
        config = SimpleNamespace(
            max_seq_len=65536,
            moe_prefill_max_tokens_per_rank=262144,
            gen_num_per_cycle=2,
            is_mtp=False,
            mm_model_config=SimpleNamespace(is_multimodal=False),
            k3_runtime_config=SimpleNamespace(dsa=object()),
        )
        owner = SimpleNamespace(
            model_config=config,
            runtime_config=runtime,
            parallelism_config=parallel,
            max_generate_batch_size=runtime.max_generate_batch_size,
            hw_kernel_config=SimpleNamespace(enable_cuda_graph=False),
            kv_cache_config=SimpleNamespace(multi_task_prompt_tokens={}),
            weight=object(),
            fmha_config=object(),
            moe_config=object(),
        )
        # Only weight/model allocation is replaced. The production startup
        # capacity calculation and production row planner/validator run intact.
        with (
            patch("rtp_llm.models_py.model_desc.kimi_k3.KimiK3Model") as construct,
            patch("torch.cuda.empty_cache"),
        ):
            KimiK3._create_python_model(owner)
        capacity = construct.call_args.kwargs["prefill_staging_tokens"]
        lengths = (8449,) * 31
        self.assertLessEqual(sum(lengths), scheduler.max_batch_tokens_size)
        self.assertLessEqual(max(lengths), config.max_seq_len)
        self.assertLessEqual(len(lengths), runtime.max_generate_batch_size)
        for rank in range(parallel.tp_size):
            with self.subTest(rank=rank):
                plan = NaturalZigzagPlan(lengths, parallel.tp_size, rank)
                validate_plan(
                    plan,
                    "to_natural",
                    parallel.tp_size,
                    rank,
                    capacity // parallel.tp_size,
                )


if __name__ == "__main__":
    unittest.main()
