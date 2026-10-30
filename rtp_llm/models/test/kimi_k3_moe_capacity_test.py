import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3, KimiK3Mtp
from rtp_llm.ops import RoleType


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
        )
        model.runtime_config = NS(
            fifo_scheduler_config=NS(
                prefill_chunk_size=chunk,
                max_context_batch_size=3,
                max_batch_tokens_size=12288,
            ),
            warm_up=False,
            warm_up_with_loss=False,
            use_batch_decode_scheduler=False,
            batch_decode_scheduler_config=NS(batch_decode_scheduler_warmup_type=0),
        )
        model.parallelism_config = NS(
            role_type=role,
            tp_size=tp,
            ffn_disaggregate_config=NS(enable_ffn_disaggregate=False),
        )
        model.hw_kernel_config = NS(
            enable_cuda_graph=False,
            decode_capture_batch_sizes=[],
            generation_prefill_capture_token_buckets=[],
        )
        model.kv_cache_config = NS(multi_task_prompt_tokens={})
        model.max_generate_batch_size = batch
        model.weight = None
        model.fmha_config = None
        model.device_resource_config = None
        model.moe_config = None
        return model

    def capacity(self, model):
        with (
            patch("rtp_llm.models_py.model_desc.kimi_k3.KimiK3Model") as target,
            patch("rtp_llm.models_py.model_desc.kimi_k3_mtp.KimiK3MtpModel") as draft,
            patch("torch.cuda.empty_cache"),
        ):
            result = model._create_python_model()
            ctor = draft if isinstance(model, KimiK3Mtp) else target
            self.assertIs(result, ctor.return_value)
            self.assertIs(model.py_model, result)
            return ctor.call_args.kwargs["max_moe_tokens"]

    def test_chunk_budget_is_not_clamped_to_fifo_admission_budget(self):
        model = self.model(chunk=8192)
        model.runtime_config.fifo_scheduler_config.max_batch_tokens_size = 7000
        self.assertEqual(self.capacity(model), 8192)
        self.assertEqual(model.model_config.moe_prefill_max_tokens_per_rank, 999999)

    def test_target_and_draft_cover_all_padded_decode_and_capture_rows(self):
        for owner in (KimiK3, KimiK3Mtp):
            for role in (RoleType.PREFILL, RoleType.DECODE, RoleType.PDFUSION):
                for tp, batch, width, bucket in ((8, 63, 3, 73), (4, 100, 4, 129)):
                    with self.subTest(owner=owner, role=role, tp=tp, width=width):
                        model = self.model(owner, role, tp, batch, width, chunk=1)
                        model.hw_kernel_config.enable_cuda_graph = True
                        model.hw_kernel_config.decode_capture_batch_sizes = [
                            1, bucket
                        ]
                        # Independently enumerate valid request padding.
                        physical_rows = []
                        for requests in range(1, max(batch, bucket) + 1):
                            for q in (1, width):
                                padded = requests
                                while padded * q % tp:
                                    padded += 1
                                physical_rows.append(padded * q)
                        self.assertEqual(self.capacity(model), max(physical_rows))

    def test_disabled_graph_buckets_do_not_inflate_decode(self):
        model = self.model(role=RoleType.DECODE, batch=9, width=5)
        model.hw_kernel_config.decode_capture_batch_sizes = [1024]
        self.assertEqual(self.capacity(model), 80)

    def test_prefill_warmup_and_generation_graph_bounds(self):
        model = self.model(width=3)
        model.runtime_config.warm_up = True
        model.runtime_config.warm_up_with_loss = True
        self.assertEqual(self.capacity(model), 12288)
        model.model_config.mm_model_config.is_multimodal = True
        self.assertEqual(self.capacity(model), 128)

        model = self.model(role=RoleType.PDFUSION)
        model.hw_kernel_config.enable_cuda_graph = True
        model.hw_kernel_config.generation_prefill_capture_token_buckets = [
            256, 1024
        ]
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


if __name__ == "__main__":
    unittest.main()
