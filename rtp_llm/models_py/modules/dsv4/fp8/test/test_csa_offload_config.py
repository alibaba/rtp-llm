import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.models.dsv4_kv_cache import build_dsv4_kv_cache_spec_descs
from rtp_llm.models_py.modules.dsv4.offload_config import CsaOffloadConfig
from rtp_llm.ops import CacheMemoryPlacement


class CsaOffloadConfigTest(unittest.TestCase):
    def test_offload_rejects_both_separated_roles(self):
        from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model
        from rtp_llm.ops import RoleType

        resource = SimpleNamespace(is_speculative=False, is_decode_role=False)
        for role in (RoleType.PREFILL, RoleType.DECODE, RoleType.PDFUSION):
            model = SimpleNamespace(
                _initialize_impl=lambda _: True,
                kv_cache=object(),
                _prefill_cp_size=1,
                parallelism_config=SimpleNamespace(role_type=role),
                v4=object(),
                _max_generate_batch_size=32,
            )
            with self.subTest(role=role), patch.dict(
                os.environ, {"DSV4_CSA_OFFLOAD": "1"}
            ), patch(
                "rtp_llm.models_py.modules.dsv4.fp8.csa_cache.initialize_csa_offload"
            ) as initialize, patch(
                "logging.error"
            ):
                if role == RoleType.PDFUSION:
                    self.assertTrue(DeepSeekV4Model.initialize(model, resource))
                    initialize.assert_called_once()
                else:
                    with self.assertRaisesRegex(ValueError, "no PD"):
                        DeepSeekV4Model.initialize(model, resource)
                    initialize.assert_not_called()

    def test_opt_in_and_invalid_capacities(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(CsaOffloadConfig.from_env())
        for env in (
            {"DSV4_CSA_OFFLOAD": "yes"},
            {"DSV4_CSA_OFFLOAD": "1", "DSV4_CSA_FETCH_CTAS": "0"},
            {"DSV4_CSA_OFFLOAD": "1", "DSV4_CSA_GPU_CACHE_MIB": "-1"},
        ):
            with patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
                CsaOffloadConfig.from_env()

    def test_only_csa_payload_moves_to_host(self):
        options = dict(
            layer_num=3,
            layer_compress_ratios=[4, 128, 4],
            fp8_kv=True,
            head_dim=512,
            indexer_head_dim=128,
        )
        layers = build_dsv4_kv_cache_spec_descs(**options, csa_offload_blocks=65537)
        for layer in layers:
            for desc in layer:
                host = (
                    desc.memory is not None
                    and desc.memory.placement == CacheMemoryPlacement.HOST_PINNED
                )
                self.assertEqual(host, desc.tag == "csa_kv", desc.tag)
                if host:
                    self.assertFalse(desc.capacity.charge_to_paged_budget)
                    self.assertEqual(desc.capacity.explicit_block_num, 65537)
        options["fp8_kv"] = False
        with self.assertRaises(ValueError):
            build_dsv4_kv_cache_spec_descs(**options, csa_offload_blocks=65537)
        options["fp8_kv"] = True
        with self.assertRaises(ValueError):
            build_dsv4_kv_cache_spec_descs(
                **options, csa_offload_blocks=65537, fixed_pool_use_host_memory=True
            )


if __name__ == "__main__":
    unittest.main()
