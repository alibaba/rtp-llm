"""CPU descriptor tests against the rebuilt RTP config extension and ATOM overlay."""
import importlib.util
import os
import pickle
import unittest
from pathlib import Path
from unittest.mock import patch
import importlib.metadata
from rtp_llm.ops import KVCacheSpecDesc, KVCacheSpecType

path = importlib.metadata.distribution("atom").locate_file("atom/plugin/rtpllm/utils/split_ssm_storage.py")
spec = importlib.util.spec_from_file_location("split_ssm_storage_tested", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class DescriptorTests(unittest.TestCase):
    def template(self):
        desc = KVCacheSpecDesc()
        desc.tag = "linear"
        desc.cache_type = KVCacheSpecType.LINEAR
        return desc

    def test_disabled_preserves_shared_templates(self):
        original = self.template()
        layers = [[original], [original]]
        with patch.dict(os.environ, {"ATOM_RTP_SPLIT_SSM_STORAGE": "0"}):
            self.assertIs(module.split_descs(layers), layers)
        self.assertIs(layers[0][0], original)
        self.assertEqual(original.linear_state_component, 0)

    def test_split_never_mutates_shared_layer_descriptor(self):
        original = self.template()
        layers = [[original] for _ in range(3)]
        with patch.dict(os.environ, {"ATOM_RTP_SPLIT_SSM_STORAGE":"1", "ROLE_TYPE":"PREFILL", "ATOM_RTP_SPLIT_REFERENCE_BLOCKS":"0"}):
            module.split_descs(layers)
        self.assertEqual(original.linear_state_component, 0)
        self.assertIsNone(original.capacity)
        self.assertEqual(len({id(layer[0]) for layer in layers}), 3)
        for state, conv in layers:
            self.assertEqual((state.linear_state_component, conv.linear_state_component), (1, 2))
            self.assertEqual(conv.tag, "gdn_conv")
            self.assertIsNone(state.capacity)
            self.assertIsNone(conv.capacity)

    def test_single_state_capacity_preserves_conv_capacity(self):
        layers = [[self.template()]]
        with patch.dict(os.environ, {"ATOM_RTP_SPLIT_SSM_STORAGE":"1", "ROLE_TYPE":"DECODE", "ATOM_RTP_SSM_STATE_SLOTS":"1", "ATOM_RTP_SSM_MAX_REQUESTS":"16", "ATOM_RTP_SPLIT_REFERENCE_BLOCKS":"512"}):
            module.split_descs(layers)
        state, conv = layers[0]
        self.assertEqual(state.capacity.explicit_block_num, 17)
        self.assertEqual(conv.capacity.explicit_block_num, 512)
        self.assertTrue(state.capacity.charge_to_paged_budget)
        self.assertTrue(conv.capacity.charge_to_paged_budget)

    def test_paged_capacity_is_independent_of_live_state_capacity(self):
        paged = KVCacheSpecDesc()
        paged.tag = "full"
        paged.cache_type = KVCacheSpecType.MHA
        layers = [[self.template(), paged]]
        with patch.dict(os.environ, {"ATOM_RTP_SPLIT_SSM_STORAGE":"1", "ROLE_TYPE":"DECODE", "ATOM_RTP_SSM_STATE_SLOTS":"1", "ATOM_RTP_SSM_MAX_REQUESTS":"16", "ATOM_RTP_SPLIT_REFERENCE_BLOCKS":"512", "ATOM_RTP_SPLIT_PAGED_BLOCKS":"8257"}):
            module.split_descs(layers)
        by_tag = {desc.tag: desc for desc in layers[0]}
        self.assertEqual(by_tag["linear"].capacity.explicit_block_num, 17)
        self.assertEqual(by_tag["gdn_conv"].capacity.explicit_block_num, 512)
        self.assertEqual(by_tag["full"].capacity.explicit_block_num, 8257)

    def test_pd_default_resolves_cpp_flags_and_runtime_capacity(self):
        from types import SimpleNamespace
        with patch.dict(os.environ, {"ROLE_TYPE": "DECODE"}, clear=True):
            config = SimpleNamespace(atom_qwen4_exp_mtp_target=True,
                                     kv_cache_spec_descs=[[self.template()]])
            module.configure_request_capacity(config, 32)
            self.assertEqual(os.environ["ATOM_RTP_SPLIT_SSM_STORAGE"], "1")
            self.assertEqual(os.environ["ATOM_RTP_SSM_STATE_SLOTS"], "1")
            state, conv = config.kv_cache_spec_descs[0]
            self.assertEqual(state.capacity.explicit_block_num, 33)
            self.assertEqual(conv.linear_state_component, 2)
            module.configure_request_capacity(config, 32)
            self.assertEqual(len(config.kv_cache_spec_descs[0]), 2)

    def test_default_does_not_enable_other_devices_or_non_mtp(self):
        from types import SimpleNamespace
        for role, hip, mtp in [("DECODE", None, True), ("DECODE", "7.2", False),
                               ("", "7.2", True)]:
            with patch.dict(os.environ, {"ROLE_TYPE": role}, clear=True), patch.object(module.torch.version, "hip", hip):
                config = SimpleNamespace(atom_qwen4_exp_mtp_target=mtp,
                                         kv_cache_spec_descs=[[self.template()]])
                module.configure_request_capacity(config, 32)
                self.assertEqual(len(config.kv_cache_spec_descs[0]), 1)
                self.assertEqual(config.kv_cache_spec_descs[0][0].linear_state_component, 0)

    def test_pd_default_preserves_explicit_disable(self):
        from types import SimpleNamespace
        with patch.dict(os.environ, {"ROLE_TYPE": "DECODE", "ATOM_RTP_SPLIT_SSM_STORAGE": "0"}, clear=True):
            config = SimpleNamespace(atom_qwen4_exp_mtp_target=True,
                                     kv_cache_spec_descs=[[self.template()]])
            module.configure_request_capacity(config, 32)
            self.assertEqual(len(config.kv_cache_spec_descs[0]), 1)
            self.assertFalse(module.enabled())

    def test_default_target_and_full_attention_draft_share_pool_policy(self):
        from types import SimpleNamespace
        paged = KVCacheSpecDesc()
        paged.tag = "full"
        paged.cache_type = KVCacheSpecType.MHA
        with patch.dict(os.environ, {"ROLE_TYPE": "DECODE",
                                    "ATOM_RTP_SPLIT_REFERENCE_BLOCKS": "512",
                                    "ATOM_RTP_SPLIT_PAGED_BLOCKS": "8257"}, clear=True):
            target = SimpleNamespace(atom_qwen4_exp_mtp_target=True,
                kv_cache_spec_descs=[[self.template(), pickle.loads(pickle.dumps(paged))]])
            draft = SimpleNamespace(is_mtp=True, kv_cache_spec_descs=[[paged]])
            module.configure_request_capacity(target, 16)
            module.configure_request_capacity(draft, 16)
            for config in (target, draft):
                full = next(d for d in config.kv_cache_spec_descs[0] if d.tag == "full")
                self.assertEqual(full.capacity.explicit_block_num, 8257)
                self.assertTrue(full.capacity.charge_to_paged_budget)
            self.assertEqual(len(draft.kv_cache_spec_descs[0]), 1)

    def test_pickle_and_legacy_tuple_compatibility(self):
        desc = self.template()
        for component in (0, 1, 2):
            desc.linear_state_component = component
            self.assertEqual(pickle.loads(pickle.dumps(desc)).linear_state_component, component)
        current = desc.__getstate__()
        for size in (20, 21):
            legacy = KVCacheSpecDesc()
            legacy.__setstate__(current[:size])
            self.assertEqual(legacy.linear_state_component, 0)

if __name__ == "__main__":
    unittest.main()
