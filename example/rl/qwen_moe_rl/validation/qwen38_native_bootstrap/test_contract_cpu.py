"""Pure CPU allocation/admission contract checks; no weights or native model."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
PATH=HERE/'overlay/rtp_llm/models/qwen3_next/qwen38_bootstrap_contract.py'
spec=importlib.util.spec_from_file_location('qwen38_bootstrap_contract_under_test',PATH)
contract_module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=contract_module
spec.loader.exec_module(contract_module)


def profile(**model_changes):
    model=SimpleNamespace(model_type='qwen38_moe_text',num_layers=92,hidden_size=8192,
        expert_num=512,moe_k=10,quant_algo=SimpleNamespace(isQuant=lambda:False),
        compute_dtype='torch.bfloat16',enable_fp32_lm_head=False,normalize_lm_head_weight=False,
        logit_scale=1.0,tie_word_embeddings=False,enable_output_vocab_pruning=False)
    model.__dict__.update(model_changes)
    parallel=SimpleNamespace(world_size=72,tp_size=8,dp_size=9,ep_size=72,pp_size=1,
                              ffn_sp_size=8,get_ffn_tp_size=lambda:1)
    hardware=SimpleNamespace(use_swizzleA=False,enable_cuda_graph=False,enable_native_cuda_graph=False)
    moe=SimpleNamespace(moe_strategy='nccl_bf16_reference',fake_balance_expert=False)
    env={key:'1' for key in ['RTP_LLM_RL_MODE','RTP_LLM_RL_WORLD72','RTP_LLM_RL_START_PAUSED','RTP_LLM_RL_METADATA_BOOTSTRAP']}
    env.update(RTP_FIXED_PHYSICAL_EP_LAYOUT='balanced_v1',RTP_RL_BORROW_SINGLETON_HF_SOURCES='1')
    return model,parallel,hardware,moe,env


class BootstrapContractTest(unittest.TestCase):
    def load_contract(self):
        return contract_module.load_bootstrap_contract(HERE/'fixtures',HERE/'fixtures/expected-canonical-policy-manifest.json')

    def test_exact_policy_sources_and_fp32_gdn_contract(self):
        contract=self.load_contract()
        self.assertEqual(len(contract.source_shapes),1590)
        self.assertFalse(any(name.startswith('mtp.') for name in contract.source_shapes))
        self.assertEqual(contract.source_shapes['model.layers.0.mlp.experts.gate_up_proj'],(512,4096,8192))
        self.assertEqual(sum(dtype=='float32' for dtype in contract.refit_dtypes.values()),138)
        self.assertIn('actual shard-header coverage is not claimed',contract.derivation)

    def test_validated_shapes_are_immutable(self):
        contract=self.load_contract()
        with self.assertRaises(TypeError):contract.source_shapes['lm_head.weight']=(1,1)

    def test_wrong_metadata_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for path in (HERE/'fixtures').iterdir():shutil.copyfile(path,root/path.name)
            (root/'config.json').write_bytes((root/'config.json').read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'config SHA256 mismatch'):
                contract_module.load_bootstrap_contract(root,root/'expected-canonical-policy-manifest.json')

    def test_only_complete_explicit_profile_is_allowed(self):
        contract_module.validate_bootstrap_profile(*profile(),native_capability=1)
        for flag in profile()[-1]:
            args=profile();args[-1].pop(flag)
            with self.subTest(flag=flag),self.assertRaisesRegex(ValueError,'initial-pause capability'):
                contract_module.validate_bootstrap_profile(*args,native_capability=1)

    def test_matching_native_marker_is_mandatory(self):
        with self.assertRaisesRegex(ValueError,'initial-pause capability'):
            contract_module.validate_bootstrap_profile(*profile(),native_capability=0)

    def test_graph_swizzle_and_value_dependent_lm_head_are_rejected(self):
        for change in ['enable_cuda_graph','enable_native_cuda_graph','use_swizzleA']:
            args=profile();setattr(args[2],change,True)
            with self.subTest(change=change),self.assertRaises(ValueError):
                contract_module.validate_bootstrap_profile(*args,native_capability=1)
        for kwargs in [{'normalize_lm_head_weight':True},{'logit_scale':2.0},{'enable_output_vocab_pruning':True}]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                contract_module.validate_bootstrap_profile(*profile(**kwargs),native_capability=1)

    def test_tp72_and_separate_full_model_ep_groups_are_rejected(self):
        for tp,dp,ep in [(72,1,72),(8,9,8),(8,9,1)]:
            args=profile();args[1].tp_size=tp;args[1].dp_size=dp;args[1].ep_size=ep
            with self.subTest(tp=tp,dp=dp,ep=ep),self.assertRaisesRegex(ValueError,'exact world72'):
                contract_module.validate_bootstrap_profile(*args,native_capability=1)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BootstrapContractTest))
    print('BOOTSTRAP_CONTRACT_CPU_RESULT',json.dumps({'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'native_model_loaded':False,'CUDA_used':False,'checkpoint_payload_loaded':False}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
