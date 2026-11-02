"""K3.1-only settings over the shared K3 launcher; no GPU execution."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('k31_launcher',Path(__file__).with_name('server_launcher.py'))
launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)


class K31LaunchContract(unittest.TestCase):
    def fixture(self,root,layers):
        target=root/'target';draft=root/'draft';target.mkdir();draft.mkdir()
        target_config={'model_type':'kimi_k31','text_config':{'num_hidden_layers':layers,'num_nextn_predict_layers':0}}
        draft_config={'model_type':'kimi_k31','text_config':{'num_hidden_layers':layers,'num_nextn_predict_layers':1},'rtp_llm_fixture':{'kind':'synthetic_dsa_mtp','synthetic':True,'native':False,'trained':False}}
        (target/'config.json').write_text(json.dumps(target_config));(draft/'config.json').write_text(json.dumps(draft_config))
        return SimpleNamespace(model_type='kimi_k31',expected_target_layers=layers,allow_synthetic_draft=True,checkpoint=target,draft_checkpoint=draft,server=sys.executable,role='PREFILL',start_port=28000,peer_port=29000,peer_ip='127.0.0.1',orthogonal_smoke=True,tp_size=8,dp_size=1,ep_size=8)

    def test_both_layers_topologies_and_roles_keep_mtp_and_dcp_contract(self):
        for layers in (4,93):
            with tempfile.TemporaryDirectory() as tmp:
                args=self.fixture(Path(tmp),layers)
                for role,tp,dp in [('PREFILL',8,1),('DECODE',8,1),('DECODE',4,2)]:
                    with self.subTest(layers=layers,role=role,tp=tp,dp=dp):
                        args.role,args.tp_size,args.dp_size=role,tp,dp
                        with patch.dict(launch.os.environ,{},clear=True), patch.object(launch.socket,'socket') as route:
                            route.return_value.getsockname.return_value=('127.0.0.1',0)
                            env,argv=launch.launch_config(args)
                        options=dict(zip(argv[1::2],argv[2::2]))
                        self.assertEqual(env['MODEL_TYPE'],'kimi_k31');self.assertEqual(env['SP_MODEL_TYPE'],'kimi_k31_mtp')
                        self.assertEqual(env['SP_TYPE'],'mtp');self.assertEqual(env['GEN_NUM_PER_CIRCLE'],'3')
                        self.assertEqual(env['RTP_MTP_ACCEPTANCE_STATS'],'1')
                        self.assertEqual(env['ACT_TYPE'],'BF16');self.assertEqual(env['SP_ACT_TYPE'],'BF16')
                        self.assertEqual(options['--seq_size_per_block'],'1024');self.assertEqual(options['--kernel_seq_size_per_block'],'128')
                        self.assertEqual(options['--max_seq_len'],'2097152')
                        self.assertGreater(int(options['--max_seq_len']),655360+256)
                        self.assertEqual(options['--prefill_cp_size'],'8');self.assertEqual(options['--prefill_cp_kv_cache_sharded'],'1')
                        self.assertEqual(options['--enable_cuda_graph'],str(int(role=='DECODE')))
                        self.assertNotIn('force_sp_accept',options)
                        self.assertEqual(options['--cp_force_single_prefill'],'0')
                        self.assertEqual(options['--concurrency_limit'],
                                         '64' if role=='PREFILL' else str(64//dp))
                        self.assertEqual(options['--max_context_batch_size'],
                                         '16' if role=='PREFILL' else str(64//dp))

    def test_synthetic_draft_is_explicit_not_native(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp),4);args.allow_synthetic_draft=False
            with self.assertRaisesRegex(ValueError,'explicitly declared untrained'):launch.launch_config(args)

    def test_fp8_sfa_or_kda_does_not_sneak_into_indexer_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp),4)
            for flag in ('fp8_gemm','fp8_kv_cache'):
                setattr(args,flag,True)
                with self.assertRaisesRegex(ValueError,'FP8 Indexer with BF16'):launch.launch_config(args)
                setattr(args,flag,False)

    def test_explicit_query_fixture_capacity_preserves_mtp_and_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp),93)
            for capacity in (None,33792):
                args.prefill_token_capacity=capacity
                with patch.object(launch.socket,'socket') as route:
                    route.return_value.getsockname.return_value=('127.0.0.1',0)
                    env,argv=launch.launch_config(args)
                options=dict(zip(argv[1::2],argv[2::2]))
                self.assertEqual(env['SP_TYPE'],'mtp')
                self.assertEqual(env['PREFILL_CHUNK_SIZE'],str(capacity or 65536))
                self.assertNotIn('RTP_LLM_FP8_COLLECTIVE_MAX_TOKENS',env)
                if capacity:
                    self.assertEqual(options['--max_batch_tokens_without_cache'],str(capacity))
                    self.assertEqual(options['--max_batch_tokens_size'],str(capacity))
            for capacity in (0,32784,66560):
                args.prefill_token_capacity=capacity
                with self.assertRaisesRegex(ValueError,'page-aligned'):
                    launch.launch_config(args)
            args.prefill_token_capacity=33792;args.role='DECODE'
            with self.assertRaisesRegex(ValueError,'page-aligned'):
                launch.launch_config(args)


if __name__=='__main__':unittest.main()
