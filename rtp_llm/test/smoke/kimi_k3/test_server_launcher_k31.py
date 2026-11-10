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
    def test_prefill_retention_changes_only_the_tail_policy_and_decode_stays_dp2_tp4(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp), 93)
            with patch.object(launch.socket, "socket") as route:
                route.return_value.getsockname.return_value = ("127.0.0.1", 0)
                original_env, original = launch.launch_config(args)
                args.reuse_cache_tail_blocks = 5
                env, command = launch.launch_config(args)
                key = command.index("--reuse_cache_tail_blocks") + 1
                self.assertEqual(command[key], "5")
                command[key] = original[key]
                self.assertEqual(command, original)
                self.assertEqual(env, original_env)
                args.role, args.tp_size, args.dp_size = "DECODE", 4, 2
                args.reuse_cache_tail_blocks = None
                _, decode = launch.launch_config(args)
                self.assertEqual(decode[decode.index("--tp_size") + 1], "4")
                self.assertEqual(decode[decode.index("--dp_size") + 1], "2")
                self.assertEqual(decode[decode.index("--reuse_cache_tail_blocks") + 1], "2147483647")
                args.reuse_cache_tail_blocks = 5
                with self.assertRaisesRegex(ValueError, "K3.1 Prefill"):
                    launch.launch_config(args)

    def test_transfer_batch_override_keeps_the_original_request_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp), 93)
            with patch.object(launch.socket, "socket") as route:
                route.return_value.getsockname.return_value = ("127.0.0.1", 0)
                original_env, original = launch.launch_config(args)
                args.memory_cache_max_descriptors_per_transfer_batch = 4
                env, command = launch.launch_config(args)
            key = "--memory_cache_max_descriptors_per_transfer_batch"
            position = command.index(key)
            self.assertEqual(command[position + 1], "4")
            self.assertEqual(command[:position] + command[position + 2:], original)
            self.assertEqual(env, original_env)
            for value, role in ((0, "PREFILL"), (True, "PREFILL"), (4, "DECODE")):
                args.memory_cache_max_descriptors_per_transfer_batch, args.role = value, role
                with self.assertRaisesRegex(ValueError, "transfer batch"):
                    launch.launch_config(args)

    def test_prefill_allocator_override_changes_only_the_allocator_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp), 93)
            with patch.object(launch.socket, "socket") as route:
                route.return_value.getsockname.return_value = ("127.0.0.1", 0)
                original_env, original_command = launch.launch_config(args)
                args.allocator_large_segment_size_mb = 12
                env, command = launch.launch_config(args)
            self.assertEqual(command, original_command)
            aliases = {"PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"}
            for key in original_env:
                if key in aliases:
                    self.assertEqual(original_env[key], "expandable_segments:True")
                    self.assertEqual(
                        env[key], "expandable_segments:True,large_segment_size_mb:12"
                    )
                else:
                    self.assertEqual(env[key], original_env[key])

    def test_prefill_allocator_override_rejects_invalid_pages_and_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp), 93)
            for size in (0, 10, 11, True, 12.0):
                args.allocator_large_segment_size_mb = size
                with self.subTest(size=size), self.assertRaisesRegex(
                    ValueError, "allocator pages"
                ):
                    launch.launch_config(args)
            args.allocator_large_segment_size_mb, args.role = 12, "DECODE"
            with self.assertRaisesRegex(ValueError, "allocator pages"):
                launch.launch_config(args)

    def test_nvfp4_switches_default_off_and_explicit_values_override_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp), 93)
            with patch.dict(launch.os.environ, {}, clear=True), patch.object(launch.socket, "socket") as route:
                route.return_value.getsockname.return_value = ("127.0.0.1", 0)
                env, _ = launch.launch_config(args)
                self.assertEqual((env["KDA_NVFP4"], env["DSA_NVFP4"]), ("0", "0"))
                self.assertEqual(env["QUANTIZATION"], "FP8_PER_BLOCK")
                for kda in (False, True):
                    for dsa in (False, True):
                        args.kda_nvfp4, args.dsa_nvfp4 = kda, dsa
                        with patch.dict(launch.os.environ, KDA_NVFP4=str(int(not kda)), DSA_NVFP4=str(int(not dsa))):
                            env, _ = launch.launch_config(args)
                        self.assertEqual((env["KDA_NVFP4"], env["DSA_NVFP4"]), (str(int(kda)), str(int(dsa))))
                        self.assertEqual(env["SP_QUANTIZATION"], "")

    def test_isolated_tiers_keep_pd_and_prefill_chunk_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp), 93)
            args.last_n_checkpoints, args.prefill_token_capacity = 5, 2048
            args.disk_cache_paths = ','.join(f'/data/rank-{rank}' for rank in range(8))
            for tier in ('DEVICE', 'HOST', 'DISK'):
                args.last_n_tier_probe = tier
                with patch.object(launch.socket, 'socket') as route:
                    route.return_value.getsockname.return_value = ('127.0.0.1', 0)
                    env, argv = launch.launch_config(args)
                options = dict(zip(argv[1::2], argv[2::2]))
                self.assertEqual(env['PREFILL_CHUNK_SIZE'], '2048')
                self.assertEqual(options['--enable_device_cache'], str(int(tier == 'DEVICE')))
                self.assertEqual(options['--enable_memory_cache'], str(int(tier == 'HOST')))
                self.assertEqual(options.get('--enable_disk_cache', '0'), str(int(tier == 'DISK')))
                self.assertEqual(options['--role_type'], 'PREFILL')
                self.assertEqual(options['--reuse_cache_tail_blocks'], '5')

    def test_calibration_keeps_b128_cp8_but_uses_unbounded_allocator(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp), 93)
            args.last_n_checkpoints, args.last_n_calibration, args.linear_pool_blocks = 5, True, 256
            with patch.object(launch.socket, 'socket') as route:
                route.return_value.getsockname.return_value = ('127.0.0.1', 0)
                _, argv = launch.launch_config(args)
            options = dict(zip(argv[1::2], argv[2::2]))
            self.assertEqual(options['--seq_size_per_block'], '128')
            self.assertEqual(options['--reuse_cache_tail_blocks'], '2147483647')
            self.assertEqual(options['--linear_pool_blocks'], '256')

    def test_b64_last_n_uses_matching_pages_on_both_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp), 93)
            args.last_n_checkpoints, args.last_n_page_tokens = 5, 64
            for role in ('PREFILL', 'DECODE'):
                args.role = role
                with self.subTest(role=role), patch.object(launch.socket, 'socket') as route:
                    route.return_value.getsockname.return_value = ('127.0.0.1', 0)
                    _, argv = launch.launch_config(args)
                options = dict(zip(argv[1::2], argv[2::2]))
                self.assertEqual(options['--seq_size_per_block'], '64')
                self.assertEqual(options['--kernel_seq_size_per_block'], '64')

    def test_last_one_through_five_uses_cp_pages_and_changes_only_prefill_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp), 93)
            for n in range(1, 6):
                for role in ("PREFILL", "DECODE"):
                    args.role, args.last_n_checkpoints = role, n
                    with self.subTest(n=n, role=role), patch.object(launch.socket, "socket") as route:
                        route.return_value.getsockname.return_value = ("127.0.0.1", 0)
                        environment, argv = launch.launch_config(args)
                    options = dict(zip(argv[1::2], argv[2::2]))
                    self.assertEqual(options["--seq_size_per_block"], "128")
                    self.assertEqual(options["--kernel_seq_size_per_block"], "128")
                    self.assertEqual(options["--prefill_cp_size"], "8")
                    self.assertEqual(options["--reuse_cache_tail_blocks"], str(n) if role == "PREFILL" else "2147483647")
                    if role == "PREFILL":
                        self.assertEqual(environment["PREFILL_CHUNK_SIZE"], "65536")
                        self.assertEqual(options["--linear_pool_blocks"], "96")
            args.role, args.tp_size, args.dp_size = "DECODE", 4, 2
            with self.assertRaisesRegex(ValueError, "TP8/EP8"):
                launch.launch_config(args)

    def test_tcp_cache_store_keeps_disaggregated_role_and_topology(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp), 93)
            args.cache_store_rdma_mode = False
            with patch.object(launch.socket, "socket") as route:
                route.return_value.getsockname.return_value = ("127.0.0.1", 0)
                _, argv = launch.launch_config(args)
            options = dict(zip(argv[1::2], argv[2::2]))
            self.assertEqual(options["--cache_store_rdma_mode"], "0")
            self.assertEqual(options["--cache_store_tcp_anet_rpc_queue_num"], "8192")
            self.assertEqual(options["--role_type"], "PREFILL")
            self.assertEqual(options["--prefill_cp_size"], "8")

    def fixture(self,root,layers):
        target=root/'target';draft=root/'draft';target.mkdir();draft.mkdir()
        target_config={'model_type':'kimi_k31','text_config':{'num_hidden_layers':layers,'num_nextn_predict_layers':0,'hidden_size':64,'vocab_size':128}}
        draft_config={'model_type':'kimi_k3','text_config':{'num_hidden_layers':93,'num_nextn_predict_layers':1,'hidden_size':64,'vocab_size':128,'mla_use_nope':True,'linear_attn_config':{'full_attn_layers':[94]}}}
        (target/'config.json').write_text(json.dumps(target_config));(draft/'config.json').write_text(json.dumps(draft_config))
        suffixes=('embed_tokens.weight','shared_head.head.weight','shared_head.norm.weight','enorm.weight','hnorm.weight','eh_proj.weight')
        mapping={f'language_model.model.layers.93.{suffix}':'draft.safetensors' for suffix in suffixes}
        (draft/'model.safetensors.index.json').write_text(json.dumps({'weight_map':mapping}))
        (draft/'draft.safetensors').write_bytes(b'fixture-index-only')
        return SimpleNamespace(model_type='kimi_k31',expected_target_layers=layers,checkpoint=target,draft_checkpoint=draft,server=sys.executable,role='PREFILL',start_port=28000,peer_port=29000,peer_ip='127.0.0.1',orthogonal_smoke=True,tp_size=8,dp_size=1,ep_size=8)

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
                        self.assertEqual(env['QUANTIZATION'],'FP8_PER_BLOCK')
                        self.assertEqual(env['FP8_KV_CACHE'],'1')
                        self.assertEqual(env['FP8_MLA'],'0')
                        self.assertEqual(env['SP_QUANTIZATION'],'')
                        self.assertEqual(options['--seq_size_per_block'],'1024');self.assertEqual(options['--kernel_seq_size_per_block'],'128')
                        self.assertEqual(options['--max_seq_len'],'2097152')
                        self.assertGreater(int(options['--max_seq_len']),655360+256)
                        self.assertEqual(options['--prefill_cp_size'],'8');self.assertEqual(options['--prefill_cp_kv_cache_sharded'],'1')
                        self.assertEqual(options['--enable_device_cache'],'1')
                        self.assertEqual(options['--enable_memory_cache'],str(int(role=='PREFILL')))
                        self.assertEqual(options['--reuse_cache'],str(int(role=='PREFILL')))
                        self.assertEqual(options['--load_method'],'fastsafetensors')
                        self.assertEqual(options['--reserver_runtime_mem_mb'],
                                         '28672' if role=='PREFILL' else '14336')
                        self.assertNotIn('--kv_cache_mem_mb', options)
                        self.assertEqual(options['--enable_cuda_graph'],str(int(role=='DECODE')))
                        if role=='PREFILL':
                            self.assertEqual(env['PREFILL_CHUNK_SIZE'],'65536')
                            self.assertNotIn('DSA_DCP_TOPK',env)
                            self.assertNotIn('DSA_DCP_COMBINE',env)
                            self.assertNotIn('DSA_DCP_PROLOGUE',env)
                            self.assertNotIn('DSA_SFA_DECODE',env)
                        else:
                            self.assertEqual(env['ENABLE_LINEAR_REPLAY'],'1')
                            self.assertEqual(env['DSA_DCP_TOPK'],'mega_topk')
                            self.assertEqual(env['DSA_DCP_COMBINE'],'mega_a2a')
                            self.assertEqual(env['DSA_DCP_PROLOGUE'],'multi_stream')
                            self.assertEqual(env['DSA_SFA_DECODE'],'metadata_free')
                            self.assertEqual(options['--decode_capture_config'],
                                             '1,2,4,8,16,32' if dp==2 else '1,2,4,8,16,32,64')
                        self.assertNotIn('force_sp_accept',options)
                        self.assertEqual(options['--cp_force_single_prefill'],'0')
                        self.assertEqual(options['--concurrency_limit'],
                                         '64' if role=='PREFILL' else str(64//dp))
                        self.assertEqual(options['--max_context_batch_size'],
                                         '16' if role=='PREFILL' else str(64//dp))

    def test_synthetic_draft_cannot_replace_trained_mla(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp),4)
            path=args.draft_checkpoint/'config.json';config=json.loads(path.read_text())
            config['rtp_llm_fixture']={'synthetic':True,'trained':False};path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'real native K3 MLA draft'):launch.launch_config(args)

    def test_default_decode_has_dp_but_cluster_can_explicitly_select_tp8(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp),93)
            for role,requested_dp,expected in (('PREFILL',None,('8','1')),
                    ('DECODE',None,('4','2')),('DECODE',1,('8','1'))):
                args.role=role;args.tp_size=None;args.dp_size=requested_dp;args.ep_size=None
                with patch.object(launch.socket,'socket') as route:
                    route.return_value.getsockname.return_value=('127.0.0.1',0)
                    _,argv=launch.launch_config(args)
                options=dict(zip(argv[1::2],argv[2::2]))
                self.assertEqual((options['--tp_size'],options['--dp_size']),expected)
                self.assertEqual(options['--ep_size'],'8')

    def test_target_only_is_a_real_non_speculative_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp),93);args.target_only=True;args.draft_checkpoint=None
            with patch.object(launch.socket,'socket') as route:
                route.return_value.getsockname.return_value=('127.0.0.1',0)
                env,argv=launch.launch_config(args)
            self.assertEqual(env['SP_TYPE'],'none')
            self.assertEqual(env['SP_CHECKPOINT_PATH'],'')
            self.assertEqual(env['SP_MODEL_TYPE'],'')
            self.assertEqual(env['GEN_NUM_PER_CIRCLE'],'0')

    def test_graph_and_mtp_modes_can_be_selected_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp),93)
            args.role='DECODE';args.tp_size=4;args.dp_size=2
            for target_only in (False,True):
                for graph in (False,True):
                    args.target_only=target_only;args.enable_cuda_graph=graph
                    with self.subTest(target_only=target_only,graph=graph):
                        with patch.object(launch.socket,'socket') as route:
                            route.return_value.getsockname.return_value=('127.0.0.1',0)
                            env,argv=launch.launch_config(args)
                        options=dict(zip(argv[1::2],argv[2::2]))
                        self.assertEqual(env['SP_TYPE'],'none' if target_only else 'mtp')
                        self.assertEqual(options['--enable_cuda_graph'],str(int(graph)))

    def test_graph_override_rejects_non_boolean_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp),93)
            for invalid in ('false',0,1):
                args.enable_cuda_graph=invalid
                with self.subTest(value=invalid),self.assertRaisesRegex(ValueError,'boolean'):
                    launch.launch_config(args)

    def test_bounded_acceptance_capacity_keeps_topology_and_precision(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp),93)
            args.role='DECODE';args.tp_size=4;args.dp_size=2
            args.max_seq_len=8192;args.request_capacity=4;args.decode_capture_batches='1,2,4'
            with patch.object(launch.socket,'socket') as route:
                route.return_value.getsockname.return_value=('127.0.0.1',0)
                env,argv=launch.launch_config(args)
            options=dict(zip(argv[1::2],argv[2::2]))
            self.assertEqual(options['--max_seq_len'],'8192')
            self.assertEqual(options['--decode_capture_config'],'1,2,4')
            self.assertEqual(options['--concurrency_limit'],'4')
            self.assertEqual(options['--max_context_batch_size'],'4')
            self.assertEqual((options['--tp_size'],options['--dp_size']),('4','2'))
            self.assertEqual((env['ACT_TYPE'],env['SP_ACT_TYPE']),('BF16','BF16'))
            args.request_capacity=2
            with self.assertRaisesRegex(ValueError,'cover all Decode graph buckets'):
                launch.launch_config(args)

    def test_linear_pool_override_keeps_paged_pool_and_mtp(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp), 93)
            args.linear_pool_blocks, args.test_block_num = 1032, 140
            args.block_tree_transfer_worker_count = 1
            args.prefill_token_capacity, args.max_seq_len = 65536, 1052672
            with patch.object(launch.socket, "socket") as route:
                route.return_value.getsockname.return_value = ("127.0.0.1", 0)
                env, argv = launch.launch_config(args)
            options = dict(zip(argv[1::2], argv[2::2]))
            self.assertEqual(options["--linear_pool_blocks"], "1032")
            self.assertEqual(options["--test_block_num"], "140")
            self.assertEqual(options["--block_tree_transfer_worker_count"], "1")
            self.assertEqual(env["PREFILL_CHUNK_SIZE"], "65536")
            self.assertEqual(options["--max_batch_tokens_without_cache"], "65536")
            self.assertEqual(options["--max_batch_tokens_size"], "65536")
            self.assertEqual(options["--max_seq_len"], "1052672")
            self.assertEqual(env["SP_QUANTIZATION"], "")
            for value in (1, -1, True, 2**31):
                args.linear_pool_blocks = value
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "linear pool blocks"):
                    launch.launch_config(args)

    def test_pairing_rejects_wrong_shapes_missing_weights_and_indexer(self):
        for field in ('hidden_size','vocab_size'):
            with tempfile.TemporaryDirectory() as tmp:
                args=self.fixture(Path(tmp),93)
                path=args.draft_checkpoint/'config.json';config=json.loads(path.read_text())
                config['text_config'][field]+=1;path.write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError,field):launch.launch_config(args)
        for mutation,error in [('missing','missing trained draft'),('indexer','Indexer')]:
            with tempfile.TemporaryDirectory() as tmp:
                args=self.fixture(Path(tmp),93)
                path=args.draft_checkpoint/'model.safetensors.index.json';index=json.loads(path.read_text())
                if mutation=='missing':index['weight_map'].pop(next(iter(index['weight_map'])))
                else:index['weight_map']['language_model.model.layers.93.self_attn.indexer.wq_b.weight']='draft.safetensors'
                path.write_text(json.dumps(index))
                with self.assertRaisesRegex(ValueError,error):launch.launch_config(args)

    def test_fp8_projections_and_kv_keep_sparse_compute_and_draft_bf16(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp),4)
            args.fp8_gemm=args.fp8_kv_cache=False
            for flag in ('fp8_gemm','fp8_kv_cache'):
                setattr(args,flag,True)
                with patch.object(launch.socket,'socket') as route:
                    route.return_value.getsockname.return_value=('127.0.0.1',0)
                    env,argv=launch.launch_config(args)
                self.assertEqual(env['QUANTIZATION'],'FP8_PER_BLOCK' if flag=='fp8_gemm' else '')
                self.assertEqual(env['FP8_KV_CACHE'],str(int(flag=='fp8_kv_cache')))
                self.assertEqual(env['FP8_MLA'],'0')
                self.assertEqual(env['SP_QUANTIZATION'],'')
                setattr(args,flag,False)

    def test_explicit_bf16_and_unspecified_precision_are_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp),4)
            for value in (None,False,True):
                args.fp8_gemm=args.fp8_kv_cache=value
                with patch.object(launch.socket,'socket') as route:
                    route.return_value.getsockname.return_value=('127.0.0.1',0)
                    env,_=launch.launch_config(args)
                self.assertEqual(env['QUANTIZATION'],'' if value is False else 'FP8_PER_BLOCK')
                self.assertEqual(env['FP8_KV_CACHE'],'0' if value is False else '1')
                self.assertEqual(env['SP_QUANTIZATION'],'')

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
