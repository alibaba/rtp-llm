"""K3.1-only settings over the shared K3 launcher; no GPU execution."""

import importlib.util
import io
import json
import shlex
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('k31_launcher',Path(__file__).with_name('server_launcher.py'))
launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)
spec = importlib.util.spec_from_file_location(
    "k31_two_host", Path(__file__).with_name("two_host_launcher.py")
)
two_host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(two_host)


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

    def test_nvfp4_switches_default_on_and_explicit_values_override_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp), 93)
            with patch.dict(launch.os.environ, {}, clear=True), patch.object(launch.socket, "socket") as route:
                route.return_value.getsockname.return_value = ("127.0.0.1", 0)
                env, _ = launch.launch_config(args)
                self.assertEqual((env["KDA_NVFP4"], env["DSA_NVFP4"]), ("1", "1"))
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
                        else:
                            self.assertEqual(env['ENABLE_LINEAR_REPLAY'],'1')
                            self.assertEqual(env['DSA_DCP_TOPK'],'mega_topk')
                            self.assertEqual(env['DSA_DCP_COMBINE'],'mega_a2a')
                            self.assertEqual(env['DSA_DCP_PROLOGUE'],'multi_stream')
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

    def two_host_config(self, root, args):
        host = dict(
            ssh_target="test-host",
            container="test-container",
            cluster="test-cluster",
            repo_root=str(root / "repo"),
            artifact_root=str(root / "artifacts"),
            checkpoint=str(args.checkpoint),
            draft_checkpoint=str(args.draft_checkpoint),
            checkpoint_root=str(root),
            storage_policy="ssd",
            ip="192.0.2.1",
            port=28000,
        )
        return dict(
            run_id="chunk-test",
            prefill=host,
            decode={**host, "ip": "192.0.2.2", "port": 29000},
        )

    def torchspec_env(self):
        return {
            "JSON_MODEL_OVERRIDE_ARGS": json.dumps(
                {
                    "hidden_state_capture_layer_ids": [1, 46, 92],
                    "hidden_state_capture_dtype": "bf16",
                }
            ),
            "MOONCAKE_MASTER_SERVER": "192.0.2.3:50051",
            "MOONCAKE_METADATA_SERVER": "http://192.0.2.3:8090/metadata",
            "MOONCAKE_LOCAL_HOSTNAME": "capture-prefill.example",
            "MOONCAKE_STORE_KEY_NAMESPACE": "K31Demo001",
            "MOONCAKE_PROTOCOL": "tcp",
            "MOONCAKE_DEVICE_NAME": "",
            "MOONCAKE_GLOBAL_SEGMENT_SIZE": "1073741824",
            "MOONCAKE_LOCAL_BUFFER_SIZE": "16777216",
            "MOONCAKE_HOST_BUFFER_SIZE": "16777216",
            "MOONCAKE_ASYNC_PUT_POOL_SIZE": "4",
            "MOONCAKE_GPU_BUFFER_SIZE": "16777216",
            "MOONCAKE_MAX_SEQ_LEN": "8192",
            "MOONCAKE_HIDDEN_DIM": "64",
            "MOONCAKE_NUM_AUX_LAYERS": "3",
            "MOONCAKE_ENABLE_GPU_DIRECT": "0",
        }

    def test_explicit_chunk_override_only_changes_chunk_not_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp), 93)
            for capacity in (None, 33792):
                args.prefill_token_capacity = capacity
                args.prefill_chunk_size = None
                with patch.object(launch.socket, "socket") as route:
                    route.return_value.getsockname.return_value = ("192.0.2.1", 0)
                    default_env, default_argv = launch.launch_config(args)
                    for chunk_size in (0, 16384):
                        with self.subTest(capacity=capacity, chunk_size=chunk_size):
                            args.prefill_chunk_size = chunk_size
                            env, argv = launch.launch_config(args)
                            self.assertEqual(argv, default_argv)
                            self.assertEqual(
                                env,
                                {**default_env, "PREFILL_CHUNK_SIZE": str(chunk_size)},
                            )

    def test_chunk_override_rejects_negative_noninteger_and_wrong_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp), 93)
            for value in (-1, True, 1.5, "0"):
                args.prefill_chunk_size = value
                with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "nonnegative integer"
                ):
                    launch.launch_config(args)
                config = self.two_host_config(Path(tmp), args)
                config["prefill"]["prefill_chunk_size"] = value
                with self.assertRaisesRegex(ValueError, "nonnegative integer"):
                    two_host.validate(config)
            args.prefill_chunk_size = 0
            args.role = "DECODE"
            with self.assertRaisesRegex(ValueError, "Prefill chunk size"):
                launch.launch_config(args)
            config = self.two_host_config(Path(tmp), args)
            config["decode"]["prefill_chunk_size"] = 0
            with self.assertRaisesRegex(ValueError, "on Prefill"):
                two_host.validate(config)
            args.role = "PREFILL"
            args.model_type = "kimi_k3"
            with self.assertRaisesRegex(ValueError, "K3.1 Prefill"):
                launch.launch_config(args)

    def test_cli_accepts_zero_chunk_without_zeroing_token_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp), 93)
            argv = [
                "server_launcher.py",
                "--role",
                "PREFILL",
                "--server",
                sys.executable,
                "--model-type",
                "kimi_k31",
                "--checkpoint",
                str(args.checkpoint),
                "--draft-checkpoint",
                str(args.draft_checkpoint),
                "--peer-ip",
                "192.0.2.2",
                "--peer-port",
                "29000",
                "--start-port",
                "28000",
                "--allow-checkpoint-root",
                tmp,
                "--run-dir",
                str(Path(tmp) / "run"),
                "--orthogonal-smoke",
                "--prefill-token-capacity",
                "33792",
                "--prefill-chunk-size",
                "0",
                "--print-config",
            ]
            output = io.StringIO()
            with patch.object(sys, "argv", argv), redirect_stdout(output), patch.object(
                launch.socket, "socket"
            ) as route:
                route.return_value.getsockname.return_value = ("192.0.2.1", 0)
                launch.main()
            result = json.loads(output.getvalue())
            options = dict(zip(result["command"][1::2], result["command"][2::2]))
            self.assertEqual(result["environment"]["PREFILL_CHUNK_SIZE"], "0")
            self.assertEqual(options["--max_batch_tokens_size"], "33792")
            self.assertEqual(options["--max_batch_tokens_without_cache"], "33792")

    def test_two_host_chunk_override_reaches_server_popen_and_preserves_pd(self):
        for capacity in (None, 33792):
            for chunk_size, capture_env in (
                (None, None),
                (0, None),
                (16384, None),
                (0, {}),
                (None, self.torchspec_env()),
                (0, self.torchspec_env()),
            ):
                with self.subTest(
                    capacity=capacity, chunk_size=chunk_size, capture=bool(capture_env)
                ), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    args = self.fixture(root, 93)
                    config = self.two_host_config(root, args)
                    if capacity is not None:
                        config["prefill"]["prefill_token_capacity"] = capacity
                    if chunk_size is not None:
                        config["prefill"]["prefill_chunk_size"] = chunk_size
                    if capture_env is not None:
                        config["prefill"]["torchspec_env"] = capture_env
                    # Exercise the same JSON round-trip as control.json on each peer.
                    config = two_host.validate(json.loads(json.dumps(config)))
                    proc_ipv4 = root / "proc-ipv4"
                    proc_ipv4.mkdir()
                    (proc_ipv4 / "ip_local_port_range").write_text("32768 60999\n")
                    (proc_ipv4 / "ip_local_reserved_ports").write_text("")
                    real_port_preflight = two_host.service_port_preflight
                    source = root / "repo"
                    binary = source / "bazel-bin/rtp_llm/rtp_llm_server"
                    binary.parent.mkdir(parents=True)
                    binary.write_bytes(b"offline-server-not-executed")
                    Path(
                        str(binary)
                        + ".runfiles/pip_gpu_cuda13_torch_torch/site-packages/torch/lib"
                    ).mkdir(parents=True)
                    rdma = root / "rdma"
                    port = rdma / "mlx5_0/ports/1"
                    port.mkdir(parents=True)
                    (port / "link_layer").write_text("InfiniBand")
                    readlink = two_host.os.readlink

                    def isolated_path(value):
                        if str(value) == "/tmp/rtp_llm/smoke/test/gpu_status":
                            return root / "locks"
                        if str(value) == "/sys/class/infiniband":
                            return rdma
                        return Path(value)

                    for role in two_host.ROLES:
                        two_host.run_dir(config, role).mkdir(parents=True)
                        with ExitStack() as stack:
                            stack.enter_context(
                                patch.object(sys, "path", sys.path.copy())
                            )
                            stack.enter_context(
                                patch.dict(
                                    sys.modules,
                                    {
                                        "server_launcher": launch,
                                        "checkpoint_guard": SimpleNamespace(
                                            preflight=lambda *a, **k: {"passed": True}
                                        ),
                                    },
                                )
                            )
                            inherited = {"PREFILL_CHUNK_SIZE": "8192"}
                            if capture_env and role == "prefill":
                                inherited.update(
                                    {key: "inherited-value" for key in capture_env}
                                )
                            elif not capture_env:
                                inherited["JSON_MODEL_OVERRIDE_ARGS"] = (
                                    '{"legacy_override": true}'
                                )
                            stack.enter_context(
                                patch.dict(two_host.os.environ, inherited, clear=True)
                            )
                            stack.enter_context(
                                patch.object(
                                    two_host, "Path", side_effect=isolated_path
                                )
                            )
                            stack.enter_context(
                                patch.object(
                                    two_host,
                                    "owned_data",
                                    side_effect=lambda path: path.resolve(strict=True),
                                )
                            )
                            port_preflight = stack.enter_context(
                                patch.object(
                                    two_host,
                                    "service_port_preflight",
                                    side_effect=lambda start_port: real_port_preflight(
                                        start_port, proc_ipv4
                                    ),
                                )
                            )
                            flock = stack.enter_context(
                                patch.object(two_host.fcntl, "flock")
                            )
                            flock.side_effect = lambda handle, *_: stack.callback(
                                handle.close
                            )
                            stack.enter_context(
                                patch.object(launch, "require_gpu_capacity")
                            )
                            stack.enter_context(
                                patch.object(
                                    launch, "validate_rdma_hcas", return_value="mlx5_0"
                                )
                            )
                            stack.enter_context(
                                patch.object(
                                    two_host,
                                    "usable_rdma_hcas",
                                    return_value=["mlx5_0"],
                                )
                            )
                            route = stack.enter_context(
                                patch.object(two_host.socket, "socket")
                            )
                            route.return_value.getsockname.return_value = (
                                "192.0.2.1",
                                0,
                            )
                            stack.enter_context(
                                patch.object(
                                    two_host.subprocess, "check_output", return_value=""
                                )
                            )
                            execute = stack.enter_context(
                                patch.object(two_host.subprocess, "Popen")
                            )
                            execute.return_value.pid = 12345
                            execute.return_value.poll.return_value = 0
                            execute.return_value.returncode = 0
                            stack.enter_context(
                                patch.object(two_host, "start_ticks", return_value=1)
                            )
                            stack.enter_context(
                                patch.object(
                                    two_host.os,
                                    "readlink",
                                    side_effect=lambda path, *a, **k: (
                                        "pid-ns"
                                        if path == "/proc/self/ns/pid"
                                        else readlink(path, *a, **k)
                                    ),
                                )
                            )
                            stack.enter_context(patch.object(two_host.signal, "signal"))
                            stack.enter_context(
                                patch.object(two_host, "stop_owned_group")
                            )
                            result = two_host.serve(config, role)
                        port_preflight.assert_called_once_with(config[role]["port"])
                        port_receipt = json.loads(
                            (
                                two_host.run_dir(config, role) / "port-preflight.json"
                            ).read_text()
                        )
                        self.assertEqual(
                            port_receipt,
                            dict(
                                service_start=config[role]["port"],
                                service_end=config[role]["port"] + 71,
                                ephemeral_start=32768,
                                ephemeral_end=60999,
                                passed=True,
                            ),
                        )
                        self.assertEqual(result["exit_code"], 0)
                        execute.assert_called_once()
                        command = execute.call_args.args[0]
                        env = execute.call_args.kwargs["env"]
                        options = dict(zip(command[1::2], command[2::2]))
                        expected_chunk = (
                            (capacity or 65536) if chunk_size is None else chunk_size
                        )
                        self.assertEqual(
                            env["PREFILL_CHUNK_SIZE"],
                            str(expected_chunk) if role == "prefill" else "8192",
                        )
                        budget = (
                            str(capacity)
                            if role == "prefill" and capacity is not None
                            else None
                        )
                        self.assertEqual(
                            options["--max_batch_tokens_size"], budget or "262144"
                        )
                        self.assertEqual(
                            options["--max_batch_tokens_without_cache"],
                            budget or "65536",
                        )
                        self.assertEqual(options["--role_type"], role.upper())
                        self.assertEqual(
                            (
                                options["--tp_size"],
                                options["--dp_size"],
                                options["--ep_size"],
                            ),
                            ("8", "1", "8") if role == "prefill" else ("4", "2", "8"),
                        )
                        self.assertEqual(options["--prefill_cp_size"], "8")
                        self.assertEqual(
                            options["--remote_server_port"],
                            str(
                                config["decode" if role == "prefill" else "prefill"][
                                    "port"
                                ]
                            ),
                        )
                        self.assertEqual(env["SP_TYPE"], "mtp")
                        receipt = json.loads(
                            (
                                two_host.run_dir(config, role) / "instance.json"
                            ).read_text()
                        )
                        if capture_env:
                            if role == "prefill":
                                self.assertEqual(
                                    {key: env[key] for key in capture_env}, capture_env
                                )
                                self.assertEqual(
                                    env["MOONCAKE_ASYNC_PUT_POOL_SIZE"], "4"
                                )
                                self.assertEqual(
                                    env["MOONCAKE_STORE_KEY_NAMESPACE"], "K31Demo001"
                                )
                                self.assertRegex(
                                    env["MOONCAKE_STORE_KEY_NAMESPACE"],
                                    r"\A[A-Za-z0-9]+\Z",
                                )
                                self.assertNotIn("--json_model_override_args", command)
                                self.assertEqual(
                                    receipt["torchspec_env_keys"], sorted(capture_env)
                                )
                            else:
                                self.assertTrue(capture_env.keys().isdisjoint(env))
                                self.assertEqual(receipt["torchspec_env_keys"], [])
                        else:
                            self.assertEqual(
                                env["JSON_MODEL_OVERRIDE_ARGS"],
                                inherited["JSON_MODEL_OVERRIDE_ARGS"],
                            )
                            self.assertEqual(receipt["torchspec_env_keys"], [])
                        self.assertTrue(
                            two_host.TORCHSPEC_ENV_KEYS.isdisjoint(
                                receipt["environment"]
                            )
                        )
                        if role == "prefill":
                            self.assertEqual(
                                receipt["environment"]["PREFILL_CHUNK_SIZE"],
                                env["PREFILL_CHUNK_SIZE"],
                            )
                        else:
                            self.assertNotIn(
                                "PREFILL_CHUNK_SIZE", receipt["environment"]
                            )

    def test_torchspec_env_rejects_invalid_map_key_value_json_and_role_before_io(self):
        invalid = [
            None,
            [],
            "not-a-map",
            True,
            {"LD_PRELOAD": "sensitive-value"},
            {"MOONCAKE_UNKNOWN": "sensitive-value"},
            {"MOONCAKE_MASTER_HOST": "sensitive-value"},
            {"MOONCAKE_MASTER_SERVER": "sensitive-value\0"},
            {"JSON_MODEL_OVERRIDE_ARGS": "sensitive-value"},
            {"JSON_MODEL_OVERRIDE_ARGS": "[]"},
            {"JSON_MODEL_OVERRIDE_ARGS": "null"},
        ]
        invalid += [{"MOONCAKE_PROTOCOL": value} for value in (0, False, None, [], {})]
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp), 93)
            for value in invalid:
                with self.subTest(value=value):
                    config = self.two_host_config(Path(tmp), args)
                    config["prefill"]["torchspec_env"] = value
                    with self.assertRaisesRegex(ValueError, "torchspec_env") as error:
                        two_host.validate(config)
                    self.assertNotIn("sensitive-value", str(error.exception))
                    with patch.object(two_host, "owned_data") as io_check, patch.object(
                        two_host.subprocess, "Popen"
                    ) as execute:
                        with self.assertRaisesRegex(ValueError, "torchspec_env"):
                            two_host.serve(config, "prefill")
                        io_check.assert_not_called()
                        execute.assert_not_called()
            for value in ({}, self.torchspec_env()):
                config = self.two_host_config(Path(tmp), args)
                config["decode"]["torchspec_env"] = value
                with self.assertRaisesRegex(ValueError, "only on K3.1 Prefill"):
                    two_host.validate(config)
                with self.assertRaisesRegex(ValueError, "only on K3.1 Prefill"):
                    two_host.serve(config, "decode")

    def test_controller_copies_capture_config_over_ssh_and_remote_prepare_roundtrip(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.fixture(root, 93)
            config = self.two_host_config(root, args)
            config["container_user"] = "offline-user"
            config["ssh_config"] = str(root / "ssh-config")
            config["prefill"].update(
                prefill_chunk_size=0, torchspec_env=self.torchspec_env()
            )
            config = two_host.validate(json.loads(json.dumps(config)))
            Path(config["prefill"]["artifact_root"]).mkdir()
            operations = []

            def transport(argv, **kwargs):
                self.assertEqual(argv[:3], ["ssh", "-F", config["ssh_config"]])
                self.assertIn("BatchMode=yes", argv)
                command = shlex.split(argv[-1])
                if command[:2] == ["docker", "inspect"]:
                    return SimpleNamespace(stdout="same-offline-image\n")
                self.assertEqual(command[:1], ["docker"])
                self.assertEqual(command[1], "exec")
                self.assertEqual(command[3:5], ["-u", "offline-user"])
                script = command.index("python3") + 1
                operation = command[script + 1]
                role = command[command.index("--role") + 1]
                operations.append((operation, role))
                self.assertEqual(argv[-2], config[role]["ssh_target"])
                control = Path(command[command.index("--config") + 1])
                self.assertEqual(
                    control, two_host.run_dir(config, role) / "control.json"
                )
                self.assertNotIn("MOONCAKE_", argv[-1])
                self.assertNotIn("JSON_MODEL_OVERRIDE_ARGS", argv[-1])
                if operation == "prepare":
                    self.assertEqual(command[2], "-i")
                    self.assertEqual(json.loads(kwargs["input"]), config)
                    output = io.StringIO()
                    with patch.object(sys, "argv", command[script:]), patch.object(
                        sys, "stdin", io.StringIO(kwargs["input"])
                    ), redirect_stdout(output):
                        two_host.main()
                    self.assertTrue(json.loads(output.getvalue())["prepared"])
                    self.assertEqual(json.loads(control.read_text()), config)
                    return SimpleNamespace(stdout=output.getvalue())
                copied = two_host.validate(json.loads(control.read_text()))
                self.assertEqual(
                    copied["prefill"]["torchspec_env"], self.torchspec_env()
                )
                self.assertNotIn("torchspec_env", copied["decode"])
                if operation == "serve":
                    self.assertEqual(command[2], "-d")
                    return SimpleNamespace(stdout="")
                self.assertEqual(operation, "status")
                return SimpleNamespace(stdout=json.dumps({"healthy": True}))

            launch_args = SimpleNamespace(
                dry_run=False,
                operation="launch",
                output=root / "controller",
                startup_timeout=10,
                keep_services=True,
            )
            output = io.StringIO()
            with patch.object(
                two_host.subprocess, "run", side_effect=transport
            ), patch.object(
                two_host,
                "owned_data",
                side_effect=lambda path: path.resolve(strict=True),
            ), patch.object(
                two_host.os, "getuid", return_value=1001
            ), redirect_stdout(
                output
            ):
                self.assertEqual(two_host.controller(launch_args, config), 0)
            self.assertEqual(
                operations[:4],
                [
                    ("prepare", "prefill"),
                    ("prepare", "decode"),
                    ("serve", "decode"),
                    ("serve", "prefill"),
                ],
            )
            self.assertEqual(
                json.loads((root / "controller/config.json").read_text()), config
            )
            self.assertNotIn("MOONCAKE_", output.getvalue())
            self.assertNotIn("JSON_MODEL_OVERRIDE_ARGS", output.getvalue())


if __name__=='__main__':unittest.main()
