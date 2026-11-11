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

    def test_synthetic_draft_cannot_replace_trained_mla(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp),4)
            path=args.draft_checkpoint/'config.json';config=json.loads(path.read_text())
            config['rtp_llm_fixture']={'synthetic':True,'trained':False};path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'real native K3 MLA draft'):launch.launch_config(args)

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
                    "hidden_state_capture_max_inflight_samples": 8,
                    "hidden_state_capture_max_inflight_bytes": 2147483648,
                    "hidden_state_capture_max_inflight_tokens": 32768,
                    "hidden_state_capture_max_pending_batches": 4,
                    "hidden_state_capture_poll_interval_ms": 250,
                    "rope_scaling": {
                        "type": "yarn",
                        "factor": 2.0,
                        "original_max_position_embeddings": 8192,
                    },
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

    def assert_torchspec_environment(self, environment):
        expected = self.torchspec_env()
        self.assertEqual({key: environment[key] for key in expected}, expected)
        for key in expected:
            self.assertIs(type(environment[key]), str)
        overrides = json.loads(environment["JSON_MODEL_OVERRIDE_ARGS"])
        expected_overrides = json.loads(expected["JSON_MODEL_OVERRIDE_ARGS"])
        self.assertIs(type(overrides), dict)
        self.assertEqual(overrides, expected_overrides)
        for key, value in expected_overrides.items():
            self.assertIs(type(overrides[key]), type(value))
        for layer_id in overrides["hidden_state_capture_layer_ids"]:
            self.assertIs(type(layer_id), int)
        for key, value in expected_overrides["rope_scaling"].items():
            self.assertIs(type(overrides["rope_scaling"][key]), type(value))

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
                    if capture_env:
                        self.assert_torchspec_environment(
                            config["prefill"]["torchspec_env"]
                        )
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
                            inherited = {
                                "PREFILL_CHUNK_SIZE": "8192",
                                "CP_ROTATE_METHOD": "ALL_GATHER",
                                "ENABLE_FFN_DISAGGREGATE": "1",
                                "FFN_SP_SIZE": "1",
                                "WORLD_SIZE": "16",
                                "JSON_MODEL_OVERRIDE_ARGS": json.dumps(
                                    {"rope_scaling": {"type": "base"}}
                                ),
                                "MOONCAKE_STORE_KEY_NAMESPACE": "Inherited001",
                            }
                            if capture_env and role == "prefill":
                                inherited.update(
                                    {key: "inherited-value" for key in capture_env}
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
                        self.assertEqual(options["--prefill_cp_kv_cache_sharded"], "1")
                        self.assertEqual(
                            options["--ffn_sp_size"], "8" if role == "prefill" else "4"
                        )
                        self.assertEqual(options["--world_size"], "8")
                        self.assertEqual(options["--local_world_size"], "8")
                        self.assertEqual(options["--enable_ffn_disaggregate"], "0")
                        self.assertNotIn("CP_ROTATE_METHOD", env)
                        self.assertNotIn("--cp_rotate_method", options)
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
                                self.assert_torchspec_environment(env)
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
                                self.assertTrue(
                                    (capture_env.keys() - inherited.keys()).isdisjoint(env)
                                )
                                self.assertEqual(receipt["torchspec_env_keys"], [])
                        else:
                            self.assertEqual(receipt["torchspec_env_keys"], [])
                        if not capture_env or role == "decode":
                            for key in (
                                "JSON_MODEL_OVERRIDE_ARGS",
                                "MOONCAKE_STORE_KEY_NAMESPACE",
                            ):
                                self.assertEqual(env[key], inherited[key])
                            self.assertEqual(
                                json.loads(env["JSON_MODEL_OVERRIDE_ARGS"]),
                                {"rope_scaling": {"type": "base"}},
                            )
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
                    prepared = json.loads(control.read_text())
                    self.assertEqual(prepared, config)
                    self.assert_torchspec_environment(
                        prepared["prefill"]["torchspec_env"]
                    )
                    self.assertNotIn("torchspec_env", prepared["decode"])
                    return SimpleNamespace(stdout=output.getvalue())
                copied = two_host.validate(json.loads(control.read_text()))
                self.assert_torchspec_environment(copied["prefill"]["torchspec_env"])
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
