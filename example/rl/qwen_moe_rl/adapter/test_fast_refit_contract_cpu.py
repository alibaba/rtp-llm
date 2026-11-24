"""CPU contract tests for RLLM_RTP_FAST_REFIT cadence and honest post-sync probe handling."""
import ast
import asyncio
import hashlib
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch


ADAPTER_DIR = Path(__file__).parent
BACKEND_PATH = ADAPTER_DIR / "rtp_verl_backend.py"
SERVER_PATH = ADAPTER_DIR / "rtp_native_server.py"


def _load_backend_adapter():
    tree = ast.parse(BACKEND_PATH.read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "RTPServerAdapter")
    methods = [
        x
        for x in cls.body
        if isinstance(x, ast.AsyncFunctionDef) and x.name in ("update_weights", "_update_weights_digest_impl")
    ]
    for m in methods:
        m.decorator_list = []
    mod = ast.fix_missing_locations(
        ast.Module(body=[ast.ClassDef(name="Adapter", bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    )
    records = []

    class Device:
        type = "cuda"

        def __init__(self, *_):
            pass

        def __eq__(self, other):
            return isinstance(other, Device)

    class Tensor:
        def __init__(self, data, shape=None, dtype="torch.bfloat16"):
            self.data = data
            self.device = Device()
            self.dtype = dtype
            self.shape = shape if shape is not None else (len(data) // 2,)

        def detach(self):
            return self

        def to(self, **kw):
            dtype = kw.get("dtype", self.dtype)
            return Tensor(self.data, shape=self.shape, dtype=dtype)

        def contiguous(self):
            return self

        def cpu(self):
            return self

        def view(self, _):
            return self

        def numpy(self):
            return self.data

        def numel(self):
            n = 1
            for d in self.shape:
                n *= d
            return n

        def element_size(self):
            return 4 if "float32" in str(self.dtype) else 2

    class Control:
        def __init__(self, owner, step):
            self.owner = owner
            self.step = step
            self.pushed = []
            self.committed = False
            self.closed = False

        async def status(self):
            return [NS(phase="running", model_version=f"v{self.step}")]

        async def pause(self):
            pass

        async def begin_update(self, **_):
            pass

        async def push_weight(self, name, metadata):
            assert len(self.owner._unacknowledged_ipc) == 1
            self.pushed.append((name, metadata))

        async def commit_update(self):
            self.committed = True
            return [
                NS(
                    model_version=f"v{self.step + 1}",
                    global_step=self.step,
                    received_weights=len(self.pushed),
                    applied_weights=len(self.pushed),
                    tp_rank=0,
                    tp_size=4,
                    phase="paused",
                )
            ]

        async def close(self):
            self.closed = True

    class Server:
        restarted = False

        class Ensure:
            async def remote(self, *_):
                return dict(restarted=Server.restarted, native_boot="boot", native_incarnation="inc")

        ensure_native_running = Ensure()

    import contextlib, time

    ns = dict(
        torch=NS(
            device=Device,
            cuda=NS(current_device=lambda: 0),
            float32="torch.float32",
            bfloat16="torch.bfloat16",
            uint8="torch.uint8",
        ),
        os=os,
        contextlib=contextlib,
        time=time,
        hashlib=hashlib,
        json=json,
        _source=lambda: None,
        ray=NS(get_actor=lambda _: Server()),
        _new_control=lambda _: ns["control_factory"](),
        _audit_root=lambda: Path("/tmp"),
        _write_json=lambda _p, x: records.append(x),
        preserve_fp32_source=lambda _: False,
    )
    exec(compile(mod, str(BACKEND_PATH), "exec"), ns)
    return ns["Adapter"], Tensor, Control, Server, records, ns


def _load_probe_method():
    tree = ast.parse(SERVER_PATH.read_text())
    method = next(
        item
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for item in cls.body
        if isinstance(item, ast.AsyncFunctionDef) and item.name == "probe_synced_policy"
    )
    helper = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "_require_real_update_for_probe"
    )
    return ast.fix_missing_locations(ast.Module(body=[helper, method], type_ignores=[]))


class FastRefitContractTests(unittest.TestCase):
    def setUp(self):
        Adapter, Tensor, Control, Server, records, ns = _load_backend_adapter()
        self.Adapter = Adapter
        self.Tensor = Tensor
        self.Control = Control
        self.Server = Server
        self.records = records
        self.ns = ns
        ipc = ModuleType("rtp_llm.utils.cuda_ipc")
        ipc.CudaIpcV2 = NS(export_tensor=lambda v: {"ptr": id(v), "shape": list(v.shape)})
        self.mods = patch.dict(
            sys.modules,
            {"rtp_llm": ModuleType("rtp_llm"), "rtp_llm.utils": ModuleType("rtp_llm.utils"), "rtp_llm.utils.cuda_ipc": ipc},
        )
        self.mods.start()
        self.addCleanup(self.mods.stop)

    def _run_step(self, step, *, restarted=False, shape_mismatch=False, tied=False, env_extra=None):
        self.Server.restarted = restarted
        adapter = self.Adapter()
        for k, v in dict(
            is_leader=True,
            replica_rank=0,
            tp_rank=0,
            tp_size=4,
            global_rank=0,
            group_ranks=[0, 1, 2, 3],
            _last_sync_version=f"v{step}",
            _last_sync_step=step - 1 if step > 0 else None,
            hf_all_sources={"a", "b"} | ({"embed_tokens.weight"} if tied else set()),
            tied_embeddings=tied,
        ).items():
            setattr(adapter, k, v)
        sources = {"a", "b"} | ({"embed_tokens.weight"} if tied else set())
        adapter._required_sources = lambda: sources
        adapter._local_ipc_binding = lambda _s: {"ok": True}
        adapter._broadcast = lambda x: x
        adapter._any_global_failure = lambda e: bool(e)

        def gather(item):
            peers = [dict(item) for _ in range(4)] if isinstance(item, dict) else [item] * 4
            if shape_mismatch and isinstance(item, dict) and item.get("name") == "b":
                peers[1]["shape"] = [999]
            return peers

        adapter._gather = gather

        async def terminate():
            adapter._unacknowledged_ipc = []

        adapter._terminate_owned_replica = terminate
        ctrl = self.Control(adapter, step)

        async def _factory():
            return ctrl

        self.ns["control_factory"] = _factory
        weights = [("a", self.Tensor(b"abcd")), ("b", self.Tensor(b"efgh"))]
        if tied:
            weights.append(("embed_tokens.weight", self.Tensor(b"ijkl")))
            weights.append(("lm_head.weight", self.Tensor(b"ijkl")))
        env = {
            "RLLM_RTP_FAST_REFIT": "1",
            "RLLM_RTP_FULL_SHA_EVERY": "5",
            "RLLM_RTP_TOTAL_STEPS": "7",
            "Q36_REFIT_BOUNDED_DIGEST": "0",
            "Q36_RTP_PROFILE_REFIT": "0",
        }
        if env_extra:
            env.update(env_extra)
        with patch.dict(os.environ, env, clear=False):
            asyncio.run(adapter.update_weights(iter(weights), global_steps=step))
        return self.records[-1]

    def test_fast_vs_full_cadence(self):
        # Step 0 must always use full SHA256.
        r0 = self._run_step(0)
        self.assertEqual(r0["digest_mode"], "full_sha256")
        self.assertRegex(r0["full_source_sha256"], r"^[0-9a-f]{64}$")

        # Intermediate Step 1 uses fast path (no host SHA256, null digest, still exports & commits all weights).
        r1 = self._run_step(1, tied=True)
        self.assertEqual(r1["digest_mode"], "fast_no_host_sha")
        self.assertIsNone(r1["full_source_sha256"])
        self.assertTrue(all(t["sha256"] is None for t in r1["tensors"]))
        self.assertEqual(r1["native_applied_weights"], 3)

        # Native restart forces full SHA even on non-mod-5 step.
        r2 = self._run_step(2, restarted=True)
        self.assertEqual(r2["digest_mode"], "full_sha256")
        self.assertRegex(r2["full_source_sha256"], r"^[0-9a-f]{64}$")

        # Mod-5 and final step force full SHA.
        self.assertEqual(self._run_step(5)["digest_mode"], "full_sha256")
        self.assertEqual(self._run_step(7)["digest_mode"], "full_sha256")

        # Shape/dtype check remains enforced on fast steps.
        with self.assertRaisesRegex(RuntimeError, "identity differs"):
            self._run_step(1, shape_mismatch=True)

    def test_probe_accepts_fast_sync_with_null_sha_and_still_requires_score_change(self):
        mod = _load_probe_method()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("test-model")
            base_vals = [-0.5]
            base_sha = hashlib.sha256(struct.pack("<f", *base_vals)).hexdigest()
            (root / "rtp_probe_rank0_step0.json").write_text(
                json.dumps(
                    {
                        "prompt_ids": [1],
                        "response_ids": [1],
                        "model_version": "v0",
                        "digest_mode": "full_sha256",
                        "full_source_sha256": "a" * 64,
                        "score_sha256": base_sha,
                    }
                )
            )
            (root / "rtp_sync_rank0_step1.json").write_text(
                json.dumps({"model_version": "v1", "digest_mode": "fast_no_host_sha", "full_source_sha256": None})
            )
            records = []
            ns = {
                "json": json,
                "os": os,
                "struct": struct,
                "hashlib": hashlib,
                "Path": Path,
                "Q36_ZERO_UPDATE_CONFIG_SHA": "other",
                "_audit_root": lambda: root,
                "_write_json": lambda _p, r: records.append(r),
            }
            exec(compile(mod, str(SERVER_PATH), "exec"), ns)

            class Tokenizer:
                def encode(self, *args, **kwargs):
                    return [1]

            cur_vals = [-0.6]

            class Server:
                tokenizer = Tokenizer()
                replica_rank = 0
                tp_size = 4
                model_path = str(root)

                async def score_tokens(self, *args):
                    return {"global_step": 1, "model_version": "v1", "log_probs": list(cur_vals)}

            server = Server()
            with patch.dict(os.environ, {"RLLM_REQUIRE_REAL_UPDATE": "1"}, clear=True):
                res = asyncio.run(ns["probe_synced_policy"](server, 1, "v1"))
                self.assertEqual(res["digest_mode"], "fast_no_host_sha")
                self.assertIsNone(res["full_source_sha256"])
                self.assertIsNone(res["changed_from_step0"])
                self.assertTrue(res["scores_changed_from_step0"])

                # Still rejects stale/unchanged policy scores!
                cur_vals[:] = list(base_vals)
                with self.assertRaisesRegex(RuntimeError, "scores did not change"):
                    asyncio.run(ns["probe_synced_policy"](server, 1, "v1"))

    def test_hf_export_preserves_gdn_fp32_params(self):
        import torch
        from safetensors.torch import load_file

        training_dir = ADAPTER_DIR.parent / "training"
        overrides_dir = training_dir / "overrides"
        verl_root = Path("/data0/caodong.cao/work/verl-poker/verl-lll")
        for p in (str(training_dir), str(overrides_dir), str(verl_root)):
            if p not in sys.path:
                sys.path.insert(0, p)

        import qwen36_training_overrides
        from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

        qwen36_training_overrides.install()

        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = NS(model_type="qwen3_5_moe")
                self.model = torch.nn.Module()
                self.model.language_model = torch.nn.Module()
                layer0 = torch.nn.Module()
                layer0.linear_attn = torch.nn.Module()
                layer0.linear_attn.A_log = torch.nn.Parameter(torch.tensor([0.12345679], dtype=torch.float32))
                layer0.linear_attn.dt_bias = torch.nn.Parameter(torch.tensor([0.23456789], dtype=torch.float32))
                layer0.self_attn = torch.nn.Module()
                layer0.self_attn.q_proj = torch.nn.Linear(2, 2, bias=False, dtype=torch.float32)
                torch.nn.init.constant_(layer0.self_attn.q_proj.weight, 0.12345679)
                self.model.language_model.layers = torch.nn.ModuleList([layer0])

            def get_expanded_tied_weights_keys(self, all_submodels=True):
                return []

        mod = MockModule()
        mgr = object.__new__(FSDPCheckpointManager)
        mgr.rank = 0
        mgr.model = mod
        mgr.hf_export_dtype = torch.bfloat16
        mgr.per_tensor_param_fn = lambda: (
            ((n, p.detach().to(qwen36_training_overrides.export_dtype(n))) for n, p in mod.named_parameters()),
            None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr._save_hf_weights_from_per_tensor_param(tmpdir)
            saved = load_file(str(Path(tmpdir) / "model-00001-of-00001.safetensors"))
            self.assertEqual(saved["model.language_model.layers.0.linear_attn.A_log"].dtype, torch.float32)
            self.assertEqual(saved["model.language_model.layers.0.linear_attn.dt_bias"].dtype, torch.float32)
            self.assertAlmostEqual(
                saved["model.language_model.layers.0.linear_attn.A_log"].item(), 0.12345679, places=7
            )
            self.assertAlmostEqual(
                saved["model.language_model.layers.0.linear_attn.dt_bias"].item(), 0.23456789, places=7
            )
            self.assertEqual(saved["model.language_model.layers.0.self_attn.q_proj.weight"].dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
