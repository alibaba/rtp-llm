"""CPU contracts for four-rank parents in the explicit TP8/DP9 world72 profile.

The stdlib path executes the actual production functions extracted via AST,
with a process/device fixture. RTP_TEST_ACTUAL_RUNTIME=1 additionally imports
the runtime module and uses its compiled ParallelismConfig type; no processes,
CUDA work, TCP stores, or checkpoint payloads are started by either mode.
"""
import ast
import importlib.util
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(os.environ.get("RTP_TEST_SOURCE_ROOT", Path(__file__).resolve().parents[2]))
BACKEND = Path(os.environ.get("RTP_TEST_BACKEND_SOURCE", ROOT / "rtp_llm/start_backend_server.py"))
PROFILE = {
    "RTP_LLM_RL_WORLD72": "1",
    "RTP_LLM_RL_MODE": "1",
    "RTP_LLM_RL_START_PAUSED": "1",
    "RTP_LLM_RL_METADATA_BOOTSTRAP": "1",
    "RTP_FIXED_PHYSICAL_EP_LAYOUT": "balanced_v1",
    "RTP_RL_BORROW_SINGLETON_HF_SOURCES": "1",
    "LOCAL_WORLD_SIZE": "4",
    "CUDA_VISIBLE_DEVICES": "fixture-0,fixture-1,fixture-2,fixture-3",
}

def functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    defs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in defs} != set(names):
        raise AssertionError("missing production functions")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *defs], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace

ACTUAL = os.environ.get("RTP_TEST_ACTUAL_RUNTIME") == "1"
if ACTUAL:
    import torch
    from rtp_llm.config.server_config_setup import set_parallelism_config
    from rtp_llm.ops import ParallelismConfig
    spec = importlib.util.spec_from_file_location("world72_candidate_start_backend", BACKEND)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = module.__dict__
    make_pc = ParallelismConfig
else:
    torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 4, is_available=lambda: False))
    calls = functions(BACKEND, ["_validate_dp_configuration", "_get_local_world_size",
        "_get_cuda_device_list", "_create_rank_processes"],
        {"os": os, "torch": torch, "logging": logging, "local_rank_start": object()})
    set_parallelism_config = functions(ROOT / "rtp_llm/config/server_config_setup.py",
        ["set_parallelism_config"], {"torch": torch, "logging": logging})["set_parallelism_config"]
    make_pc = lambda: SimpleNamespace(to_string=lambda: "CPU ParallelismConfig fixture")

def config(rank):
    pc = make_pc()
    for key, value in {"world_size":72,"tp_size":8,"dp_size":9,"ep_size":72,
        "pp_size":1,"local_world_size":4,"ffn_sp_size":8}.items():
        setattr(pc, key, value)
    set_parallelism_config(pc, rank)
    return SimpleNamespace(parallelism_config=pc)

class Pipe:
    closed = False
    def close(self):
        self.closed = True

class Process:
    def __init__(self, *, target, args, name):
        self.target, self.args, self.name = target, args, name
        self.started = False
    def start(self):
        # Explicit process fixture: capture spawn contract without creating a
        # rank, invoking local_rank_start, or touching any GPU/network state.
        self.started = True

class Context:
    writers = None
    def __init__(self):
        self.writers = []
    def Pipe(self, duplex):
        assert duplex is False
        reader, writer = Pipe(), Pipe()
        self.writers.append(writer)
        return reader, writer
    Process = Process

class World72BackendRanks(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, PROFILE)
        self.env.start()
    def tearDown(self):
        self.env.stop()

    def test_all_18_parents_spawn_exact_72_global_ranks_and_preserve_four_devices(self):
        spawned = []
        with patch.object(torch.cuda, "device_count", return_value=4):
            for parent_rank in range(0, 72, 4):
                cfg = config(parent_rank)
                ctx, processes, readers = Context(), [], []
                calls["_create_rank_processes"](object(), cfg, ctx, processes, readers)
                self.assertEqual([p.args[2] for p in processes], list(range(parent_rank, parent_rank + 4)))
                self.assertEqual(len(readers), 4)
                self.assertTrue(all(p.started for p in processes))
                self.assertTrue(all(p.target is calls["local_rank_start"] for p in processes))
                self.assertTrue(all(w.closed for w in ctx.writers))
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], PROFILE["CUDA_VISIBLE_DEVICES"])
                spawned.extend(p.args[2] for p in processes)
        self.assertEqual(spawned, list(range(72)))
        groups = {}
        for rank in spawned:
            pc = config(rank).parallelism_config
            self.assertEqual((pc.local_rank, pc.tp_rank, pc.dp_rank, pc.ep_rank, pc.ffn_tp_rank),
                (rank % 4, rank % 8, rank // 8, rank, 0))
            groups.setdefault(pc.dp_rank, []).append(pc.tp_rank)
        self.assertEqual(groups, {dp: list(range(8)) for dp in range(9)})

    def test_misaligned_and_out_of_world_parents_rejected(self):
        for rank in (-4, -1, 1, 3, 5, 71, 72, 76):
            cfg = config(0)
            cfg.parallelism_config.world_rank = rank
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                calls["_validate_dp_configuration"](cfg)

    def test_profile_sizes_are_not_a_general_dp_bypass(self):
        for name, value in {"world_size":64,"tp_size":4,"dp_size":18,"ep_size":1,
            "pp_size":2,"local_world_size":8,"ffn_sp_size":1,"ffn_tp_size":8}.items():
            cfg = config(4)
            setattr(cfg.parallelism_config, name, value)
            with self.subTest(field=name), self.assertRaises(ValueError):
                calls["_validate_dp_configuration"](cfg)

    def test_inconsistent_derived_ranks_rejected(self):
        for name, value in {"local_rank":1,"tp_rank":0,"dp_rank":1,"ep_rank":0,"ffn_tp_rank":1}.items():
            cfg = config(4)
            setattr(cfg.parallelism_config, name, value)
            with self.subTest(field=name), self.assertRaises(ValueError):
                calls["_validate_dp_configuration"](cfg)

    def test_each_required_guard_remains_mandatory(self):
        for name in PROFILE:
            if name in ("RTP_LLM_RL_WORLD72", "LOCAL_WORLD_SIZE", "CUDA_VISIBLE_DEVICES"):
                continue
            with self.subTest(env=name), patch.dict(os.environ, {name: ""}), self.assertRaises(ValueError):
                calls["_validate_dp_configuration"](config(4))

    def test_non_world72_deployments_keep_original_parent_alignment(self):
        with patch.dict(os.environ, {"RTP_LLM_RL_WORLD72": "0"}):
            calls["_validate_dp_configuration"](config(0))
            with self.assertRaises(AssertionError):
                calls["_validate_dp_configuration"](config(4))
            cfg = config(4)
            cfg.parallelism_config.dp_size = 1
            calls["_validate_dp_configuration"](cfg)

if __name__ == "__main__":
    unittest.main()
