"""CPU-only generic padding regression and replay of an existing failed batch.

The replay stops at the first real compute_log_prob RPC boundary. It does not
return fabricated log-probabilities or produce/claim an RL optimizer update.
Existing episode files are read-only and hashed before and after the replay.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

CANDIDATE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CANDIDATE / "adapter"))


def fingerprint(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def import_child():
    assert not os.environ.get("CUDA_VISIBLE_DEVICES"), "CPU check requires an empty CUDA device mask"
    import install_rtp_bridge
    assert "torch" not in sys.modules, "setup hook imported Torch before natural framework import"
    install_rtp_bridge.worker_setup()
    assert "torch" not in sys.modules, "passive setup hook imported Torch"
    import verl.utils.attention_utils as attention
    assert attention._get_attention_functions._q397_torch_padding
    import torch
    assert not torch.cuda.is_initialized()
    torch.cuda._lazy_init = lambda: (_ for _ in ()).throw(AssertionError("CPU check attempted CUDA init"))
    from rllm.trainer.verl.verl_launcher import VerlTaskRunner
    from rllm.trainer.verl.verl_backend import VerlBackend
    import ray
    functions = attention._get_attention_functions()
    assert len(functions) == 4
    assert functions[0].__module__ == "q397_torch_padding"
    assert functions[2].__module__.startswith("einops")
    assert not ray.is_initialized() and not torch.cuda.is_initialized()
    assert not torch.distributed.is_initialized()
    assert importlib.util.find_spec("flash_attn") is None
    assert "flash_attn" not in sys.modules
    return dict(taskrunner=str(VerlTaskRunner), backend_source=inspect.getsourcefile(VerlBackend),
                hook_passive=True, attention_source=fingerprint(attention.__file__),
                cuda_initialized=False, ray_initialized=False)


def helpers_regression():
    import torch
    import q397_torch_padding as shim
    upstream_path = Path(__file__).resolve().parent / "q397_padding_reference" / "bert_padding.py"
    spec = importlib.util.spec_from_file_location("_q397_upstream_padding_reference", upstream_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    checks = []

    def equal(left, right):
        if isinstance(left, torch.Tensor):
            assert left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
        else:
            assert left == right

    for dtype in (torch.float64, torch.float32, torch.bfloat16, torch.int64):
        for variant in ("holes", "unused", "noncontiguous", "all_masked", "empty_sequence"):
            shape = (3, 0, 4, 2) if variant == "empty_sequence" else (3, 7, 4, 2)
            source = torch.arange(torch.tensor(shape).prod().item()).reshape(shape).to(dtype)
            if variant == "noncontiguous":
                source = source.transpose(-1, -2)
                assert not source.is_contiguous()
            attention = torch.tensor([[0, 1, 1, 0, 1, 0, 0], [1, 0, 1, 1, 0, 1, 0], [0] * 7], dtype=torch.int32)
            if variant == "empty_sequence":
                attention = attention[:, :0]
            if variant == "all_masked":
                attention = torch.zeros_like(attention)
            unused = 1 - attention if variant == "unused" else None
            differentiable = dtype != torch.int64
            lhs = source.detach().clone(memory_format=torch.preserve_format).requires_grad_(differentiable)
            rhs = source.detach().clone(memory_format=torch.preserve_format).requires_grad_(differentiable)
            result = shim.unpad_input(lhs, attention, unused_mask=unused)
            expected = reference.unpad_input(rhs, attention, unused_mask=unused)
            assert len(result) == len(expected) == 5
            for actual, wanted in zip(result, expected, strict=True):
                equal(actual, wanted)
            padded = shim.pad_input(result[0], result[1], shape[0], shape[1])
            padded_ref = reference.pad_input(expected[0], expected[1], shape[0], shape[1])
            equal(padded, padded_ref)
            if differentiable:
                weights = torch.arange(padded.numel(), dtype=torch.float64).reshape(padded.shape).remainder(11)
                (padded.double() * weights).sum().backward()
                (padded_ref.double() * weights).sum().backward()
                equal(lhs.grad, rhs.grad)
            checks.append(f"{dtype}:{variant}:forward_offsets_used_lengths_roundtrip" + (":backward" if differentiable else ""))

    for indices in (torch.tensor([0, 4, 2]), torch.tensor([1, 1, 3]), torch.empty(0, dtype=torch.long)):
        lhs = torch.arange(30, dtype=torch.float64).reshape(5, 2, 3).requires_grad_(True)
        rhs = lhs.detach().clone().requires_grad_(True)
        actual, wanted = shim.index_first_axis(lhs, indices), reference.index_first_axis(rhs, indices)
        equal(actual, wanted)
        grad = torch.arange(actual.numel(), dtype=torch.float64).reshape(actual.shape)
        actual.backward(grad)
        wanted.backward(grad)
        equal(lhs.grad, rhs.grad)
        checks.append(f"index_autograd:{indices.tolist()}")

    # Double precision finite-difference checks exercise a complete pad/unpad
    # graph, independently of the upstream custom autograd implementation.
    mask = torch.tensor([[0, 1, 1], [1, 0, 1]], dtype=torch.int32)
    x = torch.randn(2, 3, 2, dtype=torch.float64, requires_grad=True)
    def roundtrip(value):
        unpacked = shim.unpad_input(value, mask)
        return shim.pad_input(unpacked[0], unpacked[1], 2, 3)
    assert torch.autograd.gradcheck(roundtrip, (x,), eps=1e-6, atol=1e-5, rtol=1e-4)
    checks.append("unique_mask_roundtrip_finite_difference_gradcheck")

    # Narrow exception policy: a missing nested dependency or other failure
    # must propagate rather than silently choosing a different backend.
    for name in ("flash_attn", "flash_attn.bert_padding", "einops", "flash_attn_2_cuda"):
        def original(name=name):
            raise ModuleNotFoundError("reference missing dependency", name=name)
        module = SimpleNamespace(_get_attention_functions=original)
        shim.install(module)
        installed = module._get_attention_functions
        shim.install(module)
        assert module._get_attention_functions is installed
        try:
            functions = installed()
        except ModuleNotFoundError as exc:
            assert name not in {"flash_attn", "flash_attn.bert_padding"} and exc.name == name
        else:
            assert name in {"flash_attn", "flash_attn.bert_padding"} and functions[0] is shim.index_first_axis
        checks.append(f"missing_dependency_policy:{name}")
    marker = object()
    module = SimpleNamespace(_get_attention_functions=lambda: marker)
    shim.install(module)
    assert module._get_attention_functions() is marker
    checks.append("existing_original_backend_unchanged")
    for fn in (shim.unpad_input, shim.pad_input):
        assert tuple(inspect.signature(fn).parameters) == tuple(inspect.signature(getattr(reference, fn.__name__)).parameters)
    assert tuple(inspect.signature(shim.index_first_axis).parameters) == ("input", "indices")
    return dict(checks=checks, count=len(checks), reference=fingerprint(upstream_path),
                implementation=fingerprint(shim.__file__), exact_forward_and_backward=True)


def replay(episodes_dir, logged_episodes_dir, model_path):
    import torch
    import install_rtp_bridge
    install_rtp_bridge.worker_setup()
    from omegaconf import OmegaConf
    from rllm.types import Episode
    from rllm.trainer.verl.verl_backend import VerlBackend
    from transformers import AutoTokenizer
    from verl.utils import tensordict_utils as tu

    paths = sorted(Path(episodes_dir).glob("enriched_*.json"))
    assert len(paths) == 32, f"Replay requires exactly the 32 existing r8 enriched episodes, found {len(paths)}"
    logged_paths = sorted(Path(logged_episodes_dir).glob("episode_*.json"))
    assert len(logged_paths) == 32
    logged = {json.loads(path.read_text())["episode_id"]: (path, json.loads(path.read_text())) for path in logged_paths}
    assert len(logged) == 32
    originals = [fingerprint(path) for path in paths + logged_paths]
    episodes = []
    raw_counts = []
    for path in paths:
        record = json.loads(path.read_text())
        log_path, log = logged[record["episode_id"]]
        assert record["task"] == log["task"] and record["is_correct"] == log["is_correct"]
        assert len(record["trajectories"]) == len(log["trajectories"]) == 1
        # Enriched records preserve exact token/FP32-logprob/model-output
        # payloads; EpisodeLogger preserves original trajectory UID/reward and
        # termination metadata. Rejoin these two read-only views by episode ID,
        # then require every shared step field to match before deserialization.
        trajectories = []
        for payload, metadata in zip(record["trajectories"], log["trajectories"], strict=True):
            assert payload["name"] == metadata["name"]
            assert len(payload["steps"]) == len(metadata["steps"])
            assert metadata["reward"] == record["reward"]
            for step, log_step in zip(payload["steps"], metadata["steps"], strict=True):
                for key in ("observation", "thought", "action", "reward", "done", "model_response", "chat_completions"):
                    assert step[key] == log_step[key], (record["episode_id"], key)
            trajectories.append({"uid": metadata["uid"], "name": payload["name"], "task": record["task"],
                                 "reward": metadata["reward"], "steps": payload["steps"], "info": {}})
        episode = Episode.from_dict({"id": record["episode_id"], "task": record["task"],
                  "is_correct": record["is_correct"], "termination_reason": log["termination_reason"],
                  "metrics": record["metrics"], "info": log["metadata"], "trajectories": trajectories})
        counts = [len(step.response_ids) for traj in episode.trajectories for step in traj.steps]
        used = 0
        for count in counts:
            assert count <= min(2048, 3072 - used)
            used += count
        assert used <= 3072
        assert episode.trajectories and all(traj.steps for traj in episode.trajectories)
        episodes.append(episode)
        raw_counts.append(dict(id=episode.id, response_tokens=used, rewards=[traj.reward for traj in episode.trajectories]))
    assert len({e.id for e in episodes}) == 32
    groups = {}
    for e in episodes:
        groups.setdefault(e.task_id, []).append(e.trajectories[0].reward)
    assert len(groups) == 8 and all(len(values) == 4 for values in groups.values())
    assert any(len(set(values)) > 1 for values in groups.values()), "The original batch must retain its genuine mixed rewards"

    class ReachedRealLogprobRPC(Exception):
        pass

    class CPUOnlyWorkerBoundary:
        world_size = 32
        _dispatch_info = {"actor": list(range(32))}
        received = None
        def compute_log_prob(self, data):
            self.received = data
            raise ReachedRealLogprobRPC("Stopped before the real GPU worker call; no substitute output")

    # All values mirror the recorded r8 resolved configuration. Only resource
    # construction and the GPU RPC itself are deliberately not executed.
    config = OmegaConf.create({
        "data": {"train_batch_size": 8, "max_prompt_length": 4096, "max_response_length": 4096},
        "rllm": {"mask_truncated_samples": False}, "trainer": {"balance_batch": True},
        "actor_rollout_ref": {"rollout": {"n": 4, "temperature": 1.0},
                              "actor": {"ppo_mini_batch_size": 8}},
    })
    backend = VerlBackend.__new__(VerlBackend)
    backend.config = config
    backend.rollout_engine = SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(str(model_path), local_files_only=True), processor=None)
    backend.processor = None
    backend.actor_rollout_wg = CPUOnlyWorkerBoundary()
    backend.use_reference_policy = False
    backend.use_prefix_grouper = False
    backend.algorithm_config = None
    padded_batches = []
    original_pad = backend._pad_dataproto_to_world_size
    def observe_real_padding(*args, **kwargs):
        batch = original_pad(*args, **kwargs)
        padded_batches.append(batch)
        return batch
    backend._pad_dataproto_to_world_size = observe_real_padding
    state = SimpleNamespace(episodes=episodes, trajectory_groups=None, metrics={}, timing_dict={})
    state.backend_batch = backend.transform_to_backend_batch(state)
    assert state.backend_batch.batch["input_ids"].device.type == "cpu"
    try:
        asyncio.run(backend.process_backend_batch(state))
    except ReachedRealLogprobRPC:
        pass
    else:
        raise AssertionError("Real process_backend_batch did not reach its first compute_log_prob boundary")
    packed = backend.actor_rollout_wg.received
    assert packed is not None and packed["input_ids"].is_nested
    assert len(padded_batches) == 1
    batch = padded_batches[0]
    # Read the actual batch returned by the real DP32 pad operation, including
    # newly added neutral rows, after the real balancing reorder.
    dense = batch.batch["input_ids"]
    mask = batch.batch["attention_mask"].bool()
    expected_values = dense[mask]
    assert torch.equal(packed["input_ids"].values(), expected_values)
    assert torch.equal(packed["input_ids"].offsets().diff().long(), mask.sum(-1))
    assert torch.equal(tu.get_non_tensor_data(packed, "indices", None), mask.flatten().nonzero().flatten())
    assert torch.equal(packed["loss_mask"], batch.batch["response_mask"])
    assert tu.get_non_tensor_data(packed, "calculate_entropy", None) is True
    assert tu.get_non_tensor_data(packed, "compute_loss", None) is False
    assert batch.meta_info["rllm_actor_num_updates"] == 1
    assert batch.meta_info["rllm_actor_global_batch_size"] == 32
    assert batch.meta_info["rllm_actor_mini_batch_size"] % 32 == 0
    assert batch.batch["prompts"].shape[0] % 32 == 0
    for index, seq in enumerate(packed["position_ids"].unbind()):
        assert torch.equal(seq, batch.batch["position_ids"][index][mask[index]])
    assert [fingerprint(path) for path in paths + logged_paths] == originals, "Replay modified original evidence"
    assert not torch.cuda.is_initialized() and not torch.distributed.is_initialized()
    return dict(episodes=32, original_groups=groups, raw_episode_tokens=raw_counts,
                source_files=originals, model_tokenizer_config=fingerprint(Path(model_path) / "tokenizer_config.json"),
                runtime_backend_source=fingerprint(inspect.getsourcefile(VerlBackend)),
                rows_pre_pad=batch.meta_info["rllm_actor_rows_pre_pad"],
                rows_post_pad=int(batch.batch["prompts"].shape[0]),
                mini_batch_rows=batch.meta_info["rllm_actor_mini_batch_size"],
                global_batch_rollouts=batch.meta_info["rllm_actor_global_batch_size"],
                expected_updates_per_window=batch.meta_info["rllm_actor_num_updates"],
                packed_tokens=packed["input_ids"].values().numel(),
                stop_boundary="actual_VerlBackend.process_backend_batch:first_compute_log_prob_RPC",
                replay_only=True, actual_gpu_forward=False, optimizer_updates=0,
                original_evidence_unchanged=True, cuda_initialized=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--import-child", action="store_true")
    parser.add_argument("--episodes-dir")
    parser.add_argument("--logged-episodes-dir")
    parser.add_argument("--model-path")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.import_child:
        print("Q397_PADDING_IMPORT_CHILD " + json.dumps(import_child()), flush=True)
        return
    assert args.output
    assert not os.environ.get("CUDA_VISIBLE_DEVICES"), "CPU check requires an empty CUDA mask"
    # Run the worker-setup import contract in an independent interpreter before
    # this process naturally imports Torch for the tensor regressions.
    child_env = os.environ.copy()
    child_env.pop("Q397_RUN_ROOT", None)
    child = subprocess.run([sys.executable, str(__file__), "--import-child"], env=child_env,
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
    assert child.returncode == 0, child.stdout
    import torch
    assert not torch.cuda.is_initialized()
    torch.cuda._lazy_init = lambda: (_ for _ in ()).throw(AssertionError("CPU check attempted CUDA init"))
    assert torch.__version__ == "2.11.0+cu130", torch.__version__
    torch.set_num_threads(1)
    from q397_training_deps import ensure_training_dependencies
    dependency = ensure_training_dependencies()
    # Synthetic helper contract checks do not emit run first-use proofs.
    original_run = os.environ.pop("Q397_RUN_ROOT", None)
    result = dict(schema="q397-torch-generic-padding-cpu-regression-v1", started_unix=time.time(),
                  torch_version=torch.__version__, python=sys.version, test_source=fingerprint(__file__), import_contract_stdout=child.stdout,
                  dependency=dependency, helpers=helpers_regression())
    if original_run:
        os.environ["Q397_RUN_ROOT"] = original_run
    if args.episodes_dir or args.model_path:
        assert args.episodes_dir and args.logged_episodes_dir and args.model_path
        result["original_r8_batch_replay"] = replay(args.episodes_dir, args.logged_episodes_dir, args.model_path)
    assert not torch.cuda.is_initialized()
    import ray
    assert not ray.is_initialized() and not torch.distributed.is_initialized()
    result.update(finished_unix=time.time(), status="passed_cpu_padding_only", cuda_initialized=False,
                  ray_initialized=False, optimizer_updates=0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("Q397_PADDING_CPU_PASS " + json.dumps({"output": str(output), "helper_checks": result["helpers"]["count"],
          "original_episode_replay": "original_r8_batch_replay" in result}), flush=True)


if __name__ == "__main__":
    main()
