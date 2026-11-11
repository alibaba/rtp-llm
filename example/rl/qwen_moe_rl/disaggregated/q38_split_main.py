"""Launch one two-update Qwen3.8 rLLM job on an existing, claimed Ray session."""

import argparse
import hashlib
from ipaddress import IPv4Address
import json
import os
from pathlib import Path

from q38_driver_recipe import MAX_RESPONSE, PROMPTS, ROLLOUTS, STEPS, Recipe, compose


DATASET_SHA = "28545ec7d54b1e287dbb78a8f4b47e0c7bc5d48804eb5a307236f5fe91e67c5c"
DATA_SOURCE = "deepscaler_math"
CHECKPOINT_ENV = (
    "Q38_OBJECT_MANIFEST", "Q38_POLICY_MANIFEST", "Q38_CHECKPOINT_CATALOG",
    "Q38_CHECKPOINT_CATALOG_SHA256", "Q38_CHECKPOINT_ENDPOINTS",
)
CHECKPOINT_LOCKS = {
    "Q38_OBJECT_MANIFEST": "987eb6586ccecf6cfbb29b751ebdd711b1a2ec41cde6933f166317ef0b86651a",
    "Q38_POLICY_MANIFEST": "c60d0be889e462b7665922001abfc5a57e3846f33330e33ebfdc044e6b7d3075",
    "Q38_CHECKPOINT_CATALOG": "34b083dd3e39e32b511e5459a3cfdc23ee554a9d02de99198e418913a09a6aa2",
    "Q38_CHECKPOINT_ENDPOINTS": "ddcd4eab60eeb2aa97d84e0c263a5215543421c4792f5619f7a5d67714508d9f",
}


def load_dataset(path):
    from rllm.data import Dataset

    dataset_file = Path(path)
    raw = dataset_file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATASET_SHA:
        raise ValueError("DeepScaleR rows differ from the frozen training dataset")
    rows = [json.loads(line) for line in raw.splitlines()]
    if (len(rows) != PROMPTS * STEPS or len({row["question"] for row in rows}) != len(rows)
            or any(not row["ground_truth"] or row["data_source"] != DATA_SOURCE for row in rows)):
        raise ValueError("invalid original DeepScaleR questions or final answers")
    dataset = Dataset.load_data(str(dataset_file))
    if len(dataset) != len(rows):
        raise RuntimeError("rLLM did not load the entire frozen DeepScaleR dataset")
    return dataset


def _checkpoint_env(deploy):
    env = {key: os.environ[key] for key in CHECKPOINT_ENV}
    for key, expected in CHECKPOINT_LOCKS.items():
        path = Path(env[key])
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"missing scoped checkpoint input {key}")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"checkpoint source bytes differ for {key}")
    if Path(env["Q38_POLICY_MANIFEST"]).resolve() != Path(deploy.manifest).resolve():
        raise ValueError("trainer and native bootstrap use different policy manifests")
    if env["Q38_CHECKPOINT_CATALOG_SHA256"] != CHECKPOINT_LOCKS["Q38_CHECKPOINT_CATALOG"]:
        raise ValueError("checkpoint catalog hash mismatch")
    peer_bytes = Path(__file__).with_name("DEPLOYMENT-PEERS-VERL.json").read_bytes()
    if hashlib.sha256(peer_bytes).hexdigest() != "323aac8aa01963056600a557c5763f90ba4b737f21039ea904dab720ef35c440":
        raise ValueError("checkpoint relay peer manifest hash mismatch")
    peers = json.loads(peer_bytes)
    if (peers["train"] != sorted(node.ip for node in deploy.train)
            or peers["rollout"] != [node.ip for node in deploy.rollout]):
        raise ValueError("the claimed training rank/IP or native node order differs from the relay scope")
    return env


def _tool_env(output):
    values = {
        "RLLM_PYTHON_TOOL_MAX_TURNS": "2",
        "RLLM_PYTHON_TOOL_CALL_TOKENS": str(MAX_RESPONSE // 2),
        "RLLM_PYTHON_TOOL_TOTAL_TOKENS": str(MAX_RESPONSE),
        "RLLM_PYTHON_TOOL_ENABLE_THINKING": "1",
        "RLLM_PYTHON_TOOL_REASONING_EFFORT": "low",
        "RLLM_PYTHON_TOOL_HTTP_MAX_RETRIES": "0",
        "RLLM_PYTHON_TOOL_HTTP_TIMEOUT": "920",
        "RLLM_PYTHON_TOOL_TRACE_DIR": str(output / "episodes"),
    }
    os.environ.update(values)


def _make_backend(deploy):
    import torch
    from rllm.engine.rollout.rollout_engine import RolloutEngine
    from rllm.engine.rollout.verl_engine import VerlEngine
    from rllm.gateway.manager import GatewayManager
    from rllm.trainer.verl import verl_backend

    from q38_split_backend import initialize_backend
    from q38_split_native import _write_json
    from rtp_gateway_adapter import create_official_chat_parser, create_rtp_handler

    class SplitRTPVerlEngine(VerlEngine):
        def __init__(self, config, server_manager, tokenizer, processor=None, **kwargs):
            RolloutEngine.__init__(self)
            self.config = config
            self.server_manager = server_manager
            self.tokenizer = tokenizer
            self.processor = processor
            self.chat_parser = create_official_chat_parser(tokenizer, processor=processor, disable_thinking=False)
            eos = json.loads((Path(deploy.metadata) / "generation_config.json").read_text())["eos_token_id"]
            self.chat_parser.stop_sequences = [eos] if type(eos) is int else list(eos)
            self.max_prompt_length = config.data.max_prompt_length
            self.max_response_length = config.data.max_response_length
            self.accumulate_reasoning = False
            self.router_replay_mode = "disabled"
            self.train_sampling_params = dict(temperature=1.0, top_p=1.0, top_k=0, logprobs=1)
            self.val_sampling_params = self.train_sampling_params.copy()

        async def get_token_output_from_token_input(self, token_input, **kwargs):
            output = await super().get_token_output_from_token_input(token_input, **kwargs)
            output.stop_reason = output.extra_fields["native_finish_reason"]
            return output

    original_gateway_start = GatewayManager.start

    def start_gateway(self, engine):
        if not isinstance(engine, SplitRTPVerlEngine):
            return original_gateway_start(self, engine)
        if self.cumulative_token_mode or self.tunnel_backend:
            raise RuntimeError("split RTP training needs the private, token-preserving gateway")
        self._local_handler = create_rtp_handler(engine)
        self._start_thread(local_handler=self._local_handler)
        self._train_sampling_params = engine.train_sampling_params
        self._val_sampling_params = engine.val_sampling_params

    class SplitVerlBackend(verl_backend.VerlBackend):
        def _init_colocated_workers(self):
            initialize_backend(self, deploy)

        def init_rollout_engine(self, **kwargs):
            original = verl_backend.VerlEngine
            verl_backend.VerlEngine = SplitRTPVerlEngine
            try:
                return super().init_rollout_engine(**kwargs)
            finally:
                verl_backend.VerlEngine = original

        async def generate_episodes(self, batch, agent_workflow_engine, is_validation=False, **kwargs):
            episodes = await super().generate_episodes(batch, agent_workflow_engine, is_validation, **kwargs)
            if (is_validation or len(episodes) != PROMPTS * ROLLOUTS
                    or any(episode.info.get("data_source") != DATA_SOURCE for episode in episodes)):
                raise RuntimeError("each step must consume 32 new DeepScaleR episodes")
            self.q38_batch_episodes = len(episodes)
            self.q38_tool_successes = sum(episode.metrics.get("python_tool_successes", 0) for episode in episodes)
            return episodes

        async def update_policy(self, state):
            data = state.backend_batch.batch
            scores = data["rollout_log_probs"]
            mask = data["response_mask"].bool()
            advantages = data["advantages"][mask]
            if (scores.dtype != torch.float32 or not bool(torch.isfinite(scores[mask]).all())
                    or not torch.equal(data["old_log_probs"], scores)
                    or state.backend_batch.meta_info.get("rllm_actor_num_updates") != 1):
                raise RuntimeError("the actor update requires the native FP32 policy scores and one Adam step")
            if (not bool(torch.isfinite(advantages).all()) or not bool((advantages > 0).any())
                    or not bool((advantages < 0).any())):
                raise RuntimeError("the original final-answer rewards did not produce diverse finite GRPO advantages")
            await super().update_policy(state)
            _write_json(Path(deploy.output) / f"train-step{state.global_step}.json", {
                "global_step": state.global_step, "episodes": self.q38_batch_episodes,
                "python_tool_successes": self.q38_tool_successes,
                "positive_advantage_tokens": int((advantages > 0).sum()),
                "negative_advantage_tokens": int((advantages < 0).sum()),
            })

        async def on_batch_end(self, state):
            if not (Path(deploy.output) / f"train-step{state.global_step}.json").is_file():
                raise RuntimeError("native publication requires the completed on-policy actor update")
            await super().on_batch_end(state)
            state.weight_version = state.global_step
            self.rollout_engine.weight_version = state.weight_version
            self._q38_gateway.set_weight_version(state.weight_version)

    GatewayManager.start = start_gateway
    return SplitVerlBackend, lambda: setattr(GatewayManager, "start", original_gateway_start)


def _preflight_driver(deploy, recipe, dataset_path):
    import httpx

    # Build the same verified TLS context the local gateway will use.
    # Catch missing package CA files before Ray loads model weights.
    with httpx.Client(trust_env=True):
        pass
    config = compose(recipe)
    dataset = load_dataset(dataset_path)
    if config.actor_rollout_ref.model.path != deploy.metadata:
        raise RuntimeError("Hydra resolved a different training metadata path")
    return config, dataset


def run(deploy, recipe, dataset_path, ray_address):
    import ray
    from omegaconf import OmegaConf, open_dict
    from rllm.trainer.unified_trainer import UnifiedTrainer
    from verl.single_controller.ray import RayWorkerGroup, ResourcePoolManager
    from verl.trainer.ppo.utils import Role
    from verl.utils import hf_tokenizer

    from python_math_agent import python_math_agent, python_math_evaluator
    from q38_split_actor import SplitActorWorker
    from q38_split_backend import ray_claim
    from q38_split_native import _write_json

    config, dataset = _preflight_driver(deploy, recipe, dataset_path)
    env = _checkpoint_env(deploy)
    host, port = ray_address.split(":")
    if IPv4Address(host) != IPv4Address(deploy.train[0].ip) or not 1024 <= int(port) <= 65535:
        raise ValueError("connect to the claimed training head and its dedicated GCS port")
    if ray.is_initialized():
        raise RuntimeError("start the split job in a fresh driver process")

    output = Path(deploy.output)
    output.mkdir(exist_ok=False)
    _tool_env(output)
    with open_dict(config):
        config.rllm.trainer.logger = ["console"]
        config.rllm.trainer.experiment_name = deploy.run_id
        config.trainer.logger = ["console"]
        config.trainer.experiment_name = deploy.run_id
        config.trainer.default_local_dir = str(output / "checkpoints")
    (output / "resolved-config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))

    # The launcher stages per-node Python and CUDA library paths on the Ray
    # process. The only forwarded variables here are the locked source inputs.
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env["NCCL_IB_GID_INDEX"] = "3"
    env["OMP_NUM_THREADS"] = "3"
    ray.init(address=ray_address, namespace=deploy.run_id, runtime_env={"env_vars": env})
    trainer = None
    restore = None
    try:
        ray_claim(deploy)
        if ray.util.get_node_ip_address() != deploy.train[0].ip:
            raise RuntimeError("run the driver inside the claimed training head container")
        tokenizer = hf_tokenizer(deploy.metadata, trust_remote_code=False)
        backend, restore = _make_backend(deploy)
        pool = ResourcePoolManager(
            resource_pool_spec={"q38-training": [4] * len(deploy.train)},
            mapping={Role.ActorRollout: "q38-training"}, max_colocate_count=1,
        )
        trainer = UnifiedTrainer(
            backend_cls=backend, config=config, train_dataset=dataset, val_dataset=None,
            agent_flow=python_math_agent, evaluator=python_math_evaluator, backend_name="verl",
            backend_args={
                "tokenizer": tokenizer, "processor": None,
                "role_worker_mapping": {Role.ActorRollout: ray.remote(max_restarts=0)(SplitActorWorker)},
                "resource_pool_manager": pool, "ray_worker_group_cls": RayWorkerGroup,
            },
        )
        trainer.backend._q38_gateway = trainer._gateway
        trainer.fit()
        step_rows = [json.loads((output / f"train-step{step}.json").read_bytes())
                     for step in range(1, STEPS + 1)]
        if sum(row["python_tool_successes"] for row in step_rows) <= 0:
            raise RuntimeError("the model emitted no successfully executed Python Tool call")
        result = ray.get(trainer.backend._q38_coordinator.finish.remote(), timeout=450)
        _write_json(output / "DRIVER-TRAINING-COMPLETE.json", {
            "run_id": deploy.run_id, "data_sha256": DATASET_SHA, "train_steps": step_rows,
            **result,
        })
    finally:
        try:
            if trainer is not None:
                trainer.shutdown()
        finally:
            if restore is not None:
                restore()
            ray.shutdown()


def main():
    from q38_split_backend import Deployment

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gateway-port", required=True, type=int)
    parser.add_argument("--ray-address", required=True)
    args = parser.parse_args()
    deploy = Deployment.from_json(args.deployment)
    recipe = Recipe(deploy.metadata, deploy.train[0].ip, args.gateway_port)
    run(deploy, recipe, args.dataset, args.ray_address)


if __name__ == "__main__":
    main()
