"""Explicit RTP registration; worker hook performs no CUDA initialization."""

import importlib
import importlib.abc
import importlib.machinery
import os
import sys

_DRIVER_INSTALLED = False
_WORKER_INSTALLED = False


def worker_setup():
    """Install a passive standard-library hook; never import Torch or CUDA.

    Ray calls this before assigning an actor's final device mask. Framework
    modules must import naturally, and registration runs only after their
    original loader has completed that natural import.
    """
    from q397_training_deps import ensure_training_dependencies
    ensure_training_dependencies()
    global _WORKER_INSTALLED
    if _WORKER_INSTALLED:
        return
    _WORKER_INSTALLED = True
    targets = {
        "verl.workers.rollout.base", "verl.workers.rollout.replica",
        "rllm.trainer.verl.verl_backend",
        "rllm.engine.agentflow_engine",
        "verl.utils.attention_utils",
    }

    class Loader(importlib.abc.Loader):
        def __init__(self, wrapped, fullname):
            self.wrapped = wrapped
            self.fullname = fullname

        def create_module(self, spec):
            creator = getattr(self.wrapped, "create_module", None)
            return creator(spec) if creator is not None else None

        def exec_module(self, module):
            self.wrapped.exec_module(module)
            _after_import(self.fullname, module)

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in targets:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = Loader(spec.loader, fullname)
            return spec

    sys.meta_path.insert(0, Finder())
    for name in targets:
        module = sys.modules.get(name)
        if module is not None:
            _after_import(name, module)


def _after_import(name, module):
    if name == "verl.utils.attention_utils":
        from q397_torch_padding import install as install_generic_padding
        install_generic_padding(module)
    elif name == "rllm.engine.agentflow_engine":
        from qwen36_trace_enrichment import install as install_trace_linkage
        install_trace_linkage(module)
    elif name == "verl.workers.rollout.base":
        module._ROLLOUT_REGISTRY[("rtp", "async")] = (
            "qwen38_verl_backend.World72ServerAdapter"
            if os.environ.get("RTP_LLM_RL_WORLD72") == "1" else "rtp_verl_backend.RTPServerAdapter"
        )
    elif name == "verl.workers.rollout.replica":
        module.RolloutReplicaRegistry.register(
            "rtp", lambda: (
                importlib.import_module("qwen38_verl_backend").World72Replica
                if os.environ.get("RTP_LLM_RL_WORLD72") == "1"
                else importlib.import_module("rtp_verl_backend").RTPReplica
            ),
        )
    elif name == "rllm.trainer.verl.verl_backend":
        original_init = module.VerlBackend.__init__
        if getattr(original_init, "_rtp_deferred_install", False):
            return

        def initialize(self, *args, **kwargs):
            # VerlBackend is created in the CPU trainer TaskRunner, never by
            # the early GPU worker hook. Heavy registration is safe here.
            _install_driver()
            original_init(self, *args, **kwargs)

        initialize._rtp_deferred_install = True
        module.VerlBackend.__init__ = initialize


def install():
    worker_setup()
    _install_driver()


def _install_driver():
    global _DRIVER_INSTALLED
    if _DRIVER_INSTALLED:
        return
    _DRIVER_INSTALLED = True
    from rllm.engine.rollout.rollout_engine import RolloutEngine
    from rllm.engine.rollout.verl_engine import VerlEngine
    from rllm.parser import ChatTemplateParser
    from rllm.trainer.verl import ray_runtime_env, verl_backend

    class RTPVerlEngine(VerlEngine):
        def __init__(self, config, server_manager, tokenizer, processor=None, **kwargs):
            if config.actor_rollout_ref.rollout.name != "rtp":
                raise ValueError("RTPVerlEngine requires rollout.name=rtp")
            RolloutEngine.__init__(self)
            self.config = config
            self.server_manager = server_manager
            self.tokenizer = tokenizer
            self.processor = processor
            from rtp_gateway_adapter import create_official_chat_parser
            self.chat_parser = create_official_chat_parser(
                tokenizer, processor=processor,
                disable_thinking=config.get("rllm", {}).get("disable_thinking", False),
            )
            # rLLM's generic Qwen parser pins the Qwen3 EOS ID 151645. Qwen3.6
            # reuses the text marker but assigns a different tokenizer ID.
            eos_ids = set()
            for value in (tokenizer.eos_token_id,):
                eos_ids.update(value if isinstance(value, list) else ([] if value is None else [value]))
            from pathlib import Path
            import json
            generation_path = Path(config.actor_rollout_ref.model.path) / "generation_config.json"
            if generation_path.exists():
                value = json.loads(generation_path.read_text()).get("eos_token_id")
                eos_ids.update(value if isinstance(value, list) else ([] if value is None else [value]))
            if not eos_ids:
                raise RuntimeError("RTP tokenizer did not define EOS token IDs")
            self.chat_parser.stop_sequences = sorted(eos_ids)
            self.max_prompt_length = config.data.max_prompt_length
            self.max_response_length = config.data.max_response_length
            self.accumulate_reasoning = config.get("rllm", {}).get("accumulate_reasoning", False)
            self.router_replay_mode = config.get("rllm", {}).get("algorithm", {}).get("router_replay", "disabled")
            rollout = config.actor_rollout_ref.rollout
            self.train_sampling_params = dict(
                temperature=0.0 if rollout.do_sample is False else rollout.temperature,
                top_k=rollout.top_k, top_p=rollout.top_p, logprobs=1,
            )
            self.val_sampling_params = dict(
                temperature=0.0 if rollout.val_kwargs.do_sample is False else rollout.val_kwargs.temperature,
                top_k=rollout.val_kwargs.top_k, top_p=rollout.val_kwargs.top_p, logprobs=1,
            )

    verl_backend.VerlEngine = RTPVerlEngine
    from rllm.gateway.manager import GatewayManager
    original_gateway_start = GatewayManager.start

    def gateway_start(self, rollout_engine):
        if not isinstance(rollout_engine, RTPVerlEngine):
            return original_gateway_start(self, rollout_engine)
        from rtp_gateway_adapter import create_rtp_handler

        if self.cumulative_token_mode:
            raise NotImplementedError("initial native RTP gateway requires cumulative_token_mode=false")
        self._local_handler = create_rtp_handler(rollout_engine)
        self._start_thread(local_handler=self._local_handler)
        self._train_sampling_params = rollout_engine.train_sampling_params
        self._val_sampling_params = rollout_engine.val_sampling_params
        if self.tunnel_backend and not self.public_url:
            self._start_tunnel()

    GatewayManager.start = gateway_start
    for prefix in ("RTP_", "RLLM_RTP_", "RLLM_QWEN36_", "QWEN36_", "Q38_"):
        if prefix not in ray_runtime_env.FORWARD_PREFIXES:
            ray_runtime_env.FORWARD_PREFIXES.append(prefix)
    original_env = ray_runtime_env.get_ppo_ray_runtime_env

    def runtime_env():
        result = original_env()
        previous = result.get("worker_process_setup_hook")
        if previous not in (None, "install_rtp_bridge.worker_setup"):
            raise RuntimeError("a different Ray worker setup hook is already configured")
        result["worker_process_setup_hook"] = "install_rtp_bridge.worker_setup"
        result.setdefault("env_vars", {})["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
        return result

    ray_runtime_env.get_ppo_ray_runtime_env = runtime_env
    original_shutdown = verl_backend.VerlBackend.shutdown

    def shutdown(self):
        if self.config.actor_rollout_ref.rollout.name != "rtp":
            return original_shutdown(self)
        import ray

        manager = getattr(self, "llm_server_manager", None)
        if manager is not None:
            futures = [
                server.shutdown.remote()
                for replica in manager.get_replicas()
                for server in replica.servers
            ]
            if futures:
                ray.get(futures)

    verl_backend.VerlBackend.shutdown = shutdown
