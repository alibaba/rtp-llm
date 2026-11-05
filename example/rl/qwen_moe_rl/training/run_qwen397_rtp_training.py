"""Original rLLM AgentTrainer, with explicit eight-node/32-GPU admission."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'overrides'))
import hydra
from omegaconf import DictConfig, OmegaConf


def allocated_ips(path):
    value = json.loads(Path(path).read_text())
    nodes = value.get('nodes') if isinstance(value, dict) else value
    if not isinstance(nodes, list) or len(nodes) != 8:
        raise ValueError('The dedicated cluster manifest must contain exactly eight nodes')
    ips = [node.get('container_ip', node.get('ip')) for node in nodes]
    if any(not isinstance(ip, str) or not ip for ip in ips) or len(set(ips)) != 8:
        raise ValueError('The dedicated cluster manifest must contain eight unique container IPs')
    if any(node.get('gpu_count', 4) != 4 for node in nodes):
        raise ValueError('This acceptance uses four allocated GPUs on every node')
    return set(ips)


def validate_live_cluster(ray, expected_ips):
    nodes = [node for node in ray.nodes() if node['Alive']]
    actual_ips = {node['NodeManagerAddress'] for node in nodes}
    if len(nodes) != 8 or actual_ips != expected_ips:
        raise RuntimeError(f'Connected Ray cluster is not the allocated eight nodes: {sorted(actual_ips)}')
    if any(node['Resources'].get('GPU', 0) != 4 for node in nodes):
        raise RuntimeError('All eight Ray nodes must advertise exactly four allocated GPUs')
    return [dict(node_id=node['NodeID'], container_ip=node['NodeManagerAddress'],
                 gpu_count=node['Resources']['GPU'], cpu_count=node['Resources'].get('CPU'))
            for node in nodes]


def forward_worker_environment(runtime, environ):
    """Forward project settings to the independently started eight-node Ray cluster."""
    env = runtime.setdefault('env_vars', {})
    for name in ['PYTHONPATH', 'LD_PRELOAD', 'LD_LIBRARY_PATH', 'RTP_SERVER_LAUNCHER', 'RTP_RUN_ROOT', 'RTP_PROJECT_ROOT',
                 'RLLM_QWEN36_MODEL_AUDIT_DIR', 'RLLM_QWEN36_TRACE_AUDIT_DIR', 'RLLM_OPTIMIZER_AUDIT_DIR',
                 'RLLM_PYTHON_TOOL_TRACE_DIR', 'RLLM_EPISODE_LOG_DIR', 'RLLM_REQUIRE_REAL_UPDATE',
                 'OMP_NUM_THREADS', 'TOKENIZERS_PARALLELISM', 'CUDA_DEVICE_MAX_CONNECTIONS',
                 'NCCL_CUMEM_ENABLE', 'NCCL_IB_GID_INDEX', 'RAY_DEDUP_LOGS', 'HF_HUB_OFFLINE',
                 'TRANSFORMERS_OFFLINE', 'PYTHONDONTWRITEBYTECODE']:
        if environ.get(name):
            env[name] = environ[name]
    for name, value in environ.items():
        if name.startswith(('RTP_', 'QWEN36_', 'Q397_', 'RLLM_RTP_', 'RLLM_PYTHON_TOOL_')):
            env[name] = value
    return runtime


@hydra.main(config_path='pkg://rllm.trainer.config', config_name='unified', version_base=None)
def main(config: DictConfig):
    from install_rtp_bridge import install
    install()
    import qwen36_training_overrides
    import ray
    from rllm.data import Dataset
    from rllm.trainer import AgentTrainer
    from rllm.trainer.ray_init_utils import get_ray_init_settings
    from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env
    from python_math_agent import python_math_agent, python_math_evaluator

    if config.trainer.n_gpus_per_node != 4 or config.trainer.nnodes != 8:
        raise ValueError('397B acceptance requires eight nodes with four allocated GPUs each')
    actor = config.actor_rollout_ref.actor
    if actor.strategy != 'fsdp2' or actor.fsdp_config.fsdp_size != 32:
        raise ValueError('397B acceptance requires FSDP2 sharding across all 32 GPUs')
    if actor.fsdp_config.model_dtype != 'fp32' or actor.fsdp_config.param_offload:
        raise ValueError('The accepted path keeps real FP32 master shards on the GPUs')
    if actor.optim.optimizer_impl != 'qwen36_optimizer_audit' or actor.optim.optimizer != 'CpuMomentAuditedAdamW':
        raise ValueError('The accepted CPU-moment/32 MiB AdamW implementation is required')
    if not config.trainer.balance_batch:
        raise ValueError('FSDP32 requires the original balanced/padded rLLM path for deterministic one-update windows')
    if (config.rllm.data.train_batch_size, config.rllm.rollout.n, actor.ppo_mini_batch_size,
            actor.ppo_epochs, actor.ppo_micro_batch_size_per_gpu) != (8, 4, 8, 1, 1):
        raise ValueError('Acceptance requires 8 prompts × 4 trajectories, prompt mini8, one PPO epoch and micro1')
    rollout = config.actor_rollout_ref.rollout
    if (rollout.name, rollout.tensor_model_parallel_size, rollout.data_parallel_size,
            rollout.pipeline_model_parallel_size) != ('rtp', 16, 1, 1):
        raise ValueError('397B native RTP requires TP16 × two replicas and PP1 on all 32 GPUs')
    dataset = Dataset.load_data(os.environ['RLLM_PYTHON_TOOL_TRAIN_JSONL'])
    if len(dataset) < config.rllm.data.train_batch_size:
        raise ValueError('Dataset is smaller than one complete original generation batch')
    expected_ips = allocated_ips(os.environ['Q397_NODES_FILE'])
    runtime = forward_worker_environment(get_ppo_ray_runtime_env(), os.environ)
    settings = get_ray_init_settings(config)
    address = os.environ['Q397_RAY_ADDRESS']
    if not address or address in ('auto', 'local') or settings.get('address') != address:
        raise RuntimeError('An explicit dedicated Ray head address is required; auto/local are prohibited')
    print(OmegaConf.to_yaml(config), flush=True)
    ray.init(runtime_env=runtime, **settings)
    try:
        nodes = validate_live_cluster(ray, expected_ips)
        audit = dict(schema='qwen397-32gpu-ray-allocation-v1', timestamp=time.time(),
                     driver_hostname=socket.gethostname(), driver_pid=os.getpid(), address=address,
                     fsdp_world=32, native_tp=16, native_replicas=2, native_pp=1, nodes=nodes)
        Path(os.environ['Q397_RUN_ROOT'], 'ray-allocation.json').write_text(json.dumps(audit, indent=2)+'\n')
        trainer = AgentTrainer(config=config, backend='verl', agent_flow=python_math_agent,
                               evaluator=python_math_evaluator, train_dataset=dataset, val_dataset=None)
        trainer.train()
    finally:
        try:
            from rtp_cleanup import cleanup_owned_rtp_processes
            cleanup_owned_rtp_processes(os.environ['RLLM_RTP_ACCEPTANCE_DIR'])
        finally:
            ray.shutdown()


if __name__ == '__main__':
    main()
