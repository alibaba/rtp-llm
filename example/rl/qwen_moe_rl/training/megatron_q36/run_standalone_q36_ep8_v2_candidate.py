"""Opt-in 8+4 TP4/EP8 to TP4/EP4 Q36 on-policy launch candidate; requires qualification.

The Ray cluster must already be started and owned by this run. No shell or
remote process management is performed here. The full-HF relay does not validate sharded expert redistribution.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import hydra
from omegaconf import DictConfig, OmegaConf


def validate(config):
    from q36_ep8_launch_contract_v2_candidate import validate as validate_ep8
    from q36_nonshared_launch_contract import require_real_integrations
    validate_ep8(config)
    if config.rllm.get('async_training', {}).get('enable', False):
        raise ValueError('on-policy candidate requires async_training.enable=false')
    checkpoint = config.actor_rollout_ref.rollout.checkpoint_engine
    candidates = {'rtp_full_hf_nccl': 'verl_full_hf_nccl', 'rtp_nixl_guarded': 'verl_nixl_guarded'}
    if candidates.get(checkpoint.backend) != checkpoint.custom_backend_module:
        raise ValueError('qualified standalone checkpoint engine must be selected')
    if os.environ.get('Q36_DISAGG_TRAIN_GPUS') != '8' or os.environ.get('Q36_DISAGG_ROLLOUT_GPUS') != '4':
        raise ValueError('candidate is limited to eight training and four rollout GPUs')
    if os.environ.get('RTP_EP_SIZE') != '4' or os.environ.get('RTP_RL_SHARED_BACKING', '0') != '0':
        raise ValueError('candidate requires an opt-in EP4 copying profile')
    model = Path(os.environ['Q36_MODEL_PATH'])
    model_config = json.loads((model / 'config.json').read_text())
    text_config = model_config.get('text_config', model_config)
    if (text_config.get('num_experts'), text_config.get('num_hidden_layers'),
            text_config.get('num_attention_heads'), text_config.get('num_key_value_heads'),
            text_config.get('hidden_size'), text_config.get('num_experts_per_tok')) != (256, 40, 16, 2, 2048, 8):
        raise ValueError('mounted model config is not the qualified Qwen3.6 35B text policy')
    data = Path(os.environ['RLLM_PYTHON_TOOL_TRAIN_JSONL'])
    if hashlib.sha256(data.read_bytes()).hexdigest() != os.environ['Q36_DATASET_SHA256']:
        raise ValueError('training dataset digest mismatch')
    run_name = Path(os.environ['Q36_RUN_ROOT']).name
    if not run_name.startswith('q36-disagg-') or len(run_name.rsplit('-', 1)[-1]) != 10:
        raise ValueError('fresh run identity must end in a ten-character identifier')
    if not os.environ.get('Q36_RAY_ADDRESS') or not os.environ.get('Q36_RAY_SESSION_NAME'):
        raise ValueError('explicit task-owned Ray address and session name are required')
    allowed = {x.strip() for x in os.environ.get('Q36_DISAGG_ALLOWED_NODE_IPS', '').split(',') if x.strip()}
    if not allowed or len(allowed) > 16:
        raise ValueError('explicit authorized node IPs are required')
    require_real_integrations(os.environ)
    return allowed


def validate_cluster(ray, allowed, expected_session, actual_session):
    if actual_session != expected_session:
        raise RuntimeError('Ray session is not the task-owned cluster')
    nodes = [node for node in ray.nodes() if node.get('Alive')]
    if not 1 <= len(nodes) <= 8:
        raise RuntimeError('Ray node count is outside the authorized allocation')
    if any(node.get('NodeManagerAddress') not in allowed for node in nodes):
        raise RuntimeError('Ray includes an unauthorized node')
    if any(node.get('Resources', {}).get('GPU', 0) > 4 for node in nodes):
        raise RuntimeError('Ray node advertises more than four GPUs')
    total_gpus = sum(node.get('Resources', {}).get('GPU', 0) for node in nodes)
    if not 12 <= total_gpus <= 32:
        raise RuntimeError('Ray GPU count is outside the task allocation')


@hydra.main(config_path='pkg://rllm.trainer.config', config_name='unified', version_base=None)
def main(config: DictConfig):
    allowed = validate(config)
    import q36_ep8_worker_bootstrap_candidate
    q36_ep8_worker_bootstrap_candidate.install_driver()
    import ray
    from rllm.data import Dataset
    from rllm.trainer import AgentTrainer
    from rllm.trainer.verl import ray_runtime_env
    from python_math_agent import python_math_agent, python_math_evaluator

    dataset = Dataset.load_data(os.environ['RLLM_PYTHON_TOOL_TRAIN_JSONL'])
    if len(dataset) < config.rllm.data.train_batch_size:
        raise ValueError('dataset is smaller than the requested task batch')
    runtime = ray_runtime_env.get_ppo_ray_runtime_env()
    env = runtime.setdefault('env_vars', {})
    selected = {'PYTHONPATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'CUDNN_HOME', 'NVTE_FLASH_ATTN',
                'NVTE_FUSED_ATTN', 'NVTE_UNFUSED_ATTN', 'WARM_UP', 'MODEL_WARM_UP',
                'LOAD_PYTHON_MODEL', 'ENABLE_CUDA_GRAPH', 'ENABLE_NATIVE_CUDA_GRAPH',
                'ENABLE_LAYER_MICRO_BATCH', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'PYTHONDONTWRITEBYTECODE',
                'UCX_MODULES', 'UCX_TLS', 'UCX_NET_DEVICES', 'UCX_LOG_LEVEL',
                'UCX_IB_ROCE_REACHABILITY_MODE', 'PYTORCH_CUDA_ALLOC_CONF'}
    for name, value in os.environ.items():
        if name in selected or name.startswith(('RTP_', 'RLLM_', 'Q36_', 'QWEN36_')):
            env[name] = value
    env.pop('CUDA_VISIBLE_DEVICES', None)
    run_name = Path(os.environ['Q36_RUN_ROOT']).name
    print(OmegaConf.to_yaml(config), flush=True)
    try:
        address_info = ray.init(address=os.environ['Q36_RAY_ADDRESS'], runtime_env=runtime,
                                namespace='q36-' + run_name)
        validate_cluster(ray, allowed, os.environ['Q36_RAY_SESSION_NAME'], address_info.get('session_name'))
        trainer = AgentTrainer(config=config, backend='verl', agent_flow=python_math_agent,
                               evaluator=python_math_evaluator, train_dataset=dataset, val_dataset=None)
        trainer.train()
    finally:
        primary_error = sys.exc_info()[1]
        cleanup_error = None
        try:
            from rtp_cleanup import cleanup_owned_rtp_processes
            cleanup_owned_rtp_processes(os.environ['RLLM_RTP_ACCEPTANCE_DIR'])
        except Exception as exc:
            traceback.print_exc()
            cleanup_error = exc
        finally:
            ray.shutdown()  # disconnect this driver; never stop the shared Ray cluster
        if primary_error is None and cleanup_error is not None:
            raise cleanup_error


if __name__ == '__main__': main()
