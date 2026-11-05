"""Original rLLM AgentTrainer + verl path with explicit Qwen3.6/RTP hooks."""
from __future__ import annotations
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'overrides'))
import hydra
from omegaconf import DictConfig,OmegaConf


@hydra.main(config_path='pkg://rllm.trainer.config',config_name='unified',version_base=None)
def main(config:DictConfig):
    from install_rtp_bridge import install
    install()
    import qwen36_training_overrides
    import ray
    from rllm.data import Dataset
    from rllm.trainer import AgentTrainer
    from rllm.trainer.ray_init_utils import get_ray_init_settings
    from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env
    from python_math_agent import python_math_agent,python_math_evaluator

    if config.trainer.n_gpus_per_node!=8 or config.trainer.nnodes!=1:
        raise ValueError('This acceptance deployment requires the allocated eight GPUs on one node')
    rollout=config.actor_rollout_ref.rollout
    replica_world=rollout.tensor_model_parallel_size*rollout.pipeline_model_parallel_size*rollout.data_parallel_size
    if rollout.name!='rtp' or replica_world<2 or 8%replica_world:
        raise ValueError('RTP replica parallelism must divide all eight GPUs and use multiple ranks per replica')
    dataset=Dataset.load_data(os.environ['RLLM_PYTHON_TOOL_TRAIN_JSONL'])
    if len(dataset)<config.rllm.data.train_batch_size:
        raise ValueError('Dataset is smaller than one complete generation batch')
    runtime=get_ppo_ray_runtime_env()
    env=runtime.setdefault('env_vars',{})
    # rLLM only forwards selected prefixes; propagate the opt-in source overlay
    # and adapter paths explicitly to every Ray training/rollout worker.
    for name in ['PYTHONPATH','RTP_SERVER_LAUNCHER','RTP_RUN_ROOT','RTP_PROJECT_ROOT',
                 'RLLM_QWEN36_MODEL_AUDIT_DIR','RLLM_QWEN36_TRACE_AUDIT_DIR','RLLM_OPTIMIZER_AUDIT_DIR',
                 'RLLM_PYTHON_TOOL_TRACE_DIR','RLLM_EPISODE_LOG_DIR']:
        if os.environ.get(name):env[name]=os.environ[name]
    for name,value in os.environ.items():
        if name.startswith(('RTP_','QWEN36_','RLLM_RTP_')):
            env[name]=value
    settings=get_ray_init_settings(config)
    if settings.get('address')=='auto':
        raise RuntimeError('Refusing an unrelated Ray cluster; use an isolated local runtime')
    print(OmegaConf.to_yaml(config),flush=True)
    ray.init(runtime_env=runtime,**settings)
    try:
        trainer=AgentTrainer(config=config,backend='verl',agent_flow=python_math_agent,
                             evaluator=python_math_evaluator,train_dataset=dataset,val_dataset=None)
        trainer.train()
    finally:
        try:
            from rtp_cleanup import cleanup_owned_rtp_processes
            cleanup_owned_rtp_processes(os.environ['RLLM_RTP_ACCEPTANCE_DIR'])
        finally:
            ray.shutdown()


if __name__=='__main__':main()
