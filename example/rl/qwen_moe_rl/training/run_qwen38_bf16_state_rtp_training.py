"""Dedicated 18-node / 72-GPU entry to the original rLLM AgentTrainer.

Adapted from pushed qwen_moe_rl/training/run_qwen397_rtp_training.py, with
separate Qwen3.8 meta/BF16-state runtime. This entrypoint is not an acceptance flag.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import socket
import time


def allocated_ips(path):
    value=json.loads(Path(path).read_text());nodes=value.get('nodes') if isinstance(value,dict) else value
    if not isinstance(nodes,list) or len(nodes)!=18:raise ValueError('Exactly eighteen allocated nodes are required')
    ips=[node.get('container_ip',node.get('ip')) for node in nodes]
    if any(not isinstance(ip,str) or not ip for ip in ips) or len(set(ips))!=18:
        raise ValueError('The eighteen current development-container IPs must be unique')
    if any(node.get('gpu_count',4)!=4 for node in nodes):raise ValueError('Each allocated node must have four GPUs')
    return set(ips)


def validate_live_cluster(ray,expected_ips):
    nodes=[node for node in ray.nodes() if node['Alive']]
    if len(nodes)!=18 or {node['NodeManagerAddress'] for node in nodes}!=expected_ips:
        raise RuntimeError('Ray connected to a different allocation')
    if any(node['Resources'].get('GPU',0)!=4 for node in nodes):
        raise RuntimeError('Every current Ray node must advertise exactly its four allocated GPUs')
    return [dict(node_id=node['NodeID'],container_ip=node['NodeManagerAddress'],
                 gpu_count=node['Resources']['GPU'],cpu_count=node['Resources'].get('CPU')) for node in nodes]


def forward_worker_environment(runtime,environ):
    env=runtime.setdefault('env_vars',{})
    names={'PYTHONPATH','LD_PRELOAD','LD_LIBRARY_PATH','OMP_NUM_THREADS','TOKENIZERS_PARALLELISM',
           'CUDA_DEVICE_MAX_CONNECTIONS','RAY_DEDUP_LOGS','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE',
           'PYTHONDONTWRITEBYTECODE','PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF'}
    for name,value in environ.items():
        if name in names or name in ('EP_SIZE','FFN_SP_SIZE') or name.startswith(('Q38_','RTP_','RLLM_','NCCL_')):
            env[name]=value
    if 'PYTORCH_ALLOC_CONF' in environ:
        env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    return runtime


def validate_config(config):
    if (config.trainer.n_gpus_per_node,config.trainer.nnodes)!=(4,18):
        raise ValueError('This run must use the entire dedicated 18×4 allocation')
    actor=config.actor_rollout_ref.actor;model=config.actor_rollout_ref.model
    if (actor.strategy,actor.fsdp_config.fsdp_size,actor.fsdp_config.model_dtype)!=('fsdp2',72,'bf16'):
        raise ValueError('The BF16 training masters require actual FSDP2 world72')
    if actor.fsdp_config.param_offload or actor.fsdp_config.optimizer_offload or actor.fsdp_config.offload_policy:
        raise ValueError('Generic all-parameter/off-the-shelf optimizer offload cannot hold this policy')
    if not actor.fsdp_config.reshard_after_forward:raise ValueError('Each bounded FSDP unit must reshard after forward')
    if (actor.optim.optimizer_impl,actor.optim.optimizer)!=('qwen38_bf16_training_runtime','Bf16CpuAdamW'):
        raise ValueError('The explicit BF16 CPU-state / FP32-gradient optimizer is required')
    if (config.rllm.data.train_batch_size,config.rllm.rollout.n,actor.ppo_mini_batch_size,
            actor.ppo_epochs,actor.ppo_micro_batch_size_per_gpu)!=(18,4,18,1,1):
        raise ValueError('Use 18 source prompts ×4 trajectories =72 real training sequences, prompt mini18, micro1')
    if not config.trainer.balance_batch:raise ValueError('Use the original balanced/padded rLLM episode path')
    if config.rllm.disable_thinking or os.environ.get('RLLM_PYTHON_TOOL_ENABLE_THINKING')!='1':
        raise ValueError('The official Qwen3.8 chat template requires thinking enabled')
    if (model.external_lib!='qwen38_bf16_training_runtime' or model.use_remove_padding
            or model.use_fused_kernels or model.lora_rank or actor.fsdp_config.ulysses_sequence_parallel_size!=1):
        raise ValueError('Initial validation fixes full-policy, padded, SP1, unfused HF forward')
    rollout=config.actor_rollout_ref.rollout
    if (rollout.name,rollout.tensor_model_parallel_size,rollout.data_parallel_size,
            rollout.pipeline_model_parallel_size)!=('rtp',8,9,1):
        raise ValueError('RTP TP8 ×DP9 with one global EP72 expert pool is required')
    if os.environ.get('EP_SIZE')!='72' or os.environ.get('FFN_SP_SIZE')!='8':
        raise ValueError('Native EP_SIZE=72 and FFN_SP_SIZE=8 must be explicitly enabled')
    if config.trainer.save_freq!=-1:raise ValueError('A bounded optimizer checkpoint writer is not implemented yet')


def run(config):
    from qwen38_final_policy import install
    install()
    import ray
    from rllm.data import Dataset
    from rllm.trainer import AgentTrainer
    from rllm.trainer.ray_init_utils import get_ray_init_settings
    from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env
    from python_math_agent import python_math_agent,python_math_evaluator
    validate_config(config)
    dataset=Dataset.load_data(os.environ['RLLM_PYTHON_TOOL_TRAIN_JSONL'])
    if len(dataset)<18:raise ValueError('At least eighteen original DeepScaleR source prompts are required')
    ips=allocated_ips(os.environ['Q38_NODES_FILE'])
    address=os.environ['Q38_RAY_ADDRESS'];settings=get_ray_init_settings(config)
    if not address or address in ('auto','local') or settings.get('address')!=address:
        raise ValueError('Use the exact current allocation Ray head; no local/auto discovery')
    ray.init(runtime_env=forward_worker_environment(get_ppo_ray_runtime_env(),os.environ),**settings)
    try:
        nodes=validate_live_cluster(ray,ips)
        audit={'schema':'qwen38-72gpu-ray-allocation-v1','nodes':nodes,'address':address,
               'driver_hostname':socket.gethostname(),'driver_pid':os.getpid(),'time':time.time(),
               'fsdp_world':72,'native_tp':8,'native_dp':9,'native_ep':72,'native_pp':1,
               'distributed_expert_pool_instances':1,'full_model_training_accepted':False}
        with (Path(os.environ['Q38_RUN_ROOT'])/'ray-allocation.json').open('x') as stream:
            json.dump(audit,stream,indent=2)
        trainer=AgentTrainer(config=config,backend='verl',agent_flow=python_math_agent,
            evaluator=python_math_evaluator,train_dataset=dataset,val_dataset=None)
        trainer.train()
    finally:
        try:
            from rtp_cleanup import cleanup_owned_rtp_processes
            cleanup_owned_rtp_processes(os.environ['RLLM_RTP_ACCEPTANCE_DIR'])
        finally:ray.shutdown()


def main():
    import hydra
    @hydra.main(config_path='pkg://rllm.trainer.config',config_name='unified',version_base=None)
    def entry(config):return run(config)
    entry()


if __name__=='__main__':main()
