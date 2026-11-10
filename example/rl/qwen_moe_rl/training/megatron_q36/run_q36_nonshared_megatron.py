"""Real rLLM/verl copy-refit launcher candidate; training remains gated."""
import os
import sys
import traceback
import hydra
from omegaconf import DictConfig, OmegaConf
from q36_nonshared_launch_contract import validate, require_real_integrations


@hydra.main(config_path='pkg://rllm.trainer.config', config_name='unified', version_base=None)
def main(config: DictConfig):
    validate(config, os.environ)
    require_real_integrations(os.environ)
    from install_rtp_bridge import install
    install()
    import q36_nonshared_worker_bootstrap
    from rllm.trainer.verl import ray_runtime_env
    q36_nonshared_worker_bootstrap.install_driver_hook(ray_runtime_env)
    import ray
    from rllm.data import Dataset
    from rllm.trainer import AgentTrainer
    from rllm.trainer.ray_init_utils import get_ray_init_settings
    from python_math_agent import python_math_agent, python_math_evaluator

    dataset = Dataset.load_data(os.environ['RLLM_PYTHON_TOOL_TRAIN_JSONL'])
    if len(dataset) < config.rllm.data.train_batch_size:
        raise ValueError('dataset is smaller than a rollout batch')
    runtime = ray_runtime_env.get_ppo_ray_runtime_env()
    env = runtime.setdefault('env_vars', {})
    names = {'PYTHONPATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'CUDNN_HOME', 'NVTE_FLASH_ATTN',
             'NVTE_FUSED_ATTN', 'NVTE_UNFUSED_ATTN', 'WARM_UP', 'MODEL_WARM_UP', 'LOAD_PYTHON_MODEL',
             'ENABLE_CUDA_GRAPH', 'ENABLE_NATIVE_CUDA_GRAPH', 'ENABLE_LAYER_MICRO_BATCH',
             'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'PYTHONDONTWRITEBYTECODE'}
    for name, value in os.environ.items():
        if name in names or name.startswith(('RTP_', 'RLLM_', 'Q36_', 'QWEN36_')):
            env[name] = value
    # rLLM forwards CUDA_* by default, including the driver's four-device
    # mask. Ray must install its own one-device mask for each GPU actor.
    env.pop('CUDA_VISIBLE_DEVICES', None)
    settings = get_ray_init_settings(config)
    if settings.get('address') not in (None, 'local', 'auto'):
        raise RuntimeError('refusing to attach to an unrelated Ray cluster')
    # rLLM can infer "auto" from a stale or unrelated /tmp/ray marker. Start
    # this single-node task's own cluster instead of attaching to that marker.
    settings['address'] = 'local'
    run_name = os.path.basename(os.path.abspath(os.environ['Q36_RUN_ROOT']))
    if not run_name.startswith('q36-rl35-megatron-') or len(run_name.rsplit('-', 1)[-1]) != 10:
        raise RuntimeError('expected a unique 35B run identity')
    # Ray appends a session and socket name, so the full run path exceeds the
    # Unix-domain socket limit; this short path remains unique to the run.
    settings['_temp_dir'] = '/tmp/q36ray-' + run_name.rsplit('-', 1)[-1]
    if os.path.exists(settings['_temp_dir']):
        raise RuntimeError('Ray temp directory already exists for this run')
    settings['namespace'] = 'q36-' + run_name
    settings['include_dashboard'] = False
    print(OmegaConf.to_yaml(config), flush=True)
    try:
        ray.init(runtime_env=runtime, **settings)
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
            try:
                ray.shutdown()
            except Exception as exc:
                traceback.print_exc()
                cleanup_error = cleanup_error or exc
        if primary_error is None and cleanup_error is not None:
            raise cleanup_error


if __name__ == '__main__': main()
