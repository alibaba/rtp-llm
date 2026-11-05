#!/usr/bin/env python3
"""Import the real training/worker path on CPU; never start Ray or a model.

Use the training virtualenv, not the RTP Python. Every stage gets a fresh child
process so a failed registration cannot leave a half-patched interpreter.
Passing this test proves CPU import availability only, not RL correctness.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback


STAGES = (
    'bridge_install', 'training_overrides', 'unified_trainer',
    'python_math_agent', 'fsdp_workers', 'rtp_rollout_refit',
    'checkpoint_optimizer', 'original_driver_chain',
)
PACKAGES = ('torch', 'triton', 'transformers', 'ray', 'verl', 'rllm',
            'cachetools', 'codetiming', 'peft', 'tensordict', 'safetensors')


def cpu_environment(root: Path, base: Path, environment=None):
    env = dict(os.environ if environment is None else environment)
    env.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='void',
               PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    # Pod-level WORLD_RANK=0 is not a training rank. CPU imports must not use it.
    for key in ('RANK', 'WORLD_RANK', 'LOCAL_RANK', 'WORLD_SIZE',
                'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
                'RAY_ADDRESS', 'RAY_NAMESPACE'):
        env.pop(key, None)
    paths = [root/'adapter', root/'training', root/'training/overrides', root/'references/agent', base/'agent']
    previous = [part for part in env.get('PYTHONPATH', '').split(os.pathsep) if part]
    env['PYTHONPATH'] = os.pathsep.join([*(str(p) for p in paths), *previous])
    return env


def install_guards(torch, ray):
    """Keep ordinary ray.remote decorators usable; forbid all task submission."""
    attempted = []

    def blocked(name):
        def fail(*args, **kwargs):
            attempted.append(name)
            raise RuntimeError(f'CPU import smoke forbids execution: {name}')
        return fail

    if torch.cuda.is_initialized() or ray.is_initialized() or torch.distributed.is_initialized():
        raise RuntimeError('CPU smoke started with an initialized accelerator/distributed runtime')
    torch.cuda._lazy_init = blocked('torch.cuda._lazy_init')
    torch.cuda.init = blocked('torch.cuda.init')
    torch.distributed.init_process_group = blocked('torch.distributed.init_process_group')
    ray.init = blocked('ray.init')
    for name in ('get', 'put', 'get_actor', 'wait'):
        setattr(ray, name, blocked(f'ray.{name}'))
    # A @ray.remote declaration is harmless. Actor/task .remote() is not.
    from ray.actor import ActorClass
    from ray.remote_function import RemoteFunction
    ActorClass._remote = blocked('ray.actor.ActorClass._remote')
    RemoteFunction._remote = blocked('ray.remote_function.RemoteFunction._remote')
    return attempted


def load_module(name, origins):
    module = importlib.import_module(name)
    origins[name] = str(getattr(module, '__file__', None))
    return module


def bridge(origins):
    module = load_module('install_rtp_bridge', origins)
    module.install()
    assert module._DRIVER_INSTALLED
    return module


def hooks(origins):
    module = load_module('qwen36_training_overrides', origins)
    assert module._INSTALLED
    return module


def trainer(origins):
    module = load_module('rllm.trainer', origins)
    assert module.AgentTrainer.__module__ == 'rllm.trainer.unified_trainer'
    return module.AgentTrainer


def math_agent(origins, interfaces=None):
    module = load_module('python_math_agent', origins)
    wrapper = load_module('rllm.eval.rollout_decorator', origins).AgentFlowFn
    agent, evaluator = module.python_math_agent, module.python_math_evaluator
    if not isinstance(agent, wrapper):
        raise TypeError('Original python_math_agent must be the actual rLLM AgentFlowFn')
    if (not callable(agent.run) or not callable(agent.arun)
            or agent.run.__func__ is not wrapper.run or agent.arun.__func__ is not wrapper.arun):
        raise TypeError('AgentFlowFn run/arun must retain the actual rLLM implementations')
    wrapped = inspect.unwrap(agent)
    if (wrapped is not agent._fn or not inspect.iscoroutinefunction(wrapped)
            or agent._is_async is not True or wrapped.__module__ != module.__name__
            or wrapped.__name__ != 'python_math_agent'):
        raise TypeError('Original agent must wrap its own asynchronous python_math_agent function')
    if type(evaluator) is not module.PythonMathEvaluator:
        raise TypeError('Original evaluator must be the local PythonMathEvaluator instance')
    evaluate = evaluator.evaluate
    if (not callable(evaluate) or evaluate.__self__ is not evaluator
            or evaluate.__func__ is not module.PythonMathEvaluator.evaluate):
        raise TypeError('Evaluator must expose its original bound evaluate(task, episode) method')
    contracts = {'agent.run': (agent.run, ('task', 'config')),
                 'agent.arun': (agent.arun, ('task', 'config')),
                 'agent.wrapped': (wrapped, ('task', 'config')),
                 'evaluator.evaluate': (evaluate, ('task', 'episode'))}
    signatures = {}
    for name, (function, parameters) in contracts.items():
        signature = inspect.signature(function)
        if tuple(signature.parameters) != parameters:
            raise TypeError(f'{name} has an unexpected signature: {signature}')
        signatures[name] = str(signature)
    if interfaces is not None:
        interfaces['original_math_agent'] = dict(
            agent_type=f'{type(agent).__module__}.{type(agent).__qualname__}',
            evaluator_type=f'{type(evaluator).__module__}.{type(evaluator).__qualname__}',
            signatures=signatures, wrapped_coroutine=True,
            wrapped_is_actual_fn=True, wrapped_owner=wrapped.__module__,
            original_run_arun_implementations=True, evaluator_bound_to_original_instance=True,
            evaluator_method_is_original=True,
            validation_boundary='interface introspection only; neither agent nor evaluator executed')
    return module


def perform_stage(stage, origins, interfaces=None):
    if stage == 'fsdp_workers':
        # Real workers receive this passive hook before their ordinary imports.
        load_module('install_rtp_bridge', origins).worker_setup()
        workers = load_module('verl.workers.engine_workers', origins)
        assert workers.TrainingWorker and workers.ActorRolloutRefWorker
        engine = load_module('verl.workers.engine.fsdp.transformer_impl', origins)
        assert engine.FSDPEngine
        load_module('rllm.trainer.verl.patch', origins)
        hooks(origins)
        return
    bridge(origins)
    if stage == 'bridge_install':
        return
    hooks(origins)
    if stage == 'training_overrides':
        return
    if stage == 'unified_trainer':
        trainer(origins)
    elif stage == 'python_math_agent':
        math_agent(origins, interfaces)
    elif stage == 'rtp_rollout_refit':
        for name in ('verl.workers.rollout.llm_server', 'verl.workers.rollout.replica',
                     'rtp_verl_backend', 'rtp_native_server', 'rtp_multinode_server',
                     'rtp_topology', 'rtp_gateway_adapter', 'rtp_cleanup'):
            load_module(name, origins)
    elif stage == 'checkpoint_optimizer':
        module = load_module('verl.checkpoint_engine', origins)
        assert module.CheckpointEngineManager and module.CheckpointEngineRegistry
        load_module('verl.utils.checkpoint.fsdp_checkpoint_manager', origins)
        load_module('qwen397_streamed_checkpoint', origins)
        module = load_module('qwen36_optimizer_audit', origins)
        assert module.CpuMomentAuditedAdamW
        load_module('qwen35_streamed_adamw', origins)
    elif stage == 'original_driver_chain':
        load_module('rllm.data', origins)
        trainer(origins)
        load_module('rllm.trainer.ray_init_utils', origins)
        load_module('rllm.trainer.verl.ray_runtime_env', origins)
        math_agent(origins, interfaces)
        # Import the actual Hydra entrypoint, but do not call main/AgentTrainer.
        driver = load_module('run_qwen397_rtp_training', origins)
        assert callable(driver.main)
        workers = load_module('verl.workers.engine_workers', origins)
        assert workers.ActorRolloutRefWorker
        load_module('rtp_verl_backend', origins)
    else:
        raise ValueError(stage)


def check_origins(origins, root, base):
    for name, filename in origins.items():
        if filename == 'None':
            raise RuntimeError(f'Imported module has no auditable origin: {name}')
        path = Path(filename).resolve()
        if name.startswith(('rllm.', 'verl.')):
            expected = base
        elif name in ('python_math_agent',):
            expected = root/'references/agent'
        else:
            expected = root
        if not path.is_relative_to(expected.resolve()):
            raise RuntimeError(f'{name} resolved outside {expected}: {path}')


def child(args):
    record = dict(schema='qwen397-cpu-import-stage-v1', stage=args.child_stage,
                  started_unix=time.time(), pid=os.getpid(), hostname=platform.node(),
                  executable=sys.executable, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  module_origins={}, interface_checks={}, versions={}, status='failed')
    torch = ray = None
    attempted = []
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise RuntimeError('CUDA must be hidden before torch is imported')
        for package in PACKAGES:
            try:
                record['versions'][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                record['versions'][package] = None
        import torch
        import ray
        attempted = install_guards(torch, ray)
        perform_stage(args.child_stage, record['module_origins'], record['interface_checks'])
        check_origins(record['module_origins'], args.root, args.base)
        if attempted:
            raise RuntimeError(f'An import attempted prohibited runtime execution: {attempted}')
        if torch.cuda.is_initialized() or ray.is_initialized() or torch.distributed.is_initialized():
            raise RuntimeError('Import initialized CUDA, Ray, or a process group')
        record['status'] = 'passed_cpu_imports_only'
    except BaseException as exc:
        record.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        if isinstance(exc, ModuleNotFoundError):
            record['missing_module'] = exc.name
    finally:
        record.update(finished_unix=time.time(), forbidden_runtime_attempts=attempted,
                      cuda_initialized=torch.cuda.is_initialized() if torch is not None else None,
                      ray_initialized=ray.is_initialized() if ray is not None else None,
                      process_group_initialized=torch.distributed.is_initialized() if torch is not None else None)
        args.child_output.write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n')
    return 0 if record['status'] == 'passed_cpu_imports_only' else 1


def run_stage(args, stage, destination):
    output = destination/f'{stage}.json'
    command = [sys.executable, str(Path(__file__).resolve()), '--root', str(args.root),
               '--base', str(args.base), '--child-stage', stage, '--child-output', str(output)]
    started = time.time()
    try:
        result = subprocess.run(command, cwd=args.root, env=cpu_environment(args.root, args.base),
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=args.timeout)
        (destination/f'{stage}.log').write_text(result.stdout)
        record = json.loads(output.read_text()) if output.is_file() else dict(status='no_child_record')
        record.update(returncode=result.returncode, command=command)
    except subprocess.TimeoutExpired as exc:
        raw = exc.stdout or ''
        (destination/f'{stage}.log').write_text(raw.decode(errors='replace') if isinstance(raw, bytes) else raw)
        record = dict(status='timeout', timeout_seconds=args.timeout, command=command)
    except BaseException as exc:
        record = dict(status='launcher_failed', error=repr(exc), traceback=traceback.format_exc(), command=command)
    record.update(stage=stage, parent_started_unix=started, parent_finished_unix=time.time())
    print('Q397_CPU_IMPORT_STAGE', stage, record['status'], record.get('missing_module', ''), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--stage', action='append', choices=STAGES)
    parser.add_argument('--child-stage', choices=STAGES, help=argparse.SUPPRESS)
    parser.add_argument('--child-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.root, args.base = args.root.resolve(), args.base.resolve()
    if args.child_stage:
        if not args.child_output:
            parser.error('--child-output required for a child')
        return child(args)
    if not args.output or args.jobs < 1 or args.jobs > 8 or args.timeout <= 0:
        parser.error('--output required; jobs must be 1..8 and timeout positive')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    destination = args.output.parent/(args.output.stem+'.stages')
    destination.mkdir(exist_ok=False)
    stages = args.stage or STAGES
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        records = list(pool.map(lambda stage: run_stage(args, stage, destination), stages))
    passed = all(r.get('returncode') == 0 and r['status'] == 'passed_cpu_imports_only' for r in records)
    result = dict(schema='qwen397-cpu-training-import-smoke-v1',
                  status='passed_cpu_imports_only' if passed else 'failed',
                  host=platform.node(), executable=sys.executable, root=str(args.root), base=str(args.base),
                  all_stages_requested=not bool(args.stage), stages=records,
                  missing_modules=sorted({r['missing_module'] for r in records if r.get('missing_module')}),
                  scope='CPU imports only; no Ray start, actor construction, CUDA context, model load, or training')
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print('Q397_CPU_IMPORT_SMOKE', result['status'], str(args.output), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
