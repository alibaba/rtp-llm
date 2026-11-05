"""Exercise pinned rLLM padding and verl minibatch bodies on CPU tensors.

The 32 dispatch positions are executed sequentially with a scalar autograd
engine, so this test never constructs a model or initializes CUDA/NCCL/Ray.
It uses real DataProto/TensorDict padding/chunking/iterators, and the unchanged
runtime method bodies extracted without their Ray dispatch decorators.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import nullcontext
import copy
import hashlib
import itertools
import json
import logging
import math
import os
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''


def method(path, name, namespace, class_name=None):
    text = Path(path).read_text()
    tree = ast.parse(text)
    body = tree.body
    if class_name is not None:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name).body
    original = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    function = copy.deepcopy(original)
    function.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import ray.cloudpickle as cloudpickle
    from codetiming import Timer
    from tensordict import NonTensorData
    import rllm
    import verl
    from verl.protocol import DataProto, BatchData
    from verl.utils import tensordict_utils as tu
    import verl.utils.torch_functional as verl_F

    if torch.cuda.is_initialized():
        raise RuntimeError('CPU contract regression must never initialize CUDA')
    rllm_root = Path(rllm.__file__).resolve().parent
    verl_root = Path(verl.__file__).resolve().parent
    files = dict(backend=rllm_root/'trainer/verl/verl_backend.py',
                 worker=verl_root/'workers/engine_workers.py',
                 engine=verl_root/'workers/engine/base.py',
                 micro=verl_root/'workers/engine/utils.py',
                 loss=verl_root/'trainer/ppo/core_algos.py')
    common = dict(math=math, np=np, torch=torch, tu=tu, logger=logging.getLogger('q397-batch-cpu'),
                  Timer=Timer, NonTensorData=NonTensorData, chain=itertools.chain,
                  maybe_fix_3d_position_ids=tu.maybe_fix_3d_position_ids, verl_F=verl_F)
    pad = method(files['backend'], '_pad_dataproto_to_world_size', dict(common), 'VerlBackend')
    remove_padding = method(files['backend'], '_remove_padding', dict(common), 'VerlBackend')
    train_mini = method(files['worker'], 'train_mini_batch', dict(common), 'TrainingWorker')
    train_batch = method(files['worker'], 'train_batch', dict(common), 'TrainingWorker')
    engine_batch = method(files['engine'], 'train_batch', dict(common), 'BaseEngine')
    prepare_micro = method(files['micro'], 'prepare_micro_batches', dict(common))
    aggregate = method(files['loss'], 'agg_loss', dict(common))
    config = SimpleNamespace(data=SimpleNamespace(train_batch_size=8),
                             actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=4),
                                                              actor=SimpleNamespace(ppo_mini_batch_size=8)))
    backend = SimpleNamespace(config=config, _get_aggregate_dp_size=lambda: 32)
    backend._remove_padding = lambda data: remove_padding(backend, data)
    cases = []

    for real_rows in (32, 33, 47, 63, 64, 65, 95, 96):
        mask = torch.arange(4)[None, :] < (torch.arange(real_rows) % 4+1)[:, None]
        coefficients = (torch.arange(real_rows*4, dtype=torch.float32).reshape(real_rows, 4)+1)/100
        batch = DataProto.from_dict(tensors=dict(prompts=torch.ones((real_rows, 2), dtype=torch.int64),
                                                 response_mask=mask, coefficients=coefficients),
                                    non_tensors=dict(is_pad_step=np.zeros(real_rows, dtype=bool),
                                                     is_last_step=np.ones(real_rows, dtype=bool),
                                                     is_valid=np.ones(real_rows, dtype=bool)))
        padded = pad(backend, batch)
        expected_rows = math.ceil(real_rows/32)*32
        metadata = padded.meta_info
        if (len(padded), metadata['rllm_actor_mini_batch_size'], metadata['rllm_actor_global_batch_size'],
                metadata['rllm_actor_num_updates']) != (expected_rows, expected_rows, 32, 1):
            raise AssertionError('Actual rLLM padding metadata differs from the 1-update/32-rollout contract')
        if padded.batch['response_mask'][real_rows:].any().item():
            raise AssertionError('Actual rLLM padding left nonzero response loss masks')
        if padded.non_tensor_batch['is_valid'][real_rows:].any():
            raise AssertionError('Actual rLLM padding marked a repeated pad row valid')
        global_tokens = int(padded.batch['response_mask'].sum())
        mode_results = []
        for mode in ('token-mean', 'seq-mean-token-sum', 'seq-mean-token-mean'):
            td = padded.to_tensordict()
            tu.assign_non_tensor(td, mini_batch_size=metadata['rllm_actor_mini_batch_size'],
                                 global_batch_size=metadata['rllm_actor_global_batch_size'], epochs=1,
                                 seed=17, dataloader_kwargs={'shuffle': True})
            chunks = BatchData(td).chunk(32)
            local_results = []
            for rank, chunk in enumerate(chunks):
                # Real BatchData dispatch consolidates/locks its payload; the
                # worker receives a serialized/materialized value rather than
                # that same in-process sender object. Preserve this boundary
                # even though Ray actors are intentionally absent in this test.
                sender_locked = bool(chunk.is_locked)
                chunk = cloudpickle.loads(cloudpickle.dumps(chunk))
                receiver_locked = bool(chunk.is_locked)
                if receiver_locked:
                    raise RuntimeError('Serialized dispatch payload remains locked; review actual RPC materialization before training')
                value = torch.tensor(0.713, requires_grad=True)
                state = dict(optimizer_calls=0, zero_grad_calls=0, worker_batch_calls=0, micro_calls=0,
                             scheduler_calls=0, gradient=None)
                engine = SimpleNamespace(get_data_parallel_size=lambda: 32,
                                         get_data_parallel_rank=lambda rank=rank: rank,
                                         is_mp_src_rank_with_outputs=lambda: False,
                                         train_mode=lambda **kwargs: nullcontext())
                def zero_grad():
                    state['zero_grad_calls'] += 1
                    value.grad = None
                def forward_backward(data, loss_function, forward_only=False):
                    state['worker_batch_calls'] += 1
                    tu.assign_non_tensor(data, dp_size=32, batch_num_tokens=global_tokens, sp_size=1)
                    micros, _ = prepare_micro(data)
                    for micro in micros:
                        state['micro_calls'] += 1
                        loss = aggregate(value*micro['coefficients'], micro['response_mask'], mode,
                                         dp_size=32, batch_num_tokens=global_tokens, global_batch_size=32)
                        loss.backward()
                    return {'metrics': {}}
                def optimizer_step():
                    state['optimizer_calls'] += 1
                    state['gradient'] = float(value.grad)
                    return value.grad.abs()
                def scheduler_step():
                    state['scheduler_calls'] += 1
                    return 1e-6
                engine.optimizer_zero_grad = zero_grad
                engine.forward_backward_batch = forward_backward
                engine.optimizer_step = optimizer_step
                engine.lr_scheduler_step = scheduler_step
                engine.train_batch = lambda data, loss_function: engine_batch(engine, data, loss_function)
                worker = SimpleNamespace(engine=engine, loss_fn=lambda: None,
                    model_config={'use_remove_padding': False},
                    engine_config=SimpleNamespace(forward_only=False, use_dynamic_bsz=False,
                                                   max_token_len_per_gpu=8192, micro_batch_size_per_gpu=1,
                                                   use_fused_kernels=False))
                worker.train_batch = lambda data: train_batch(worker, data)
                train_mini(worker, chunk)
                expected_micro = expected_rows//32
                if (state['optimizer_calls'], state['worker_batch_calls'], state['micro_calls'],
                        state['scheduler_calls']) != (1, 1, expected_micro, 1):
                    raise AssertionError(f'Unexpected actual worker mini/micro/update counts: {state}')
                state.update(sender_locked=sender_locked, receiver_locked=receiver_locked)
                local_results.append(state)
            # FSDP averages gradients across DP. Compare the sum of all local
            # micro losses with the unchanged global objective, including pads.
            reference_value = torch.tensor(0.713, requires_grad=True)
            reference_loss = aggregate(reference_value*coefficients, mask, mode, dp_size=1,
                                       batch_num_tokens=global_tokens, global_batch_size=32)
            reference_loss.backward()
            averaged = sum(row['gradient'] for row in local_results)/32
            expected = float(reference_value.grad)
            if not math.isclose(averaged, expected, rel_tol=2e-6, abs_tol=2e-6):
                raise AssertionError(f'DP/micro/padding normalization changed {mode}: {averaged} != {expected}')
            mode_results.append(dict(mode=mode, logical_ranks=32, calls_per_rank=1,
                                     micro_per_rank=expected_rows//32, global_batch_rollouts=32,
                                     dp_average_gradient=averaged, global_reference_gradient=expected))
        cases.append(dict(real_rows=real_rows, padded_rows=expected_rows,
                          local_mini_batch=expected_rows//32, actual_updates_per_rank_per_epoch=1,
                          modes=mode_results))
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU contract regression initialized CUDA')
    result = dict(schema='qwen397-pinned-batch-contract-cpu-v1', passed=True,
                  execution='actual pinned method bodies; logical32 sequential CPU dispatch; scalar autograd engine',
                  torch_version=torch.__version__, cuda_initialized=False,
                  config=dict(prompt_batch=8, rollout_n=4, prompt_ppo_mini_batch=8, dp=32,
                              micro_per_gpu=1, epochs=1, target_windows=2, target_optimizer_updates=2),
                  source_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()},
                  cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Q397_PINNED_BATCH_CONTRACT_CPU_PASSED', json.dumps({k: v for k, v in result.items() if k != 'cases'}), flush=True)


if __name__ == '__main__':
    main()
