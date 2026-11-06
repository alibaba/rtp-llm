"""Execute real recipe/owner/launcher prefixes, intercept both GPU entry points.

The exact recipe dataset validator and exact launcher's standard-library
validator run unmodified. The training module, Ray actor construction, native
server main and GPU allocation never run. A supplied CPU checker receives the
native argument/environment capture as its one positional JSON path.
"""
from __future__ import annotations
import argparse,asyncio,hashlib,json,os
from pathlib import Path
import subprocess,sys
from unittest.mock import patch


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def profile(environment=None):
    environment=os.environ if environment is None else environment
    names=('PYTHONPATH','LD_PRELOAD','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES',
           'PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF','OMP_NUM_THREADS',
           'WORLD_SIZE','WORLD_RANK','LOCAL_WORLD_SIZE','TP_SIZE','DP_SIZE',
           'EP_SIZE','FFN_SP_SIZE','PP_SIZE','GANG_CONFIG_STRING',
           'HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTHONDONTWRITEBYTECODE')
    def allowed(key):
        sensitive=('SECRET','PASSWORD','CREDENTIAL','AUTH','API_KEY','ACCESS_KEY','COOKIE','BEARER')
        return (key in names or key.startswith(('RTP_','RLLM_','Q38_','NCCL_'))) \
            and not any(part in key.upper() for part in sensitive) and not key.upper().endswith('_TOKEN')
    return {key:value for key,value in environment.items() if allowed(key)}


def shim(role):
    args=sys.argv[3:];real=os.environ['Q38_CHAIN_REAL_PYTHON'];root=Path(os.environ['Q38_CHAIN_OUTPUT'])
    target=['-m','run_qwen38_bf16_state_rtp_training'] if role=='training' else ['-m','rtp_llm.start_server']
    if args[:2]!=target:
        os.execv(real,[real,*args])  # Preserve the real stdin validators exactly.
    record=dict(role=role,argv=args,environment=profile(),entrypoint_intercepted_before_execution=True)
    capture=root/(role+'-entrypoint-capture.json');write(capture,record)
    if role=='training':
        os.environ['RTP_PYTHON']=os.environ['Q38_CHAIN_NATIVE_SHIM']
        os.execv(real,[real,str(Path(__file__).resolve()),'--owner-prefix'])
    checker=os.environ['Q38_CHAIN_CPU_CHECKER']
    os.execv(real,[real,checker,str(capture)])


def owner_prefix():
    """Explicit CPU fixture; stop actual WorldNodeOwner.start at Popen."""
    import ray,torch
    from qwen38_world_contract import World72Topology
    from qwen38_world_server import WorldNodeOwner
    root=Path(os.environ['Q38_CHAIN_OUTPUT']);inputs=json.loads(Path(os.environ['Q38_CHAIN_INPUT']).read_text())
    index=int(os.environ['Q38_CHAIN_NODE_INDEX'])
    assert not torch.cuda.is_initialized() and not ray.is_initialized()
    topology=World72Topology(gpu_ids=tuple('GPU-'+value for value in inputs['rank_gpu_uuids']),
        gpu_uuids=tuple(inputs['rank_gpu_uuids']),rank_hosts=tuple(node['container_ip'] for node in inputs['nodes'] for _ in range(4)),
        start_port=inputs['start_port'])
    # No constructor: its identity uses a live Ray runtime. The real pre-Popen
    # method only needs these explicit fixture fields and the real topology.
    owner=WorldNodeOwner.__new__(WorldNodeOwner)
    owner.topology=topology;owner.node_index=index;owner.node=topology.node_groups[index]
    owner.model_path=inputs['model_path'];owner.main_audit_root=Path(os.environ['RLLM_RTP_ACCEPTANCE_DIR']).resolve()
    owner.poisoned=False;owner.process=None;owner.log=None;owner._cleanup_receipt=None
    captured={}
    class CapturedPopen(Exception):pass
    def intercept(*args,**kwargs):
        captured.update(argv=args[0],environment=kwargs['env'],start_new_session=kwargs.get('start_new_session'))
        raise CapturedPopen()
    try:
        with patch('qwen38_world_server.subprocess.Popen',intercept):
            try:asyncio.run(owner.start(0,'cpu-prefix-qualification-no-native-incarnation'))
            except CapturedPopen:pass
    finally:
        if owner.log is not None:owner.log.close()
    assert captured and not torch.cuda.is_initialized() and not ray.is_initialized()
    write(root/'actual-owner-prefix-capture.json',dict(captured,environment=profile(captured['environment']),scope='Actual WorldNodeOwner.start through pre-Popen; explicit CPU fixture, no owner identity or Ray/native health claim',
        ray_initialized=False,cuda_initialized=False))
    # Native Python is the terminal CPU-checker shim. The exact launcher still
    # runs its real stdin profile validator before reaching the interception.
    result=subprocess.run(captured['argv'],env=captured['environment'])
    write(root/'chain-CPU-result.json',dict(actual_exit_code=result.returncode,
        training_driver_executed=False,Ray_actor_or_HF_worker_created=False,
        actual_owner_prefix_executed_before_Popen=True,native_main_executed=False,
        GPU_allocation_requested=False,scope='CPU recipe/dataset/owner-prefix/launcher checks only; no native bootstrap or RL acceptance'))
    return result.returncode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',type=Path,required=True);parser.add_argument('--real-python',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--node-index',type=int,required=True)
    parser.add_argument('--cpu-checker',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();assert 0<=args.node_index<18
    for path in (args.recipe,args.real_python,args.input,args.cpu_checker):assert path.is_file()
    root=args.output_dir.resolve();root.mkdir(parents=True,exist_ok=False)
    script=Path(__file__).resolve()
    for role in ('training','native'):
        shim_path=root/(role+'-python-shim')
        shim_path.write_text('#!/usr/bin/env python3\nimport os\n'+f'os.execv({str(args.real_python.resolve())!r},[{str(args.real_python.resolve())!r},{str(script)!r},"--shim",{role!r},*__import__("sys").argv[1:]])\n')
        shim_path.chmod(0o700)
    env=dict(os.environ,Q38_CHAIN_REAL_PYTHON=str(args.real_python.resolve()),Q38_CHAIN_OUTPUT=str(root),
        Q38_CHAIN_INPUT=str(args.input.resolve()),Q38_CHAIN_NODE_INDEX=str(args.node_index),
        Q38_CHAIN_CPU_CHECKER=str(args.cpu_checker.resolve()),Q38_CHAIN_NATIVE_SHIM=str(root/'native-python-shim'),
        Q38_TRAIN_PYTHON=str(root/'training-python-shim'),Q38_RUN_ROOT=str(root/'recipe-run'),CUDA_VISIBLE_DEVICES='')
    write(root/'SOURCE.json',dict(recipe_sha256=digest(args.recipe),driver_sha256=digest(script),input_sha256=digest(args.input),
        CPU_checker_sha256=digest(args.cpu_checker),entrypoints_intercepted=['training -m','native -m'],native_start_authorized=False))
    return subprocess.run(['/bin/bash',str(args.recipe.resolve())],env=env).returncode


if __name__=='__main__':
    if sys.argv[1:2]==['--shim']:shim(sys.argv[2])
    elif sys.argv[1:]==['--owner-prefix']:raise SystemExit(owner_prefix())
    else:raise SystemExit(main())
