"""Run the launcher's preflight and inspect its actual exported environment.

Only --print-command is allowed. A standard-library shim records the child
environment before execing the current interpreter; no RTP/native/CUDA import.
"""
from __future__ import annotations
import argparse
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import uuid


def qualify(launcher):
    profile=dict(WORLD_SIZE='72',LOCAL_WORLD_SIZE='4',TP_SIZE='8',DP_SIZE='9',EP_SIZE='72',FFN_SP_SIZE='8',PP_SIZE='1',
        RTP_LLM_RL_MODE='1',RTP_LLM_RL_WORLD72='1',RTP_LLM_RL_START_PAUSED='1',RTP_LLM_RL_METADATA_BOOTSTRAP='1',
        RTP_FIXED_PHYSICAL_EP_LAYOUT='balanced_v1',RTP_RL_BORROW_SINGLETON_HF_SOURCES='1',
        RTP_NCCL_EP_REFERENCE_MAX_PADDED_TOKENS='73728',
        PYTORCH_ALLOC_CONF='backend:native,expandable_segments:False',PYTORCH_CUDA_ALLOC_CONF=None)
    with tempfile.TemporaryDirectory(prefix='q38-world-launcher-env-') as tmp:
        root=Path(tmp);capture=root/'child-profile.json';shim=root/'stdlib-python';model=root/'model'
        model.mkdir();(model/'config.json').write_text('{}')
        policy=root/'policy.json';policy.write_text('{}');mapping=root/'fixed map.json';mapping.write_text('[]')
        shim.write_text('#!'+sys.executable+'\nimport json,os\nfrom pathlib import Path\n'
            +f'Path({str(capture)!r}).write_text(json.dumps({{key:os.environ.get(key) for key in {tuple(profile)!r}}}))\n'
            +f'os.execv({sys.executable!r},[{sys.executable!r},*__import__("sys").argv[1:]])\n')
        shim.chmod(0o700)
        env=dict(os.environ,Q38_DEPLOY_ROOT=str(root),RTP_SOURCE=str(root),RTP_PYTHON=str(shim),
            RTP_MODEL_PATH=str(model),RTP_WEIGHT_MANIFEST_DIR=str(root/'manifests'),
            RTP_RL_NATIVE_LIBRARY_SHA256='a'*64,RTP_RL_BOOTSTRAP_POLICY_MANIFEST=str(policy),RTP_PHY2LOG_PATH=str(mapping),
            RLLM_Q38_NCCL_LIBRARY_PATH=ctypes.util.find_library('c') or '/usr/lib/libSystem.B.dylib',
            RLLM_Q38_NCCL_LIBRARY_SHA256='b'*64,RTP_LLM_RL_RUN_ID='explicit-stdlib-CPU-fixture',RTP_WORLD_RANK='0',
            GANG_CONFIG_STRING=';'.join(f'name:node{i},ip:192.0.2.{i+1},port:24000' for i in range(18)),
            CUDA_VISIBLE_DEVICES=','.join('GPU-'+str(uuid.UUID(int=i+1)) for i in range(4)),
            RLLM_RTP_ACCEPTANCE_DIR=str(root/'audit'))
        for key in (*profile,'BASH_ENV'):env.pop(key,None)
        env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
        result=subprocess.run(['/bin/bash',str(launcher),'--print-command'],env=env,
                              stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=30)
        if result.returncode:raise RuntimeError(f'Actual launcher preflight failed: {result.stdout}')
        actual=json.loads(capture.read_text())
        if actual!=profile:raise RuntimeError(f'The actual launcher child environment differs: {actual}')
        args=shlex.split(result.stdout)
        if '--pp_size' in args or '--json_model_override_args' in args:raise ValueError('Unsupported/incorrect profile override flag')
        if args[args.index('--phy2log_path')+1]!=str(mapping):raise ValueError('The fixed map path was not passed as one exact argument')
        return dict(schema='q38-native-launcher-actual-stdlib-env-qualification-v1',actual_exit_code=result.returncode,
            launcher_sha256=hashlib.sha256(launcher.read_bytes()).hexdigest(),actual_child_profile=actual,
            path_with_space_preserved=True,command=result.stdout,
            scope='Actual Bash exports and standard-library preflight only; no RTP/native/CUDA/model execution')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launcher',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=qualify(args.launcher.resolve())
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(actual_exit_code=result['actual_exit_code'],launcher_sha256=result['launcher_sha256'],
        profile_keys=len(result['actual_child_profile']),scope=result['scope'])))
