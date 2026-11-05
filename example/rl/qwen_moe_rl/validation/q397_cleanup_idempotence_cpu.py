"""Read-only ARM/Ray regression of candidate cleanup against closed R9 identities.

Calls _cleanup_multinode directly. Never calls cleanup_owned_rtp_processes and
never changes the original exit.json / driver-cleanup.json. A live named actor
causes immediate refusal before any shutdown/stop RPC can be sent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid


def digest(path):
    data=Path(path).read_bytes()
    return dict(path=str(Path(path).resolve()),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def source_state(root):
    commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    status=subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=no'],text=True)
    return dict(commit=commit,tracked_status=status)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--candidate-root',type=Path,required=True)
    parser.add_argument('--audit-dir',type=Path,required=True)
    parser.add_argument('--ray-address',required=True)
    parser.add_argument('--expected-source-commit',required=True)
    parser.add_argument('--expected-latest-source-sha256',required=True)
    parser.add_argument('--expected-latest-step',type=int,default=2)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    project=args.project_root.resolve(strict=True)
    candidate=args.candidate_root.resolve(strict=True)
    audit=args.audit_dir.resolve(strict=True)
    output=args.output.resolve()
    if output.is_relative_to(audit.parent) or output.exists():
        raise ValueError('Regression output must be a fresh path outside the original run directory')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or 'torch' in sys.modules:
        raise RuntimeError('Start this CPU-only regression with CUDA_VISIBLE_DEVICES empty and no Torch import')
    sys.path.insert(0,str(candidate/'adapter'))
    import rtp_cleanup
    import rtp_cleanup_completed
    if Path(rtp_cleanup.__file__).resolve()!=candidate/'adapter/rtp_cleanup.py':
        raise RuntimeError('The regression did not import the reviewed candidate cleanup')
    originals=[audit.parent/'exit.json',audit/'driver-cleanup.json']
    before=[digest(path) for path in originals]
    source_before=source_state(project)
    if source_before!={'commit':args.expected_source_commit,'tracked_status':''}:
        raise RuntimeError('Original source state differs from the explicitly pinned run')
    if json.loads(originals[0].read_text())['returncode']!=1 or json.loads(originals[1].read_text())['passed'] is not False:
        raise RuntimeError('This regression expects the preserved failed R9 cleanup originals')
    report=dict(schema='q397-closed-r9-cleanup-idempotence-cpu-v1',passed=False,started_unix=time.time(),
                scope='Post-run cleanup code regression only; never a new RL run or rewrite of the original failed exit.',
                counts_as_new_rl_success=False,originals_before=before,source_before=source_before,
                candidate_sources=[digest(path) for path in (candidate/'adapter/rtp_cleanup.py',candidate/'adapter/rtp_cleanup_completed.py',Path(__file__))],
                signals_sent=False,native_control_rpcs_sent=False,replicas=[])
    import ray
    real_get_actor=ray.get_actor
    def read_only_lookup(*arguments,**keywords):
        real_get_actor(*arguments,**keywords)
        raise RuntimeError('CPU regression refuses a live named actor; no control RPC was sent')
    try:
        inherited=os.environ.get('PYTHONPATH','')
        ray.init(address=args.ray_address,namespace='q397-cleanup-regression-'+uuid.uuid4().hex,
                 runtime_env={'env_vars':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(candidate/'adapter')+os.pathsep+inherited}})
        ray.get_actor=read_only_lookup
        for replica in range(2):
            identity_path=audit/f'rtp_cluster_{replica}.json'
            identity=json.loads(identity_path.read_text())
            result=rtp_cleanup._cleanup_multinode(identity,audit)
            proof=result['rpc_receipt']
            if (result['method']!='verified_completed_multinode_shutdown' or
                    proof['latest_sync_step']!=args.expected_latest_step or
                    proof['latest_source_sha256']!=args.expected_latest_source_sha256 or
                    len(proof['node_inspections'])!=4 or not all(row['passed'] for row in proof['node_inspections'])):
                raise RuntimeError('Candidate did not positively verify the exact closed R9 policy on all4 replica nodes')
            report['replicas'].append(dict(identity_file=digest(identity_path),result=result))
        if 'torch' in sys.modules:
            raise RuntimeError('CPU driver unexpectedly imported Torch')
        report['passed']=True
    except BaseException as error:
        report.update(error=repr(error),traceback=traceback.format_exc())
    finally:
        ray.get_actor=real_get_actor
        ray.shutdown()
        report.update(finished_unix=time.time(),source_after=source_state(project),originals_after=[digest(path) for path in originals],torch_imported='torch' in sys.modules)
        report['originals_unchanged']=report['originals_before']==report['originals_after']
        report['source_unchanged']=report['source_before']==report['source_after']
        report['passed']=report['passed'] and report['originals_unchanged'] and report['source_unchanged'] and not report['torch_imported']
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({key:report[key] for key in ('passed','originals_unchanged','source_unchanged','torch_imported','signals_sent','native_control_rpcs_sent')}))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
