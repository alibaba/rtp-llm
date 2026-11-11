"""CPU-only real Ray test of the actual decorated rLLM TaskRunner hook.

The run is isolated to one private node1 Ray session with zero GPUs.  An
invalid run call deliberately enters HOLD before any train/rollout creation;
the test releases only its own actor handle and shuts down its local Ray.
"""
import hashlib
import json
import os
from pathlib import Path
import time

import psutil
import ray
from ray.exceptions import GetTimeoutError

import q36_managed_shutdown_candidate as managed
import rllm.trainer.verl.verl_launcher as launcher

root=Path(os.environ['Q36_RUN_ROOT']).resolve(strict=True)
assert os.environ['Q36_USE_MANAGED_NIXL']=='1'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert not list(root.glob('managed-hold-*.json'))
actor_class=launcher.VerlTaskRunner
metadata=actor_class.__ray_metadata__
assert metadata.last_export_cluster_and_job is None
managed.install_outer(launcher)
wrapped=metadata.modified_class.run
assert wrapped._q36_managed_outer is True
assert metadata.method_meta.methods['run'] is wrapped
import ray.cloudpickle
roundtrip=ray.cloudpickle.loads(ray.cloudpickle.dumps(metadata.modified_class))
assert roundtrip.run._q36_managed_outer is True

result={'schema':'q36-real-ray-taskrunner-hold-v1',
        'managed_sha256':hashlib.sha256(Path(managed.__file__).read_bytes()).hexdigest(),
        'launcher_sha256':hashlib.sha256(Path(launcher.__file__).read_bytes()).hexdigest(),
        'ray_version':ray.__version__,'actor_max_restarts':metadata.max_restarts,
        'actual_rllm_actor_class':True,'gpu_count':0}
actor=None
try:
    ray.init(address='local',num_cpus=1,num_gpus=0,include_dashboard=False,
             _temp_dir=os.environ['Q36_RAY_TEST_TEMP'],log_to_driver=True)
    actor=actor_class.options(max_restarts=0).remote()
    result['actor_id']=actor._actor_id.hex()
    pending=actor.run.remote(None,None,{})
    deadline=time.monotonic()+90
    receipts=[]
    while time.monotonic()<deadline:
        receipts=list(root.glob('managed-hold-*.json'))
        if receipts:break
        ready,_=ray.wait([pending],timeout=1)
        if ready:
            ray.get(pending)
            raise AssertionError('unmanaged task runner exited before HOLD')
    assert len(receipts)==1,'actual actor did not enter HOLD'
    receipt=json.loads(receipts[0].read_text())
    assert receipt['phase']=='task_runner_outer'
    process=psutil.Process(receipt['pid'])
    result.update(hold_receipt=receipt,actor_process_create_time=process.create_time())
    try:ray.get(pending,timeout=1)
    except GetTimeoutError:result['call_still_pending_in_hold']=True
    else:raise AssertionError('HOLD returned unexpectedly')
    ray.kill(actor,no_restart=True);actor=None
    gone,alive=psutil.wait_procs([process],timeout=20)
    assert not alive,'owned actor did not exit after ray.kill'
    result['actor_exited_after_kill']=True
finally:
    if actor is not None:
        ray.kill(actor,no_restart=True)
    ray.shutdown()

assert not ray.is_initialized()
result['ray_driver_disconnected']=True
tmp=root/'RESULT.tmp'
tmp.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
tmp.replace(root/'RESULT.json')
print('REAL_RAY_TASKRUNNER_HOLD_OK '+json.dumps(result,sort_keys=True),flush=True)
