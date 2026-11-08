"""Supervise one exact node-owned host cgroup sampler and record actual wait."""
from pathlib import Path
import json,os,subprocess,sys,time
cid,node,run_id,job_dir,sampler_path=sys.argv[1:]
job=Path(job_dir);job.mkdir(parents=True,exist_ok=False)
def identity(pid):
 p=Path('/proc')/str(pid)
 return dict(pid=pid,start_ticks=(p/'stat').read_text().split(') ',1)[1].split()[19],
             boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
with (job/'supervisor-identity.json').open('x') as f:json.dump(identity(os.getpid()),f)
command=[sys.executable,sampler_path,'--container-id',cid,'--node',node,'--run-id',run_id,'--interval','1']
with (job/'sampler.log').open('x') as log:
 p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 child=identity(p.pid);child['command']=command
 with (job/'sampler-identity.json').open('x') as f:json.dump(child,f)
 code=p.wait()
with (job/'actual-exit.json').open('x') as f:
 json.dump(dict(process=child,actual_wait_exit_code=code,finished_unix=time.time()),f)
