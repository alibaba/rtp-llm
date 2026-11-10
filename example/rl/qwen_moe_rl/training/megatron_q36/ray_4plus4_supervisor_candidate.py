"""Scoped Ray start --block supervisor; no global ray stop or PID reuse kills.

Runs in a unique node-local run directory. Polls descendants and persists
PID/create-time identities. An explicit run-scoped stop request is accepted
only after the operator has independently verified probe cleanup.
"""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import psutil


root = Path(os.environ['Q36_RAY_SUPERVISOR_DIR']).resolve(strict=True)
run = os.environ['Q36_4PLUS4_RUN_ID']
role = os.environ['Q36_RAY_ROLE']
if role not in ('head', 'worker') or not re.fullmatch(r'q36-disagg-[a-z0-9-]+-[a-f0-9]{10}', run):
    raise ValueError('invalid run or Ray role')
if root.name != run:
    raise ValueError('Ray supervisor directory does not match run id')
ray_executable = os.environ['Q36_RAY_EXECUTABLE']
node_ip = os.environ['Q36_RAY_NODE_IP']
head_ip = os.environ['Q36_PROBE_NODE1_IP']
node_gpus = '4'
ray_tmp = Path('/tmp') / ('q36r-' + run.rsplit('-', 1)[-1])
if ray_tmp.exists():
    raise RuntimeError('fresh run Ray temporary directory already exists')
arguments = [ray_executable, 'start', '--block', '--node-ip-address', node_ip,
             '--num-gpus', node_gpus, '--num-cpus', '32', '--object-store-memory', str(2 * 1024**3),
             '--node-manager-port', '31101', '--object-manager-port', '31102',
             '--dashboard-agent-listen-port', '31103', '--dashboard-agent-grpc-port', '31104',
             '--runtime-env-agent-port', '31105', '--metrics-export-port', '31106',
             '--min-worker-port', '31200', '--max-worker-port', '31399',
             '--temp-dir', str(ray_tmp), '--disable-usage-stats']
if role == 'head':
    arguments += ['--head', '--port', '31100', '--include-dashboard=false', '--dashboard-port', '31107']
else:
    arguments += ['--address', head_ip + ':31100']


def write_json(name, value):
    temporary = root / (name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, root / name)


def snapshot_tree(proc, owned):
    try:
        parent = psutil.Process(proc.pid)
        descendants = parent.children(recursive=True)
        for process in [parent] + descendants:
            try:
                item = {'pid':process.pid, 'create_time':process.create_time(),
                        'name':process.name(), 'ppid':process.ppid()}
                key = str(item['pid']) + ':' + str(item['create_time'])
                owned[key] = item
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except psutil.NoSuchProcess:
        pass


def alive(item):
    try:
        process = psutil.Process(item['pid'])
        return (abs(process.create_time() - item['create_time']) < .001 and process.is_running()
                and process.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        return False


with (root / f'ray-{role}.raw.log').open('x') as output:
    proc = subprocess.Popen(arguments, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    owned = {}
    started = time.time_ns()
    requested = False
    signalled = set()
    while True:
        snapshot_tree(proc, owned)
        live = [item for item in owned.values() if alive(item)]
        result = proc.poll()
        status = {'run_id':run, 'role':role, 'supervisor_pid':os.getpid(),
                  'supervisor_create_time':psutil.Process().create_time(), 'started_ns':started,
                  'ray_start_pid':proc.pid, 'ray_start_returncode':result,
                  'state':'STOPPING' if requested else ('RUNNING' if result is None else 'HOLD_UNEXPECTED_EXIT'),
                  'owned_processes':list(owned.values()), 'live':live}
        write_json(f'ray-{role}.status.json', status)
        request_file = root / f'ray-{role}.stop-request.json'
        if not requested and request_file.is_file():
            try:
                request = json.loads(request_file.read_text())
                requested = (request.get('run_id') == run and request.get('role') == role and
                             request.get('action') == 'stop_owned_ray' and
                             request.get('probe_cleanup_verified') is True)
            except (OSError, ValueError, TypeError):
                pass
        if requested:
            # Only identities captured as descendants are signalled. Recheck
            # creation time immediately before signalling each process.
            # Stop the task-owned launcher first so it cannot create new Ray
            # children while the captured descendants are being terminated.
            ordered = [item for item in live if item['pid'] == proc.pid] + [
                item for item in reversed(live) if item['pid'] != proc.pid]
            for item in ordered:
                key = str(item['pid']) + ':' + str(item['create_time'])
                if key in signalled:
                    continue
                try:
                    process = psutil.Process(item['pid'])
                    if abs(process.create_time() - item['create_time']) < .001:
                        process.send_signal(signal.SIGTERM)
                        signalled.add(key)
                except psutil.NoSuchProcess:
                    pass
            if proc.poll() is not None and not [item for item in owned.values() if alive(item)]:
                status.update(state='STOPPED', ray_start_returncode=proc.wait(), live=[])
                write_json(f'ray-{role}.status.json', status)
                print('RAY_SUPERVISOR_STOPPED', json.dumps(status, sort_keys=True), flush=True)
                break
        time.sleep(1)
