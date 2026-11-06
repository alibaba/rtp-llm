"""Execute exact controller methods with CPU RPC fixtures and owned Popen jobs.

No Ray cluster, GPU, native model, or real owner identity is created. The
standalone AST extraction preserves the production method bodies byte-for-byte
while replacing only their surrounding external RPC/process dependencies.
"""
from pathlib import Path
import argparse,ast,asyncio,hashlib,json,os,subprocess,sys,time,uuid


def require(value,message):
    if not value:raise ValueError(message)


class Remote:
    def __init__(self,function):self.remote=function


class Owner:
    def __init__(self,index,fail_first=False):
        self.index=index;self.fail_first=fail_first;self.poisoned=False
        self.release=asyncio.Event();self.child=None;self.actual_exits=[]
        self.permanent_stops=self.normal_stops=0;self.start_calls=0
        self.identity=dict(pid=20000+index,create_time=float(30000+index),audit_dir=f'CPU-fixture-owner-{index}')
        self.start=Remote(self._start);self.stop_if_owned=Remote(self._permanent_stop)
        self.stop_for_training=Remote(self._normal_stop)

    async def _start(self,boot,incarnation):
        self.start_calls+=1
        if self.index>=9:await self.release.wait()  # Queued/not-yet-allocated start.
        require(not self.poisoned,'permanently stopped CPU owner rejects a late start')
        command='import sys;sys.exit(7)' if self.fail_first else 'import time;time.sleep(30)'
        self.child=subprocess.Popen([sys.executable,'-c',command],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if self.fail_first:
            await asyncio.sleep(.04)
            actual=await asyncio.to_thread(self.child.wait);self.actual_exits.append(actual)
            raise RuntimeError('CPU fixture owner failed with actual child exit7')
        await self.release.wait()
        require(not self.poisoned,'CPU owner startup was permanently stopped')
        raise RuntimeError('CPU fixture has no success/native-manifest route')

    async def _wait_child(self):
        if self.child is not None:
            if self.child.poll() is None:self.child.terminate()
            actual=await asyncio.to_thread(self.child.wait,timeout=5)
            if not self.actual_exits:self.actual_exits.append(actual)

    async def _permanent_stop(self,pid,created,audit_dir):
        require((pid,created,audit_dir)==tuple(self.identity[k] for k in ('pid','create_time','audit_dir')),
                'CPU fixture cleanup identity differs')
        self.poisoned=True;self.permanent_stops+=1;self.release.set()
        await self._wait_child()
        return self.receipt()

    async def _normal_stop(self):
        self.normal_stops+=1;self.release.set();await self._wait_child();return self.receipt()

    def receipt(self):
        return dict(node_index=self.index,process_tree_terminated=True,
            checked_native_ranks=list(range(4*self.index,4*self.index+4)),
            remaining_owned_gpu_contexts=[],owned_pids=([self.child.pid] if self.child else []))


def controller_class(source):
    syntax=ast.parse(source.read_text());original=next(node for node in syntax.body
        if isinstance(node,ast.ClassDef) and node.name=='World72RTPServer')
    methods=[node for node in original.body if isinstance(node,ast.AsyncFunctionDef)
             and node.name in ('_stop_owners','ensure_native_running')]
    assert {node.name for node in methods}=={'_stop_owners','ensure_native_running'}
    cls=ast.ClassDef(name='CPUController',bases=[],keywords=[],body=methods,decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[]))
    namespace=dict(asyncio=asyncio,uuid=uuid,require=require,time=time)
    exec(compile(module,str(source),'exec'),namespace)
    return namespace['CPUController']


def make_controller(cls,owners):
    server=cls();server.owners=owners;server._owner_identities=[o.identity for o in owners]
    server.poisoned=False;server._committed_version=None;server._started=False
    server._generation_ready=False;server._owners_active=False;server._incarnation_failed=False
    server._lifecycle_lock=asyncio.Lock();server.boot=-1;server.incarnation=None;server._last_suspend=None
    async def close_admission(reason):server._generation_ready=False
    async def close_clients():pass
    server._close_admission=close_admission;server._close_clients=close_clients
    return server


async def failure_case(cls):
    owners=[Owner(i,fail_first=(i==0)) for i in range(18)]
    server=make_controller(cls,owners);started=time.monotonic()
    try:
        try:await asyncio.wait_for(server.ensure_native_running(0,None),timeout=2)
        except RuntimeError as error:assert 'actual child exit7' in str(error)
        else:raise AssertionError('The real failed owner exception did not propagate')
        latency=time.monotonic()-started
        await asyncio.sleep(.02)
        assert latency<2 and server.poisoned and server._incarnation_failed
        assert all(o.poisoned and o.permanent_stops==1 and o.normal_stops==0 for o in owners)
        assert all(o.child is None for o in owners[9:])
        try:await owners[-1]._start(0,'late-start-after-cleanup')
        except ValueError:pass
        else:raise AssertionError('A queued late owner start could allocate after cleanup')
        assert server._last_suspend['all_owned_native_processes_exited'] is True
        assert server._last_suspend['training_admission_safe'] is False
        assert server._last_suspend['checked_native_ranks']==list(range(72))
        actual=[o.actual_exits[0] for o in owners if o.child is not None]
        assert actual[0]==7 and all(code==-15 for code in actual[1:]),actual
        return dict(first_failure_observed_without_waiting_for_pending_starts=True,
            failure_to_complete_cleanup_seconds=latency,actual_owned_CPU_child_exit_codes=actual,
            permanent_identity_bound_stops=18,late_queued_allocations=0,training_admission_safe=False)
    finally:
        for owner in owners:await owner._wait_child()


async def cancellation_case(cls):
    owners=[Owner(i) for i in range(18)];server=make_controller(cls,owners)
    job=asyncio.create_task(server.ensure_native_running(0,None))
    try:
        for _ in range(100):
            if all(owner.child is not None for owner in owners[:9]):break
            await asyncio.sleep(.005)
        assert all(owner.child is not None for owner in owners[:9])
        job.cancel()
        try:await asyncio.wait_for(job,2)
        except asyncio.CancelledError:pass
        else:raise AssertionError('Caller cancellation was swallowed')
        assert all(owner.poisoned and owner.permanent_stops==1 for owner in owners)
        assert all(owner.child is None for owner in owners[9:])
        assert all(owner.actual_exits==[-15] for owner in owners[:9])
        assert server._last_suspend['training_admission_safe'] is False
        return dict(caller_cancelled=True,permanent_identity_bound_stops=18,
            actual_owned_CPU_child_exit_codes=[owner.actual_exits[0] for owner in owners[:9]],
            late_queued_allocations=0,training_admission_safe=False)
    finally:
        if not job.done():job.cancel()
        for owner in owners:await owner._wait_child()


async def normal_stop_case(cls):
    owners=[Owner(i) for i in range(18)];server=make_controller(cls,owners)
    receipts=await server._stop_owners()
    assert len(receipts)==18 and all(o.normal_stops==1 and o.permanent_stops==0 and not o.poisoned for o in owners)
    return dict(normal_sleep_remains_nonpermanent=True,normal_stops=18,permanent_stops=0)


async def run(args):
    cls=controller_class(args.source)
    result=dict(schema='q38-owner-failfast-exact-methods-owned-CPU-fixture-v1',
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        failed_owner=await failure_case(cls),caller_cancellation=await cancellation_case(cls),
        normal_sleep=await normal_stop_case(cls),passed=True,
        scope='Exact production controller methods with CPU RPC fixtures and actual owned Popen waits; no Ray/GPU/native/real owner allocation',
        GPU_or_Ray_or_model_execution=False,full_RL_accepted=False)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);asyncio.run(run(parser.parse_args()))
