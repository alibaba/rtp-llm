"""Private manager transaction gate for the pinned coupled NIXL interface.

Caller must reserve *before* prepare/build_process_group. This is a candidate
for integration with CheckpointEngineManager, not a replacement of verl.
"""
import asyncio
import math
import threading
import uuid


class ProductionManagerGate:
    def __init__(self):
        self.poisoned=False
        self.active=None
        self.last_step=None
        self.shutdown_started=False
        self.shutdown_completed=False
        self.shutdown_refs=[]
        self.participants=None
        self.worker_incarnations=None
        self._lock=threading.RLock()

    def bind_participants(self,participants):
        with self._lock:
            if self.poisoned:raise RuntimeError('manager transaction is poisoned')
            if (not isinstance(participants,tuple) or len(participants)!=2 or
                not all(isinstance(group,tuple) and group for group in participants) or
                len(set(participants[0]+participants[1]))!=len(participants[0])+len(participants[1])):
                self.poisoned=True
                raise RuntimeError('invalid or overlapping worker identities')
            if self.participants is None:self.participants=participants
            elif self.participants!=participants:
                self.poisoned=True
                raise RuntimeError('managed worker identities changed')

    def begin_shutdown(self,participants):
        with self._lock:
            if self.poisoned or self.shutdown_started or self.active is not None:
                raise RuntimeError('cannot shut down active or poisoned manager contexts')
            self.bind_participants(participants)
            self.shutdown_started=True

    def bind_worker_incarnations(self,rows,expected_shutdown=False):
        with self._lock:
            if self.poisoned:raise RuntimeError('manager transaction is poisoned')
            if self.participants is None or len(rows)!=sum(map(len,self.participants)):
                self.poisoned=True;raise RuntimeError('worker incarnation count differs')
            keys=('node_ip','pid','create_time','engine_nonce','agent_name')
            snapshots=[]
            for row in rows:
                if (not isinstance(row,dict) or any(key not in row for key in keys) or
                    not isinstance(row['node_ip'],str) or not row['node_ip'] or
                    type(row['pid']) is not int or row['pid']<=0 or
                    type(row['create_time']) not in (int,float) or not math.isfinite(row['create_time']) or row['create_time']<=0 or
                    not isinstance(row['engine_nonce'],str) or len(row['engine_nonce'])!=32 or
                    not isinstance(row['agent_name'],str) or not row['agent_name'] or
                    row.get('shutdown_started') is not expected_shutdown):
                    self.poisoned=True;raise RuntimeError('invalid worker incarnation')
                snapshots.append(tuple(row[key] for key in keys))
            snapshots=tuple(snapshots)
            if len({x[3] for x in snapshots})!=len(snapshots):
                self.poisoned=True;raise RuntimeError('duplicate engine incarnation')
            if self.worker_incarnations is None:self.worker_incarnations=snapshots
            elif self.worker_incarnations!=snapshots:
                self.poisoned=True;raise RuntimeError('worker process or engine incarnation changed')

    def poison_without_active(self):
        with self._lock:self.poisoned=True

    def dispatch(self,tx,action):
        # Serializes the decision to start a side effect against poison. An
        # action accepted before poison can still complete afterwards.
        with self._lock:
            if self.poisoned or self.shutdown_started or self.active is not tx:
                raise RuntimeError('cannot dispatch on poisoned or inactive transaction')
            return action()

    def dispatch_shutdown(self,action):
        with self._lock:
            if self.poisoned or not self.shutdown_started or self.active is not None:
                raise RuntimeError('cannot dispatch shutdown on poisoned manager')
            return action()

    def reserve(self,step,actor_wg,rollout):
        with self._lock:
            if (self.poisoned or self.shutdown_started or self.active is not None or type(step) is not int or step<0 or
                (self.last_step is not None and step<=self.last_step)):
                raise RuntimeError('manager transaction is poisoned, active or out of order')
            self.active={'step':step,'token':uuid.uuid4().hex,'actor':actor_wg,'rollout':rollout,
                         'trainer_refs':None,'rollout_refs':None,'gather':None,'phase':'prepare'}
            return self.active

    def fail(self,error):
        with self._lock:
            if self.active is None:raise RuntimeError('no active manager transaction')
            self.poisoned=True
            self.active['phase']='failed'
            self.active['error']=repr(error)
        # Keep all references in self.active for external identity-bound
        # reconciliation. This does not stop tasks or prove NIC quiescence.

    @staticmethod
    def _command(group,name,*args):
        count=group.world_size
        if type(count) is not int or count<=0:raise RuntimeError('invalid worker count')
        return group.execute_checkpoint_engine([name]*count,*([[arg]*count for arg in args]))

    async def commit_after_prepare(self,trainer_refs,resume,merge=None,verify=None):
        tx=self.active
        if tx is None or self.poisoned or tx['phase']!='prepare':
            raise RuntimeError('manager transaction was not reserved')
        actor,rollout=tx['actor'],tx['rollout']
        tx['trainer_refs']=trainer_refs
        try:
            tx['phase']='transferring'
            def start_rollouts():tx['rollout_refs']=rollout.update_weights(global_steps=tx['step'])
            self.dispatch(tx,start_rollouts)
            tx['gather']=asyncio.gather(*tx['trainer_refs'],*tx['rollout_refs'])
            results=await asyncio.shield(tx['gather'])
            # All actor/rollout coroutines must return successfully. A rollout
            # adapter may return only after all its native source ACKs/commit.
            tx['phase']='authorize'
            if verify is not None:await verify()
            tx['authorization_refs']=[]
            def authorize():
                tx['authorization_refs'].extend(self._command(actor,'authorize_finalize',tx['step'],tx['token']))
                tx['authorization_refs'].extend(self._command(rollout,'authorize_finalize',tx['step'],tx['token']))
            self.dispatch(tx,authorize)
            tx['authorization_gather']=asyncio.gather(*tx['authorization_refs'])
            await asyncio.shield(tx['authorization_gather'])
            tx['phase']='finalize'
            if verify is not None:await verify()
            tx['finalize_refs']=[]
            def finalize():
                tx['finalize_refs'].extend(self._command(actor,'finalize'))
                tx['finalize_refs'].extend(self._command(rollout,'finalize'))
            self.dispatch(tx,finalize)
            tx['finalize_gather']=asyncio.gather(*tx['finalize_refs'])
            await asyncio.shield(tx['finalize_gather'])
            result=merge(results) if merge is not None else results
            tx['phase']='resume'
            if verify is not None:await verify()
            tx['resume_task']=self.dispatch(tx,lambda:asyncio.create_task(resume()))
            await asyncio.shield(tx['resume_task'])
            tx['phase']='committed'
            with self._lock:
                if self.poisoned or self.active is not tx:
                    raise RuntimeError('manager state changed before completion')
                self.last_step=tx['step'];self.active=None
            return result
        except BaseException as error:
            self.fail(error)
            raise
