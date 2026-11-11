"""Private production candidate: per-worker NIXL/ZeroMQ context ownership.

Requires manager to call authorize_finalize after *all* trainer and rollout
workers have completed native-ACKed transfers, before calling finalize.
The server PULL context survives generations. A failed authorization,
connection or finalize remains poisoned and must not be retried automatically.
Not registered into the production backend or qualified for real RL.
"""
import hashlib
import os
from pathlib import Path
import socket
import uuid
import psutil
import zmq
from verl.checkpoint_engine import nixl_checkpoint_engine as upstream
from verl_nixl_guarded import RtpGuardedNixl

GUARD_SHA='b14be8fa0b69bc8c33dff4ae0694beed5e34bc0849517f9a4ca6464e48191406'


class ProductionManagedNixl(RtpGuardedNixl):
    def __init__(self,*args,**kwargs):
        import verl_nixl_guarded
        path=Path(verl_nixl_guarded.__file__)
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=GUARD_SHA:
            raise RuntimeError('unqualified guarded NIXL source')
        super().__init__(*args,**kwargs)
        self.pull_socket=self.agent.socket
        self.pull_context=self.pull_socket.context
        self.peer_contexts={}
        self.context_quarantine=[]
        self.transaction_step=None
        self.authorization=None
        self.finalization_started=False
        self.shutdown_started=False
        self.engine_nonce=uuid.uuid4().hex
        self.process_identity=(socket.gethostbyname(socket.gethostname()),os.getpid(),psutil.Process().create_time())
        # The fixed upstream engine calls this per-instance method while
        # creating its communication topology; keep the real agent object.
        self.agent.add_remote_agent=self._managed_add_remote_agent

    def managed_identity(self):
        node,pid,created=self.process_identity
        if os.getpid()!=pid or psutil.Process().create_time()!=created:
            raise RuntimeError('managed engine process identity changed')
        return {'node_ip':node,'pid':pid,'create_time':created,'engine_nonce':self.engine_nonce,
                'agent_name':self.agent.agent_name,'shutdown_started':self.shutdown_started}

    def _managed_add_remote_agent(self,metadata):
        try:
            name=self.agent.agent.add_remote_agent(metadata.agent_metadata).decode('utf-8')
            if name!=metadata.agent_name or name in self.peer_contexts:
                raise RuntimeError('unexpected or duplicate remote NIXL agent')
            context=zmq.Context()
            self.context_quarantine.append(context)
            sock=context.socket(zmq.PUSH)
            # Retain both before network setup: failed connect is uncertain.
            self.context_quarantine.append(sock)
            if upstream.is_valid_ipv6_address(metadata.zmq_ip):
                sock.setsockopt(zmq.IPV6,1);address=f'tcp://[{metadata.zmq_ip}]:{metadata.zmq_port}'
            else:address=f'tcp://{metadata.zmq_ip}:{metadata.zmq_port}'
            sock.connect(address)
            self.peer_contexts[name]=(context,sock)
            self.agent.zmq_clients[name]=sock
            return name
        except BaseException:
            self.poisoned=True
            raise

    def prepare(self):
        if self.poisoned or self.shutdown_started or self.peer_contexts or self.authorization is not None:
            raise RuntimeError('previous managed context transaction not closed')
        self.transaction_step=None;self.finalization_started=False
        return super().prepare()

    def init_process_group(self,*args,**kwargs):
        try:return super().init_process_group(*args,**kwargs)
        except BaseException:
            self.poisoned=True
            raise

    def _claim_step(self,step):
        if type(step) is not int or step<0 or self.transaction_step is not None:
            raise RuntimeError('invalid or duplicate transaction step')
        self.transaction_step=step

    async def send_weights(self,weights,global_steps=None):
        self._claim_step(global_steps)
        return await super().send_weights(weights,global_steps=global_steps)

    def receive_weights(self,global_steps=None):
        self._claim_step(global_steps)
        return super().receive_weights(global_steps=global_steps)

    def authorize_finalize(self,step,commit_token):
        if (self.poisoned or not self.active or not self.finished or step!=self.transaction_step or
            self.authorization is not None or self.finalization_started or
            not isinstance(commit_token,str) or len(commit_token)<16):
            raise RuntimeError('global completed generation authorization required')
        self.authorization=(step,commit_token)

    def finalize(self):
        if self.authorization is None or self.finalization_started:
            raise RuntimeError('managed finalization not authorized or already attempted')
        self.finalization_started=True
        try:
            if set(self.peer_contexts)!=set(self.agent.zmq_clients):
                raise RuntimeError('PUSH context ownership differs')
            for name,(context,sock) in self.peer_contexts.items():
                if self.agent.zmq_clients[name] is not sock or sock.context is not context or set(context._sockets)!={sock}:
                    raise RuntimeError('PUSH context changed before finalization')
                sock.setsockopt(zmq.LINGER,0)
            super().finalize()
            if self.agent.zmq_clients:raise RuntimeError('PUSH sockets not removed')
            for context,sock in self.peer_contexts.values():
                if not sock.closed or context._sockets:raise RuntimeError('PUSH socket still active')
                context.term()
            self.peer_contexts.clear();self.context_quarantine.clear()
            self.authorization=None;self.transaction_step=None
        except BaseException:
            self.poisoned=True
            raise

    def shutdown_managed_context(self):
        # A bounded external supervisor must own this call; ctx.term can block.
        if self.shutdown_started or self.active or self.poisoned or self.peer_contexts or self.context_quarantine:
            raise RuntimeError('worker context still active or shutdown already attempted')
        self.shutdown_started=True
        if self.pull_socket.context is not self.pull_context or set(self.pull_context._sockets)!={self.pull_socket}:
            raise RuntimeError('PULL context ownership changed')
        self.pull_socket.close(linger=0)
        if self.pull_context._sockets:raise RuntimeError('PULL context still has sockets')
        self.pull_context.term()
