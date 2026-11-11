"""CPU context ownership tests; transport and native ACK are simulated."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import zmq
import zmq.asyncio


def load_candidate():
    guard=types.ModuleType('verl_nixl_guarded')
    guard.__file__=str(Path(__file__).with_name('verl_nixl_guarded_partial_lease.py'))
    upstream=types.ModuleType('verl.checkpoint_engine.nixl_checkpoint_engine')
    upstream.is_valid_ipv6_address=lambda _ip:False
    ce=types.ModuleType('verl.checkpoint_engine');ce.nixl_checkpoint_engine=upstream
    class Raw:
        def add_remote_agent(self,blob):return blob
    class Base:
        def __init__(self,*_,**__):
            context=zmq.asyncio.Context();socket=context.socket(zmq.PULL);socket.bind('tcp://127.0.0.1:*')
            self.agent=types.SimpleNamespace(socket=socket,agent=Raw(),zmq_clients={},agent_name='fake-agent')
            self.active=False;self.finished=False;self.poisoned=False
        def prepare(self):
            if getattr(self,'inject_prepare_failure',False):
                self.poisoned=True;raise RuntimeError('injected partial registration')
            self.active=True;self.finished=False
        def init_process_group(self,metadata):self.agent.add_remote_agent(metadata)
        def finalize(self):
            for socket in self.agent.zmq_clients.values():socket.close()
            self.agent.zmq_clients.clear();self.active=False
    guard.RtpGuardedNixl=Base
    modules={'verl_nixl_guarded':guard,'verl':types.ModuleType('verl'),'verl.checkpoint_engine':ce,
             'verl.checkpoint_engine.nixl_checkpoint_engine':upstream}
    scope=patch.dict(sys.modules,modules);scope.start()
    spec=importlib.util.spec_from_file_location('production_context_candidate',
        Path(__file__).with_name('production_nixl_context_candidate.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,upstream,scope


class Tests(unittest.TestCase):
    def setUp(self):
        self.m,self.upstream,scope=load_candidate();self.addCleanup(scope.stop)
    @staticmethod
    def metadata():return types.SimpleNamespace(agent_name='peer',agent_metadata=b'peer',zmq_ip='127.0.0.1',zmq_port=1)

    def test_three_generations_preserve_pull_until_shutdown(self):
        e=self.m.ProductionManagedNixl();pull=e.pull_context;identity=e.managed_identity()
        for step in range(3):
            e.prepare();e.init_process_group(self.metadata());e._claim_step(step);e.finished=True
            e.authorize_finalize(step,'completion-token-1234');e.finalize()
            self.assertIs(e.pull_context,pull);self.assertFalse(pull.closed);self.assertFalse(e.peer_contexts)
            self.assertEqual(e.managed_identity(),identity)
        e.shutdown_managed_context();self.assertTrue(pull.closed)

    def test_no_global_authorization_blocks_finalize(self):
        e=self.m.ProductionManagedNixl();e.prepare();e.init_process_group(self.metadata());e._claim_step(0);e.finished=True
        with self.assertRaises(RuntimeError):e.finalize()
        self.assertTrue(e.active);self.assertEqual(len(e.peer_contexts),1)
        e.authorize_finalize(0,'completion-token-1234');e.finalize();e.shutdown_managed_context()

    def test_partial_connection_is_quarantined_and_poisoned(self):
        e=self.m.ProductionManagedNixl();e.prepare()
        self.upstream.is_valid_ipv6_address=lambda _:(_ for _ in ()).throw(ValueError('injected after socket create'))
        with self.assertRaises(ValueError):e.init_process_group(self.metadata())
        self.assertTrue(e.poisoned);self.assertEqual(len(e.context_quarantine),2)
        with self.assertRaises(RuntimeError):e.finalize()
        with self.assertRaises(RuntimeError):e.shutdown_managed_context()
        # Test-only isolation cleanup; production keeps these in quarantine.
        context,sock=e.context_quarantine;sock.close(linger=0);context.term()
        e.pull_socket.close(linger=0);e.pull_context.term()

    def test_partial_prepare_poison_survives_different_manager_reference(self):
        engine=self.m.ProductionManagedNixl();manager_a=engine;manager_b=engine
        engine.inject_prepare_failure=True
        with self.assertRaisesRegex(RuntimeError,'partial registration'):manager_a.prepare()
        self.assertTrue(engine.poisoned);self.assertFalse(engine.active)
        engine.inject_prepare_failure=False
        with self.assertRaises(RuntimeError):manager_b.prepare()
        self.assertTrue(engine.poisoned)
        engine.pull_socket.close(linger=0);engine.pull_context.term()  # isolated mock cleanup

    def test_failed_finalization_cannot_retry(self):
        e=self.m.ProductionManagedNixl();e.prepare();e.init_process_group(self.metadata());e._claim_step(0);e.finished=True
        e.authorize_finalize(0,'completion-token-1234')
        # Force ownership check failure after the one-shot gate is armed.
        sock=e.agent.zmq_clients.pop('peer')
        with self.assertRaises(RuntimeError):e.finalize()
        self.assertTrue(e.poisoned);self.assertTrue(e.finalization_started)
        e.agent.zmq_clients['peer']=sock
        with self.assertRaises(RuntimeError):e.finalize()
        context,sock=e.peer_contexts['peer'];sock.close(linger=0);context.term()
        e.pull_socket.close(linger=0);e.pull_context.term()


if __name__=='__main__':unittest.main()
