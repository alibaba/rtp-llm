"""Synthetic status contract: no native server or GPU is started."""
import asyncio
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


class ResidentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        (root/'config.json').write_text('{"test":"model"}')
        adapter = Path(os.environ['Q36_RTP_ADAPTER_SOURCE_DIR']).resolve()
        original = types.ModuleType('standalone_rtp_adapter_v4')
        original.__file__ = str(adapter/'standalone_rtp_adapter_v4.py')
        events=[]
        class Replica:
            async def release_kv_cache(self): events.append('real-release')
            async def resume_kv_cache(self): events.append('real-restore')
        original.StandaloneRTPReplica=Replica
        original.StandaloneRTPAdapter=object
        native=types.ModuleType('rtp_native_server')
        async def control(replica): return self.control
        native._new_control=control
        env={'Q36_ADAPTER_DIR':str(adapter),'RTP_MODEL_PATH':str(root),
             'Q36_ONPOLICY_NO_CONCURRENT_ROLLOUT':'1','RLLM_RTP_KEEP_KV_CACHE':'1',
             'RTP_LLM_RL_MODE':'1','RTP_RL_SHARED_BACKING':'0','RTP_TP_SIZE':'4',
             'RTP_EP_SIZE':'1','RLLM_RTP_GPUS_PER_NODE':'4','RLLM_RTP_EXPECTED_REPLICAS':'1'}
        self.ep=patch.dict(os.environ,env); self.ep.start();self.addCleanup(self.ep.stop)
        self.pp=patch.object(sys,'path',[str(adapter)]+sys.path);self.pp.start();self.addCleanup(self.pp.stop)
        self.mp=patch.dict(sys.modules,{'standalone_rtp_adapter_v4':original,'rtp_native_server':native})
        self.mp.start();self.addCleanup(self.mp.stop)
        path=adapter/'standalone_rtp_adapter_v5.py'
        spec=importlib.util.spec_from_file_location('private_resident_v5',path)
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        self.module.MODEL_CONFIG_SHA256=hashlib.sha256((root/'config.json').read_bytes()).hexdigest()
        self.replica=object.__new__(self.module.StandaloneRTPReplica)
        self.replica.world_size=4;self.replica.replica_rank=0;self.events=events
        self.boot=1;self.incarnation='boot-uuid';self.version='run:0';self.step=0
        async def pause():
            self.events.append('pause')
            return {'paused':True,'kv_pool_retained':True,'native_processes_released':False,
                    'native_boot':self.boot,'native_incarnation':self.incarnation}
        self.replica._server_handle=types.SimpleNamespace(pause_for_training=types.SimpleNamespace(remote=pause))
        outer=self
        class Control:
            async def status(self):
                outer.events.append('status')
                return [types.SimpleNamespace(protocol_version=2,tp_rank=i,tp_size=4,phase='paused',
                    quiescent=True,kv_cache_ready=True,kv_cache_bytes=1024,model_version=outer.version,
                    global_step=outer.step,update_id='',torch_allocated_bytes=2048,torch_reserved_bytes=4096)
                    for i in range(4)]
            async def close(self): outer.events.append('close')
        self.control=Control()

    def test_disabled_profile_preserves_original_release_and_restore(self):
        async def run():
            await self.replica.release_kv_cache();await self.replica.resume_kv_cache()
        with patch.dict(os.environ, {'Q36_DISAGG_RESIDENT_KV':'0'}): asyncio.run(run())
        self.assertEqual(self.events,['real-release','real-restore'])

    def test_invalid_resident_flag_never_calls_release(self):
        async def run():
            with self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
                await self.replica.release_kv_cache()
            with self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
                await self.replica.resume_kv_cache()
        with patch.dict(os.environ, {'Q36_DISAGG_RESIDENT_KV':'invalid'}): asyncio.run(run())
        self.assertEqual(self.events,[])

    def test_resident_protocol_before_and_after_commit(self):
        async def run():
            await self.replica.release_kv_cache()
            self.version='run:1';self.step=1
            await self.replica.resume_kv_cache()
        with patch.dict(os.environ, {'Q36_DISAGG_RESIDENT_KV':'1'}): asyncio.run(run())
        self.assertEqual(self.events,['pause','status','close','pause','status','close'])
        self.assertIsNone(self.replica._resident_kv_lease)

    def test_zero_boot_is_valid_and_missing_or_negative_is_rejected(self):
        async def run():
            self.boot=0
            await self.replica.release_kv_cache()
            self.assertEqual(self.replica._resident_kv_lease['native_boot'],0)
            self.replica._resident_kv_lease=None
            for bad in (None,-1,True):
                self.boot=bad
                with self.subTest(boot=bad), self.assertRaisesRegex(RuntimeError,'live resident paused replica'):
                    await self.replica.release_kv_cache()
        with patch.dict(os.environ, {'Q36_DISAGG_RESIDENT_KV':'1'}): asyncio.run(run())

    def test_missing_profile_and_unpaired_restore_fail_closed(self):
        async def run():
            with patch.dict(os.environ, {'RLLM_RTP_KEEP_KV_CACHE':'0'}):
                with self.assertRaisesRegex(RuntimeError,'explicit Q36'):
                    await self.replica.release_kv_cache()
            with self.assertRaisesRegex(RuntimeError,'lease is missing'):
                await self.replica.resume_kv_cache()
        with patch.dict(os.environ, {'Q36_DISAGG_RESIDENT_KV':'1'}): asyncio.run(run())
        self.assertEqual(self.events,[])

    def test_rank_bad_or_uncommitted_policy_never_restores(self):
        async def run():
            await self.replica.release_kv_cache()
            with self.assertRaisesRegex(RuntimeError,'did not follow an update'):
                await self.replica.resume_kv_cache()
            self.version='run:1'; self.incarnation='different-process'
            with self.assertRaisesRegex(RuntimeError,'same native incarnation'):
                await self.replica.resume_kv_cache()
        with patch.dict(os.environ, {'Q36_DISAGG_RESIDENT_KV':'1'}): asyncio.run(run())
        self.assertNotIn('real-release',self.events)

    def test_status_rejects_missing_rank_and_unquiescent(self):
        async def run():
            rows=await self.control.status()
            with self.assertRaisesRegex(RuntimeError,'every TP rank'): self.module._rank_proof(rows[:-1],4)
            rows[2].quiescent=False
            with self.assertRaisesRegex(RuntimeError,'quiescent'): self.module._rank_proof(rows,4)
            rows[2].quiescent=True; rows[3].model_version='other'
            with self.assertRaisesRegex(RuntimeError,'disagree'): self.module._rank_proof(rows,4)
        asyncio.run(run())


if __name__=='__main__':unittest.main()
