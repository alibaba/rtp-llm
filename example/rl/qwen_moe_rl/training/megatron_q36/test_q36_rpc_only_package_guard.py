import hashlib
import os
from pathlib import Path
import pickle
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import q36_rpc_only_package_guard as guard


class GuardTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); p=self.root/'rtp_llm';p.mkdir();init=p/'__init__.py';init.write_text('# locked\n')
        self.module=types.ModuleType('rtp_llm');self.module.__file__=str(init);self.module.__path__=[str(p)]
        self.calls=[]
        def original(name):
            self.calls.append(name)
            if name in guard.ALLOWED_LAZY:return 'known-'+name
            raise RuntimeError('unexpected native import')
        self.module.__getattr__=original
        self.patches=[patch.dict(os.environ,{'Q36_USE_RPC_OVERLAY':'1'}),patch.object(sys,'version_info',(3,12,0)),
                      patch.object(guard,'RTP_INIT_SHA256',hashlib.sha256(init.read_bytes()).hexdigest()),
                      patch.dict(sys.modules,{'rtp_llm':self.module})]
        for p in self.patches:p.start();self.addCleanup(p.stop)

    def test_unknown_attribute_is_standard_and_known_lazy_delegates(self):
        guard.install(self.root);guard.install(self.root)
        with self.assertRaises(AttributeError):getattr(self.module,'arbitrary_non_dunder_name')
        self.assertEqual(getattr(self.module,'_ft_pickler'),'known-_ft_pickler')
        self.assertEqual(self.calls,['_ft_pickler'])
        self.assertNotIn('rtp_llm.ops',sys.modules)

    def test_actual_pickle_walks_module_without_native_import(self):
        guard.install(self.root)
        def q36_unique_rpc_pickle_probe():return 1
        q36_unique_rpc_pickle_probe.__module__=None
        q36_unique_rpc_pickle_probe.__qualname__='q36_unique_rpc_pickle_probe'
        host=types.ModuleType('zz_q36_pickle_probe_host')
        host.q36_unique_rpc_pickle_probe=q36_unique_rpc_pickle_probe
        with patch.dict(sys.modules,{'zz_q36_pickle_probe_host':host}):
            payload=pickle.dumps(q36_unique_rpc_pickle_probe)
            self.assertIs(pickle.loads(payload),q36_unique_rpc_pickle_probe)
        self.assertNotIn('rtp_llm.ops',sys.modules)
        self.assertNotIn('q36_unique_rpc_pickle_probe',self.calls)

    def test_wrong_profile_source_or_native_import_fails_closed(self):
        with patch.dict(os.environ,{'Q36_USE_RPC_OVERLAY':'0'}):
            with self.assertRaises(RuntimeError):guard.install(self.root)
        with patch.object(guard,'RTP_INIT_SHA256','0'*64):
            with self.assertRaises(RuntimeError):guard.install(self.root)
        with patch.dict(sys.modules,{'rtp_llm.ops':types.ModuleType('rtp_llm.ops')}):
            with self.assertRaises(RuntimeError):guard.install(self.root)

if __name__=='__main__':unittest.main()
