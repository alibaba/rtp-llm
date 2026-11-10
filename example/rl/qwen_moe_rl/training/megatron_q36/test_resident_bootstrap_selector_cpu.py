import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


class SelectorTest(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).resolve().parent
        self.adapter=Path(os.environ['Q36_RTP_ADAPTER_SOURCE_DIR']).resolve()
        def load(name):
            spec=importlib.util.spec_from_file_location('private_'+name,self.root/(name+'.py'))
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
        self.selector=load('q36_standalone_bootstrap_selector')
        self.bootstrap=load('standalone_bootstrap_v5')
        env=patch.dict(os.environ,{'Q36_CODE_DIR':str(self.root),'Q36_ADAPTER_DIR':str(self.adapter)},clear=True)
        env.start();self.addCleanup(env.stop)
        paths=patch.object(sys,'path',[str(self.root),str(self.adapter)]+sys.path);paths.start();self.addCleanup(paths.stop)

    def test_opt_in_selects_v5_and_defaults_stay_v3_v4(self):
        with patch.object(self.selector.importlib,'import_module',side_effect=lambda name:name):
            self.assertEqual(self.selector.select_bootstrap(),'standalone_bootstrap_v3')
            with patch.dict(os.environ,{'Q36_ONPOLICY_NO_CONCURRENT_ROLLOUT':'1'}):
                self.assertEqual(self.selector.select_bootstrap(),'standalone_bootstrap_v4')
                with patch.dict(os.environ,{'Q36_DISAGG_RESIDENT_KV':'1'}):
                    self.assertEqual(self.selector.select_bootstrap(),'standalone_bootstrap_v5')
            with patch.dict(os.environ,{'Q36_DISAGG_RESIDENT_KV':'1'}):
                with self.assertRaisesRegex(ValueError,'no-concurrent'):self.selector.select_bootstrap()

    def test_adapter_source_lock_rejects_tampered_file(self):
        self.bootstrap._verify_v5()
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'standalone_rtp_adapter_v5.py'
            target.write_bytes((self.adapter/target.name).read_bytes()+b'\n# tamper\n')
            with patch.dict(os.environ,{'Q36_ADAPTER_DIR':tmp}),patch.object(sys,'path',[tmp]+sys.path):
                with self.assertRaisesRegex(RuntimeError,'source identity'):
                    self.bootstrap._verify_v5()


if __name__=='__main__':unittest.main()
