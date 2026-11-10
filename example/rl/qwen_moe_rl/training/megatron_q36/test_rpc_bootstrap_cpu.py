import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


path=Path(__file__).with_name('standalone_bootstrap_rpc_v6.py')
spec=importlib.util.spec_from_file_location('private_rpc_bootstrap',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class RpcBootstrapTest(unittest.TestCase):
    def test_worker_installs_proto_before_existing_bootstrap(self):
        events=[]
        with patch.object(module,'_install_proto',lambda:events.append('proto')), \
             patch.object(module,'_load_v5',lambda:types.SimpleNamespace(worker_setup=lambda:events.append('v5'))):
            module.worker_setup()
        self.assertEqual(events,['proto','v5'])

    def test_v5_bootstrap_identity_lock(self):
        root=Path(__file__).resolve().parent
        with patch.dict(os.environ,{'Q36_CODE_DIR':str(root)}),patch.object(sys,'path',[str(root)]+sys.path):
            self.assertEqual(module._load_v5().__name__,'standalone_bootstrap_v5')
        sys.modules.pop('standalone_bootstrap_v5',None)
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'standalone_bootstrap_v5.py'
            target.write_bytes((root/target.name).read_bytes()+b'\n# tampered\n')
            with patch.dict(os.environ,{'Q36_CODE_DIR':tmp}),patch.object(sys,'path',[tmp]+sys.path):
                with self.assertRaisesRegex(RuntimeError,'source identity'):module._load_v5()


if __name__=='__main__':unittest.main()
