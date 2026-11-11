"""Exercise the actual import hook against minimal importable module files."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


class ImportOrderTest(unittest.TestCase):
    def test_preloaded_and_later_imports_are_wrapped(self):
        code_dir=Path(__file__).resolve().parent
        for order in ('before','after'):
            with self.subTest(order=order),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                for path in ('rllm/__init__.py','rllm/trainer/__init__.py','rllm/trainer/verl/__init__.py'):
                    p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('')
                (root/'rllm/trainer/unified_trainer.py').write_text(
                    'class UnifiedTrainer:\n def __init__(self): pass\n def shutdown(self): pass\n')
                (root/'rllm/trainer/verl/verl_launcher.py').write_text(
                    'class VerlTaskRunner:\n def run(self): pass\n')
                script=textwrap.dedent('''
                    import importlib,sys,types
                    import q36_nonshared_worker_bootstrap as b
                    sys.modules['full_tensor_refit']=types.SimpleNamespace(worker_setup=lambda:None)
                    names=('rllm.trainer.unified_trainer','rllm.trainer.verl.verl_launcher')
                    if sys.argv[1]=='before':
                        for name in names:importlib.import_module(name)
                    b.worker_setup()
                    u=importlib.import_module(names[0]);v=importlib.import_module(names[1])
                    assert u.UnifiedTrainer.__init__._q36_managed_outer
                    assert u.UnifiedTrainer.shutdown._q36_managed_outer
                    assert v.VerlTaskRunner.run._q36_managed_outer
                ''')
                env=dict(os.environ,Q36_CODE_DIR=str(code_dir),Q36_USE_MANAGED_NIXL='1',
                         PYTHONPATH=str(root)+os.pathsep+str(code_dir))
                p=subprocess.run([sys.executable,'-c',script,order],env=env,text=True,capture_output=True)
                self.assertEqual(p.returncode,0,p.stderr)


if __name__=='__main__':unittest.main()
