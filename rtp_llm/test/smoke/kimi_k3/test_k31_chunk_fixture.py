"""Exercise the real KV planner through the smoke-only chunk fixture."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import k31_chunk_fixture as fixture


class ChunkFixtureTest(unittest.TestCase):
    def module(self):
        path=Path(__file__).parents[3]/'models_py/modules/factory/attention/cuda_mla_impl/dsa_kv_chunk_plan.py'
        spec=importlib.util.spec_from_file_location('fixture_planner',path)
        planner=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=planner
        # This stdlib-only test also runs on the Mac's Python 3.9.
        exec(compile('from __future__ import annotations\n'+path.read_text(),str(path),'exec'),planner.__dict__)
        module=SimpleNamespace(plan_dsa_kv_chunks=planner.plan_dsa_kv_chunks)
        class Execution:
            def __init__(self,params,name):
                self.plan=module.plan_dsa_kv_chunks(params.kv_lens_host,page_tokens=1024,
                    budget_bytes=params.workspace_owner.budgets[name],fixed_bytes=4096,
                    temporary_bytes=lambda block:block.tokens*2)
                if params.fail:raise ValueError('descriptor failed')
        module.DsaKVExecution=Execution
        return module

    def params(self,rows=1026,*,prefill=True,fp8=True,length=655363,fail=False):
        return SimpleNamespace(is_prefill=prefill,is_cuda_graph=False,
            workspace_owner=SimpleNamespace(sfa_use_fp8=fp8,budgets={'sfa':12*1024**3,'indexer':8*1024**3}),
            query_positions=SimpleNamespace(numel=lambda:rows),kv_lens_host=[length],fail=fail)

    def test_padded_cp_rows_use_real_chunks_with_original_budget(self):
        module=self.module();fixture.install(module)
        for fp8, rows in ((True, 1024), (True, 1025), (True, 1026),
                          (True, 1058), (False, 8192)):
            params=self.params(rows, fp8=fp8)
            execution=module.DsaKVExecution(params,'sfa')
            self.assertGreaterEqual(len(execution.plan.blocks),2)
            self.assertEqual(sum(block.tokens for block in execution.plan.blocks),655363)
            self.assertEqual(execution.plan.budget_bytes,12*1024**3)
            self.assertEqual(params.workspace_owner.budgets['sfa'],12*1024**3)

    def test_other_phases_shapes_and_exception_do_not_inherit_cap(self):
        module=self.module();fixture.install(module);fixture.install(module)
        for params,name in ((self.params(),'indexer'),(self.params(prefill=False),'sfa'),
                            (self.params(rows=8192),'sfa'),
                            (self.params(length=65536),'sfa')):
            self.assertEqual(len(module.DsaKVExecution(params,name).plan.blocks),1)
        with self.assertRaisesRegex(ValueError,'descriptor failed'):
            module.DsaKVExecution(self.params(fail=True),'sfa')
        plan=module.plan_dsa_kv_chunks([655363],page_tokens=1024,budget_bytes=12*1024**3,
            fixed_bytes=4096,temporary_bytes=lambda block:block.tokens*2)
        self.assertEqual(len(plan.blocks),1)

    def test_new_worker_bootstraps_fixture_before_loading_model_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            package=root/'packages'
            path=package
            for name in fixture.MODULE.split('.')[:-1]:
                path=path/name;path.mkdir(parents=True,exist_ok=True)
                (path/'__init__.py').touch()
            (path/'dsa_kv_chunk_execution.py').write_text(
                'def plan_dsa_kv_chunks(*args,**kwargs): pass\n'
                'class DsaKVExecution:\n def __init__(self,*args): pass\n')
            env={**os.environ,'PYTHONPATH':str(package)}
            env.update(fixture.fixture_environment(root,env))
            child=subprocess.check_output([sys.executable,'-c',
                f'import {fixture.MODULE} as m; print(m.DsaKVExecution._k31_smoke_chunk_fixture)'],
                cwd=root,env=env,text=True)
            self.assertEqual(child.strip(),'True')


if __name__=='__main__':unittest.main()
