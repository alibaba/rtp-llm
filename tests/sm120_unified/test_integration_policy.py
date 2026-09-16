"""CPU-only integration contracts; executes real selected source functions.

No claim of CUDA kernel correctness: tensor math, imports/JIT and service profiles
have separate qualification gates. Run with CUDA_VISIBLE_DEVICES=.
"""
from __future__ import annotations
import ast
import os
from pathlib import Path
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
ARCH = 'rtp_llm/models_py/utils/arch.py'
FP8 = 'rtp_llm/models_py/modules/dsv4/fp8/'
LINEAR = 'rtp_llm/models_py/modules/factory/linear/impl/cuda/'


def source_fn(path, name, ns, cls=None):
    """Compile the actual function, injecting only its dependencies."""
    nodes = ast.parse((ROOT / path).read_text()).body
    if cls:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    mod = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), str(ROOT/path), 'exec'), ns)
    return ns[name]


class IntegrationPolicyTest(unittest.TestCase):
    def test_native_dense_default_off_and_exact_arch(self):
        fn = source_fn(ARCH, 'sm120_native_fp8_enabled', {'os': os, 'is_sm120': lambda d: d == 120})
        for flag in ('0', '1'):
            for arch in (90, 100, 120, 121):
                with self.subTest(flag=flag, arch=arch), patch.dict(os.environ, {'DSV4_SM120_NATIVE_FP8': flag}, clear=True):
                    self.assertEqual(fn(arch), flag == '1' and arch == 120)

    def test_native_dense_can_handle_does_not_open_other_sm12x(self):
        torch = types.SimpleNamespace(float8_e4m3fn='fp8', float8_e4m3fnuz='fp8uz')
        ns = {'torch': torch, 'os': os, 'is_sm120': lambda d: d == 120, 'is_sm12x': lambda d: d in (120, 121)}
        source_fn(ARCH, 'sm120_native_fp8_enabled', ns)
        fn = source_fn(LINEAR+'fp8_deepgemm_linear.py', 'can_handle', ns, cls='CudaFp8DeepGEMMLinear')
        for flag in ('0', '1'):
            for arch in (90, 100, 120, 121):
                with self.subTest(flag=flag, arch=arch), patch.dict(os.environ, {'DSV4_SM120_NATIVE_FP8':flag}, clear=True):
                    w=types.SimpleNamespace(device=arch, dtype='fp8')
                    q=types.SimpleNamespace(get_method=lambda:'FP8_PER_BLOCK')
                    self.assertEqual(fn(None,q,w,object()), arch in (90,100) or (flag=='1' and arch==120))
                    self.assertFalse(fn(None,q,w,None))

    def test_actual_factory_registration_selects_only_one_sm120_backend(self):
        tree=ast.parse((ROOT/(LINEAR+'__init__.py')).read_text())
        outer=next(n for n in tree.body if isinstance(n,ast.If) and ast.unparse(n.test)=='is_cuda()')
        node=next(n for n in outer.body if isinstance(n,ast.If) and ast.unparse(n.test)=='is_sm120()')
        class RemoveImports(ast.NodeTransformer):
            def visit_ImportFrom(self,node): return ast.Pass()
        node=RemoveImports().visit(node)
        for native in (False,True):
            calls=[]
            ns={'is_sm120':lambda:True,'sm120_native_fp8_enabled':lambda:native,
                'LinearFactory':types.SimpleNamespace(register=calls.append),
                'CudaFp8DeepGEMMLinear':'native', 'CudaFp8VllmBlockwiseLinear':'baseline'}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])), '<factory>', 'exec'),ns)
            self.assertEqual(calls,['native' if native else 'baseline'])

    def test_paged_native_rejects_wrong_geometry_and_missing_provider(self):
        ns={'os':os,'_HAS_DEEP_GEMM':True}
        fn=source_fn(FP8+'_indexer_score.py','sm120_paged_deepgemm_ready',ns)
        with patch.dict(os.environ,{},clear=True):
            self.assertFalse(fn(32)); self.assertFalse(fn(64))
        with patch.dict(os.environ,{'DSV4_INDEXER_FP8_DEEPGEMM_PAGED':'1'},clear=True):
            self.assertTrue(fn(64))
            for page in (0,2,32,128,256):
                with self.assertRaises(ValueError): fn(page)
            ns['_HAS_DEEP_GEMM']=False
            with self.assertRaises(RuntimeError):fn(64)

    def test_shared_fusion_requires_native_scale_path(self):
        ns={'os':os,'is_sm120':lambda d:d==120,'sm120_native_fp8_enabled':lambda d:os.environ.get('DSV4_SM120_NATIVE_FP8')=='1'}
        fn=source_fn('rtp_llm/models_py/modules/dsv4/moe/shared_expert.py','_requires_sm120_linear',ns)
        x=types.SimpleNamespace(is_cuda=True,device=120)
        with patch.dict(os.environ,{},clear=True):self.assertTrue(fn(x))
        with patch.dict(os.environ,{'DSV4_SM120_SHARED_EXPERT_FUSED':'1'},clear=True):
            with self.assertRaises(RuntimeError):fn(x)
            os.environ['DSV4_SM120_NATIVE_FP8']='1'
            self.assertFalse(fn(x))

    def test_baseline_scale_cache_not_installed_on_native_linear(self):
        ns={'os':os,'is_sm120':lambda d:True, 'sm120_native_fp8_enabled':lambda d:True}
        fn=source_fn('rtp_llm/models_py/modules/dsv4/utils.py','_enable_sm120_cached_weight_scale',ns)
        x=types.SimpleNamespace(weight=types.SimpleNamespace(is_cuda=True,device=120))
        self.assertIs(fn(x),x)  # missing weight_scales would expose a wrong fallthrough

    def test_woa_and_warmup_choose_same_arm_at_invocation(self):
        def decision(path,method,needle,cls=None):
            ns={'os':os,'is_sm120':lambda d:True}
            tree=ast.parse((ROOT/path).read_text())
            nodes=tree.body if not cls else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls).body
            # Locate by method anywhere in the file; class name is not part of the contract.
            f=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==method)
            test=next(n.test for n in ast.walk(f) if isinstance(n,ast.If) and needle in ast.unparse(n.test))
            return eval(compile(ast.Expression(test),'<actual-guard>','eval'),dict(ns,device=120,o_fp8=types.SimpleNamespace(device=120)))
        for flag in ('0','1'):
            with patch.dict(os.environ,{'DSV4_SM120_WOA_EINSUM':flag},clear=True):
                self.assertEqual(decision(FP8+'attention.py','_wo_a_einsum_from_fp8','DSV4_SM120_WOA_EINSUM'),flag=='0')
                self.assertEqual(decision('rtp_llm/models_py/modules/dsv4/dsv4_kernel_jit_warmup.py','warmup_batched_fp8_einsum_jit','DSV4_SM120_WOA_EINSUM'),flag=='0')

    def test_authoritative_rank_layout_profiles(self):
        import importlib.util,sys
        p=ROOT/'rtp_llm/models_py/distributed/rank_layout.py'
        spec=importlib.util.spec_from_file_location('_unified_rank_layout',p)
        mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
        for pp,dp,tp in ((4,1,2),(2,1,4),(1,4,1)):
            layout=mod.RankLayout(pp_size=pp,dp_size=dp,tp_size=tp)
            self.assertEqual([layout.world_rank_of(layout.coord_of(r)) for r in range(pp*dp*tp)],list(range(pp*dp*tp)))
        layout=mod.RankLayout(pp_size=2,dp_size=1,tp_size=4)
        self.assertEqual(layout.groups(mod.Group.STAGE),[[0,1,2,3],[4,5,6,7]])
        self.assertEqual([layout.ep_rank_of(r,4) for r in range(8)],[0,1,2,3,0,1,2,3])

    def test_wire_hint_is_not_duplicated_by_automatic_merge(self):
        text=(ROOT/'rtp_llm/cpp/models/ModelTypes.h').read_text()
        enum=text.split('enum GptModelInputIndex')[1].split('};')[0]
        fields=[line.split('//')[0].strip().rstrip(',') for line in enum.splitlines()]
        self.assertEqual(fields.count('pdSeparation'),1)
        text=(ROOT/'rtp_llm/cpp/models/ModelTypes.cc').read_text()
        self.assertEqual(text.count('shape_hints[GptModelInputIndex::pdSeparation]'),1)
        self.assertEqual(text.count('inputs.pd_separation                   ='),1)


if __name__=='__main__':
    unittest.main(verbosity=2)
