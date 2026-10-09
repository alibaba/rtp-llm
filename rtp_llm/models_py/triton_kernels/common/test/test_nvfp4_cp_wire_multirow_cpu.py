"""Execute the actual R8/R16 scatter AST against the independent page-layout oracle."""
import functools
import importlib.util
from pathlib import Path
import types
import unittest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('wire_reference_cpu', HERE / 'test_nvfp4_cp_wire_cpu.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
MULTI = HERE.parent / 'nvfp4_cp_wire.py'
ROWS_PER_CTA = 8


def kernels():
    tl = old.TL()
    ns = {'tl': tl}
    old.load_ast(old.fn(old.CACHE, '_scale_128x4_offset'), ns)
    old.load_ast(old.fn(MULTI, '_scatter_wire_multirow'), ns)
    return tl, functools.partial(ns['_scatter_wire_multirow'], ROWS_PER_CTA=ROWS_PER_CTA)


old.kernels = kernels


class MultirowR8Tests(old.WireTests):
    def setUp(self):
        global ROWS_PER_CTA
        ROWS_PER_CTA = 8


class MultirowR16Tests(old.WireTests):
    def setUp(self):
        global ROWS_PER_CTA
        ROWS_PER_CTA = 16


class MultirowWrapperTests(unittest.TestCase):
    def test_launch_contract_r8_r16_and_default_layout(self):
        ns, _, _ = old.wrappers()
        launch = old.Launch()
        ns.update({'triton': types.SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
                   '_scatter_wire_multirow': launch})
        old.load_ast(old.fn(MULTI, 'scatter_cp_nvfp4_wire'), ns)
        maps = [old.MetaTensor((129,), 'i64') for _ in range(3)]
        for rows in (8, 16):
            for fi in (False, True):
                ns['scatter_cp_nvfp4_wire'](old.MetaTensor((512, 648)), *maps,
                    old.planes(5), old.planes(3), rows_per_cta=rows, fi_working_layout=fi)
                grid, args, kw = launch.calls[-1]
                self.assertEqual(grid, ((129 + rows - 1) // rows, 9))
                self.assertEqual(kw['ROWS_PER_CTA'], rows)
                self.assertEqual(kw['FI_WORKING_LAYOUT'], fi)
                self.assertTrue(all(t.dtype == 'u8' for t in args[4:16]))
        for invalid in (1, 4, 32, True):
            with self.assertRaises(ValueError):
                ns['scatter_cp_nvfp4_wire'](old.MetaTensor((512, 648)), *maps,
                    old.planes(5), old.planes(3), rows_per_cta=invalid)


if __name__ == '__main__':
    unittest.main()
