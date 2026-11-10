"""CPU regression against the checked-in, pinned MCore metadata implementation."""
import ast
import re
import os
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
TE_SOURCE = Path(os.environ.get('MEGATRON_CORE_SOURCE', '/nonexistent')) / 'megatron/core/extensions/transformer_engine.py'


class Parameter:
    pass


class Module:
    def __init__(self):
        self.weight = Parameter()

    def named_parameters(self, recurse=False):
        return [('weight', self.weight)]


class MetadataTest(unittest.TestCase):
    def test_pinned_mcore_marks_single_etp_expert_weight_parallel(self):
        if not TE_SOURCE.is_file():
            self.skipTest('provide MEGATRON_CORE_SOURCE to execute pinned source metadata test')
        tree = ast.parse(TE_SOURCE.read_text())
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_set_expert_parameter_attributes')
        func.returns = None
        for arg in func.args.args:
            arg.annotation = None
        module = ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[]))
        namespace = {'_EXPERT_PARAMETER_NAME_PATTERN': re.compile(r'^(weight|bias)(?:\d+)?$')}
        exec(compile(module, str(TE_SOURCE), 'exec'), namespace)
        apply_attributes = namespace[func.name]
        for mode, axis in [('row', 1), ('column', 0)]:
            with self.subTest(mode=mode):
                expert = Module()
                apply_attributes(expert, mode, True)
                self.assertTrue(expert.weight.tensor_model_parallel)
                self.assertFalse(expert.weight.allreduce)
                self.assertEqual(expert.weight.partition_dim, axis)

    def test_v2_uses_verified_etp_group_not_tp_flag_for_experts(self):
        binder = (HERE / 'q36_ep8_sr_binder_v2.py').read_text()
        factory = (HERE / 'bf16_megatron_factory_q36_ep8_v2.py').read_text()
        self.assertIn('dist.get_world_size(ps.get_expert_tensor_parallel_group()) != 1', binder)
        self.assertIn('if not is_expert:', binder)
        self.assertNotIn("not is_expert or bool(getattr(param, 'tensor_model_parallel'", binder)
        self.assertIn('dist.get_world_size(ps.get_expert_tensor_parallel_group())!=1', factory)
        self.assertNotIn('if (parallel or found is None', factory)
        self.assertIn("source_global_expert=row.get('source_global_expert')", binder)


if __name__ == '__main__':
    unittest.main()
