"""Ray 2.59 session-source contract without initializing Ray or CUDA."""
import ast
from pathlib import Path
import types
import unittest

HERE = Path(__file__).resolve().parent
FILES = ('run_standalone_q36_candidate_v3.py', 'run_standalone_q36_ep8_v2_candidate.py')


def load_checker(path):
    source = path.read_text()
    tree = ast.parse(source)
    func = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'validate_cluster')
    module = ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[]))
    context = {}
    exec(compile(module, str(path), 'exec'), context)
    return context['validate_cluster'], tree


class RaySessionTest(unittest.TestCase):
    def test_session_name_is_read_from_runtime_context(self):
        for filename in FILES:
            with self.subTest(filename=filename):
                _, tree = load_checker(HERE / filename)
                main = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'main')
                calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name) and node.func.id == 'validate_cluster']
                self.assertEqual(len(calls), 1)
                argument = calls[0].args[3]
                self.assertEqual(ast.unparse(argument), 'ray.get_runtime_context().get_session_name()')
                self.assertNotIn("address_info.get('session_name')", (HERE / filename).read_text())

    def test_real_session_must_match_and_all_nodes_must_be_authorized(self):
        for filename in FILES:
            with self.subTest(filename=filename):
                validate, _ = load_checker(HERE / filename)
                count = 3 if 'ep8' in filename else 2
                ips = [f'10.44.0.{i + 1}' for i in range(count)]
                nodes = [{'Alive': True, 'NodeManagerAddress': ip, 'Resources': {'GPU': 4}} for ip in ips]
                ray = types.SimpleNamespace(nodes=lambda: nodes)
                validate(ray, set(ips), 'session_real_owned', 'session_real_owned')
                for got in (None, '', 'session_other'):
                    with self.subTest(actual=got), self.assertRaisesRegex(RuntimeError, 'task-owned'):
                        validate(ray, set(ips), 'session_real_owned', got)
                with self.assertRaisesRegex(RuntimeError, 'unauthorized'):
                    validate(ray, set(ips[:-1]), 'session_real_owned', 'session_real_owned')
                nodes[0]['Resources']['GPU'] = 5
                with self.assertRaisesRegex(RuntimeError, 'more than four'):
                    validate(ray, set(ips), 'session_real_owned', 'session_real_owned')


if __name__ == '__main__': unittest.main()
