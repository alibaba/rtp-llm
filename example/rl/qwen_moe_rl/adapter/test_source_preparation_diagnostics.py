import ast
from pathlib import Path
import unittest


source = Path(__file__).with_name('standalone_rtp_adapter_v3.py').read_text()
tree = ast.parse(source)
fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == '_source_preparation_failures')
scope = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<isolated>', 'exec'), scope)


class SourceDiagnosticsTest(unittest.TestCase):
    def test_report_rank_name_and_stage_without_exception_or_tensor(self):
        rows = [{'name':'model.layers.2.foo', 'error':None},
                {'name':'model.layers.2.foo', 'error':'tensor contents secret', 'error_stage':'export_ipc',
                 'error_type':'RuntimeError', 'metadata':'tensor contents secret',
                 'ipc_details':{'source_is_view':True,'source_storage_offset':4}},
                {'name':object(), 'error':'private payload', 'error_stage':'check_expected_name',
                 'error_type':'ValueError'}]
        got = scope['_source_preparation_failures'](rows)
        self.assertEqual(got, [{'rank':1,'name':'model.layers.2.foo','stage':'export_ipc','error_type':'RuntimeError',
                                'ipc_details':{'source_is_view':True,'source_storage_offset':4}},
                               {'rank':2,'name':'<invalid-name>','stage':'check_expected_name','error_type':'ValueError',
                                'ipc_details':None}])
        self.assertNotIn('secret', repr(got))
        self.assertNotIn('payload', repr(got))

    def test_no_failures_reports_empty(self):
        self.assertEqual(scope['_source_preparation_failures']([{'error':None}]), [])


if __name__ == '__main__': unittest.main()
