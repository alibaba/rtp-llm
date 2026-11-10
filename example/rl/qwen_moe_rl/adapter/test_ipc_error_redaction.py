import ast
from pathlib import Path
import re
import unittest

source = Path(__file__).with_name('standalone_rtp_adapter_v3.py').read_text()
tree = ast.parse(source)
fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == '_safe_ipc_error')
scope = {'re': re}
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<isolated>', 'exec'), scope)
safe = scope['_safe_ipc_error']

class IpcErrorRedactionTest(unittest.TestCase):
    def test_ninja_message_is_preserved(self):
        msg = 'Ninja is required to load C++ extensions (pip install ninja to get it)'
        self.assertEqual(safe(RuntimeError(msg)), {'text':msg, 'cuda_error_code':None})

    def test_driver_status_is_extracted(self):
        msg = 'cuMemGetAddressRange failed (1); IPC v2 requires a CUDA allocation exportable by cudaIpcGetMemHandle'
        self.assertEqual(safe(RuntimeError(msg))['cuda_error_code'],1)

    def test_tensor_handle_and_arbitrary_error_are_redacted(self):
        for msg in ('tensor([1,2,3]) ipc_handle=deadbeef',
                    'cudaIpcGetMemHandle: tensor 12345', 'Error building extension secret payload'):
            result = safe(RuntimeError(msg))
            self.assertEqual(result['text'],'[redacted: unrecognized IPC error]')
            self.assertNotIn('12345',str(result))
            self.assertNotIn('deadbeef',str(result))

if __name__ == '__main__': unittest.main()
