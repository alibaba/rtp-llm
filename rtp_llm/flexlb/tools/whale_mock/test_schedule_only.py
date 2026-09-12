from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

from rtp_llm.flexlb.tools.whale_mock.schedule_only import acknowledge


class ScheduleOnlyTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(__file__).resolve().parents[3] / 'dash_sc' / 'proto'
        subprocess.run([sys.executable, '-m', 'grpc_tools.protoc', '-I', str(root),
                        '--python_out=' + cls.temp.name,
                        str(root / 'model_config.proto'), str(root / 'predict_v2.proto')], check=True)
        sys.path.insert(0, cls.temp.name)
        import predict_v2_pb2
        cls.proto = predict_v2_pb2

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(cls.temp.name)
        cls.temp.cleanup()

    async def call(self, visitor, value):
        package = types.ModuleType('rtp_llm.dash_sc.proto')
        package.predict_v2_pb2 = self.proto
        with patch.dict(sys.modules, {'rtp_llm.dash_sc.proto': package}):
            return await acknowledge(visitor, value, types.SimpleNamespace(id='copied', model_name='mock'))

    async def test_ack_preserves_request_and_never_fetches_or_claims_completion(self):
        value = types.SimpleNamespace(prompt_length=100, request_id=123,
                                      enqueued_by_master=True,
                                      generate_config=types.SimpleNamespace(validate=Mock(), max_new_tokens=393216))
        visitor = types.SimpleNamespace(fill_request_info=Mock(), route_ips=AsyncMock(), enqueue=AsyncMock())
        response = await self.call(visitor, value)
        visitor.route_ips.assert_awaited_once_with(value)
        visitor.enqueue.assert_not_called()
        self.assertEqual(393216, value.generate_config.max_new_tokens)
        self.assertTrue(response.infer_response.parameters['schedule_accepted'].bool_param)
        self.assertFalse(response.infer_response.parameters['inference_completed'].bool_param)
        self.assertEqual(0, len(response.infer_response.outputs))
        self.assertEqual(0, len(response.infer_response.raw_output_contents))

    async def test_non_batch_is_not_accepted(self):
        value = types.SimpleNamespace(prompt_length=100, request_id=123, enqueued_by_master=False,
                                      generate_config=types.SimpleNamespace(validate=Mock()))
        visitor = types.SimpleNamespace(fill_request_info=Mock(), route_ips=AsyncMock())
        with self.assertRaisesRegex(ValueError, 'BATCH'):
            await self.call(visitor, value)

    async def test_route_failure_is_propagated(self):
        value = types.SimpleNamespace(prompt_length=100, request_id=123,
                                      generate_config=types.SimpleNamespace(validate=Mock()))
        visitor = types.SimpleNamespace(fill_request_info=Mock(), route_ips=AsyncMock(side_effect=TimeoutError('route')))
        with self.assertRaises(TimeoutError):
            await self.call(visitor, value)
