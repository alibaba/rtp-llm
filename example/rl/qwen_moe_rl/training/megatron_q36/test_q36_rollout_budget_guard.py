import asyncio
import types
import unittest
from q36_rollout_budget_guard import install


class BudgetGuardTests(unittest.TestCase):
    def test_preserves_valid_request_and_rejects_oversized_requests(self):
        calls = []
        class Server:
            async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
                calls.append((request_id, prompt_ids, sampling_params, kwargs)); return 'ok'
        install(types.SimpleNamespace(RTPServer=Server))
        server = Server()
        self.assertEqual(asyncio.run(server.generate('a', prompt_ids=[1]*400,
                          sampling_params={'max_tokens': 512, 'temperature': 1})), 'ok')
        self.assertEqual(calls[0][2]['max_tokens'], 512)
        with self.assertRaisesRegex(ValueError, 'response budget'):
            asyncio.run(server.generate('b', prompt_ids=[1], sampling_params={'max_tokens': 30720}))
        with self.assertRaisesRegex(ValueError, '8192'):
            asyncio.run(server.generate('c', prompt_ids=[1]*8000, sampling_params={'max_tokens': 512}))
        self.assertEqual(len(calls), 1)
