import unittest
from rtp_llm.flexlb.tools.mock_bundle import master

class MasterLifecycleTest(unittest.TestCase):
    def test_online_waits_for_process_then_calls_start_once(self):
        from unittest.mock import Mock, patch
        from rtp_llm.flexlb.tools.mock_bundle import master
        child = Mock()
        child.poll.return_value = None
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = b'success'
        with patch.object(master, 'get', side_effect=[OSError(), b'ok']) as ready, \
                patch.object(master.time, 'sleep'), \
                patch.object(master.urllib.request, 'urlopen', return_value=response) as start:
            master.online(child, 28080)
        self.assertEqual(ready.call_count, 2)
        start.assert_called_once_with('http://127.0.0.1:28080/hook/after_start', timeout=60)

    def test_failed_child_cannot_become_online(self):
        from unittest.mock import Mock, patch
        from rtp_llm.flexlb.tools.mock_bundle import master
        child = Mock()
        child.poll.return_value = 1
        with patch.object(master.urllib.request, 'urlopen') as start:
            with self.assertRaisesRegex(RuntimeError, 'exited'):
                master.online(child, 28080)
        start.assert_not_called()


class MasterDiscoveryTest(unittest.TestCase):
    def test_control_ports_expand_to_all_logical_workers(self):
        env = dict(MOCK_PREFILL_COUNT='2', MOCK_DECODE_COUNT='1', MOCK_PREFILL_DOMAIN='p', MOCK_DECODE_DOMAIN='d', MOCK_VIT_DOMAIN='v')
        self.assertEqual(master.expand([('10.0.0.1', 29129)], [('10.0.0.2', 22160)], env),
                         dict(p=['10.0.0.1:29129','10.0.0.1:29130'],d=['10.0.0.1:29131'],v=['10.0.0.2:22160']))
