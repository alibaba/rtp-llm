"""Run the actual peer methods with fixed in-memory V6D registration state."""
import ast
import copy
import enum
import os
import pathlib
import sys
import types
import unittest


class Mode(enum.Enum):
    DISABLED = 'disabled'
    REPORT_EVENT_ONLY = 'report-event-only'
    REPORT_EVENT_AND_CACHE_TRANSFER = 'report-event-and-cache-transfer'


class BasePeer:
    def get_peer_info(self):
        return {'peer_address': 'http://isolated-v6d', 'base_field': True}


def load_peer(path):
    tree = ast.parse(path.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TieredVineyardPeer')
    methods = [copy.deepcopy(n) for n in original.body if isinstance(n, ast.FunctionDef)
               and n.name in ['get_peer_info', 'get_tair_kvcm_query_config']]
    assert len(methods) == 2
    cls = ast.ClassDef(name='PeerUnderTest', bases=[ast.Name(id='BasePeer', ctx=ast.Load())],
                       keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    env = {'BasePeer': BasePeer, 'TairKVCMMode': Mode}
    exec(compile(module, str(path), 'exec'), env)
    return env['PeerUnderTest']


class PeerQueryMetadataTest(unittest.TestCase):
    def make_peer(self, mode=Mode.REPORT_EVENT_AND_CACHE_TRANSFER, ready=True):
        peer = PEER()
        peer.config = types.SimpleNamespace(tair_kvcm_mode=mode)
        peer.peer_type = 'vineyard'
        peer.socket_path = '/isolated/vineyard.sock'
        peer._kvcs_tier = None
        peer._is_lazy_strategy = lambda: False
        peer._is_tair_kvcm_enabled = lambda: mode != Mode.DISABLED
        peer._is_tair_kvcm_report_event_ready = lambda: ready
        peer._is_tair_kvcm_cache_transfer_ready = lambda: ready
        peer._tair_kvcm_store = types.SimpleNamespace(spec_received=True, client=types.SimpleNamespace(
            is_group_aware=True, instance_id='native_instance_4096',
            manager_endpoint='spectrum://v-isolated:6382', block_size=4096, node_count=4,
            _manager_client_options={'discovery_refresh_interval_seconds': 30},
            group_specs_by_id={1: types.SimpleNamespace(spec_type='MambaSpec', object_size_bytes=4096),
                               0: types.SimpleNamespace(spec_type='FullAttentionSpec', object_size_bytes=512)},
            spec_name_by_group_node={(group, rank): f'actual_group_{group}_N{rank}'
                                     for group in range(2) for rank in range(4)},
        ))
        return peer

    def test_eager_mode_exports_actual_registration_and_all_pp_spec_names(self):
        info = self.make_peer().get_peer_info()
        config = info['tair_kvcm_query_config']
        self.assertTrue(info['base_field'])
        self.assertNotIn('tair_kvcm', info)  # Existing lazy transfer initialization is untouched.
        self.assertEqual(config['manager_endpoint'], 'spectrum://v-isolated:6382')
        self.assertEqual(config['instance_id'], 'native_instance_4096')
        self.assertEqual(config['hash_block_size'], 4096)
        self.assertEqual(config['node_count'], 4)
        self.assertEqual(config['discovery_refresh_interval_ms'], 30000)
        self.assertEqual([g['group_id'] for g in config['groups']], [0, 1])
        self.assertEqual(config['groups'][1]['location_spec_names'],
                         [f'actual_group_1_N{rank}' for rank in range(4)])

    def test_disabled_and_report_only_never_expose_remote_query_config(self):
        for mode in [Mode.DISABLED, Mode.REPORT_EVENT_ONLY]:
            peer = self.make_peer(mode)
            peer._tair_kvcm_store = None
            self.assertNotIn('tair_kvcm_query_config', peer.get_peer_info())

    def test_query_configuration_does_not_depend_on_transfer_startup_liveness(self):
        info = self.make_peer(ready=False).get_peer_info()
        self.assertIn('tair_kvcm_query_config', info)
        self.assertFalse(info['kvcm_status']['cache_transfer_ready'])

    def test_unregistered_or_incomplete_pp_layout_is_not_exported(self):
        peer = self.make_peer()
        peer._tair_kvcm_store.spec_received = False
        self.assertNotIn('tair_kvcm_query_config', peer.get_peer_info())
        peer = self.make_peer()
        del peer._tair_kvcm_store.client.spec_name_by_group_node[(1, 3)]
        self.assertNotIn('tair_kvcm_query_config', peer.get_peer_info())

    def test_getter_does_not_expose_mutable_registration_maps(self):
        peer = self.make_peer()
        result = peer.get_peer_info()['tair_kvcm_query_config']
        result['groups'][0]['location_spec_names'][0] = 'changed'
        self.assertEqual(peer._tair_kvcm_store.client.spec_name_by_group_node[(0, 0)],
                         'actual_group_0_N0')


if __name__ == '__main__':
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        source_path = pathlib.Path(sys.argv.pop(1))
    else:
        source_path = (pathlib.Path(os.environ['TEST_SRCDIR']) / os.environ['TEST_WORKSPACE'] /
                       'aios/navi_sched/packaging/dashllm_r50/runtime_overlay/v6d/server/peers/tiered_vineyard/peer.py')
    PEER = load_peer(source_path)
    unittest.main()
