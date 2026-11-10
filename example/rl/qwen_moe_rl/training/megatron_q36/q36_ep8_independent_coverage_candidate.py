"""Independently derive Q36 EP8 expert coverage from fixed source evidence.

CPU-only review candidate. The EP1 provider ledger and Bridge registry were
produced before the EP8 ledger under test. No model tensor payload is read.
This does not qualify the runtime, optimizer, or source transport.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

LOCKS = {
    'config': '93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99',
    'index': '41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83',
    'provider_ep1': 'd824d3ac60b76ac7699dfe36f3935580fdf4033780e3ebb9f8b311e10ac0d4e0',
    'bridge_registry': '8d13ee243a1c914d6e98d909006fefbf161af5cbef3935e71e3af5da38d58be6',
}
NAME = re.compile(r'^chunk0/decoder\.layers\.(\d+)\.mlp\.experts\.linear_fc([12])\.weight(\d+)$')


def read_locked(path, label):
    path = Path(path)
    if path.is_symlink() or not path.is_file(): raise ValueError('missing locked evidence: ' + label)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != LOCKS[label]: raise ValueError('evidence SHA mismatch: ' + label)
    return json.loads(raw)


def derive_expected(config_path, index_path, provider_path, registry_path):
    config = read_locked(config_path, 'config')
    config = config.get('text_config', config)
    actual = tuple(config.get(k) for k in ('num_hidden_layers', 'hidden_size', 'num_experts',
                                         'num_experts_per_tok', 'moe_intermediate_size'))
    if actual != (40, 2048, 256, 8, 512): raise ValueError('unexpected Q36 model geometry')
    index = read_locked(index_path, 'index')['weight_map']
    registry = read_locked(registry_path, 'bridge_registry')
    if registry.get('bridge_commit') != 'c0e164ed2aedac4ad1c877780e2564a19d5d54ec':
        raise ValueError('unexpected independently recorded Bridge revision')
    provider = read_locked(provider_path, 'provider_ep1')
    if (provider.get('binding_scope') != 'full-provider-ddp-live-source-verified' or
        (provider.get('tp'), provider.get('ep'), provider.get('etp')) != (4, 1, 4)):
        raise ValueError('independent EP1 provider evidence has wrong topology')
    old = {}
    for row in provider['parameters']:
        match = NAME.fullmatch(row.get('name', ''))
        if not match: continue
        layer, fc, expert = map(int, match.groups())
        key = (layer, fc, expert)
        if key in old: raise ValueError('duplicate independent provider expert')
        old[key] = row
    full = {(layer, fc, expert) for layer in range(40) for fc in (1, 2) for expert in range(256)}
    if set(old) != full: raise ValueError('independent provider expert coverage is incomplete')
    expected = {str(rank): {} for rank in range(8)}
    for layer, fc, global_expert in sorted(full):
        source_kind = 'gate_up_proj' if fc == 1 else 'down_proj'
        mapping_kind = 'FusedGatedExpertMapping' if fc == 1 else 'FusedExpertMapping'
        source = f'model.language_model.layers.{layer}.mlp.experts.{source_kind}'
        if source not in index: raise ValueError('expert HF source absent from fixed checkpoint index')
        bridge = registry['sources'].get(source, {})
        pattern = f'language_model.decoder.layers.{layer}.mlp.experts.linear_fc{fc}.weight*'
        if (bridge.get('mapping_type') != mapping_kind or bridge.get('hf_sources_pattern') != source or
                bridge.get('megatron_parameter_pattern') != pattern):
            raise ValueError('independent Bridge expert mapping differs from fixed checkpoint')
        previous = old[(layer, fc, global_expert)]
        mapped = previous.get('bridge_mapping', {})
        if mapped.get('type') != mapping_kind or mapped.get('source_names') != [source]:
            raise ValueError('EP1 provider expert mapping differs from independent Bridge evidence')
        axis = 0 if fc == 1 else 1
        shape = list(previous['shape'])
        if previous.get('partition_dim') != axis or previous.get('partition_stride') != 1:
            raise ValueError('unexpected qualified EP1 expert tensor partition')
        shape[axis] *= 4  # EP1 ETP4 shard -> EP8 ETP1 full individual expert.
        if shape != ([1024, 2048] if fc == 1 else [2048, 512]):
            raise ValueError('independent provider shape disagrees with fixed model config')
        rank, local = divmod(global_expert, 32)
        name = f'chunk0/decoder.layers.{layer}.mlp.experts.linear_fc{fc}.weight{local}'
        logical = 'q36/' + name.removeprefix('chunk0/')
        expected[str(rank)][name] = {'name': name, 'kind': 'expert', 'logical_name': logical,
            'shape': shape, 'layer': layer, 'local_expert': local, 'source_global_expert': global_expert,
            'tensor_kind': source_kind, 'bridge_mapping': {'type': mapping_kind, 'source_names': [source]}}
    return expected


def audit_ledger(ledger, expected):
    if (ledger.get('schema') != 'q36-ep8-live-provider-sr-v1' or
        ledger.get('binding_scope') != 'independent-source-and-live-provider-reviewed' or
        (ledger.get('tp'), ledger.get('ep'), ledger.get('etp'), ledger.get('pp')) != (4, 8, 1, 1) or
        set(ledger.get('rank_parameters', {})) != set(expected)):
        raise ValueError('unexpected EP8 ledger header or rank coverage')
    for rank, independent in expected.items():
        rows = ledger['rank_parameters'][rank]
        if not isinstance(rows, list): raise ValueError('invalid rank ledger')
        names = [row.get('name') for row in rows]
        if len(names) != len(set(names)): raise ValueError('duplicate rank parameter name')
        expert_rows = {row['name']: row for row in rows if row.get('kind') == 'expert' or
                       '.mlp.experts.' in str(row.get('name', ''))}
        if set(expert_rows) != set(independent):
            raise ValueError('rank ' + rank + ' expert names do not match independent provider coverage')
        for name, expected_row in independent.items():
            row = expert_rows[name]
            for key, value in expected_row.items():
                if row.get(key) != value:
                    raise ValueError('rank ' + rank + ' ' + name + ' mismatches independent ' + key)
    return {'ranks': 8, 'layers': 40, 'experts_per_rank': 32,
            'expert_parameters_per_rank': 2560, 'global_expert_parameters': 20480}


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'index', 'provider-ep1', 'bridge-registry', 'ep8-ledger'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    expected = derive_expected(args.config, args.index, args.provider_ep1, args.bridge_registry)
    raw = args.ep8_ledger.read_bytes()
    result = audit_ledger(json.loads(raw), expected)
    print(json.dumps({'passed': True, 'ledger_sha256': hashlib.sha256(raw).hexdigest(),
                      'independent_sources_sha256': LOCKS, **result}, sort_keys=True))


if __name__ == '__main__': main()
