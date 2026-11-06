"""CPU contracts for one TP8/attention-DP9/global-EP72 Qwen3.8 policy."""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import uuid

CONTROL_TOPOLOGY = dict(world_size=72, tp_size=8, dp_size=9, ep_size=72)


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical_uuid(value):
    return str(uuid.UUID(str(value).lower().removeprefix('gpu-')))


def fixed_layer_source_map(layers=92):
    """Match the native fixed pool: 512 logical experts, 576 physical slots.

    Eight ranks hold eight primary experts; the other 64 hold seven primary
    experts plus one never-routed padding slot. There is one global policy.
    """
    require(type(layers) is int and layers == 92, 'the pinned Qwen3.8 policy has 92 MoE layers')
    physical=[]; logical=0
    for rank in range(72):
        for _ in range(7+int(rank<8)):
            physical.append(logical); logical+=1
        while len(physical)<(rank+1)*8:
            physical.append(logical%512)
    require(logical==512 and len(physical)==576 and set(physical)==set(range(512)),
            'the fixed physical expert map is incomplete')
    return [list(physical) for _ in range(layers)]


@dataclass(frozen=True)
class World72Topology:
    gpu_ids: tuple[str, ...]
    rank_hosts: tuple[str, ...]
    start_port: int
    gpu_uuids: tuple[str, ...] = ()
    rank_port_stride: int = 9
    replica_rank: int = 0
    world_size: int = 72
    tp_size: int = 8
    dp_size: int = 9
    ep_size: int = 72
    ffn_sp_size: int = 8
    pp_size: int = 1

    def __post_init__(self):
        require(all(type(getattr(self, field)) is int for field in
                    ('replica_rank', 'world_size', 'tp_size', 'dp_size', 'ep_size', 'ffn_sp_size', 'pp_size')),
                'world profile dimensions must be integers')
        require((self.replica_rank, self.world_size, self.tp_size, self.dp_size,
                 self.ep_size, self.ffn_sp_size, self.pp_size) == (0, 72, 8, 9, 72, 8, 1),
                'only one explicit TP8/DP9/EP72/FFN-SP8/PP1 policy is qualified')
        require(len(self.rank_hosts) == len(self.gpu_ids) == 72,
                'exactly one host and GPU selector per world rank is required')
        require(all(isinstance(x, str) and x for x in self.rank_hosts + self.gpu_ids),
                'host and GPU selectors must be nonempty strings')
        hosts = list(dict.fromkeys(self.rank_hosts))
        require(len(hosts) == 18, 'one world requires exactly 18 distinct nodes')
        require(len(set(zip(self.rank_hosts, self.gpu_ids))) == 72,
                'one physical host/GPU selector cannot serve two world ranks')
        for node, host in enumerate(hosts):
            require([i for i, value in enumerate(self.rank_hosts) if value == host] == list(range(4 * node, 4 * node + 4)),
                    'each node must own one aligned contiguous four-rank group')
        require(self.rank_port_stride >= 9 and self.start_port - 11 >= 1024
                and self.start_port + 4 * self.rank_port_stride + 12 < 65536,
                'native local port block must reserve all rank endpoints')
        if self.gpu_uuids:
            require(len(self.gpu_uuids) == 72 and len({canonical_uuid(x) for x in self.gpu_uuids}) == 72,
                    'the allocation requires 72 distinct physical GPU UUIDs')

    @property
    def node_groups(self):
        groups = []
        for node in range(18):
            ranks = list(range(node * 4, node * 4 + 4))
            groups.append(dict(node_index=node, host=self.rank_hosts[ranks[0]], ranks=ranks,
                               gpu_ids=[self.gpu_ids[r] for r in ranks],
                               gpu_uuids=[canonical_uuid(self.gpu_uuids[r]) for r in ranks] if self.gpu_uuids else []))
        return groups

    @property
    def control_addresses(self):
        return [f'{self.rank_hosts[r]}:{self.start_port + (r % 4) * self.rank_port_stride + 1}' for r in range(72)]

    @property
    def addresses(self):
        return self.control_addresses

    @property
    def generation_addresses(self):
        return [self.control_addresses[r] for r in range(0, 72, 8)]

    def rank(self, world_rank):
        require(type(world_rank) is int and 0 <= world_rank < 72, 'invalid world rank')
        return dict(world_rank=world_rank, world_size=72, tp_rank=world_rank % 8, tp_size=8,
                    dp_rank=world_rank // 8, dp_size=9, ep_rank=world_rank, ep_size=72,
                    node_index=world_rank // 4, local_rank=world_rank % 4,
                    owner_ip=self.rank_hosts[world_rank])

    def with_gpu_uuids(self, values):
        return replace(self, gpu_uuids=tuple(canonical_uuid(x) for x in values))

    def node_environment(self, node_index, incarnation):
        require(len(self.gpu_uuids) == 72, 'bind all 72 current physical UUIDs before native allocation')
        node = self.node_groups[int(node_index)]
        gang = ';'.join(f'name:q38_world_part{i},ip:{group["host"]},port:{self.start_port}'
                        for i, group in enumerate(self.node_groups))
        values = dict(WORLD_SIZE=72, WORLD_RANK=node['ranks'][0], LOCAL_WORLD_SIZE=4,
                      TP_SIZE=8, DP_SIZE=9, EP_SIZE=72, FFN_SP_SIZE=8,
                      RTP_WORLD_SIZE=72, RTP_WORLD_RANK=node['ranks'][0], RTP_LOCAL_WORLD_SIZE=4,
                      RTP_TP_SIZE=8, RTP_DP_SIZE=9, RTP_EP_SIZE=72, RTP_FFN_SP_SIZE=8,
                      RTP_START_PORT=self.start_port, RTP_REPLICA_RANK=0)
        result = {key: str(value) for key, value in values.items()}
        result.update(CUDA_VISIBLE_DEVICES=','.join('GPU-' + value for value in node['gpu_uuids']),
                      RTP_LLM_RL_MODE='1', RTP_LLM_RL_WORLD72='1', RTP_LLM_RL_START_PAUSED='1',
                      RTP_LLM_RL_METADATA_BOOTSTRAP='1',
                      RTP_FIXED_PHYSICAL_EP_LAYOUT='balanced_v1', RTP_RL_BORROW_SINGLETON_HF_SOURCES='1',
                      RTP_NCCL_EP_REFERENCE_MAX_PADDED_TOKENS='73728',
                      RTP_LLM_RL_RUN_ID=str(incarnation), GANG_CONFIG_STRING=gang)
        return result

    def to_dict(self):
        result = asdict(self)
        result.update(schema='qwen38-one-world72-topology-v1',
                      control_addresses=self.control_addresses,
                      generation_addresses=self.generation_addresses,
                      generation_world_ranks=list(range(0, 72, 8)),
                      node_groups=self.node_groups, logical_model_replicas=1)
        # JSON canonical types make in-memory and persisted contracts identical.
        return json.loads(json.dumps(result))

    @classmethod
    def from_dict(cls, record):
        allowed = {field.name for field in __import__('dataclasses').fields(cls)}
        values = {key: record[key] for key in allowed if key in record}
        for key in ('gpu_ids', 'rank_hosts', 'gpu_uuids'):
            if key in values:
                values[key] = tuple(values[key])
        result = cls(**values)
        expected = result.to_dict()
        for key in ('schema', 'control_addresses', 'generation_addresses', 'generation_world_ranks',
                    'node_groups', 'logical_model_replicas'):
            require(key not in record or record[key] == expected[key],
                    f'recorded {key} differs from the one-world rank mapping')
        return result


def validate_native_manifest(record, topology, world_rank, required_sources, *, owner_fields=False,
                             boot=None, incarnation=None):
    expected = topology.rank(world_rank)
    for name in ('world_rank', 'world_size', 'tp_rank', 'tp_size', 'dp_rank', 'dp_size', 'ep_rank', 'ep_size'):
        require(type(record.get(name)) is int and record[name] == expected[name],
                f'native manifest {name} differs from world rank {world_rank}')
    require(type(record.get('pid')) is int and record['pid'] > 0, 'native PID is missing')
    require(len(topology.gpu_uuids) == 72 and canonical_uuid(record.get('gpu_uuid')) == canonical_uuid(topology.gpu_uuids[world_rank]),
            'native manifest physical UUID differs from the bound allocation')
    names = record.get('names')
    require(isinstance(names, list) and len(names) == len(set(names)) == 1590
            and set(names) == set(required_sources), 'native manifest must cover all 1590 main-policy sources')
    require(record.get('names_sha256') == hashlib.sha256('\n'.join(names).encode()).hexdigest(),
            'native source-name digest differs from its actual names')
    if incarnation is not None:
        require(record.get('incarnation_id') == incarnation, 'native process did not inherit this incarnation ID')
    if owner_fields:
        require(record.get('owner_ip') == expected['owner_ip'] and record.get('owner_node_index') == expected['node_index'],
                'native manifest belongs to a different owner')
        require(record.get('boot_id') == boot and record.get('incarnation') == incarnation,
                'native manifest belongs to another native incarnation')
        require(record.get('metadata_bootstrap') is True, 'native must allocate through metadata bootstrap')
        library = record.get('bootstrap_native_library', {})
        validate_native_library_proof(library, record, library.get('native_library_sha256'))
    return record


def validate_native_library_proof(proof, manifest, expected_sha256):
    """Bind the bootstrap's actual loaded-library proof to this live rank PID.

    The producer hashes and verifies its mapped inode before target allocation;
    the owner additionally checks that PID's identity-bound process ancestry.
    """
    require(isinstance(expected_sha256, str) and len(expected_sha256) == 64
            and all(c in '0123456789abcdef' for c in expected_sha256),
            'the accepted native library SHA256 is required before allocation')
    require(proof.get('schema') == 'qwen38-actual-native-library-proof-v1'
            and proof.get('native_initial_pause_version') == 1
            and proof.get('native_library_sha256') == expected_sha256,
            'native bootstrap proof differs from the accepted fresh initial-pause library')
    for field in ('pid', 'world_rank', 'world_size', 'tp_size', 'dp_size', 'ep_size'):
        require(type(proof.get(field)) is int and proof[field] == manifest[field],
                f'native library proof {field} does not bind this world-rank process')
    require(isinstance(proof.get('native_library_path'), str)
            and proof['native_library_path'].startswith('/')
            and all(type(proof.get(field)) is int and proof[field] > 0
                    for field in ('native_library_inode', 'native_library_device', 'native_library_bytes')),
            'native bootstrap proof lacks an actual mapped-file identity')
    return proof


def validate_world_manifests(records, topology, required_sources, *, boot, incarnation):
    require(len(records) == 72 and {r.get('world_rank') for r in records} == set(range(72)),
            'native manifests must identify every world rank exactly once')
    result = sorted(records, key=lambda row: row['world_rank'])
    for rank, record in enumerate(result):
        validate_native_manifest(record, topology, rank, required_sources,
                                 owner_fields=True, boot=boot, incarnation=incarnation)
    require(len({canonical_uuid(row['gpu_uuid']) for row in result}) == 72,
            'a physical GPU serves multiple native world ranks')
    require(len({row['bootstrap_native_library']['native_library_sha256'] for row in result}) == 1,
            'all 72 ranks must map the same accepted fresh native library')
    return result
