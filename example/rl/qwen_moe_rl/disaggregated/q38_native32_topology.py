"""One Qwen3.8 inference policy over 8 four-GPU rollout containers."""

from dataclasses import dataclass
from ipaddress import IPv4Address
import re
import uuid


WORLD = 32
TP = 8
DP = 4
LOCAL = 4
EP = 32
LAYERS = 92
EXPERTS = 512
PORT_STRIDE = 9
CONTROL_TOPOLOGY = dict(world_size=WORLD, tp_size=TP, dp_size=DP, ep_size=EP)


def expert_map():
    """The 512 physical slots are identical to logical expert IDs."""
    return [list(range(EXPERTS)) for _ in range(LAYERS)]


@dataclass(frozen=True)
class Native32Topology:
    hosts: tuple
    gpu_uuids: tuple
    start_port: int

    def __post_init__(self):
        hosts = tuple(str(IPv4Address(h)) for h in self.hosts)
        if len(hosts) != WORLD//LOCAL or len(set(hosts)) != len(hosts):
            raise ValueError('one distinct IPv4 address per rollout container is required')
        if any(not IPv4Address(h).is_private or IPv4Address(h).is_loopback for h in hosts):
            raise ValueError('native rollout owners must be on the explicit private allocation')
        if len(self.gpu_uuids) != len(hosts) or any(len(g) != LOCAL for g in self.gpu_uuids):
            raise ValueError('exactly four UUIDs per owner are required')
        devices = tuple(tuple('GPU-'+str(uuid.UUID(str(x).lower().removeprefix('gpu-'))) for x in group)
                        for group in self.gpu_uuids)
        if len({x for group in devices for x in group}) != WORLD:
            raise ValueError('one physical GPU cannot serve two rollout ranks')
        if type(self.start_port) is not int or not 1035 <= self.start_port < 65536-48:
            raise ValueError('invalid per-node native port reservation')
        object.__setattr__(self, 'hosts', hosts)
        object.__setattr__(self, 'gpu_uuids', devices)

    @property
    def control_addresses(self):
        return [f'{self.hosts[r//LOCAL]}:{self.start_port+(r%LOCAL)*PORT_STRIDE+1}' for r in range(WORLD)]

    @property
    def generation_addresses(self):
        return self.control_addresses[::TP]

    def node_environment(self, node, incarnation, *, local_experts):
        if type(node) is not int or not 0 <= node < len(self.hosts):
            raise ValueError('invalid rollout node index')
        if not isinstance(incarnation, str) or not re.fullmatch(r'q38-[a-z0-9-]{6,85}', incarnation):
            raise ValueError('a fresh scoped Qwen3.8 incarnation is required')
        if type(local_experts) is not bool:
            raise ValueError('select the expert wire format explicitly')
        return {
            'RTP_NODE_IP': self.hosts[node],
            'RTP_WORLD_RANK': str(node*LOCAL),
            'RTP_START_PORT': str(self.start_port),
            'CUDA_VISIBLE_DEVICES': ','.join(self.gpu_uuids[node]),
            'GANG_CONFIG_STRING': ';'.join(
                f'name:q38_world_part{i},ip:{host},port:{self.start_port}' for i, host in enumerate(self.hosts)),
            'RTP_LLM_RL_RUN_ID': incarnation,
            'RTP_LLM_RL_WORLD32_LOCAL_EXPERTS': '1' if local_experts else '0',
        }
