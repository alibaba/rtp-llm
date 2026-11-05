"""CPU-only topology contract for a tensor-parallel RTP replica."""
from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class ReplicaTopology:
    replica_rank: int
    tp_size: int
    gpu_ids: tuple[str, ...]
    start_port: int
    rank_port_stride: int = 9
    host: str = "127.0.0.1"
    ep_size: int = 1
    rank_hosts: tuple[str, ...] = ()

    def __post_init__(self):
        if self.replica_rank < 0 or self.tp_size < 1:
            raise ValueError("invalid replica rank or tensor-parallel size")
        hosts = self.rank_hosts or (self.host,) * self.tp_size
        if len(hosts) != self.tp_size or len(self.gpu_ids) != self.tp_size:
            raise ValueError("one host and GPU ID per RTP TP rank is required")
        if len(set(zip(hosts, self.gpu_ids))) != self.tp_size:
            raise ValueError("a physical host/GPU pair cannot serve two RTP ranks")
        if len({len(node['ranks']) for node in self.node_groups})!=1:
            raise ValueError("native distributed launch requires the same local world size on every node")
        for node in self.node_groups:
            ranks = node['ranks']
            if ranks != list(range(ranks[0], ranks[0] + len(ranks))) or ranks[0] % len(ranks):
                raise ValueError("native node ranks must form an aligned contiguous group")
        if self.ep_size < 1 or self.tp_size % self.ep_size:
            raise ValueError("expert parallelism must divide this replica's world size")
        if self.rank_port_stride < 9:
            raise ValueError("RTP rank port stride must reserve all nine native endpoints")
        if not 1024 <= self.start_port or self.start_port + self.tp_size*self.rank_port_stride+11 >= 65536:
            raise ValueError("RTP port block exceeds the usable TCP range")

    @property
    def addresses(self):
        result = []
        for node in self.node_groups:
            result.extend(f"{node['host']}:{self.start_port+local*self.rank_port_stride+1}"
                          for local in range(len(node['ranks'])))
        return result

    @property
    def node_groups(self):
        hosts = self.rank_hosts or (self.host,) * self.tp_size
        groups = []
        for host in dict.fromkeys(hosts):
            ranks = [rank for rank, value in enumerate(hosts) if value == host]
            groups.append(dict(host=host, ranks=ranks, gpu_ids=[self.gpu_ids[rank] for rank in ranks]))
        return groups

    def node_environment(self,index,incarnation):
        if self.ep_size!=1:
            raise NotImplementedError('Native RL requires EP=1')
        node=self.node_groups[int(index)]
        gang=';'.join(f'name:rl_part{i},ip:{group["host"]},port:{self.start_port}'
                      for i,group in enumerate(self.node_groups))
        return dict(CUDA_VISIBLE_DEVICES=','.join(node['gpu_ids']),
            RTP_START_PORT=str(self.start_port),RTP_REPLICA_RANK=str(self.replica_rank),
            RTP_TP_SIZE=str(self.tp_size),RTP_EP_SIZE=str(self.ep_size),
            RTP_LOCAL_WORLD_SIZE=str(len(node['ranks'])),LOCAL_WORLD_SIZE=str(len(node['ranks'])),
            RTP_WORLD_RANK=str(node['ranks'][0]),RTP_LLM_RL_MODE='1',
            RTP_LLM_RL_RUN_ID=str(incarnation),GANG_CONFIG_STRING=gang)

    def to_dict(self):
        value=asdict(self)
        value.update(schema="rtp-tp-topology-v1", control_addresses=self.addresses,
                     generation_address=self.addresses[0], world_size=self.tp_size,
                     dp_size=1, pp_size=1, node_groups=self.node_groups)
        return value

    @classmethod
    def from_dict(cls,value):
        result=cls(replica_rank=int(value["replica_rank"]),tp_size=int(value["tp_size"]),
                   gpu_ids=tuple(map(str,value["gpu_ids"])),start_port=int(value["start_port"]),
                   rank_port_stride=int(value.get("rank_port_stride",9)),
                   host=value.get("host","127.0.0.1"),ep_size=int(value.get("ep_size",1)),
                   rank_hosts=tuple(value.get("rank_hosts", ())))
        if value.get("control_addresses",result.addresses) != result.addresses:
            raise ValueError("recorded native endpoint mapping differs from the TP rank layout")
        return result


def training_rank_layout(rank, world_size, tp_size, internal_dp=1, pp_size=1):
    if internal_dp != 1 or pp_size != 1:
        raise NotImplementedError("use separate TP replicas for rollout DP; pipeline parallelism is not implemented")
    if world_size < 1 or tp_size < 1 or world_size % tp_size or not 0 <= rank < world_size:
        raise ValueError("training ranks must form complete contiguous RTP TP groups")
    return dict(replica_rank=rank//tp_size,tp_rank=rank%tp_size,tp_size=tp_size,
                replica_count=world_size//tp_size,
                group_ranks=list(range((rank//tp_size)*tp_size,(rank//tp_size+1)*tp_size)))


def preserve_fp32_source(name):
    return name.endswith((".linear_attn.A_log", ".linear_attn.dt_bias"))


def required_source_names(directory, tp_size):
    """All native ranks must agree on their loader's exact HF source names."""
    records=[json.loads((directory/f"rank-{rank:05d}.json").read_text()) for rank in range(tp_size)]
    expected=None
    for rank,record in enumerate(records):
        if int(record["tp_rank"]) != rank or int(record["tp_size"]) != tp_size:
            raise ValueError("native loader manifest does not match this TP group")
        names=record["names"]
        if len(names) != len(set(names)) or not names:
            raise ValueError("native source manifest is empty or contains duplicate keys")
        current=set(names)
        if expected is not None and current != expected:
            raise ValueError("native ranks require different HF source key sets")
        expected=current
    return expected
