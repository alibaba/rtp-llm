"""Runtime topology and communication settings for a backend worker."""

import json
from datetime import timedelta

from torch.distributed import distributed_c10d

from rtp_llm.config.layer_partition import validate_pp_partition_snapshots
from rtp_llm.config.py_config_modules import COORDINATOR_INFO_PORT_NUM, PyEnvConfigs
from rtp_llm.distribute.worker_info import WorldInfo
from rtp_llm.ops import NcclCommConfig, ParallelismConfig


def _build_nccl_comm_config(ip: str, base_port: int, dp_rank: int) -> NcclCommConfig:
    """Build NcclCommConfig from ip, base_port, dp_rank (same port layout as former NodeCommInfo)."""
    rank_base_port = base_port - dp_rank * COORDINATOR_INFO_PORT_NUM
    return NcclCommConfig(
        nccl_ip=ip,
        tp_nccl_port=rank_base_port - 2,
        dp_tp_nccl_port=base_port - 10,
        ffn_tp_nccl_port=rank_base_port - 5,
    )


class WorkerTopology:
    """Members, master address and communication ports for a backend worker."""

    PP_PARTITION_SNAPSHOT_KEY = "pp_partition_snapshot_"

    def __init__(self, py_env_configs: PyEnvConfigs):
        dc = py_env_configs.distribute_config
        pc = py_env_configs.parallelism_config
        world = dc.world_info
        if world is None:
            if pc.world_size != 1:
                raise RuntimeError(
                    "Multi-rank backend requires a topology from prepare_host()"
                )
            world = WorldInfo.for_single_rank(py_env_configs.server_config)

        self.world_info = world
        collective_ip, collective_base_port = world.collective_master
        self._nccl_comm_config = _build_nccl_comm_config(
            collective_ip, collective_base_port, pc.dp_rank
        )
        timeout_seconds = dc.dist_comm_timeout
        self._store_timeout = timedelta(
            seconds=300 if timeout_seconds is None else timeout_seconds
        )

    def get_nccl_comm_config(self) -> NcclCommConfig:
        """Return the NCCL communication config (ip and ports)."""
        return self._nccl_comm_config

    def get_nccl_init_port(self) -> int:
        """Return the port used for torch.distributed init (base_port - 11)."""
        _, base_port = self.world_info.collective_master
        return base_port - 11

    def validate_pp_partition_consistency(
        self, parallelism_config: ParallelismConfig
    ) -> None:
        """Compare the finalized PP partitions using the initialized world group."""
        pp_size = int(parallelism_config.pp_size)
        if pp_size <= 1:
            return

        store = distributed_c10d._get_default_store()
        keys = [
            self.PP_PARTITION_SNAPSHOT_KEY + str(rank)
            for rank in range(parallelism_config.world_size)
        ]
        snapshot = [int(count) for count in parallelism_config.pp_stage_layer_counts]
        store.set(keys[parallelism_config.world_rank], json.dumps(snapshot))
        # Bound only this exchange; the shared Store also serves the process group.
        try:
            store.wait(keys, self._store_timeout)
        except RuntimeError as error:
            for rank, key in enumerate(keys):
                if not store.check([key]):
                    raise RuntimeError(
                        f"timed out waiting for PP partition snapshot from world_rank={rank}"
                    ) from error
            raise
        validate_pp_partition_snapshots([json.loads(v) for v in store.multi_get(keys)])
