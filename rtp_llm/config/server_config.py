"""Frontend server configuration and port layout; no compute runtime required."""

import math
import re

DEFAULT_START_PORT = 8088
COORDINATOR_INFO_PORT_NUM = 11
MIN_WORKER_INFO_PORT_NUM = 9
WORKER_INFO_PORT_NUM = MIN_WORKER_INFO_PORT_NUM
DASH_SC_GRPC_SERVER_PORT_OFFSET = 8


class ServerConfig:
    """Port layout: base = start_port + rank_id * worker_info_port_num."""

    def __init__(self):
        self.frontend_metrics_enable = False
        self.frontend_server_count = 4
        self.vit_server_count = 1
        self.start_port = DEFAULT_START_PORT
        self.timeout_keep_alive = 5
        self.frontend_server_id = 0
        self.vit_server_id = 0
        self.rank_id = 0
        self.ip: str = ""
        self.worker_info_port_num: int = MIN_WORKER_INFO_PORT_NUM
        self.shutdown_timeout: int = 600  # graceful drain budget (seconds)
        self.startup_health_check_timeout: int = (
            3600  # startup readiness bound (seconds)
        )
        self.monitor_interval: int = 1  # Monitor interval in seconds
        self.frontend_pre_stop_drain_seconds: float = 120.0
        self.dash_sc_grpc_pre_stop_drain_seconds: float = 120.0
        self.pre_stop_drain_headroom_seconds: float = -1.0
        self.pre_stop_drain_signal: bool = True
        self.backend_post_frontend_drain_seconds: float = -1.0
        self.enable_torch_allocator_dump: bool = False
        self.torch_allocator_dump_auth_token: str = ""
        self.torch_allocator_dump_auth_header: str = "X-RTP-LLM-Allocator-Dump-Token"
        self.torch_allocator_dump_cooldown_seconds: float = 60.0

    def validate_allocator_dump_config(self) -> None:
        cooldown = self.torch_allocator_dump_cooldown_seconds
        if not math.isfinite(cooldown) or cooldown < 0:
            raise ValueError(
                "torch_allocator_dump_cooldown_seconds must be finite and non-negative"
            )
        if not self.enable_torch_allocator_dump:
            return
        if not self.torch_allocator_dump_auth_token:
            raise ValueError(
                "torch_allocator_dump_auth_token is required when allocator dumps are enabled"
            )
        if not re.fullmatch(
            r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+",
            self.torch_allocator_dump_auth_header,
        ):
            raise ValueError(
                "torch_allocator_dump_auth_header is not a valid HTTP header name"
            )

    def _server_base(self) -> int:
        return self.start_port + self.rank_id * self.worker_info_port_num

    @property
    def server_port(self) -> int:
        """Port for main server (base + 0). Equals start_port + rank_id * worker_info_port_num."""
        return self._server_base() + 0

    @property
    def rpc_server_port(self) -> int:
        return self._server_base() + 1

    @property
    def cache_store_listen_port(self) -> int:
        return self._server_base() + 2

    @property
    def cache_store_rdma_listen_port(self) -> int:
        return self._server_base() + 4

    @property
    def http_port(self) -> int:
        return self._server_base() + 5

    @property
    def embedding_rpc_server_port(self) -> int:
        return self._server_base() + 7

    @property
    def dash_sc_grpc_server_port(self) -> int:
        """DashSc gRPC listen port (ModelStreamInfer, wire: predict_v2.proto)."""
        return self._server_base() + DASH_SC_GRPC_SERVER_PORT_OFFSET

    def set_local_rank(self, local_rank: int):
        """Update rank_id in place; server_port-related properties reflect new values."""
        self.rank_id = local_rank

    def validate_port_layout(self, *, dash_sc_enabled: bool) -> None:
        if dash_sc_enabled and self.worker_info_port_num < MIN_WORKER_INFO_PORT_NUM:
            raise ValueError(
                "worker_info_port_num must be at least "
                f"{MIN_WORKER_INFO_PORT_NUM} when DashSc gRPC is enabled; "
                f"got {self.worker_info_port_num}. DashSc uses port offset "
                f"{DASH_SC_GRPC_SERVER_PORT_OFFSET}, which overlaps the next "
                "rank's port block with a smaller stride."
            )

    # update_from_args 方法已不再需要
    # 配置绑定现在通过声明式 bind_to 参数在 add_argument 时自动处理

    def to_string(self):
        return (
            f"frontend_server_count: {self.frontend_server_count}\n"
            f"vit_server_count: {self.vit_server_count}\n"
            f"start_port: {self.start_port}\n"
            f"timeout_keep_alive: {self.timeout_keep_alive}\n"
            f"frontend_server_id: {self.frontend_server_id}\n"
            f"vit_server_id: {self.vit_server_id}\n"
            f"rank_id: {self.rank_id}\n"
            f"worker_info_port_num: {self.worker_info_port_num}\n"
            f"shutdown_timeout: {self.shutdown_timeout}\n"
            f"monitor_interval: {self.monitor_interval}\n"
            f"frontend_pre_stop_drain_seconds: {self.frontend_pre_stop_drain_seconds}\n"
            f"dash_sc_grpc_pre_stop_drain_seconds: {self.dash_sc_grpc_pre_stop_drain_seconds}\n"
            f"pre_stop_drain_headroom_seconds: {self.pre_stop_drain_headroom_seconds}\n"
            f"pre_stop_drain_signal: {self.pre_stop_drain_signal}\n"
            f"backend_post_frontend_drain_seconds: {self.backend_post_frontend_drain_seconds}\n"
            f"enable_torch_allocator_dump: {self.enable_torch_allocator_dump}\n"
            f"torch_allocator_dump_auth_configured: {bool(self.torch_allocator_dump_auth_token)}\n"
            f"torch_allocator_dump_auth_header: {self.torch_allocator_dump_auth_header}\n"
            f"torch_allocator_dump_cooldown_seconds: {self.torch_allocator_dump_cooldown_seconds}\n"
            f"server_port: {self.server_port}\n"
            f"rpc_server_port: {self.rpc_server_port}\n"
            f"cache_store_listen_port: {self.cache_store_listen_port}\n"
            f"cache_store_rdma_listen_port: {self.cache_store_rdma_listen_port}\n"
            f"http_port: {self.http_port}\n"
            f"embedding_rpc_server_port: {self.embedding_rpc_server_port}\n"
            f"dash_sc_grpc_server_port: {self.dash_sc_grpc_server_port}"
        )
