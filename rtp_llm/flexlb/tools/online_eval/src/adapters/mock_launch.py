"""Mock JVM 的公共启动规格；入口负责配置准入和进程所有权。"""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Union

PathValue = Union[str, Path]


@dataclass(frozen=True)
class MockLaunchSpec:
    java: str
    jar: PathValue
    n_prefill: int
    n_decode: int
    base_grpc_port: int
    jvm_options: tuple[str, ...] = ()
    whale: Optional[bool] = None
    whale_bundle: Optional[bool] = None
    kmonitor: Optional[bool] = None
    host: Optional[str] = None
    unique_engine_ips: Optional[bool] = None
    event_loop_threads: Optional[int] = None
    completion_threads: Optional[int] = None
    rpc_threads: Optional[int] = None
    rpc_max_calls: Optional[int] = None
    auto_fetch: Optional[bool] = None
    fetch_attach_timeout_ms: Optional[int] = None
    cache_status_compact_encoding: Optional[bool] = None
    performance: Optional[PathValue] = None
    master_config: Optional[PathValue] = None
    stats_stdout: Optional[bool] = None
    events_file: Optional[PathValue] = None
    block_size: Optional[int] = None
    prefill_block_size: Optional[int] = None
    decode_block_size: Optional[int] = None
    prefill_kv_pool_blocks: Optional[int] = None
    decode_kv_pool_blocks: Optional[int] = None
    decode_max_concurrency: Optional[int] = None
    endpoint_file: Optional[PathValue] = None
    env_file: Optional[PathValue] = None
    discovery_file: Optional[PathValue] = None
    extra_args: tuple[str, ...] = ()

    def argv(self) -> list[str]:
        args = [self.java, *self.jvm_options, "-jar", str(self.jar)]
        for field in fields(self):
            if field.name in {"java", "jar", "jvm_options", "extra_args", "discovery_file"}:
                continue
            value = getattr(self, field.name)
            if value is not None:
                args.extend(
                    ["--" + field.name.replace("_", "-"), str(value).lower() if type(value) is bool else str(value)]
                )
        args.extend(self.extra_args)
        # Local launchers reserve discovery ownership after caller-supplied flags.
        if self.discovery_file is not None:
            args.extend(["--discovery-file", str(self.discovery_file)])
        return args
