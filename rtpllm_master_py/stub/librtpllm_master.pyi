from __future__ import annotations
import typing
__all__ = ['MasterInitParameter', 'PyEstimatorConfig', 'PyLoadbalanceConfig', 'PySubscribeConfig', 'PySubscribeConfigType', 'RtpLLMMasterEntry']
class MasterInitParameter:
    estimator_config: PyEstimatorConfig
    load_balance_config: PyLoadbalanceConfig
    port: int
    def __init__(self) -> None:
        ...
class PyEstimatorConfig:
    estimator_config_map: dict[str, str]
    def __init__(self) -> None:
        ...
class PyLoadbalanceConfig:
    subscribe_config: PySubscribeConfig
    sync_status_interval_ms: int
    update_interval_ms: int
    def __init__(self) -> None:
        ...
class PySubscribeConfig:
    cluster_name: str
    local_http_port: int
    local_ip: str
    local_rpc_port: int
    type: PySubscribeConfigType
    zk_host: str
    zk_path: str
    zk_timeout_ms: int
    def __init__(self) -> None:
        ...
class PySubscribeConfigType:
    """
    Members:
    
      CM2
    
      LOCAL
    """
    CM2: typing.ClassVar[PySubscribeConfigType]  # value = <PySubscribeConfigType.CM2: 0>
    LOCAL: typing.ClassVar[PySubscribeConfigType]  # value = <PySubscribeConfigType.LOCAL: 1>
    __members__: typing.ClassVar[dict[str, PySubscribeConfigType]]  # value = {'CM2': <PySubscribeConfigType.CM2: 0>, 'LOCAL': <PySubscribeConfigType.LOCAL: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class RtpLLMMasterEntry:
    def __init__(self) -> None:
        ...
    def init(self, arg0: ...) -> bool:
        ...
