from . import _navi_sched_core
from collections.abc import Mapping


navi_sched = _navi_sched_core
DEFAULT_LOG_FILE_NAME = "./logs/navi_sched.log"
DEFAULT_SNAPSHOT_DIR = "navi_sched"
DEFAULT_MASTER_SNAPSHOT_DIR = "."
DEFAULT_MASTER_RUNTIME_NAME = "master_snapshot"
DEFAULT_MASTER_SNAPSHOT_KEEP_COUNT = 3000
DEFAULT_AGENT_RUNTIME_NAME = "sched_agent"
DEFAULT_CLIENT_RUNTIME_NAME = "sched_client"


def naviSchedModule():
    return navi_sched


def makeServiceConfig(navi_sched, name, options):
    options = dict(options)
    config = navi_sched.NaviSchedConfig()
    config.name = str(name)
    config.log_file_name = DEFAULT_LOG_FILE_NAME
    config.snapshot_dir = DEFAULT_SNAPSHOT_DIR
    gig_options = options.pop("gig", {})
    if not isinstance(gig_options, Mapping):
        raise TypeError("gig must be a mapping")
    _setFields(config, {
        key: value for key, value in options.items() if value is not None
    }, "service config")
    gig = navi_sched.NaviSchedGigConfig()
    _setFields(gig, {
        key: value for key, value in gig_options.items() if value is not None
    }, "gig config")
    config.gig = gig
    return config


def _setFields(target, values, object_name):
    for name, value in values.items():
        if not hasattr(target, name):
            raise ValueError(f"unknown {object_name} option: {name}")
        setattr(target, name, value)


def _makeRuntimeDetails(navi, values):
    if values is None:
        values = {}
    if not isinstance(values, Mapping):
        raise TypeError("runtime_config must be a mapping")
    values = dict(values)
    runtime = navi.NaviRuntimeConfig()
    _setFields(runtime, {
        name: values.pop(name)
        for name in (
            "thread_num",
            "queue_size",
            "processing_size",
            "disable_perf",
            "disable_symbol_table",
        )
        if name in values
    }, "runtime_config")

    nodes = []
    for fields in values.pop("local_nodes", []):
        if not isinstance(fields, Mapping):
            raise TypeError("each local_nodes entry must be a mapping")
        node = navi.NaviLocalNode()
        _setFields(node, fields, "local node")
        nodes.append(node)
    runtime.local_nodes = nodes

    subscriptions = []
    for fields in values.pop("cm2_subscriptions", []):
        if not isinstance(fields, Mapping):
            raise TypeError("each cm2_subscriptions entry must be a mapping")
        subscription = navi.NaviCm2Subscription()
        _setFields(subscription, fields, "CM2 subscription")
        subscriptions.append(subscription)
    runtime.cm2_subscriptions = subscriptions

    vip_fields = values.pop("vip_subscription", {})
    if not isinstance(vip_fields, Mapping):
        raise TypeError("vip_subscription must be a mapping")
    if vip_fields:
        vip = navi.NaviVipSubscription()
        _setFields(vip, vip_fields, "VIP subscription")
        runtime.vip_subscription = vip
    if values:
        raise ValueError(
            f"unknown runtime_config option: {next(iter(values))}"
        )
    return runtime


def makeRuntimeConfig(navi_sched, config_type, biz_name, runtime_config):
    navi = navi_sched.navi
    runtime_config = _makeRuntimeDetails(navi, runtime_config)
    config = config_type()
    if hasattr(config, "runtime"):
        runtime_config = navi_sched.NaviSchedRuntimeConfig(runtime_config)
        runtime_config.biz_name = str(biz_name)
        config.runtime = runtime_config
    else:
        config = config_type(runtime_config)
        config.biz_name = str(biz_name)
    return config


class RoleFacade:
    def __init__(self, navi_sched, service):
        self._navi_sched = navi_sched
        self._service = service
        self._closed = False

    @property
    def grpc_port(self):
        return self._service.get_grpc_port()

    @property
    def http_port(self):
        getter = getattr(self._service, "get_http_port", None)
        return getter() if getter else -1

    def init(self):
        if not self._service.init():
            raise RuntimeError("failed to start navi sched role")
        return self

    def stop(self):
        if not self._closed:
            self._service.stop()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop()

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
