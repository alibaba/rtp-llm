import argparse
import json
import os
import signal
import threading
from dataclasses import dataclass
from typing import Mapping, Optional

from .env import (
    NAVI_MASTER_ARPC_PORT_ENV,
    NAVI_MASTER_BIZ_NAME_ENV,
    NAVI_MASTER_CONFIG_FILE_ENV,
    NAVI_MASTER_CONFIG_JSON_ENV,
    NAVI_MASTER_ENABLE_KMON_ENV,
    NAVI_MASTER_GRPC_PORT_ENV,
    NAVI_MASTER_HTTP_PORT_ENV,
    NAVI_MASTER_LOG_FILE_ENV,
    NAVI_MASTER_NAME_ENV,
    NAVI_MASTER_SNAPSHOT_DIR_ENV,
    NAVI_MASTER_VIP_SUBSCRIPTION_JSON_ENV,
)
from .master import NaviSchedMaster
from .spectrum_kmon import spectrum_kmon_tags
from .util import (
    DEFAULT_LOG_FILE_NAME,
    DEFAULT_MASTER_RUNTIME_NAME,
    DEFAULT_MASTER_SNAPSHOT_DIR,
)

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_RUNTIME_OPTION_NAMES = (
    "thread_num",
    "queue_size",
    "processing_size",
    "disable_perf",
    "disable_symbol_table",
    "local_nodes",
    "cm2_subscriptions",
    "vip_subscription",
    "spectrum_subscription",
)


def _read_bool(values: Mapping[str, str], name: str, default: bool) -> bool:
    value = values.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE_VALUES


@dataclass(frozen=True)
class MasterProcessConfig:
    master_biz_name: str
    config_file: str
    config_json: str = ""
    name: str = DEFAULT_MASTER_RUNTIME_NAME
    http_port: int = 0
    arpc_port: int = 0
    grpc_port: int = 0
    log_file_name: str = DEFAULT_LOG_FILE_NAME
    snapshot_dir: str = DEFAULT_MASTER_SNAPSHOT_DIR
    enable_kmon: bool = True

    @classmethod
    def from_env(
        cls, environ: Optional[Mapping[str, str]] = None
    ) -> "MasterProcessConfig":
        values = environ if environ is not None else os.environ
        return cls(
            master_biz_name=values.get(NAVI_MASTER_BIZ_NAME_ENV, "").strip(),
            config_file=values.get(NAVI_MASTER_CONFIG_FILE_ENV, "").strip(),
            config_json=values.get(NAVI_MASTER_CONFIG_JSON_ENV, "").strip(),
            name=values.get(NAVI_MASTER_NAME_ENV, DEFAULT_MASTER_RUNTIME_NAME).strip(),
            http_port=int(values.get(NAVI_MASTER_HTTP_PORT_ENV, "0") or 0),
            arpc_port=int(values.get(NAVI_MASTER_ARPC_PORT_ENV, "0") or 0),
            grpc_port=int(values.get(NAVI_MASTER_GRPC_PORT_ENV, "0") or 0),
            log_file_name=values.get(
                NAVI_MASTER_LOG_FILE_ENV, DEFAULT_LOG_FILE_NAME
            ).strip(),
            snapshot_dir=values.get(
                NAVI_MASTER_SNAPSHOT_DIR_ENV, DEFAULT_MASTER_SNAPSHOT_DIR
            ).strip(),
            enable_kmon=_read_bool(values, NAVI_MASTER_ENABLE_KMON_ENV, True),
        )

    def validate(self) -> None:
        if not self.master_biz_name:
            raise ValueError(f"{NAVI_MASTER_BIZ_NAME_ENV} is required")
        if not self.config_json and not self.config_file:
            raise ValueError(
                f"{NAVI_MASTER_CONFIG_JSON_ENV} or "
                f"{NAVI_MASTER_CONFIG_FILE_ENV} is required"
            )
        for name, port in (
            (NAVI_MASTER_HTTP_PORT_ENV, self.http_port),
            (NAVI_MASTER_ARPC_PORT_ENV, self.arpc_port),
            (NAVI_MASTER_GRPC_PORT_ENV, self.grpc_port),
        ):
            if port < 0 or port > 65535:
                raise ValueError(f"{name} must be between 0 and 65535")


def load_master_config(config_file: str = "", config_json: str = ""):
    if config_json:
        try:
            config = json.loads(config_json)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{NAVI_MASTER_CONFIG_JSON_ENV} must be valid JSON: {error.msg}"
            ) from error
    else:
        with open(config_file, encoding="utf-8") as stream:
            config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("master config must be a JSON object")
    return config


def make_runtime_config(options, environ=None):
    values = options.pop("runtime_config", {})
    if not isinstance(values, dict):
        raise ValueError("runtime_config must be a JSON object")
    values = dict(values)
    for name in _RUNTIME_OPTION_NAMES:
        if name not in options:
            continue
        if name in values:
            raise ValueError(f"duplicate runtime config option: {name}")
        values[name] = options.pop(name)
    environment = os.environ if environ is None else environ
    subscription_json = environment.get(NAVI_MASTER_VIP_SUBSCRIPTION_JSON_ENV)
    if subscription_json is not None:
        try:
            subscription = json.loads(subscription_json)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{NAVI_MASTER_VIP_SUBSCRIPTION_JSON_ENV} must be valid JSON"
            ) from error
        if not isinstance(subscription, dict):
            raise ValueError(
                f"{NAVI_MASTER_VIP_SUBSCRIPTION_JSON_ENV} must be a JSON object"
            )
        values["vip_subscription"] = subscription
    return values


def make_scheduler_config(options):
    value = options.pop("scheduler_config", {})
    if not isinstance(value, dict):
        raise ValueError("scheduler_config must be a JSON object")
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def make_master_kmon_tags(options, environ=None):
    """Build dashboard-stable tags for the standalone Master process."""
    values = os.environ if environ is None else environ
    configured_tags = options.pop("kmon_tags", {})
    if not isinstance(configured_tags, dict):
        raise ValueError("kmon_tags must be a JSON object")
    tags = {str(name): str(value) for name, value in configured_tags.items()}
    tags.setdefault("role", "master")
    deployment_name = (
        values.get("DEPLOYMENT_NAME", "").strip()
        or values.get("HIPPO_ROLE", "").strip()
        or values.get("SPECTRUM_DEPLOYMENT_NAME", "").strip()
    )
    if deployment_name:
        tags.setdefault("deployment_name", deployment_name)
    for env_name, tag_name in (
        ("HIPPO_APP", "hippo_app"),
        ("HIPPO_ROLE", "hippo_role"),
        ("C2_GROUP", "hippo_group"),
        ("HIPPO_GROUP", "hippo_group"),
        ("HIPPO_SLOT_ID", "slot_id"),
    ):
        value = values.get(env_name, "").strip()
        if value:
            tags.setdefault(tag_name, value)
    tags.update(spectrum_kmon_tags(values))
    physical_host_ip = values.get("HIPPO_SLAVE_IP", "").strip()
    instance_ip = values.get("POD_IP", "").strip() or physical_host_ip
    if instance_ip:
        tags.setdefault("host", instance_ip)
        tags.setdefault("host_ip", instance_ip)
        tags.setdefault("container_ip", instance_ip)
    if physical_host_ip:
        tags.setdefault("physical_host_ip", physical_host_ip)
    return tags


def wait_for_stop() -> None:
    stopped = threading.Event()

    def stop(signum, frame):
        stopped.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    stopped.wait()


def run_master(config: MasterProcessConfig, role_options=None) -> None:
    config.validate()
    if role_options is None:
        role_options = load_master_config(config.config_file, config.config_json)
    else:
        role_options = dict(role_options)
    role_options.pop("biz_name", None)
    scheduler_config = make_scheduler_config(role_options)
    runtime_config = make_runtime_config(role_options)
    kmon_tags = make_master_kmon_tags(role_options)
    gig = dict(role_options.pop("gig", {}))
    gig.update(
        {
            "http_port": config.http_port,
            "arpc_port": config.arpc_port,
            "grpc_port": config.grpc_port,
            "snapshot_log_dir": config.snapshot_dir,
        }
    )
    with NaviSchedMaster(
        config.master_biz_name,
        name=config.name,
        gig=gig,
        log_file_name=config.log_file_name,
        snapshot_dir=config.snapshot_dir,
        enable_kmon=config.enable_kmon,
        kmon_tags=kmon_tags,
        wait_no_query=False,
        runtime_config=runtime_config,
        scheduler_config=scheduler_config,
        **role_options,
    ) as master:
        master.init()
        wait_for_stop()


def main() -> None:
    run_master(MasterProcessConfig.from_env())


if __name__ == "__main__":
    main()
