"""Side-effect-free SSH execution schema; machine credentials remain external."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping, Optional, Tuple

from rtp_llm.test.perf_test.config_schema import (
    ConfigError,
    _non_empty_string,
    _object,
    _positive_int,
    _reject_unknown,
    _reject_unsupported_forms,
    _required,
)


@dataclass(frozen=True)
class SshHost:
    ssh_target: str
    advertise_host: str


@dataclass(frozen=True)
class SshSourceConfig:
    worktree: Optional[str] = None
    ref: str = "auto"
    build_policy: str = "build"
    build_driver: Optional[str] = None
    bazel_config: str = "cuda13"
    bazel_targets: Tuple[str, ...] = ("//:th_transformer", "//rtp_llm:rtp_llm_lib")
    bazel_args: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SshExecutionConfig:
    backend: str
    hosts: Tuple[SshHost, ...]
    gpus_per_machine: int
    source: SshSourceConfig
    run_user: Optional[str] = None
    ssh_port: Optional[int] = None
    start_port: int = 12333
    python_bin: str = "/opt/conda310/bin/python"
    work_dir: Optional[str] = None


def _path(name: str, value: Any) -> str:
    parsed = _non_empty_string(name, value)
    if (
        not PurePosixPath(parsed).is_absolute()
        or "\x00" in parsed
        or ".." in PurePosixPath(parsed).parts
    ):
        raise ConfigError(f"{name}: expected an absolute path")
    return parsed


def _strings(name: str, value: Any, *, nonempty: bool = False) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or (nonempty and not value):
        raise ConfigError(f"{name}: expected {'non-empty ' if nonempty else ''}array")
    return tuple(
        _non_empty_string(f"{name}[{index}]", item) for index, item in enumerate(value)
    )


def _parse_source(value: Any) -> SshSourceConfig:
    raw = _object("execution.source", value)
    _reject_unknown("execution.source", raw, set(SshSourceConfig.__dataclass_fields__))
    policy = raw.get("build_policy", "build")
    if policy not in ("build_with_deps", "build", "reuse"):
        raise ConfigError(
            "execution.source.build_policy: expected build_with_deps, build, or reuse"
        )
    driver = raw.get("build_driver")
    if policy != "reuse" and not driver:
        raise ConfigError(
            "execution.source.build_driver: required for build/build_with_deps"
        )
    args = _strings("execution.source.bazel_args", raw.get("bazel_args", []))
    for arg in args:
        if arg.startswith(
            (
                "--output_user_root",
                "--output_base",
                "--disk_cache",
                "--config",
                "--run_under",
            )
        ) or arg in ("--", "build", "test", "run"):
            raise ConfigError(
                f"execution.source.bazel_args: runtime-owned option {arg!r}"
            )
    platform = _non_empty_string(
        "execution.source.bazel_config", raw.get("bazel_config", "cuda13")
    )
    if not re.fullmatch(r"[A-Za-z0-9_]+", platform):
        raise ConfigError(
            "execution.source.bazel_config: expected a platform config name"
        )
    return SshSourceConfig(
        worktree=(
            None
            if raw.get("worktree") is None
            else _non_empty_string("execution.source.worktree", raw["worktree"])
        ),
        ref=_non_empty_string("execution.source.ref", raw.get("ref", "auto")),
        build_policy=policy,
        build_driver=(
            None if driver is None else _path("execution.source.build_driver", driver)
        ),
        bazel_config=platform,
        bazel_targets=_strings(
            "execution.source.bazel_targets",
            raw.get("bazel_targets", SshSourceConfig.bazel_targets),
            nonempty=True,
        ),
        bazel_args=args,
    )


def parse_execution(value: Any) -> SshExecutionConfig:
    raw = _object("execution", value)
    _reject_unsupported_forms(raw, "execution")
    _reject_unknown("execution", raw, set(SshExecutionConfig.__dataclass_fields__))
    if raw.get("backend") != "ssh":
        raise ConfigError("execution.backend: expected ssh")
    items = _required("execution", raw, "hosts")
    if not isinstance(items, (list, tuple)) or not items:
        raise ConfigError("execution.hosts: expected a non-empty array")
    hosts = []
    for index, item in enumerate(items):
        path = f"execution.hosts[{index}]"
        host = _object(path, item)
        _reject_unknown(path, host, {"ssh_target", "advertise_host"})
        target = _non_empty_string(
            path + ".ssh_target", _required(path, host, "ssh_target")
        )
        address = _non_empty_string(
            path + ".advertise_host", _required(path, host, "advertise_host")
        )
        # SSH operands must never be parsed as options; advertised addresses are
        # encoded in the existing engine name:...,ip:...,port:... grammar.
        if target.startswith("-") or any(c.isspace() or c == "\x00" for c in target):
            raise ConfigError(path + ".ssh_target: invalid SSH destination")
        if any(c in address for c in (",", ";", ":", "\x00")) or any(
            c.isspace() for c in address
        ):
            raise ConfigError(path + ".advertise_host: invalid gang address")
        hosts.append(SshHost(target, address))
    if len({host.ssh_target for host in hosts}) != len(hosts) or len(
        {host.advertise_host for host in hosts}
    ) != len(hosts):
        raise ConfigError("execution.hosts: duplicate members")
    port = _positive_int("execution.start_port", raw.get("start_port", 12333))
    if port + 1025 > 65535:
        raise ConfigError(
            "execution.start_port: derived decode control port exceeds 65535"
        )
    ssh_port = (
        None
        if raw.get("ssh_port") is None
        else _positive_int("execution.ssh_port", raw["ssh_port"])
    )
    if ssh_port is not None and ssh_port > 65535:
        raise ConfigError("execution.ssh_port: must not exceed 65535")
    user = raw.get("run_user")
    if user is not None and (
        not isinstance(user, str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", user) is None
        or user == "root"
    ):
        raise ConfigError("execution.run_user: expected a non-root username")
    return SshExecutionConfig(
        backend="ssh",
        hosts=tuple(hosts),
        gpus_per_machine=_positive_int(
            "execution.gpus_per_machine",
            _required("execution", raw, "gpus_per_machine"),
        ),
        source=_parse_source(_required("execution", raw, "source")),
        run_user=user,
        ssh_port=ssh_port,
        start_port=port,
        python_bin=_path(
            "execution.python_bin", raw.get("python_bin", "/opt/conda310/bin/python")
        ),
        work_dir=(
            None
            if raw.get("work_dir") is None
            else _path("execution.work_dir", raw["work_dir"])
        ),
    )
