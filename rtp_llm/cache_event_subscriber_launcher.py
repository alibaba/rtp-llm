import logging
import multiprocessing
import os
import shlex
import signal
import subprocess
import threading
from collections.abc import Mapping, Sequence
from multiprocessing.process import BaseProcess
from typing import Callable, Optional

_COMMAND_ENV = "KVCM_CACHE_EVENT_SUBSCRIBER_COMMAND"
_OWNER_RANK_ENV = "KVCM_CACHE_EVENT_SUBSCRIBER_OWNER_RANK"
_REQUIRED_ENV = "KVCM_CACHE_EVENT_SUBSCRIBER_REQUIRED"
_RESTART_DELAY_ENV = "KVCM_CACHE_EVENT_SUBSCRIBER_RESTART_DELAY_S"


def _bool_env(environ: Mapping[str, str], name: str, default: bool) -> bool:
    value = environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def subscriber_owner_rank(environ: Mapping[str, str] = os.environ) -> int:
    rank = int(environ.get(_OWNER_RANK_ENV, "0"))
    if rank < 0:
        raise ValueError(f"{_OWNER_RANK_ENV} must be non-negative")
    return rank


def subscriber_required(environ: Mapping[str, str] = os.environ) -> bool:
    return _bool_env(environ, _REQUIRED_ENV, False)


def subscriber_command(environ: Mapping[str, str] = os.environ) -> tuple[str, ...]:
    """Return the configured argv without invoking a shell."""

    raw_command = environ.get(_COMMAND_ENV, "").strip()
    if not raw_command:
        return ()
    command = tuple(shlex.split(raw_command))
    if not command:
        raise ValueError(f"{_COMMAND_ENV} does not contain an executable")
    return command


def _child_environment(
    environ: Mapping[str, str], world_rank: int, default_endpoint: str | None
) -> dict[str, str]:
    child_environ = dict(environ)
    child_environ["RTP_CACHE_EVENT_WORLD_RANK"] = str(world_rank)
    if default_endpoint:
        child_environ["RTP_CACHE_EVENT_ENDPOINT"] = default_endpoint
    return child_environ


def _expand_command(
    command: Sequence[str], world_rank: int, default_endpoint: str | None
) -> tuple[str, ...]:
    replacements = {"{world_rank}": str(world_rank)}
    if default_endpoint:
        replacements["{rtp_endpoint}"] = default_endpoint
    expanded: list[str] = []
    for argument in command:
        for marker, value in replacements.items():
            argument = argument.replace(marker, value)
        if "{rtp_endpoint}" in argument:
            raise ValueError(
                "KVCM_CACHE_EVENT_SUBSCRIBER_COMMAND uses {rtp_endpoint}, "
                "but no default RTP endpoint is available"
            )
        expanded.append(argument)
    return tuple(expanded)


def _exec_subscriber(command: Sequence[str], child_environ: Mapping[str, str]) -> None:
    argv = list(command)
    logging.info("[CACHE_EVENT_SUBSCRIBER] exec argv=%s", argv)
    os.execvpe(argv[0], argv, dict(child_environ))


def _supervise_subscriber(
    command: Sequence[str],
    child_environ: Mapping[str, str],
    restart_delay_s: float,
) -> None:
    """Keep an optional subscriber alive without making RTP fail closed."""

    stopping = threading.Event()
    child: subprocess.Popen[bytes] | None = None

    def stop(_signum: int, _frame: object) -> None:
        stopping.set()
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    while not stopping.is_set():
        try:
            child = subprocess.Popen(list(command), env=dict(child_environ))
        except OSError:
            logging.exception(
                "[CACHE_EVENT_SUBSCRIBER] optional child failed to start; retrying"
            )
        else:
            return_code = child.wait()
            child = None
            if stopping.is_set():
                return
            logging.warning(
                "[CACHE_EVENT_SUBSCRIBER] optional child exited rc=%s; retrying",
                return_code,
            )
        stopping.wait(restart_delay_s)


def launch_cache_event_subscriber(
    *,
    environ: Mapping[str, str] = os.environ,
    world_rank: int = 0,
    default_endpoint: str | None = None,
    process_factory: Callable[..., BaseProcess] = multiprocessing.Process,
) -> Optional[BaseProcess]:
    """Start the optional subscriber as a ProcessManager-compatible child."""

    command = subscriber_command(environ)
    if not command:
        logging.info("[CACHE_EVENT_SUBSCRIBER] disabled; %s is empty", _COMMAND_ENV)
        return None
    owner_rank = subscriber_owner_rank(environ)
    if world_rank != owner_rank:
        logging.info(
            "[CACHE_EVENT_SUBSCRIBER] skipped on world_rank=%s; owner_rank=%s",
            world_rank,
            owner_rank,
        )
        return None
    command = _expand_command(command, world_rank, default_endpoint)
    child_environ = _child_environment(environ, world_rank, default_endpoint)
    required = subscriber_required(environ)
    restart_delay_s = float(environ.get(_RESTART_DELAY_ENV, "1"))
    if restart_delay_s < 0:
        raise ValueError(f"{_RESTART_DELAY_ENV} must be non-negative")
    target = _exec_subscriber if required else _supervise_subscriber
    args: tuple[object, ...] = (
        (command, child_environ)
        if required
        else (command, child_environ, restart_delay_s)
    )
    process = process_factory(
        target=target,
        args=args,
        name="cache_event_subscriber",
    )
    process.start()
    logging.info(
        "[CACHE_EVENT_SUBSCRIBER] started pid=%s required=%s",
        process.pid,
        required,
    )
    return process
