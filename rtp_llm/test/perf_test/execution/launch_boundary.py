"""Ownership of the perf launcher/server boundary.

The controller can import the constants without loading the engine. Registry
inspection is deferred until an engine launch is projected on a worker.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Iterable, Mapping, Optional

from rtp_llm.test.perf_test.config_schema import (
    ConfigError,
    validate_engine_argument_names,
)
from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.execution.devices import GPU_LOCKED_DEVICES_ENV

RUNTIME_ENVIRONMENT_KEYS = frozenset(
    {
        "WORLD_RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "TP_SIZE",
        "DP_SIZE",
        "PP_SIZE",
        "EP_SIZE",
        "GANG_CONFIG_STRING",
        "START_PORT",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        GPU_LOCKED_DEVICES_ENV,
        "TORCH_CUDA_PROFILER_DIR",
        "TEST_UNDECLARED_OUTPUTS_DIR",
        "MAGA_SERVER_WORK_DIR",
    }
)
FORCED_ENGINE_ENVIRONMENT = {
    "USE_BATCH_DECODE_SCHEDULER": "1",
    "FAKE_BALANCE_EXPERT": "1",
}


@lru_cache(maxsize=1)
def _server_argument_parser():
    # Importing server_args loads config/ops; never do this on controller import.
    from rtp_llm.config.py_config_modules import PyEnvConfigs
    from rtp_llm.server.server_args.server_args import (
        EnvArgumentParser,
        ensure_backend_entrypoint_loaded,
        init_all_group_args,
    )

    ensure_backend_entrypoint_loaded()
    parser = EnvArgumentParser(add_help=False)
    config = PyEnvConfigs()
    parser.set_root_config(config)
    init_all_group_args(parser, config)
    return parser


def server_environment_exclusions() -> tuple[str, ...]:
    """Registered server parameters must come from the frozen JSON or runtime."""
    return tuple(
        sorted(
            set(_server_argument_parser().get_env_mappings().values())
            - RUNTIME_ENVIRONMENT_KEYS
        )
    )


def configure_perf_environment(
    explicit: Optional[Mapping[str, str]] = None, *, preserve: Iterable[str] = ()
) -> None:
    """Remove inherited server settings before parsing, then apply JSON env.

    Initial SSH parsing preserves the model fields only to check them against
    the JSON. Subsequent parsing and server spawning use the frozen projection.
    """
    preserved = set(preserve)
    for name in server_environment_exclusions():
        if name not in preserved:
            os.environ.pop(name, None)
    os.environ.update(explicit or {})


def validate_engine_overrides(
    args: Mapping[str, Any],
    environment: Mapping[str, str],
    first_class: Mapping[str, Any],
) -> None:
    """Resolve the actual argparse destinations, including aliases/abbreviations."""
    try:
        validate_engine_argument_names(args)
    except ConfigError as error:
        raise PlanningError(str(error)) from error
    parser = _server_argument_parser()
    mappings = parser.get_env_mappings()
    owned = (
        set(first_class)
        | {name.lower() for name in RUNTIME_ENVIRONMENT_KEYS}
        | {name.lower() for name in FORCED_ENGINE_ENVIRONMENT}
        | {
            dest
            for dest, name in mappings.items()
            if name in RUNTIME_ENVIRONMENT_KEYS or name in FORCED_ENGINE_ENVIRONMENT
        }
    )
    for name in args:
        if name.upper() == GPU_LOCKED_DEVICES_ENV:
            raise PlanningError(
                f"engine.args.{name} is owned by the GPU runtime wrapper"
            )
        option = "--" + name
        action = parser._option_string_actions.get(option)
        destinations = (
            {action.dest}
            if action is not None
            else {
                candidate.dest
                for spelling, candidate in parser._option_string_actions.items()
                if parser.allow_abbrev
                and spelling.startswith("--")
                and spelling.startswith(option)
            }
        )
        if destinations & owned:
            raise PlanningError(
                f"engine.args.{name} duplicates a runtime-owned or first-class "
                f"config field ({', '.join(sorted(destinations & owned))})"
            )

    forced = {
        **FORCED_ENGINE_ENVIRONMENT,
        "WORLD_SIZE": str(first_class["world_size"]),
    }
    for name, value in environment.items():
        if name in forced:
            if str(value) != forced[name]:
                raise PlanningError(
                    f"engine.env.{name} conflicts with the required runtime value "
                    f"{forced[name]!r}"
                )
        elif name in RUNTIME_ENVIRONMENT_KEYS:
            raise PlanningError(f"engine.env.{name} is owned by the runtime launcher")
