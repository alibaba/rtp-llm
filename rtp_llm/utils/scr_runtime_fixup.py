"""Repair process-local runtime state before releasing an SCR template.

Restore environment input is supplied as data, never shell code. A future
platform file adapter should be registered before arrival and read the file
inside the provider on *each* invocation, not while constructing the seed.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rtp_llm.utils.scr_local_comm import current_pod_ip

if TYPE_CHECKING:
    from rtp_llm.utils.scr_template_lifecycle import TemplateLifecycle


LOGGER = logging.getLogger(__name__)
# Identity/metrics inputs only. GPU topology, model config, SCR control state,
# credentials and filesystem paths require their own explicit repair contracts.
RESTORE_ENV_KEYS = frozenset(
    {
        "RequestedIP",
        "HIPPO_SLAVE_IP",
        "HIPPO_ROLE",
        "HIPPO_ROLE_SHORT_NAME",
        "HIPPO_APP",
        "HIPPO_SERVICE_NAME",
        "kmonitorSinkAddress",
        "kmonitorPort",
    }
)
RestoreEnvProvider = Callable[[str], Mapping[str, str | None]]
_restore_env_provider: RestoreEnvProvider | None = None


@dataclass(frozen=True)
class RestoreRuntimeIdentity:
    generation: str
    pod_ip: str
    environment_keys: tuple[str, ...]


_runtime_identity: RestoreRuntimeIdentity | None = None


def get_restore_runtime_identity() -> RestoreRuntimeIdentity | None:
    """Latest repaired identity in this process; never used as the next input."""
    return _runtime_identity


def register_restore_env_provider(provider: RestoreEnvProvider | None) -> None:
    """Install a process-local adapter before SCR arrival; None disables it.

    The adapter must validate the file's generation/current Pod identity and select
    supported keys. Returning None for an allowed key removes its seed value.
    Each participating process needs its own registration. No format/path is
    assumed until the platform defines the restore environment file contract.
    """
    global _restore_env_provider
    _restore_env_provider = provider


def _validate_environment(values: Mapping[str, str | None]) -> dict[str, str | None]:
    if not isinstance(values, Mapping):
        raise ValueError("restore environment provider must return a mapping")
    result = dict(values)
    if result.keys() - RESTORE_ENV_KEYS:
        raise ValueError("restore environment contains unsupported keys")
    for value in result.values():
        if value is not None and (
            not isinstance(value, str)
            or "\0" in value
            or "\n" in value
            or "\r" in value
        ):
            raise ValueError(
                "restore environment values must be single-line strings or None"
            )
    return result


def fixup_runtime_after_restore(
    generation: str,
    lifecycle: TemplateLifecycle,
    *,
    restore_env: Mapping[str, str | None] | None = None,
) -> RestoreRuntimeIdentity:
    """Fresh inputs -> Logger/identity -> registered component repairs.

    Called only after a successful template barrier, before any release hook.
    Also runs when a checkpoint seed resumes. No cached phase/previous identity
    is trusted to decide whether the process needs repair. Failures propagate;
    the caller must not release serving after a partial fixup.
    """
    global _runtime_identity
    if restore_env is None:
        restore_env = _restore_env_provider(generation) if _restore_env_provider else {}
    updates = _validate_environment(restore_env)
    # Only an explicitly fresh provider may override namespace discovery.
    # os.environ['RequestedIP'] by itself can still be the seed's value.
    supplied_ip = updates.get("RequestedIP")
    pod_ip = current_pod_ip() if supplied_ip is None else supplied_ip
    parsed = ipaddress.IPv4Address(pod_ip)
    if parsed.is_loopback or parsed.is_unspecified or parsed.is_multicast:
        raise ValueError("restore identity requires a non-loopback Pod IP")
    pod_ip = str(parsed)
    extension = sys.modules.get("libth_transformer")
    refresh_logger = getattr(extension, "refresh_logger_after_scr", None)
    if extension is not None and not callable(refresh_logger):
        raise RuntimeError(
            "loaded native library lacks SCR Logger fixup; rebuild the complete image"
        )

    previous = _runtime_identity
    for key, value in updates.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    os.environ["RequestedIP"] = pod_ip
    if refresh_logger is not None:
        refresh_logger(pod_ip)
    identity = RestoreRuntimeIdentity(generation, pod_ip, tuple(sorted(updates)))
    _runtime_identity = identity
    # This class caches identity at import time. Refresh even before the Python
    # reporter decides whether this is a Hippo environment on release.
    hippo_module = sys.modules.get(
        "rtp_llm.aios.kmonitor.python_client.kmonitor.utils.hippo_helper"
    )
    if hippo_module is not None:
        hippo_module.HippoHelper.refresh_runtime_identity()
    # Log only identity and key names, never the file body/environment values.
    LOGGER.info(
        "SCR runtime identity fixed generation=%s previous_runtime_ip=%s pod_ip=%s env_keys=%s",
        generation,
        previous.pod_ip if previous else "unrecorded",
        pod_ip,
        identity.environment_keys,
    )
    if os.environ.get("HIPPO_ROLE") and "HIPPO_SLAVE_IP" not in updates:
        LOGGER.warning(
            "SCR restore input did not supply HIPPO_SLAVE_IP; "
            "host identity retained from environment, freshness unverified"
        )
    lifecycle.restore_fixup(generation)
    return identity
