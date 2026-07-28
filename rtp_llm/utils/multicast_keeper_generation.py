"""Keeper-only restart fencing for multicast process groups.

The fence is activated only for Level3 sleep with the multicast keeper enabled.
It uses private TCPStore keys and does not modify DistributedServer bootstrap.
Its single purpose is to reject a restarted rank before that rank can reattach
GPUs to a surviving FABRIC multicast object.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, Mapping, Optional

from rtp_llm.utils.multicast_keeper import ENABLE_ENV

if TYPE_CHECKING:
    from rtp_llm.config.py_config_modules import PyEnvConfigs

_LOGGER = logging.getLogger(__name__)

PROTOCOL = "v1"
ROOT = f"rtp_llm/multicast_keeper/restart_fence/{PROTOCOL}"
CURRENT_EPOCH_KEY = f"{ROOT}/current_epoch"
MONITOR_INTERVAL_SECONDS = 0.25
GANG_EPOCH_ENV = "RTP_LLM_MC_GANG_EPOCH"


class MulticastGenerationError(RuntimeError):
    """The multicast keeper generation is stale or has been aborted."""


def _decode_store_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def generation_guard_enabled(
    py_env_configs: "PyEnvConfigs",
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Enable only for an explicitly configured Level3 keeper runtime."""

    source = os.environ if env is None else env
    runtime_config = py_env_configs.runtime_config
    return (
        bool(getattr(runtime_config, "enable_sleep_mode", False))
        and int(getattr(runtime_config, "sleep_mode_level", 1) or 1) == 3
        and source.get(ENABLE_ENV) == "1"
    )


class MulticastGenerationGuard:
    """Fence a restarted rank out of a surviving multicast generation."""

    @staticmethod
    def epoch_prefix(epoch: str) -> str:
        return f"{ROOT}/epoch/{epoch}"

    @classmethod
    def rank_incarnation_key(cls, epoch: str, rank: int) -> str:
        return f"{cls.epoch_prefix(epoch)}/rank/{rank}/incarnation"

    @classmethod
    def abort_key(cls, epoch: str) -> str:
        return f"{cls.epoch_prefix(epoch)}/abort"

    @classmethod
    def from_config(
        cls,
        py_env_configs: "PyEnvConfigs",
        *,
        store: Any,
        rank: int,
        world_size: int,
        env: Optional[Mapping[str, str]] = None,
    ) -> Optional["MulticastGenerationGuard"]:
        if not generation_guard_enabled(py_env_configs, env):
            return None
        return cls(store=store, rank=rank, world_size=world_size)

    def __init__(
        self,
        *,
        store: Any,
        rank: int,
        world_size: int,
        incarnation: Optional[str] = None,
    ):
        self.store = store
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.world_size <= 0 or self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(
                f"invalid multicast generation rank {self.rank}/{self.world_size}"
            )
        self.incarnation = incarnation or uuid.uuid4().hex
        self.epoch = ""
        self._joined = False
        self._abort_event = threading.Event()
        self._abort_reason = ""
        self._monitor_stop = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

    def _get_text(self, key: str) -> str:
        return _decode_store_value(self.store.get(key))

    def _set_local_abort(self, reason: str) -> None:
        if not self._abort_event.is_set():
            self._abort_reason = reason
            _LOGGER.error(
                "multicast keeper restart fence aborted: epoch=%s rank=%d "
                "incarnation=%s reason=%s",
                self.epoch,
                self.rank,
                self.incarnation,
                reason,
            )
        self._abort_event.set()

    @property
    def abort_reason(self) -> str:
        return self._abort_reason

    def is_aborted(self) -> bool:
        return self._abort_event.is_set()

    def abort(self, reason: str) -> None:
        """Poison the epoch so surviving ranks leave before a peer restarts."""

        reason = str(reason or "unspecified keeper failure").replace("\x00", "")[
            :2048
        ]
        self._set_local_abort(reason)
        if not self.epoch:
            return
        payload = json.dumps(
            {
                "epoch": self.epoch,
                "rank": self.rank,
                "incarnation": self.incarnation,
                "reason": reason,
                "time_ns": time.time_ns(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            self.store.compare_set(self.abort_key(self.epoch), "", payload)
        except Exception:
            _LOGGER.exception(
                "failed to publish multicast keeper restart ABORT: "
                "epoch=%s rank=%d",
                self.epoch,
                self.rank,
            )

    def _raise_if_aborted(self) -> None:
        if self._abort_event.is_set():
            raise MulticastGenerationError(
                "multicast keeper generation is aborted: "
                f"epoch={self.epoch} reason={self._abort_reason}"
            )
        abort_key = self.abort_key(self.epoch)
        try:
            if self.store.check([abort_key]):
                reason = self._get_text(abort_key)
                self._set_local_abort(reason)
                raise MulticastGenerationError(
                    "multicast keeper generation is aborted: "
                    f"epoch={self.epoch} reason={reason}"
                )
        except MulticastGenerationError:
            raise
        except Exception as e:
            reason = f"multicast keeper TCPStore unavailable: {e}"
            self._set_local_abort(reason)
            raise MulticastGenerationError(reason) from e

    def _start_monitor(self) -> None:
        if self._monitor_thread is not None:
            return

        def monitor() -> None:
            abort_key = self.abort_key(self.epoch)
            while not self._monitor_stop.wait(MONITOR_INTERVAL_SECONDS):
                try:
                    if self.store.check([abort_key]):
                        self._set_local_abort(self._get_text(abort_key))
                        return
                except Exception as e:
                    self._set_local_abort(
                        f"multicast keeper TCPStore monitor failed: {e}"
                    )
                    return

        self._monitor_thread = threading.Thread(
            target=monitor,
            name=f"multicast_restart_fence_rank_{self.rank}",
            daemon=True,
        )
        self._monitor_thread.start()

    def join(self) -> str:
        """Claim the rank slot before NCCL/SymmetricMemory initialization."""

        if self._joined:
            return self.epoch
        if self.rank == 0:
            proposed_epoch = uuid.uuid4().hex
            observed = _decode_store_value(
                self.store.compare_set(CURRENT_EPOCH_KEY, "", proposed_epoch)
            )
            if observed != proposed_epoch:
                self.epoch = observed
                reason = (
                    "multicast keeper TCPStore already contains an active "
                    f"epoch: proposed={proposed_epoch} active={observed}"
                )
                self.abort(reason)
                raise MulticastGenerationError(reason)
            self.epoch = proposed_epoch
        else:
            self.epoch = self._get_text(CURRENT_EPOCH_KEY)
        if not self.epoch:
            raise MulticastGenerationError("multicast keeper epoch is empty")

        os.environ[GANG_EPOCH_ENV] = self.epoch
        self._raise_if_aborted()
        incarnation_key = self.rank_incarnation_key(self.epoch, self.rank)
        observed = _decode_store_value(
            self.store.compare_set(incarnation_key, "", self.incarnation)
        )
        if observed != self.incarnation:
            reason = (
                f"stale multicast keeper rank join rejected for rank {self.rank}: "
                f"epoch={self.epoch} active_incarnation={observed} "
                f"joining_incarnation={self.incarnation}"
            )
            self.abort(reason)
            raise MulticastGenerationError(reason)

        self._raise_if_aborted()
        self._joined = True
        self._start_monitor()
        _LOGGER.info(
            "multicast keeper restart fence JOINED before collective "
            "initialization: epoch=%s rank=%d/%d incarnation=%s",
            self.epoch,
            self.rank,
            self.world_size,
            self.incarnation,
        )
        return self.epoch

    def stop(self) -> None:
        self._monitor_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=1.0)
            self._monitor_thread = None


__all__ = [
    "CURRENT_EPOCH_KEY",
    "GANG_EPOCH_ENV",
    "MulticastGenerationError",
    "MulticastGenerationGuard",
    "generation_guard_enabled",
]
