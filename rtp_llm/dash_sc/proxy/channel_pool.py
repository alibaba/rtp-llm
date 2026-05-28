"""Loop-bound outbound gRPC channel pool for the dash_sc proxy."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
from typing import Optional

import grpc

from rtp_llm.dash_sc.proto import predict_v2_pb2_grpc

FORWARD_ADDR_ENV_KEY = "DASH_SC_GRPC_FORWARD_ADDR"
CHANNELS_PER_ADDR_ENV_KEY = "DASH_SC_GRPC_FORWARD_CHANNELS_PER_ADDR"
DEFAULT_CHANNELS_PER_ADDR = 1

# Keepalive for forwarder -> backend frontend. These channels carry long-lived
# streams; PING while calls are active so LBS idle timeout does not look like a
# backend failure.
FORWARD_CHANNEL_OPTIONS: tuple[tuple[str, int], ...] = (
    ("grpc.keepalive_time_ms", 30000),
    ("grpc.keepalive_timeout_ms", 10000),
    ("grpc.keepalive_permit_without_calls", 0),
    ("grpc.http2.max_pings_without_data", 0),
)


def parse_forward_addrs(env_value: str) -> list[str]:
    """Parse a single addr, comma-separated addrs, or a JSON array."""
    env_value = env_value.strip()
    if not env_value:
        return []
    if env_value.startswith("["):
        try:
            addrs = json.loads(env_value)
        except json.JSONDecodeError:
            addrs = None
        if isinstance(addrs, list):
            return [str(a).strip() for a in addrs if str(a).strip()]
    return [a.strip() for a in env_value.split(",") if a.strip()]


def parse_channels_per_addr(env_value: str) -> int:
    try:
        value = int(env_value.strip())
    except (TypeError, ValueError, AttributeError):
        return DEFAULT_CHANNELS_PER_ADDR
    return value if value >= 1 else DEFAULT_CHANNELS_PER_ADDR


@dataclasses.dataclass(frozen=True)
class ForwardChannelPoolConfig:
    addrs: tuple[str, ...]
    channels_per_addr: int = DEFAULT_CHANNELS_PER_ADDR
    options: tuple[tuple[str, int], ...] = FORWARD_CHANNEL_OPTIONS

    @classmethod
    def from_env(
        cls,
        *,
        forward_addrs: Optional[list[str]] = None,
        channels_per_addr: Optional[int] = None,
    ) -> "ForwardChannelPoolConfig":
        if forward_addrs is None:
            forward_addrs = parse_forward_addrs(os.environ.get(FORWARD_ADDR_ENV_KEY, ""))
        addrs = tuple(a for a in forward_addrs if a)
        if not addrs:
            raise ValueError(
                f"No forward addresses provided. Set {FORWARD_ADDR_ENV_KEY} env or pass forward_addrs."
            )
        if channels_per_addr is None:
            channels_per_addr = parse_channels_per_addr(
                os.environ.get(CHANNELS_PER_ADDR_ENV_KEY, "")
            )
        elif channels_per_addr < 1:
            channels_per_addr = DEFAULT_CHANNELS_PER_ADDR
        return cls(addrs=addrs, channels_per_addr=channels_per_addr)


@dataclasses.dataclass(frozen=True)
class ForwardEndpoint:
    stub: predict_v2_pb2_grpc.GRPCInferenceServiceStub
    addr: str
    addr_index: int


class LoopBoundForwardChannelPool:
    """Owns grpc.aio channels for exactly one asyncio event loop."""

    def __init__(self, config: ForwardChannelPoolConfig):
        self._config = config
        self._channels: list[grpc.aio.Channel] = []
        self._endpoints: list[ForwardEndpoint] = []
        self._owner_loop: Optional[asyncio.AbstractEventLoop] = None
        self._next_index = 0
        self._closed = False

    @property
    def config(self) -> ForwardChannelPoolConfig:
        return self._config

    @property
    def is_open(self) -> bool:
        return bool(self._endpoints)

    async def open(self) -> None:
        """Open all channels on the current running loop.

        grpc.aio channels are loop-affine. This method is intentionally called
        from ``DashScGrpcServer.start`` on the server owner loop instead of from
        the servicer constructor.
        """
        loop = asyncio.get_running_loop()
        if self._closed:
            raise RuntimeError("forward channel pool is already closed")
        if self._endpoints:
            self._ensure_owner_loop(loop)
            return

        channels: list[grpc.aio.Channel] = []
        endpoints: list[ForwardEndpoint] = []
        for addr_index, addr in enumerate(self._config.addrs):
            for _ in range(self._config.channels_per_addr):
                channel = grpc.aio.insecure_channel(
                    addr, options=list(self._config.options)
                )
                channels.append(channel)
                endpoints.append(
                    ForwardEndpoint(
                        stub=predict_v2_pb2_grpc.GRPCInferenceServiceStub(channel),
                        addr=addr,
                        addr_index=addr_index,
                    )
                )

        self._channels = channels
        self._endpoints = endpoints
        self._owner_loop = loop
        self._next_index = 0
        logging.info(
            "[DashScGrpc] forward channel pool opened: %d addresses x %d channels/addr = %d stubs: %s",
            len(self._config.addrs),
            self._config.channels_per_addr,
            len(self._endpoints),
            list(self._config.addrs),
        )

    def pick(self) -> Optional[ForwardEndpoint]:
        """Return the next endpoint, or ``None`` after shutdown."""
        if not self._endpoints:
            return None
        self._ensure_owner_loop(asyncio.get_running_loop())
        i = self._next_index
        self._next_index = (i + 1) % len(self._endpoints)
        return self._endpoints[i]

    async def close(self) -> None:
        """Close all channels. Safe to call more than once on the owner loop."""
        loop = asyncio.get_running_loop()
        if self._owner_loop is not None:
            self._ensure_owner_loop(loop)
        channels = self._channels
        self._channels = []
        self._endpoints = []
        self._owner_loop = None
        self._closed = True
        for channel in channels:
            try:
                await channel.close()
            except Exception as e:
                logging.warning("[DashScGrpc] forward channel close failed: %s", e)

    def _ensure_owner_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._owner_loop is not None and self._owner_loop is not loop:
            raise RuntimeError(
                "forward channel pool belongs to a different asyncio event loop"
            )
