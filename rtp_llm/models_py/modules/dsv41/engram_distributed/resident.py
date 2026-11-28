# SPDX-License-Identifier: Apache-2.0
"""Fabric shards owned independently of the RTP worker processes."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import stat
import time
from pathlib import Path
from uuid import UUID

from .checkpoint import ShardPublisher, checkpoint_samples
from .config import (
    EXTERNAL_GRAPH_MODE_ENV,
    REVISION_ENV,
    EngramOptions,
    ShardConfig,
    ShardConfigError,
    StorageMode,
    StoragePolicy,
    capacity_from_environ,
    gpu_reserve_bytes,
    parse_numa,
    resolve_checkpoint,
    startup_timeout,
)
from .exchange import (
    RESIDENT_DIR_ENV,
    RESIDENT_GENERATION_ENV,
    ExchangeError,
)
from .manifest import prepare_manifest

_LOGGER = logging.getLogger(__name__)


def _check_private(info, is_expected_type):
    if (
        not is_expected_type(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o077
    ):
        raise ExchangeError("UNSAFE_RESIDENT_ENGRAM_METADATA")


class ResidentGeneration:
    """One immutable set of owner shards, read only during worker startup."""

    def __init__(self, env, *, checkpoint=None):
        from .resident_network import OwnerClients, STATE_ENV, TRANSPORT_ENV

        policy = StoragePolicy.from_environ(env)
        if policy.mode not in (StorageMode.HOST_SHARED, StorageMode.GPU_SHARED):
            raise ShardConfigError("RESIDENT_ENGRAM_REQUIRES_FABRIC_SHARED")
        if RESIDENT_DIR_ENV in env or env.get(TRANSPORT_ENV, "tcp") != "tcp":
            raise ShardConfigError("RESIDENT_C2_TCP_REQUIRED")
        root = Path(env.get(STATE_ENV, ""))
        if not root.is_absolute():
            raise ShardConfigError("RESIDENT_ENGRAM_REQUIRES_ABSOLUTE_DIRECTORY")
        _check_private(root.lstat(), stat.S_ISDIR)
        generation = env.get(RESIDENT_GENERATION_ENV, "")
        try:
            if str(UUID(generation)) != generation:
                raise ValueError()
        except (ValueError, TypeError, AttributeError):
            raise ShardConfigError("INVALID_RESIDENT_ENGRAM_GENERATION") from None
        self.revision = env.get(REVISION_ENV, "")
        if not 0 < len(self.revision) <= 256:
            raise ShardConfigError("RESIDENT_ENGRAM_REQUIRES_MODEL_REVISION")
        self.options = EngramOptions(
            policy,
            ShardConfig.from_environ(env),
            env.get(EXTERNAL_GRAPH_MODE_ENV, policy.default_graph_mode),
        )
        self.generation = generation
        self.checkpoint = resolve_checkpoint(env, checkpoint)
        self.timeout = startup_timeout(env)
        self.network = OwnerClients(env, generation, self.revision, root / generation)

    def load(self):
        deadline = time.monotonic() + self.timeout
        count = self.options.shards.shard_count
        documents = self.network.load(deadline)
        if any(
            doc.get("memory_kind") != self.options.policy.memory_kind
            for doc in documents
        ):
            raise ExchangeError("RESIDENT_ENGRAM_DESCRIPTOR_MISMATCH")
        manifest, handles = prepare_manifest(
            documents, count, self.generation, self.revision, None
        )
        samples = checkpoint_samples(self.checkpoint, manifest.manifest)
        self.network.pin(documents)
        _LOGGER.info(
            "engram event=resident_generation_staged generation=%s shards=%d",
            self.generation,
            count,
        )
        return dict(
            manifest=manifest,
            generation=self.generation,
            model_revision=self.revision,
            capabilities=handles,
            samples=samples,
        )


def serve_shard(generation, env, shard, host_id, device, should_stop, gpu_devices=None):
    from .resident_network import OwnerEndpoint

    if not 0 <= shard < generation.options.shards.shard_count:
        raise ShardConfigError("INVALID_RESIDENT_ENGRAM_SHARD")
    memory_kind = generation.options.policy.memory_kind
    if memory_kind == "gpu":
        gpu_devices = tuple(gpu_devices or (device,))
        reserve = gpu_reserve_bytes(env)
    else:
        if gpu_devices is not None:
            raise ShardConfigError("RESIDENT_ENGRAM_GPU_DEVICES_REQUIRE_GPU_SHARED")
        reserve = 0
    numa = parse_numa(env.get("RTP_ENGRAM_LOCAL_NUMA"))
    allocation_limit, total_limit = capacity_from_environ(env)
    with OwnerEndpoint(generation, shard) as endpoint:
        publisher = ShardPublisher()
        try:
            document = publisher.publish(
                generation.checkpoint,
                generation.revision,
                host_id,
                generation.generation,
                shard,
                generation.options.shards.shard_count,
                device=device,
                numa=numa,
                memory_kind=memory_kind,
                gpu_devices=gpu_devices,
                gpu_reserve_bytes=reserve,
                max_allocation_bytes=allocation_limit,
                budget_bytes=total_limit,
                release_input_page_cache=env.get("RTP_ENGRAM_RELEASE_INPUT_PAGE_CACHE")
                == "1",
                deadline=time.monotonic() + generation.timeout,
                reclaim_legacy=False,
            )
            if should_stop():
                return
            endpoint.publish(document)
            _LOGGER.info(
                "engram event=resident_shard_ready generation=%s shard=%d",
                generation.generation,
                shard,
            )
            while not should_stop():
                time.sleep(0.25)
        finally:
            endpoint.withdraw()
            if publisher.release(generation.generation):
                raise ExchangeError("RESIDENT_ENGRAM_SHARD_CLEANUP_INCOMPLETE")


def main():
    parser = argparse.ArgumentParser(
        description="Hold one resident Fabric Engram shard."
    )
    parser.add_argument("--shard", type=int, help="zero-based shard index")
    parser.add_argument("--host-id", help="unique storage node identity")
    parser.add_argument(
        "--discover",
        action="store_true",
        help="discover this network Owner from gang membership",
    )
    parser.add_argument("--device", type=int, default=0, help="visible CUDA ordinal")
    parser.add_argument(
        "--gpu-devices", type=int, nargs="+", help="GPU backing ordinals"
    )
    args = parser.parse_args()
    if args.discover and (args.shard is not None or args.host_id is not None):
        parser.error("--discover cannot be combined with --shard/--host-id")
    if not args.discover and (
        args.shard is None or not args.host_id or len(args.host_id) > 256
    ):
        parser.error("--shard and --host-id are required without --discover")
    if args.device < 0:
        parser.error("invalid storage host ID or device ordinal")
    logging.basicConfig(level=logging.INFO)
    stopping = False

    def request_stop(*_):
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    generation = ResidentGeneration(os.environ)
    if args.discover:
        from .config import _instance_index

        args.shard = _instance_index(
            os.environ, generation.network.peers, generation.network.source_ip
        )
        args.host_id = generation.network.peers[args.shard][0]
        if generation.options.policy.memory_kind == "gpu" and args.gpu_devices is None:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            local_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
            if world_size > 1 and local_size == 1:
                import torch

                local_size = min(world_size, torch.cuda.device_count())
            if not 1 <= local_size <= world_size:
                raise ShardConfigError("INVALID_RESIDENT_ENGRAM_GPU_WORLD")
            args.gpu_devices = tuple(range(local_size))
    serve_shard(
        generation,
        os.environ,
        args.shard,
        args.host_id,
        args.device,
        lambda: stopping,
        gpu_devices=args.gpu_devices,
    )


if __name__ == "__main__":
    main()
