# SPDX-License-Identifier: Apache-2.0
"""Host Fabric shards owned independently of the RTP worker processes."""
from __future__ import annotations

import argparse
import logging
import os
import signal
import stat
import tempfile
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
    parse_numa,
    resolve_checkpoint,
    startup_timeout,
)
from .exchange import (
    MAX_MESSAGE,
    RESIDENT_DIR_ENV,
    RESIDENT_GENERATION_ENV,
    ExchangeError,
    decode,
    encode,
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


def _read_descriptor(path):
    with os.fdopen(
        os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb"
    ) as source:
        _check_private(os.fstat(source.fileno()), stat.S_ISREG)
        return decode(source.read(MAX_MESSAGE + 1))


def _publish_descriptor(path, document):
    # The directory was exclusively claimed before allocating. Rename publishes
    # one complete record even when the owner is killed during the write.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as target:
        tmp = Path(target.name)
        try:
            target.write(encode(document))
            target.flush()
            os.fsync(target.fileno())
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


class ResidentGeneration:
    """One immutable set of owner shards, read only during worker startup."""

    def __init__(self, env, *, checkpoint=None):
        policy = StoragePolicy.from_environ(env)
        if policy.mode != StorageMode.HOST_SHARED:
            raise ShardConfigError("RESIDENT_ENGRAM_REQUIRES_HOST_SHARED")
        root = Path(env.get(RESIDENT_DIR_ENV, ""))
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
        self.directory = root / generation
        self.checkpoint = resolve_checkpoint(env, checkpoint)
        self.timeout = startup_timeout(env)

    def _slot(self, shard):
        return self.directory / f"shard-{shard}"

    def claim(self, shard):
        if not 0 <= shard < self.options.shards.shard_count:
            raise ShardConfigError("INVALID_RESIDENT_ENGRAM_SHARD")
        self.directory.mkdir(mode=0o700, exist_ok=True)
        _check_private(self.directory.lstat(), stat.S_ISDIR)
        slot = self._slot(shard)
        try:
            slot.mkdir(mode=0o700)
        except FileExistsError:
            raise ExchangeError("RESIDENT_ENGRAM_SHARD_ALREADY_CLAIMED") from None
        return slot / "ready.json"

    def load(self):
        deadline = time.monotonic() + self.timeout
        count = self.options.shards.shard_count
        while True:
            try:
                _check_private(self.directory.lstat(), stat.S_ISDIR)
                documents = []
                for shard in range(count):
                    slot = self._slot(shard)
                    _check_private(slot.lstat(), stat.S_ISDIR)
                    doc = _read_descriptor(slot / "ready.json")
                    if doc.get("shard") != shard or doc.get("memory_kind") != "host":
                        raise ExchangeError("RESIDENT_ENGRAM_DESCRIPTOR_MISMATCH")
                    documents.append(doc)
                break
            except FileNotFoundError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ExchangeError("RESIDENT_ENGRAM_STARTUP_TIMEOUT") from None
                time.sleep(min(0.25, remaining))
        manifest, handles = prepare_manifest(
            documents, count, self.generation, self.revision, None
        )
        samples = checkpoint_samples(self.checkpoint, manifest.manifest)
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


def serve_shard(generation, env, shard, host_id, device, should_stop):
    numa = parse_numa(env.get("RTP_ENGRAM_LOCAL_NUMA"))
    allocation_limit, total_limit = capacity_from_environ(env)
    path = generation.claim(shard)
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
            max_allocation_bytes=allocation_limit,
            budget_bytes=total_limit,
            release_input_page_cache=env.get("RTP_ENGRAM_RELEASE_INPUT_PAGE_CACHE")
            == "1",
            deadline=time.monotonic() + generation.timeout,
            reclaim_legacy=False,
        )
        if should_stop():
            return
        _publish_descriptor(path, document)
        _LOGGER.info(
            "engram event=resident_shard_ready generation=%s shard=%d",
            generation.generation,
            shard,
        )
        while not should_stop():
            time.sleep(0.25)
    finally:
        try:
            path.unlink(missing_ok=True)
        finally:
            if publisher.release(generation.generation):
                raise ExchangeError("RESIDENT_ENGRAM_SHARD_CLEANUP_INCOMPLETE")


def main():
    parser = argparse.ArgumentParser(description="Hold one resident host Engram shard.")
    parser.add_argument(
        "--shard", type=int, required=True, help="zero-based shard index"
    )
    parser.add_argument("--host-id", required=True, help="unique storage node identity")
    parser.add_argument("--device", type=int, default=0, help="visible CUDA ordinal")
    args = parser.parse_args()
    if not 0 < len(args.host_id) <= 256 or args.device < 0:
        parser.error("invalid storage host ID or device ordinal")
    logging.basicConfig(level=logging.INFO)
    stopping = False

    def request_stop(*_):
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    generation = ResidentGeneration(os.environ)
    serve_shard(
        generation, os.environ, args.shard, args.host_id, args.device, lambda: stopping
    )


if __name__ == "__main__":
    main()
