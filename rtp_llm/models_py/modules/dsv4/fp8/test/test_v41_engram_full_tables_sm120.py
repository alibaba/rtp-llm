"""Opt-in full NAS Engram staging and shared mapped-host lookup on eight GPUs."""

import hashlib
import json
import os
import shutil
import socket
import tempfile
import time
import unittest
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.model_desc.deepseek_v41_model import _SharedTableEmbedding
from rtp_llm.models_py.modules.dsv4.v41_engram_host import (
    _PinnedSharedTable,
    plan_stage_engram_tables,
)


def sampled_rows(source, indices):
    """Read only selected checkpoint rows, independently of the staged mapping."""
    source.check_sources()
    rows, dim = source.weight.shape
    result = torch.zeros((*indices.shape, dim), dtype=torch.bfloat16)
    with open(source.weight.path, "rb") as weights, open(
        source.scales.path, "rb"
    ) as scales:
        for offset, row in enumerate(indices.flatten().tolist()):
            if not 0 <= row < rows:
                continue
            weights.seek(source.weight.offset + row * dim)
            scales.seek(source.scales.offset + row * (dim // 32))
            payload = bytearray(weights.read(dim))
            exponents = bytearray(scales.read(dim // 32))
            assert len(payload) == dim and len(exponents) == dim // 32
            value = torch.frombuffer(payload, dtype=torch.float8_e4m3fn).float()
            scale = torch.frombuffer(exponents, dtype=torch.float8_e8m0fnu).float()
            result.flatten(0, 1)[offset] = (
                value * scale.repeat_interleave(32)
            ).bfloat16()
    source.check_sources()
    return result


def run_rank(rank, port, sources, directory, identities):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() == (12, 0)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=600),
    )
    try:
        with ExitStack() as owners:
            tables = []
            for source, identity in zip(sources, identities):
                started = time.monotonic()
                table = owners.enter_context(
                    _PinnedSharedTable(
                        source, max_bytes=source.nbytes, directory=directory
                    )
                )
                stat = table._path.stat()
                assert (stat.st_dev, stat.st_ino) == identity
                allocated = torch.cuda.memory_allocated()
                table.register_cuda(torch.device("cuda", rank))
                assert torch.cuda.memory_allocated() - allocated < 2**20
                tables.append(table)
                print(
                    f"rank={rank} layer={source.layer_id} registered "
                    f"bytes={source.nbytes} seconds={time.monotonic() - started:.2f}",
                    flush=True,
                )
            # Hold both complete tables registered on every device at once.
            dist.monitored_barrier(timeout=timedelta(seconds=600))
            for table in tables:
                source = table.source
                rows = source.weight.shape[0]
                generator = torch.Generator().manual_seed(
                    41970 + rank + source.layer_id
                )
                for phase in range(2):
                    indices = torch.randint(rows, (3, 24), generator=generator)
                    indices[0, :6] = torch.tensor([0, 1, rows // 2, rows - 1, -1, rows])
                    expected = sampled_rows(source, indices)
                    actual = _SharedTableEmbedding(table)(
                        indices.cuda(), torch.device("cuda", rank)
                    ).cpu()
                    torch.testing.assert_close(
                        actual, expected, rtol=0, atol=0, equal_nan=True
                    )
                    print(
                        f"rank={rank} layer={source.layer_id} phase={phase} "
                        "sampled NAS lookup parity PASS",
                        flush=True,
                    )
            dist.monitored_barrier(timeout=timedelta(seconds=600))
        assert all(table._closed and table._mapping is None for table in tables)
        dist.monitored_barrier(timeout=timedelta(seconds=600))
    finally:
        dist.destroy_process_group()


class V41EngramFullTablesSM120Test(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("DSV41_TEST_FULL_ENGRAM") == "1",
        "full Engram staging requires an explicit resource admission",
    )
    def test_full_tables_shared_registration_lookup_and_teardown(self):
        self.assertEqual(torch.cuda.device_count(), 8)
        checkpoint = Path(os.environ["DSV41_TEST_CHECKPOINT"]).resolve()
        directory = Path(os.environ["DSV41_ENGRAM_CACHE_DIR"]).resolve(strict=True)
        config = json.loads((checkpoint / "config.json").read_text())["text_config"]
        identity = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        sources = list(
            plan_stage_engram_tables(
                SimpleNamespace(text=config), checkpoint, [1, 14], identity
            ).values()
        )
        self.assertEqual([source.layer_id for source in sources], [1, 14])
        total = sum(source.nbytes for source in sources)
        self.assertGreater(total, 180 * 2**30)
        self.assertLessEqual(total, int(os.environ["DSV41_ENGRAM_HOST_BYTES"]))
        self.assertGreater(shutil.disk_usage(directory).free, total + 64 * 2**30)
        memory = {
            key: int(value.split()[0]) * 1024
            for key, value in (
                line.split(":", 1)
                for line in Path("/proc/meminfo").read_text().splitlines()
            )
        }
        # Admit source page cache plus pinned destination with 128 GiB headroom.
        self.assertGreater(memory["MemAvailable"], 2 * total + 128 * 2**30)
        # This large manual test requires the inspected, unconstrained v1 container.
        # Constrained/v2 containers need a separate hierarchical resource review.
        cgroup = Path("/sys/fs/cgroup/memory/memory.stat")
        limits = dict(line.split() for line in cgroup.read_text().splitlines())
        self.assertGreaterEqual(
            int(limits["hierarchical_memory_limit"]), memory["MemTotal"]
        )
        print(
            f"admitted bytes={total} available_ram={memory['MemAvailable']} "
            f"free_storage={shutil.disk_usage(directory).free}",
            flush=True,
        )
        with tempfile.TemporaryDirectory(
            prefix="v41-full-engram-", dir=directory
        ) as root:
            with ExitStack() as owners:
                tables = []
                for source in sources:
                    started = time.monotonic()
                    print(
                        f"staging layer={source.layer_id} bytes={source.nbytes}",
                        flush=True,
                    )
                    tables.append(
                        owners.enter_context(
                            _PinnedSharedTable(
                                source, max_bytes=source.nbytes, directory=root
                            )
                        )
                    )
                    print(
                        f"staged layer={source.layer_id} "
                        f"seconds={time.monotonic() - started:.2f}",
                        flush=True,
                    )
                paths = [table._path for table in tables]
                identities = [
                    (path.stat().st_dev, path.stat().st_ino) for path in paths
                ]
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                mp.spawn(
                    run_rank,
                    args=(port, sources, root, identities),
                    nprocs=8,
                    join=True,
                )
                self.assertTrue(all(path.exists() for path in paths))
                self.assertEqual(len(list(Path(root).glob("*.lease.*"))), 2)
            self.assertTrue(all(not path.exists() for path in paths))
            self.assertEqual(list(Path(root).glob("*.json")), [])
            self.assertEqual(list(Path(root).glob("*.lease.*")), [])
            print("full L1/L14 Engram DP8 lookup and teardown PASS", flush=True)


if __name__ == "__main__":
    unittest.main()
