"""Small real-file/multiprocess tests for Engram table ownership and lookup."""

import errno
import json
import multiprocessing
import os
import pickle
import struct
import sys
import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models.test.dsv41_cache_codec_test import load_codec_source

host = load_codec_source("_dsv41_host_test", "v41_engram_host.py")
Source = host.V41EngramTableSource
Table = host._PinnedSharedTable


def write_fixture(root, rows=7, dim=64, marker=0):
    weights = bytes((i + marker) % 127 for i in range(rows * dim))
    scales = bytes([125, 128] * rows)
    names = ["layers.1.engram.embed.weight", "layers.1.engram.embed.scale"]
    headers = {
        names[0]: dict(
            dtype="F8_E4M3", shape=[rows, dim], data_offsets=[0, len(weights)]
        ),
        names[1]: dict(
            dtype="F8_E8M0",
            shape=[rows, dim // 32],
            data_offsets=[len(weights), len(weights) + len(scales)],
        ),
    }
    raw = json.dumps(headers).encode()
    (root / "table.safetensors").write_bytes(
        struct.pack("<Q", len(raw)) + raw + weights + scales
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "table.safetensors" for name in names}})
    )
    return weights, scales


def attach_child(root, shared, pipe):
    source = Source.from_checkpoint(root, 1, 7, 64, "a" * 64)
    with Table(source, max_bytes=4096, directory=shared) as table:
        embedding = host.HostEngramEmbedding(table)
        result = embedding(
            torch.tensor([[6, 0]]), torch.tensor([True]), backend="eager"
        )
        pipe.send((table._path.stat().st_ino, result.tolist()))
        pipe.recv()
    pipe.close()


def scalar_row(weight_bytes, scale_bytes, dtype=torch.bfloat16):
    values = []
    for channel, code in enumerate(weight_bytes):
        exponent = scale_bytes[channel // 32]
        scale = float("nan") if exponent == 255 else 2.0 ** (exponent - 127)
        magnitude = code & 127
        value = (
            magnitude * 2.0**-9
            if magnitude < 8
            else (1 + (magnitude & 7) / 8) * 2.0 ** ((magnitude >> 3) - 7)
        )
        if magnitude == 127:
            value = float("nan")
        if code & 128:
            value = -value
        values.append(value * scale)
    return torch.tensor(values).to(dtype)


class V41EngramHostTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.shared = self.root / "shared"
        self.shared.mkdir()
        self.weights, self.scales = write_fixture(self.root)
        self.source = Source.from_checkpoint(self.root, 1, 7, 64, "a" * 64)

    def open(self, **kwargs):
        table = Table(
            kwargs.pop("source", self.source),
            max_bytes=kwargs.pop("max_bytes", 4096),
            directory=self.shared,
            **kwargs
        )
        self.addCleanup(table.close)
        return table

    def cuda_mocks(self, table):
        runtime = SimpleNamespace(
            cudaHostRegister=Mock(return_value=(0,)),
            cudaHostGetDevicePointer=Mock(side_effect=lambda p, flags: (0, p)),
            cudaHostUnregister=Mock(return_value=(0,)),
        )
        stack = ExitStack()
        stack.enter_context(
            patch.dict(
                sys.modules,
                {
                    "cuda": SimpleNamespace(bindings=SimpleNamespace(runtime=runtime)),
                    "cuda.bindings": SimpleNamespace(runtime=runtime),
                },
            )
        )
        stack.enter_context(
            patch.object(torch.cuda, "device", side_effect=lambda d: nullcontext())
        )
        stack.enter_context(
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False)
        )
        sync = stack.enter_context(patch.object(torch.cuda, "synchronize"))
        stack.enter_context(
            patch.object(
                torch,
                "as_tensor",
                side_effect=lambda view, device: (
                    table._weight
                    if view.__cuda_array_interface__["data"][0]
                    == table._weight.data_ptr()
                    else table._scales
                ),
            )
        )
        return stack, runtime, sync

    def test_exact_staging_shares_one_file_and_deletes_after_last_owner(self):
        with patch.object(host, "_COPY_BYTES", 17):
            first = self.open()
        path = first._path
        with patch.object(
            host, "_copy_span", side_effect=AssertionError("second owner recopied")
        ):
            second = self.open()
        self.assertEqual(path.read_bytes(), self.weights + self.scales)
        self.assertEqual(len(list(self.shared.glob("*.lease.*"))), 2)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        first.close()
        self.assertTrue(path.exists())
        second.close()
        self.assertFalse(path.exists())
        self.assertFalse(second._ready_path.exists())
        self.assertEqual(list(self.shared.glob("*.lease.*")), [])

    def test_reader_process_keeps_allocation_alive(self):
        parent = self.open()
        context = multiprocessing.get_context("spawn")
        receive, send = context.Pipe()
        child = context.Process(
            target=attach_child, args=(str(self.root), str(self.shared), send)
        )
        child.start()
        try:
            self.assertTrue(receive.poll(20), "child attachment timed out")
            inode, result = receive.recv()
            self.assertEqual(inode, parent._path.stat().st_ino)
            expected = host.HostEngramEmbedding(parent)(
                torch.tensor([[6, 0]]), torch.tensor([True]), backend="eager"
            )
            self.assertEqual(result, expected.tolist())
            parent.close()
            self.assertTrue(parent._path.exists())
            receive.send("close")
            child.join(20)
            self.assertEqual(child.exitcode, 0)
            self.assertFalse(parent._path.exists())
        finally:
            if child.is_alive():
                child.kill()
                child.join(10)
            receive.close()
            send.close()

    def test_killed_worker_lease_is_reclaimed_on_next_attach(self):
        context = multiprocessing.get_context("spawn")
        receive, send = context.Pipe()
        child = context.Process(
            target=attach_child, args=(str(self.root), str(self.shared), send)
        )
        child.start()
        try:
            self.assertTrue(receive.poll(20))
            inode, _ = receive.recv()
            child.kill()
            child.join(10)
            recovered = self.open()
            self.assertEqual(recovered._path.stat().st_ino, inode)
            self.assertEqual(len(list(self.shared.glob("*.lease.*"))), 1)
            recovered.close()
            self.assertEqual(list(self.shared.glob("*.lease.*")), [])
        finally:
            if child.is_alive():
                child.kill()
                child.join(10)
            receive.close()
            send.close()

    def test_concurrent_first_readers_share_one_publication(self):
        context = multiprocessing.get_context("spawn")
        connections, children = [], []
        try:
            for _ in range(2):
                receive, send = context.Pipe()
                connections.append((receive, send))
                child = context.Process(
                    target=attach_child, args=(str(self.root), str(self.shared), send)
                )
                children.append(child)
                child.start()
            inodes = []
            for receive, _ in connections:
                self.assertTrue(receive.poll(20), "concurrent attachment timed out")
                inode, _ = receive.recv()
                inodes.append(inode)
            self.assertEqual(inodes[0], inodes[1])
            self.assertEqual(len(list(self.shared.glob("*.lease.*"))), 2)
            for receive, _ in connections:
                receive.send("close")
            for child in children:
                child.join(20)
                self.assertEqual(child.exitcode, 0)
            self.assertEqual(
                [p for p in self.shared.iterdir() if p.suffix != ".lock"], []
            )
        finally:
            for child in children:
                if child.is_alive():
                    child.kill()
                    child.join(10)
            for receive, send in connections:
                receive.close()
                send.close()

    def test_interrupted_copy_never_publishes_and_retry_recovers(self):
        original = host._copy_span

        def fail_scale(span, destination):
            if span.dtype == "F8_E8M0":
                raise OSError("injected read failure")
            original(span, destination)

        with patch.object(host, "_copy_span", side_effect=fail_scale):
            with self.assertRaisesRegex(OSError, "injected"):
                self.open()
        self.assertEqual([p for p in self.shared.iterdir() if p.suffix != ".lock"], [])
        table = self.open()
        self.assertEqual(table._path.read_bytes(), self.weights + self.scales)

    def test_capacity_and_sync_failure_leave_no_ready_or_mapped_table(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            self.open(max_bytes=1)
        with patch.object(
            host.shutil, "disk_usage", return_value=SimpleNamespace(free=0)
        ):
            with self.assertRaisesRegex(RuntimeError, "insufficient"):
                self.open()
        with patch.object(
            host.os, "fsync", side_effect=OSError(errno.ENOSPC, "no space")
        ), patch.object(host.mmap, "mmap") as mapping:
            with self.assertRaisesRegex(OSError, "no space"):
                self.open()
            mapping.assert_not_called()
        self.assertEqual(list(self.shared.glob("*.partial")), [])
        self.assertEqual(list(self.shared.glob("*.json")), [])

    def test_partial_write_and_short_copy_never_publish_or_map(self):
        def write_failure(span, destination):
            destination.write(b"partial")
            raise OSError(errno.ENOSPC, "injected write exhaustion")

        for copy, error in ((write_failure, OSError), (lambda *args: None, ValueError)):
            with patch.object(host, "_copy_span", side_effect=copy), patch.object(
                host.mmap, "mmap"
            ) as mapping:
                with self.assertRaises(error):
                    self.open()
                mapping.assert_not_called()
            self.assertEqual(
                [p for p in self.shared.iterdir() if p.suffix != ".lock"], []
            )
        with patch.object(
            host.os,
            "posix_fallocate",
            side_effect=AssertionError("emulated preallocation"),
        ):
            table = self.open()
        self.assertEqual(table._path.read_bytes(), self.weights + self.scales)

    def test_bad_live_metadata_cannot_replace_a_readers_storage(self):
        table = self.open()
        inode = table._path.stat().st_ino
        table._ready_path.write_bytes(b"bad metadata")
        with self.assertRaisesRegex(RuntimeError, "readers are active"):
            self.open()
        self.assertEqual(table._path.stat().st_ino, inode)
        table.close()
        with self.open() as rebuilt:
            self.assertEqual(rebuilt._path.read_bytes(), self.weights + self.scales)

    def test_failed_attachment_cleans_unowned_table_and_preserves_live_owner(self):
        for owner, method in ((host.mmap, "mmap"), (host.torch, "frombuffer")):
            with patch.object(owner, method, side_effect=RuntimeError("attach failed")):
                with self.assertRaisesRegex(RuntimeError, "attach failed"):
                    self.open()
            self.assertEqual(
                [p for p in self.shared.iterdir() if p.suffix != ".lock"], []
            )
            with self.open() as table:
                with patch.object(
                    owner, method, side_effect=RuntimeError("attach failed")
                ):
                    with self.assertRaisesRegex(RuntimeError, "attach failed"):
                        self.open()
                self.assertTrue(table._path.exists())
                self.assertEqual(len(list(self.shared.glob("*.lease.*"))), 1)
                self.assertEqual(table._path.read_bytes(), self.weights + self.scales)

    def test_checkpoint_mutation_and_identity_changes_are_fenced(self):
        other = Source.from_checkpoint(self.root, 1, 7, 64, "b" * 64)
        self.assertNotEqual(other.key, self.source.key)
        replacement = self.root / "replacement"
        replacement.mkdir()
        write_fixture(replacement, marker=1)
        os.replace(replacement / "table.safetensors", self.root / "table.safetensors")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            self.open()
        newer = Source.from_checkpoint(self.root, 1, 7, 64, "a" * 64)
        self.assertNotEqual(newer.key, self.source.key)
        with self.open(source=newer) as table:
            self.assertEqual(table._path.read_bytes()[0], 1)

    def test_header_geometry_index_and_stage_ownership(self):
        for rows, dim in ((6, 64), (7, 32), (7, 63)):
            with self.assertRaises(ValueError):
                Source.from_checkpoint(self.root, 1, rows, dim, "a" * 64)
        config = SimpleNamespace(
            text=dict(
                num_hidden_layers=40,
                engram_layer_ids=[1, 14],
                engram_num_embeddings=[7, 7],
                engram_head_dim=64,
            )
        )
        self.assertEqual(
            host.plan_stage_engram_tables(
                config, "/absent-checkpoint", range(21, 40), "a" * 64
            ),
            {},
        )
        plan = host.plan_stage_engram_tables(config, self.root, [1], "a" * 64)
        self.assertEqual(plan[1], self.source)
        with self.assertRaises(ValueError):
            host.plan_stage_engram_tables(config, self.root, [1, 1], "a" * 64)
        index = self.root / "model.safetensors.index.json"
        index.write_text(
            json.dumps({"weight_map": {"layers.1.engram.embed.weight": "../outside"}})
        )
        with self.assertRaisesRegex(ValueError, "checkpoint directory"):
            Source.from_checkpoint(self.root, 1, 7, 64, "a" * 64)

    def test_lookup_matches_scalar_fp8_oracle_and_skips_dead_hashes(self):
        table = self.open()
        embedding = host.HostEngramEmbedding(table)
        indices = torch.tensor([[6, 0], [7, -1], [3, 4]], dtype=torch.int32)
        result = embedding(indices, torch.tensor([True, True, False]), backend="eager")
        for column, row in enumerate((6, 0)):
            expected = scalar_row(
                self.weights[row * 64 : (row + 1) * 64],
                self.scales[row * 2 : (row + 1) * 2],
            )
            torch.testing.assert_close(result[0, column], expected, rtol=0, atol=0)
        self.assertTrue(bool((result[1:] == 0).all()))
        self.assertEqual(
            embedding(
                torch.empty(0, 24, dtype=torch.int64),
                torch.empty(0, dtype=torch.bool),
                backend="eager",
            ).shape,
            (0, 24, 64),
        )

    def test_host_table_stays_outside_module_buffers_and_closed_handles_fail(self):
        table = self.open()
        module = torch.nn.Module()
        module.embedding = host.HostEngramEmbedding(table)
        module.register_buffer("small", torch.ones(1))
        module.to("meta")
        self.assertEqual(table._weight.device.type, "cpu")
        self.assertEqual(list(module.named_buffers())[0][0], "small")
        with self.assertRaises(TypeError):
            pickle.dumps(table)
        table.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            module.embedding(
                torch.zeros(1, 2, dtype=torch.int64),
                torch.ones(1, dtype=torch.bool),
                backend="eager",
            )

    def test_cuda_drain_precedes_unregister_and_last_lease_release(self):
        table = self.open()
        stack, runtime, sync = self.cuda_mocks(table)
        with stack:
            table.register_cuda("cuda:0")
            table.register_cuda("cuda:0")
            runtime.cudaHostRegister.assert_called_once()
            events = []
            sync.side_effect = lambda d: events.append("drain")
            runtime.cudaHostUnregister.side_effect = lambda p: (
                events.append("unregister") or 0,
            )
            table.close()
            self.assertEqual(events, ["drain", "unregister"])
            self.assertFalse(table._path.exists())

    def test_failed_cuda_drain_or_unregister_retains_mapping_until_retry(self):
        for operation in ("drain", "unregister"):
            table = self.open()
            stack, runtime, sync = self.cuda_mocks(table)
            with stack:
                table.register_cuda("cuda:0")
                if operation == "drain":
                    sync.side_effect = RuntimeError("injected drain")
                else:
                    runtime.cudaHostUnregister.return_value = (1,)
                with self.assertRaises(RuntimeError):
                    table.close()
                self.assertIsNotNone(table._mapping)
                self.assertTrue(table._lease_path.exists())
                with self.assertRaisesRegex(RuntimeError, "failed CUDA teardown"):
                    table._check_open()
                sync.side_effect = None
                runtime.cudaHostUnregister.return_value = (0,)
                table.close()
                self.assertFalse(table._path.exists())

    def test_registration_and_device_view_failures_release_correct_resources(self):
        table = self.open()
        stack, runtime, sync = self.cuda_mocks(table)
        with stack:
            runtime.cudaHostRegister.return_value = (1,)
            with self.assertRaisesRegex(RuntimeError, "cudaHostRegister"):
                table.register_cuda("cuda:0")
            runtime.cudaHostUnregister.assert_not_called()
            self.assertTrue(table._path.exists())
            runtime.cudaHostRegister.return_value = (0,)
            runtime.cudaHostGetDevicePointer.side_effect = None
            runtime.cudaHostGetDevicePointer.return_value = (1, 0)
            with self.assertRaisesRegex(RuntimeError, "cudaHostGetDevicePointer"):
                table.register_cuda("cuda:0")
            sync.assert_called_once()
            runtime.cudaHostUnregister.assert_called_once()
            self.assertFalse(table._path.exists())


@unittest.skipUnless(
    os.environ.get("TRITON_INTERPRET") == "1", "Set TRITON_INTERPRET=1"
)
class V41EngramLookupInterpreterTest(unittest.TestCase):
    def test_lookup_strides_tail_mask_and_ue8m0_special_codes(self):
        kernels = load_codec_source(
            "_dsv41_host_interpreter", "_v41_engram_lookup_triton.py"
        )
        raw = torch.tensor(
            [0, 1, 7, 8, 56, 126, 128, 184] * 8, dtype=torch.uint8
        ).repeat(7, 1)
        scales = torch.tensor(
            [
                [0, 127],
                [125, 128],
                [254, 255],
                [127, 127],
                [120, 130],
                [0, 0],
                [255, 255],
            ],
            dtype=torch.uint8,
        )
        indices = torch.tensor(
            [
                [0, 99, 1, 99, 2, 99],
                [3, 99, 4, 99, 5, 99],
                [6, 99, 0, 99, 1, 99],
                [-1, 99, 7, 99, 1, 99],
            ]
        )[:, ::2]
        mask = torch.tensor([True, True, False, True])
        # Interpreter BF16 conversion corrupts subnormals. Check the actual
        # arithmetic in FP32 here; the hardware test retains BF16 special scales.
        actual = torch.empty(4, 3, 64, dtype=torch.float32)
        kernels._lookup_host_kernel[(1,)](
            raw,
            scales,
            indices,
            mask,
            actual,
            4,
            7,
            indices.stride(0),
            indices.stride(1),
            HEADS=3,
            DIM=64,
            BLOCK_D=64,
            BLOCK_R=16,
        )
        expected = torch.zeros_like(actual)
        for token in range(4):
            for head in range(3):
                row = int(indices[token, head])
                if mask[token] and 0 <= row < 7:
                    expected[token, head] = scalar_row(
                        raw[row].tolist(), scales[row].tolist(), torch.float32
                    )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        normal = kernels.lookup_rows(raw, torch.full_like(scales, 127), indices, mask)
        for token in range(4):
            for head in range(3):
                row = int(indices[token, head])
                target = (
                    scalar_row(raw[row].tolist(), [127, 127])
                    if mask[token] and 0 <= row < 7
                    else torch.zeros(64, dtype=torch.bfloat16)
                )
                torch.testing.assert_close(normal[token, head], target, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
