# SPDX-License-Identifier: Apache-2.0
"""CPU protocol/geometry gates; CUDA Fabric reachability is tested separately."""

import base64
import json
import os
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.models_py.modules.dsv41 import engram_checkpoint as checkpoint
from rtp_llm.models_py.modules.dsv41 import engram_local_cuda as storage
from rtp_llm.models_py.modules.dsv41.engram_manifest import ExpectedLayout
from rtp_llm.models_py.modules.dsv41.engram_shard import plan_shards


def _write_checkpoint(root, revision="a"):
    """Write a real, small safetensors file without allocating any GPU tensor."""
    header, weight_map, payload = (
        {"__metadata__": {"revision": revision}},
        {},
        bytearray(),
    )
    for role, columns, dtype in (("weight", 64, "F8_E4M3"), ("scale", 2, "U8")):
        name = f"layers.1.engram.embed.{role}"
        start = len(payload)
        payload.extend(bytes([0x38 if role == "weight" else 127]) * (8 * columns))
        header[name] = dict(
            dtype=dtype, shape=[8, columns], data_offsets=[start, len(payload)]
        )
        weight_map[name] = "engram.safetensors"
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    (root / "engram.safetensors").write_bytes(
        len(raw).to_bytes(8, "little") + raw + payload
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )


class DeviceFabricGeometryTest(unittest.TestCase):
    def test_real_two_layer_32dp_allocation_is_per_rank_not_replicated(self):
        plans = [
            plan_shards(rows, 256, 32, granularity=2 << 20)
            for rows in (384006168, 384016682)
        ]
        for plan in plans:
            self.assertEqual(plan.shard_rows, 12058624)
            self.assertEqual(plan.weight_bytes, 3087007744)
            self.assertEqual(plan.scale_bytes, 96468992)
            self.assertEqual(plan.weight_bytes % (2 << 20), 0)
            self.assertEqual(plan.scale_bytes % (2 << 20), 0)
            cursor = 0
            for rank in range(32):
                begin, end = plan.row_range(rank)
                self.assertEqual(begin, cursor)
                self.assertLess(begin, end)
                self.assertEqual(plan.shard_of_row(begin), rank)
                self.assertEqual(plan.shard_of_row(end - 1), rank)
                cursor = end
            self.assertEqual(cursor, plan.total_rows)
        per_rank = sum(plan.shard_bytes for plan in plans)
        self.assertEqual(per_rank, 6366953472)
        self.assertEqual(per_rank / (1 << 30), 5.9296875)


class PortableCheckpointIdentityTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.first, self.second = self.root / "host-a", self.root / "host-b"
        self.first.mkdir()
        self.second.mkdir()
        _write_checkpoint(self.first)
        for source in self.first.iterdir():
            destination = self.second / source.name
            shutil.copyfile(source, destination)
            os.utime(destination, ns=(1000000000, 1000000000))
        self.layout = ExpectedLayout(
            layers=((1, 8),),
            head_dim=64,
            model_revision="portable-v1",
            hash_abi="sha256:" + "a" * 64,
        )
        self.plan = checkpoint.inspect_checkpoint(self.first, self.layout)

    def identity(self, plan=None, layout=None):
        return storage._checkpoint_identity(plan or self.plan, layout or self.layout)

    def test_distinct_local_stat_tokens_and_roots_do_not_change_identity(self):
        replica = checkpoint.inspect_checkpoint(self.second, self.layout)
        self.assertNotEqual(self.plan.root, replica.root)
        self.assertNotEqual(self.plan.index.stat_token, replica.index.stat_token)
        self.assertNotEqual(self.plan.files[0].stat_token, replica.files[0].stat_token)
        # A remote filesystem may also have a different device/inode pair.
        remote = replace(
            replica,
            index=replace(
                replica.index, stat_token=(91, 92, replica.index.stat_token[2], 93, 94)
            ),
            files=tuple(
                replace(file, stat_token=(81, 82, file.stat_token[2], 83, 84))
                for file in replica.files
            ),
        )
        self.assertRegex(self.identity(), r"^[0-9a-f]{64}$")
        self.assertEqual(self.identity(), self.identity(remote))

    def test_changed_safetensors_header_and_index_change_identity(self):
        _write_checkpoint(self.second, revision="b")
        changed = checkpoint.inspect_checkpoint(self.second, self.layout)
        self.assertEqual(
            self.plan.files[0].stat_token[2], changed.files[0].stat_token[2]
        )
        self.assertNotEqual(self.identity(), self.identity(changed))
        _write_checkpoint(self.second)
        index = self.second / "model.safetensors.index.json"
        index.write_text(index.read_text() + "\n")
        self.assertNotEqual(
            self.identity(),
            self.identity(checkpoint.inspect_checkpoint(self.second, self.layout)),
        )

    def test_layout_tensor_schema_and_file_size_are_part_of_identity(self):
        for layout in (
            replace(self.layout, layers=((1, 9),)),
            replace(self.layout, head_dim=128),
            replace(self.layout, model_revision="portable-v2"),
            replace(self.layout, hash_abi="sha256:" + "b" * 64),
        ):
            with self.subTest(layout=layout):
                self.assertNotEqual(self.identity(), self.identity(layout=layout))
        layer = self.plan.layers[0]
        tensor = replace(
            layer.tensors[0], data_range=(1, layer.tensors[0].data_range[1] + 1)
        )
        changed = replace(
            self.plan, layers=(replace(layer, tensors=(tensor, layer.tensors[1])),)
        )
        self.assertNotEqual(self.identity(), self.identity(changed))
        file = self.plan.files[0]
        stat = list(file.stat_token)
        stat[2] += 1
        self.assertNotEqual(
            self.identity(),
            self.identity(
                replace(self.plan, files=(replace(file, stat_token=tuple(stat)),))
            ),
        )


class DeviceFabricHandleProtocolTest(unittest.TestCase):
    def setUp(self):
        self.handles = [bytes(range(64)), bytes(reversed(range(64)))]
        self.document = dict(
            location_type="DEVICE",
            handle_type="FABRIC",
            handles=[base64.b64encode(raw).decode("ascii") for raw in self.handles],
        )

    def assert_invalid(self, document, count=2, handle_bytes=64):
        with self.assertRaisesRegex(ValueError, "INVALID_DEVICE_FABRIC_HANDLES"):
            storage._decode_fabric_handles(document, count, handle_bytes)

    def test_real_wire_json_round_trip_preserves_opaque_bytes(self):
        wire = json.loads(json.dumps(self.document))
        self.assertEqual(storage._decode_fabric_handles(wire, 2, 64), self.handles)

    def test_only_exact_device_fabric_metadata_is_accepted(self):
        for document in (
            None,
            [],
            {
                key: value
                for key, value in self.document.items()
                if key != "handle_type"
            },
            {**self.document, "fd": 3},
            {**self.document, "location_type": "HOST_NUMA"},
            {**self.document, "handle_type": "POSIX_FD"},
            {**self.document, "location_type": "device"},
            {**self.document, "handles": tuple(self.document["handles"])},
            {**self.document, "handles": self.document["handles"][:1]},
        ):
            with self.subTest(document=document):
                self.assert_invalid(document)

    def test_base64_is_strict_canonical_and_exactly_fixed_size(self):
        encoded = self.document["handles"][0]
        noncanonical = base64.b64encode(bytes(64)).decode("ascii")[:-3] + "B=="
        for invalid in (
            None,
            encoded.encode(),
            encoded[:-1],
            "*" + encoded[1:],
            encoded[:10] + "\n" + encoded[11:],
            noncanonical,
            base64.b64encode(bytes(63)).decode("ascii"),
            base64.b64encode(bytes(65)).decode("ascii"),
        ):
            with self.subTest(value=invalid):
                self.assert_invalid(
                    {**self.document, "handles": [invalid, self.document["handles"][1]]}
                )

    def test_count_and_handle_size_require_positive_integers(self):
        for invalid in (None, 0, -1, True, 1.0, "2"):
            with self.subTest(value=invalid):
                self.assert_invalid(self.document, count=invalid)
                self.assert_invalid(self.document, handle_bytes=invalid)
        self.assert_invalid(self.document, count=1)
        self.assert_invalid(self.document, count=3)


class DeviceFabricFailClosedTest(unittest.TestCase):
    def test_model_attach_consumes_matching_staged_device_storage_and_pins_graph(self):
        """CPU control flow only: model/layer/hash/storage factories are fakes."""
        from rtp_llm.models_py.modules.dsv41 import engram_bootstrap as bootstrap
        from rtp_llm.models_py.modules.dsv41 import engram_model_attach as attach

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = str(Path(directory).resolve(strict=True))
            expected = ExpectedLayout(
                layers=((1, 8),),
                head_dim=64,
                model_revision="staged-v1",
                hash_abi="sha256:" + "a" * 64,
            )
            layout = SimpleNamespace(layer_ids=(1,), num_embeddings=(8,), head_dim=64)
            for mode in ("cuda_local", "cuda_fabric"):
                for matching in (True, False):
                    with self.subTest(mode=mode, matching=matching):
                        model = SimpleNamespace(
                            _v4_args=SimpleNamespace(
                                v41_config={"checkpoint_path": checkpoint_path}
                            ),
                            v4=SimpleNamespace(
                                embed=SimpleNamespace(
                                    weight=SimpleNamespace(device="cuda:0")
                                ),
                                layers=[SimpleNamespace(engram=None) for _ in range(2)],
                            ),
                            py_hw_kernel_config=SimpleNamespace(enable_cuda_graph=True),
                            _is_decode_role=True,
                        )
                        staged = SimpleNamespace(
                            storage_mode=(
                                mode
                                if matching
                                else (
                                    "cuda_fabric"
                                    if mode == "cuda_local"
                                    else "cuda_local"
                                )
                            ),
                            checkpoint_path=checkpoint_path,
                            expected_layout=expected,
                            require_model_layout=Mock(),
                            _pin_graph=Mock(),
                            _unpin_graph=Mock(),
                            close=Mock(),
                        )
                        module, hash_state = (
                            SimpleNamespace(release_external_views=Mock()),
                            object(),
                        )
                        layer_factory, retain = Mock(return_value=module), Mock()
                        fallback = Mock(
                            side_effect=AssertionError("must consume staged storage")
                        )
                        kwargs = dict(
                            layout_factory=Mock(return_value=layout),
                            hash_factory=Mock(return_value=hash_state),
                            layer_factory=layer_factory,
                            attach_factory=fallback,
                            retain=retain,
                            environ={"RTP_ENGRAM_STORAGE": mode},
                        )
                        with patch.object(
                            bootstrap, "staged_package", return_value=None
                        ), patch.object(
                            bootstrap,
                            "hash_state_digest",
                            return_value=expected.hash_abi,
                        ), patch.object(
                            storage, "staged_local_cuda_storage", return_value=staged
                        ) as getter:
                            if not matching:
                                with self.assertRaisesRegex(
                                    attach.EngramModelInitializationError,
                                    "GPU_ENGRAM_STORAGE_MODE_MISMATCH",
                                ):
                                    attach.initialize_model_engram(model, **kwargs)
                                self.assertEqual(
                                    model._engram_attachment.phase, "FAILED"
                                )
                                staged.close.assert_called_once_with()
                                staged._pin_graph.assert_not_called()
                                layer_factory.assert_not_called()
                                retain.assert_not_called()
                            else:
                                state = attach.initialize_model_engram(model, **kwargs)
                                self.assertEqual(
                                    (state.phase, state.graph_mode),
                                    ("READY", "process_lifetime"),
                                )
                                self.assertTrue(
                                    state.external
                                    and state.graph_pin_held
                                    and state.process_retained
                                )
                                self.assertIs(state.storage, staged)
                                self.assertIs(model.v4.layers[1].engram, module)
                                staged.require_model_layout.assert_called_once_with(
                                    (1,), (8,), 64
                                )
                                layer_factory.assert_called_once_with(
                                    model._v4_args.v41_config,
                                    1,
                                    checkpoint_path,
                                    "cuda:0",
                                    external_storage=staged,
                                )
                                staged._pin_graph.assert_called_once_with()
                                retain.assert_called_once_with(staged)
                                self.assertIs(
                                    attach.initialize_model_engram(model, **kwargs),
                                    state,
                                )
                                staged._pin_graph.assert_called_once_with()
                                attach.require_engram_runtime(
                                    model, is_cuda_graph=True, is_target_verify=True
                                )
                                model._is_decode_role = False
                                with self.assertRaisesRegex(
                                    attach.EngramModelInitializationError,
                                    "DECODE_GRAPH_ONLY",
                                ):
                                    attach.require_engram_runtime(
                                        model, is_cuda_graph=True, is_prefill=True
                                    )
                            getter.assert_called_once_with()
                            fallback.assert_not_called()

    def test_local_mode_still_rejects_cross_host_before_native_or_checkpoint(self):
        with patch(
            "rtp_llm.models_py.modules.dsv41.engram_bootstrap._load_native"
        ) as native, patch.object(checkpoint, "inspect_checkpoint") as inspect:
            with self.assertRaisesRegex(ValueError, "cross-host"):
                storage.LocalCudaEngramStorage.from_checkpoint(
                    "/unused",
                    None,
                    device=0,
                    store=object(),
                    rank=0,
                    world_size=2,
                    gang_hosts=["host-a", "host-b"],
                )
            native.assert_not_called()
            inspect.assert_not_called()

    def test_unsupported_fabric_stops_before_checkpoint_gpu_query_and_allocation(self):
        native = SimpleNamespace(
            DeviceOwner=Mock(),
            DeviceFabricImport=Mock(),
            device_granularity=Mock(),
            device_fabric_supported=Mock(return_value=False),
            fabric_handle_bytes=64,
        )
        store = Mock()
        store.add.return_value = 1
        with patch(
            "rtp_llm.models_py.modules.dsv41.engram_bootstrap._load_native",
            return_value=native,
        ), patch.object(checkpoint, "inspect_checkpoint") as inspect, patch(
            "rtp_llm.models_py.modules.dsv41.engram_publish.load_rows"
        ) as load, patch(
            "torch.cuda.get_device_properties"
        ) as properties, patch(
            "torch.cuda.mem_get_info"
        ) as memory, patch(
            "torch.cuda.synchronize"
        ) as synchronize:
            with self.assertRaisesRegex(RuntimeError, "CUDA_FABRIC_DEVICE_UNSUPPORTED"):
                storage.LocalCudaEngramStorage.from_checkpoint(
                    "/unused",
                    None,
                    device=0,
                    store=store,
                    rank=0,
                    world_size=2,
                    gang_hosts=["host-a", "host-b"],
                    storage_mode="cuda_fabric",
                )
            native.device_fabric_supported.assert_called_once_with(0)
            for operation in (
                inspect,
                load,
                properties,
                memory,
                synchronize,
                native.DeviceOwner,
                native.DeviceFabricImport,
                native.device_granularity,
                store.wait,
            ):
                operation.assert_not_called()
        self.assertTrue(
            store.set.called, "failed startup must publish an error receipt to peers"
        )


if __name__ == "__main__":
    unittest.main()
