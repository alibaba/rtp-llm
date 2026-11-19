"""CPU dispatch and real multi-rank Gloo tests for GLM Prefill collectives."""

import contextlib
import copy
import datetime
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.distributed import collective_utils
from rtp_llm.models_py.distributed import glm53_collective_gemm as glm
from rtp_llm.models_py.distributed import symm_mem
from rtp_llm.models_py.modules.dsv4 import forward_metadata


def parallelism(rank=0, tp=4, dp=1, pp=1, local=8, role="PREFILL"):
    return SimpleNamespace(
        world_size=tp * dp * pp,
        world_rank=rank,
        local_world_size=local,
        local_rank=rank % local,
        tp_size=tp,
        tp_rank=rank % tp,
        dp_size=dp,
        pp_size=pp,
        role_type=role,
    )


class NativeAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(glm, "_STATE", None))
        self.group = MagicMock()
        self.group.size.return_value = 4

    def contract(self, pc, members=None, graph=False, sm=(10, 3)):
        members = members or list(
            range(
                pc.world_rank // pc.tp_size * pc.tp_size,
                (pc.world_rank // pc.tp_size + 1) * pc.tp_size,
            )
        )
        with (
            patch.object(glm.dist, "get_backend", return_value="nccl"),
            patch.object(
                glm.dist,
                "get_world_size",
                side_effect=lambda g=None: (
                    len(members) if g is self.group else pc.world_size
                ),
            ),
            patch.object(
                glm.dist,
                "get_rank",
                side_effect=lambda g=None: (
                    members.index(pc.world_rank) if g is self.group else pc.world_rank
                ),
            ),
            patch.object(
                glm.dist, "get_global_rank", side_effect=lambda g, rank: members[rank]
            ),
            patch.object(glm.torch.cuda, "current_device", return_value=pc.local_rank),
            patch.object(glm.torch.cuda, "get_device_capability", return_value=sm),
            patch.object(glm.torch.cuda, "can_device_access_peer", return_value=True),
            patch.object(glm.socket, "gethostname", return_value="same-node"),
            patch.object(glm.Path, "read_text", return_value="same-boot"),
            patch.object(symm_mem, "torch_symm_mem_available", True),
        ):
            return glm._native_ag_contract(self.group, pc, graph)

    def test_actual_membership_and_node_boundary_not_just_tp_size(self):
        for tp in (2, 4, 8):
            self.assertTrue(self.contract(parallelism(tp=tp))["admitted"])
        # A TP4 group can straddle nodes even though local_world_size >= TP.
        self.assertFalse(
            self.contract(parallelism(rank=4, tp=4, dp=3, local=6))["admitted"]
        )
        # Group membership differs from the declared PP/DP/TP grid.
        self.assertFalse(
            self.contract(parallelism(rank=4, tp=4, dp=2), members=[4, 5, 6, 0])[
                "admitted"
            ]
        )
        self.assertFalse(self.contract(parallelism(), graph=True)["admitted"])
        self.assertFalse(self.contract(parallelism(), sm=(9, 0))["admitted"])
        self.assertFalse(self.contract(parallelism(role="DECODE"))["admitted"])

    def test_any_peer_miss_or_disagreement_declines_before_construction(self):
        local = self.contract(parallelism())
        good = [dict(copy.deepcopy(local), device=rank) for rank in range(4)]
        cases = []
        for key, value in (("admitted", False), ("device", 0)):
            peers = copy.deepcopy(good)
            peers[1][key] = value
            cases.append(peers)
        peers = copy.deepcopy(good)
        peers[1]["shared"][-1][1] = "different-boot"
        cases.append(peers)
        peers = copy.deepcopy(good)
        peers[1]["shared"][-2] = True
        cases.append(peers)
        for peers in cases:
            with self.subTest(peers=peers):
                self.assertFalse(glm._native_group_agrees(local, peers))
        self.assertTrue(glm._native_group_agrees(local, good))

    def test_rendezvous_exception_and_group_disable(self):
        with patch.object(
            symm_mem,
            "TorchSymmMemCommunicator",
            side_effect=RuntimeError("rendezvous failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "rendezvous failed"):
                glm._configure_native_all_gather(self.group, 0)
        with patch.object(
            symm_mem,
            "TorchSymmMemCommunicator",
            return_value=SimpleNamespace(disabled=True),
        ):
            self.assertIsNone(glm._configure_native_all_gather(self.group, 0))

    def configure_for_peers(
        self, pc, *, alter=None, features=(None, None, None, 0), graph=False
    ):
        local = self.contract(pc, graph=graph)

        def exchange(group, vote, tag):
            peers = [dict(copy.deepcopy(vote), device=i) for i in range(4)]
            if alter is not None:
                alter(peers[-1])
            return peers

        with patch.object(glm, "_native_ag_contract", return_value=local), patch.object(
            glm, "_local_collective_features", return_value=features
        ), patch.object(collective_utils, "exchange_group_state", side_effect=exchange):
            glm.configure_glm53_collective_gemm(
                self.group, 128, pc, enable_cuda_graph=graph, max_batch_tokens=7
            )

    def test_decode_pdfusion_and_cuda_graph_remain_unconfigured(self):
        with patch.object(glm, "_configure_native_all_gather") as construct:
            for role, graph in (
                ("DECODE", False),
                ("PDFUSION", False),
                ("PREFILL", True),
            ):
                self.configure_for_peers(parallelism(role=role), graph=graph)
                self.assertIsNone(glm._STATE)
            construct.assert_not_called()

    def test_one_rank_without_native_features_declines_for_every_rank(self):
        features = (MagicMock(), MagicMock(), MagicMock(), 2176)
        with patch.object(glm, "_configure_native_all_gather") as construct:
            self.configure_for_peers(
                parallelism(),
                alter=lambda p: p.update(ag=False, rs_bytes=0),
                features=features,
            )
            self.assertFalse(glm._STATE.all_gather)
            self.assertIsNone(glm._STATE.reduce_partials)
            self.assertIsNone(glm._STATE.workspace)
            construct.assert_not_called()
            features[1].GemmRSBuffer.assert_not_called()

    def test_capacity_disagreement_before_allocation(self):
        with self.assertRaisesRegex(ValueError, "agree"):
            self.configure_for_peers(
                parallelism(), alter=lambda p: p.update(shape=[128, 64])
            )
        self.assertIsNone(glm._STATE)

    def test_probe_requires_reducer_memory_and_supported_workspace_layout(self):
        import torch.distributed._symmetric_memory as symm_mem_ops

        reducer = MagicMock()
        native = SimpleNamespace(bf16_gemm_rs_reduce=reducer)
        size = MagicMock(return_value=2176)
        gemm = SimpleNamespace(
            _C=native,
            GemmRSBuffer=MagicMock(),
            bf16_gemm_rs_nn=MagicMock(),
            get_bf16_gemm_rs_workspace_size=size,
        )
        with patch.dict(sys.modules, {"deep_gemm": gemm}), patch.object(
            torch.cuda, "mem_get_info", return_value=(4096, 8192)
        ):
            self.assertEqual(
                glm._local_collective_features(0, 8, 128),
                (symm_mem_ops, gemm, reducer, 2176),
            )
            for nbytes in (2048, 2112, 8192):
                size.return_value = nbytes
                self.assertEqual(
                    glm._local_collective_features(0, 8, 128),
                    (symm_mem_ops, None, None, 0),
                )
            del native.bf16_gemm_rs_reduce
            self.assertEqual(
                glm._local_collective_features(0, 8, 128), (symm_mem_ops, None, None, 0)
            )
            gemm.GemmRSBuffer.assert_not_called()

    def test_committed_workspace_error_propagates_before_publication(self):
        gemm = SimpleNamespace(
            GemmRSBuffer=MagicMock(
                side_effect=RuntimeError("workspace construction failed")
            )
        )
        with self.assertRaisesRegex(RuntimeError, "workspace construction failed"):
            self.configure_for_peers(
                parallelism(), features=(None, gemm, object(), 2176)
            )
        self.assertIsNone(glm._STATE)
        gemm.GemmRSBuffer.assert_called_once()

    def test_transport_error_propagates_without_reduction_retry(self):
        with patch.object(
            glm, "_STATE", glm._State(self.group, 8, 4, False)
        ), patch.object(
            glm.dist,
            "all_to_all_single",
            side_effect=RuntimeError("group transport failed"),
        ) as send:
            with self.assertRaisesRegex(RuntimeError, "group transport failed"):
                glm.reduce_scatter_glm53(torch.zeros((5, 4), dtype=torch.bfloat16))
            send.assert_called_once()

    def test_idempotence_and_workspace_before_publication(self):
        calls = []
        workspace = SimpleNamespace(
            num_bytes=2176, buffer_ptrs=[1, 2, 3, 4], device=torch.device("cpu")
        )
        ag = SimpleNamespace(enable_symm_mem_for_group=lambda g: calls.append("group"))
        gemm = SimpleNamespace(
            GemmRSBuffer=lambda *a, **kw: calls.append("workspace") or workspace
        )
        reducer = object()
        native = object()

        def initialize_native(*args):
            self.assertIsNone(glm._STATE)
            calls.append("native")
            return native

        with patch.object(
            glm, "_configure_native_all_gather", side_effect=initialize_native
        ), patch.object(torch.cuda, "current_stream"):
            self.configure_for_peers(parallelism(), features=(ag, gemm, reducer, 2176))
            state = glm._STATE
            self.assertIs(state.workspace, workspace)
            self.assertIs(state.native_all_gather, native)
            self.assertIs(state.reduce_partials, reducer)
            self.assertEqual(state.max_tokens, 8)
            self.configure_for_peers(parallelism())
            self.assertIs(glm._STATE, state)
            self.assertEqual(calls, ["group", "workspace", "native"])
            with self.assertRaisesRegex(
                RuntimeError, "already initialized differently"
            ):
                self.configure_for_peers(parallelism(), graph=True)


class NativeRoutingTest(unittest.TestCase):
    def setUp(self):
        from rtp_llm.models_py.distributed import collective_torch

        self.collective = collective_torch
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.metadata = self.stack.enter_context(
            patch.object(forward_metadata, "metadata_cache", return_value={})
        )
        self.capture = self.stack.enter_context(
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False)
        )
        self.fallback = self.stack.enter_context(
            patch.object(self.collective, "all_gather")
        )
        # Use the real communicator eligibility policy with a CPU stand-in;
        # numerical GPU communication remains the responsibility of integration.
        comm = object.__new__(symm_mem.TorchSymmMemCommunicator)
        comm.disabled, comm.dtype = False, torch.bfloat16
        comm.device, comm.world_size, comm.max_size = torch.device("cpu"), 2, 64 << 20
        self.native = SimpleNamespace(communicator=comm, gather=MagicMock())
        self.state = glm._State(object(), 1024, 4, True, native_all_gather=self.native)
        self.stack.enter_context(patch.object(glm, "_STATE", self.state))

    def test_shape_and_trim_keep_rank_major_values(self):
        shard = torch.arange(1, 9, dtype=torch.bfloat16).reshape(2, 4)
        gathered = torch.cat((shard, shard + 10))
        self.native.gather.return_value = gathered
        output = glm.all_gather_trim(shard, 3, self.collective.Group.TP)
        self.assertTrue(torch.equal(output, gathered[:3]))
        self.assertEqual(output.shape, (3, 4))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.fallback.assert_not_called()
        for logical in (-1, 5):
            with self.assertRaisesRegex(ValueError, "logical token count"):
                glm.all_gather_trim(shard, logical, self.collective.Group.TP)

    def test_typed_payload_and_capacity_misses_use_ordinary_path(self):
        values = [
            torch.tensor([0, 255], dtype=torch.uint8),
            torch.ones(2, 3),
            torch.ones(3, dtype=torch.bfloat16),
            torch.empty(0, 2, dtype=torch.bfloat16),
            torch.ones(2, 2, dtype=torch.bfloat16).T,
            torch.ones(1025, 2, dtype=torch.bfloat16),
        ]
        for value in values:
            with self.subTest(dtype=value.dtype, shape=value.shape):
                sentinel = object()
                self.fallback.return_value = sentinel
                self.assertIs(glm.all_gather(value, self.collective.Group.TP), sentinel)
                self.fallback.assert_called_with(value, self.collective.Group.TP)
        self.native.gather.assert_not_called()
        self.native.communicator.max_size = (
            32  # gathered bytes equal capacity: strict miss
        )
        value = torch.ones(2, 4, dtype=torch.bfloat16)
        glm.all_gather(value, self.collective.Group.TP)
        self.native.gather.assert_not_called()
        self.fallback.assert_called_with(value, self.collective.Group.TP)

    def test_vector_alignment_misses_preserve_ordinary_path(self):
        # Even when total gathered bytes are 16-aligned, an 8-byte shard
        # starts rank1 at an address unsuitable for a 16-byte vector store.
        values = [torch.ones(n, dtype=torch.bfloat16) for n in (2, 4, 6)]
        for value in values:
            with self.subTest(shape=value.shape, offset=value.storage_offset()):
                self.assertTrue(
                    self.native.communicator.should_torch_symm_mem_allgather(value)
                )
                sentinel = object()
                self.fallback.return_value = sentinel
                self.assertIs(glm.all_gather(value, self.collective.Group.TP), sentinel)
                self.fallback.assert_called_with(value, self.collective.Group.TP)
        self.native.gather.assert_not_called()

    def test_uninitialized_and_decode_verify_contexts_never_use_native(self):
        value = torch.ones(2, 4, dtype=torch.bfloat16)
        for state, metadata, group in (
            (None, {}, self.collective.Group.TP),
            (self.state, None, self.collective.Group.TP),
            (self.state, {}, self.collective.Group.DP),
        ):
            with patch.object(glm, "_STATE", state), patch.object(
                forward_metadata, "metadata_cache", return_value=metadata
            ):
                glm.all_gather(value, group)
        self.native.gather.assert_not_called()

    def test_cp_typed_fallback_preserves_role_and_input(self):
        value = torch.tensor([1, 255], dtype=torch.uint8)
        with patch("rtp_llm.models_py.distributed.pynccl_cp.all_gather") as cp:
            self.assertIs(
                glm.all_gather(value, self.collective.Group.TP, role="mla_history"),
                cp.return_value,
            )
            cp.assert_called_once_with(
                value, group=self.collective.Group.TP, role="mla_history"
            )
        self.native.gather.assert_not_called()
        self.fallback.assert_not_called()

    def test_illegal_capture_in_native_contract_fails_not_local_fallback(self):
        self.capture.return_value = True
        with self.assertRaisesRegex(RuntimeError, "non-Graph Prefill"):
            glm.all_gather(
                torch.ones(2, 4, dtype=torch.bfloat16), self.collective.Group.TP
            )
        self.native.gather.assert_not_called()
        self.fallback.assert_not_called()


class NativeBufferLifetimeTest(unittest.TestCase):
    def test_lock_and_completion_cover_copy_out_and_next_stream(self):
        trace = []
        event = SimpleNamespace(last=None)
        event.record = lambda stream: (
            trace.append(("record", stream)),
            setattr(event, "last", stream),
        )
        a, b = MagicMock(name="a"), MagicMock(name="b")
        a.wait_event.side_effect = lambda e: trace.append(("wait-a", e.last))
        b.wait_event.side_effect = lambda e: trace.append(("wait-b", e.last))
        comm = SimpleNamespace(device="cuda:0", world_size=2)
        owner = None

        def gather(tensor):
            self.assertFalse(
                owner.lock.acquire(blocking=False),
                "submission must hold the owner lock",
            )
            trace.append(("copy-out", tensor.value))
            return torch.tensor([tensor.value, tensor.value + 10])

        comm.all_gather = gather
        with patch.object(torch.cuda, "Event", return_value=event), patch.object(
            torch.cuda, "current_stream", side_effect=[a, a, b]
        ), patch.object(torch.cuda, "device", return_value=contextlib.nullcontext()):
            owner = glm._NativeAllGather(comm)
            first = SimpleNamespace(
                shape=(1,), value=3, record_stream=MagicMock(), data_ptr=lambda: 16
            )
            second = SimpleNamespace(
                shape=(1,), value=8, record_stream=MagicMock(), data_ptr=lambda: 16
            )
            result = owner.gather(first)
            owner.gather(second)
        self.assertEqual(result.tolist(), [3, 13])
        self.assertEqual(
            trace,
            [
                ("record", a),
                ("wait-a", a),
                ("copy-out", 3),
                ("record", a),
                ("wait-b", a),
                ("copy-out", 8),
                ("record", b),
            ],
        )
        first.record_stream.assert_called_once_with(a)
        second.record_stream.assert_called_once_with(b)

    def test_misaligned_pointer_is_staged_without_a_different_collective(self):
        event, stream = MagicMock(), MagicMock()
        staged = SimpleNamespace(record_stream=MagicMock())
        value = SimpleNamespace(
            shape=(1,),
            data_ptr=lambda: 2,
            record_stream=MagicMock(),
            clone=MagicMock(return_value=staged),
        )
        comm = SimpleNamespace(
            device="cuda:0",
            world_size=2,
            all_gather=MagicMock(return_value=torch.tensor([3, 13])),
        )
        with patch.object(torch.cuda, "Event", return_value=event), patch.object(
            torch.cuda, "current_stream", return_value=stream
        ), patch.object(torch.cuda, "device", return_value=contextlib.nullcontext()):
            owner = glm._NativeAllGather(comm)
            self.assertEqual(owner.gather(value).tolist(), [3, 13])
        value.clone.assert_called_once_with()
        value.record_stream.assert_called_once_with(stream)
        staged.record_stream.assert_called_once_with(stream)
        comm.all_gather.assert_called_once_with(staged)

    def test_failed_submission_poisoned_owner_does_not_retry_or_fall_back(self):
        event = MagicMock()
        stream = MagicMock()
        comm = SimpleNamespace(
            device="cuda:0",
            world_size=2,
            all_gather=MagicMock(side_effect=RuntimeError("launch failed")),
        )
        with patch.object(torch.cuda, "Event", return_value=event), patch.object(
            torch.cuda, "current_stream", return_value=stream
        ), patch.object(torch.cuda, "device", return_value=contextlib.nullcontext()):
            owner = glm._NativeAllGather(comm)
            value = SimpleNamespace(
                shape=(1,), record_stream=MagicMock(), data_ptr=lambda: 16
            )
            with self.assertRaisesRegex(RuntimeError, "launch failed"):
                owner.gather(value)
            with self.assertRaisesRegex(RuntimeError, "previously failed"):
                owner.gather(value)
        comm.all_gather.assert_called_once()


def _ordered_worker(rank, world, store):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method="file://" + store,
        rank=rank,
        world_size=world,
        timeout=datetime.timedelta(seconds=45),
    )
    try:
        group = dist.group.WORLD
        glm.configure_glm53_collective_gemm(
            group,
            4,
            parallelism(rank=rank, tp=world, local=world),
            enable_cuda_graph=False,
            max_batch_tokens=8,
        )
        assert glm._STATE.reduce_partials is None
        cancellation = torch.tensor(
            [
                [2**24, 1, 256, 0.5],
                [1, 256, 1, 0.5],
                [-(2**24), -256, -256, 0.5],
                [3, 0.5, 0.5, 0.5],
            ]
        )

        def partial(source, rows):
            storage = torch.empty((rows, 8), dtype=torch.bfloat16)
            values = storage[:, ::2]
            values.copy_(cancellation[source].expand(rows, 4))
            if rows:
                values[:, 3].copy_(torch.arange(rows) + source)
            return values

        # Force multiple stripes for both the no-native and over-capacity paths.
        with patch.object(glm, "_ORDERED_RS_BUFFER_BYTES", 160):
            for rows in (0, 1, 4, 5, 17, 33):
                source = partial(rank, rows)
                original = source.clone()
                expected = torch.zeros((rows, 4), dtype=torch.float32)
                for peer in range(world):
                    expected.add_(partial(peer, rows).float())
                shard = (rows + world - 1) // world
                padded = torch.zeros(shard * world, 4, dtype=torch.bfloat16)
                padded[:rows].copy_(expected)
                reference = padded[rank * shard : (rank + 1) * shard]
                actual = glm.reduce_scatter_glm53(source)
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                torch.testing.assert_close(source, original, rtol=0, atol=0)
                if rows > glm._STATE.max_tokens:
                    with patch.object(glm._STATE, "reduce_partials", object()):
                        overflow = glm.reduce_scatter_glm53(source)
                    torch.testing.assert_close(overflow, reference, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class OrderedReduceScatterTest(unittest.TestCase):
    def test_four_gloo_ranks_fp32_order_chunking_and_zero_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _ordered_worker,
                args=(4, str(Path(directory) / "store")),
                nprocs=4,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
