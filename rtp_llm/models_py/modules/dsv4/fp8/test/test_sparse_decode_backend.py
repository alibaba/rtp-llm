"""Device binding and planner lifetime for sparse decode backends."""

import threading
import unittest
from unittest.mock import ANY, Mock, patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import (
    attention_backend,
    backend_selection,
    sm120_sparse_mla,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention_backend import (
    SparseAttentionWorkspace,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode import fp8_sparse_attn_decode_op
from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op import (
    SparseAttnV4DecodeFp8Op,
)


class SparseDecodeBackendTest(unittest.TestCase):
    def test_construction_does_not_load_provider_or_choose_current_device(self):
        with (
            patch.object(fp8_sparse_attn_decode_op, "_load_flash_mla") as load,
            patch.object(
                torch.cuda,
                "current_device",
                side_effect=AssertionError("implicit device"),
            ),
        ):
            SparseAttnV4DecodeFp8Op(8, 512, 1.0)
        load.assert_not_called()

    def test_metadata_and_forward_share_binding_but_not_step_metadata(self):
        op = SparseAttnV4DecodeFp8Op(8, 512, 1.0)
        query = torch.zeros(1, 1, 8, 512, dtype=torch.bfloat16)
        cache, sink, indices = object(), object(), object()
        step_metadata = [object(), object()]
        schedulers = [object(), object()]
        output = object()
        with (
            patch.object(
                fp8_sparse_attn_decode_op, "_load_flash_mla", return_value=True
            ) as load,
            patch.object(op, "_forward_flash_mla", return_value=output) as forward,
            patch(
                "rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata.get_or_build_sched_meta",
                side_effect=schedulers,
            ) as build,
        ):
            for metadata, expected in zip(step_metadata, schedulers):
                scheduler = op.get_sched_meta(
                    query.device,
                    metadata,
                    batch_size=1,
                    q_len=1,
                    topk=128,
                    extra_attn_type="HCA_KV",
                )
                self.assertIs(scheduler, expected)
                self.assertIs(
                    op.forward(query, cache, sink, indices, scheduler), output
                )
                build.assert_called_with(
                    metadata,
                    batch_size=1,
                    q_len=1,
                    num_heads=8,
                    topk=128,
                    extra_attn_type="HCA_KV",
                )
            load.assert_called_once_with()
            self.assertEqual(build.call_count, 2)
            self.assertEqual(forward.call_count, 2)
            self.assertIs(forward.call_args.kwargs["sched_meta"], schedulers[1])
            self.assertIs(forward.call_args.kwargs["q"], query)

    def test_device_change_fails_before_backend_execution(self):
        op = SparseAttnV4DecodeFp8Op(8, 512, 1.0)
        with (
            patch.object(
                fp8_sparse_attn_decode_op, "_load_flash_mla", return_value=True
            ),
            patch.object(op, "_forward_flash_mla") as forward,
        ):
            op.bind_device(torch.device("cpu"))
            query = Mock(device=torch.device("cuda:1"))
            with self.assertRaisesRegex(ValueError, "bound to cpu, got cuda:1"):
                op.forward(query, None, None, None, None)
            forward.assert_not_called()

    def test_indexless_cuda_device_is_not_inferred_from_current_device(self):
        op = SparseAttnV4DecodeFp8Op(8, 512, 1.0)
        with (
            patch.object(op, "_prepare_backend") as prepare,
            patch.object(
                torch.cuda,
                "current_device",
                side_effect=AssertionError("implicit device"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "explicit CUDA device index"):
                op.bind_device(torch.device("cuda"))
        prepare.assert_not_called()

    def test_missing_provider_fails_before_planning_and_does_not_bind(self):
        op = SparseAttnV4DecodeFp8Op(8, 512, 1.0)
        with (
            patch.object(
                fp8_sparse_attn_decode_op, "_load_flash_mla", return_value=False
            ),
            patch(
                "rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata.get_or_build_sched_meta"
            ) as build,
        ):
            with self.assertRaisesRegex(RuntimeError, "FlashMLA is required"):
                op.get_sched_meta(
                    torch.device("cpu"),
                    object(),
                    batch_size=1,
                    q_len=1,
                    topk=128,
                    extra_attn_type=None,
                )
            build.assert_not_called()
        # Failed initialization must not leave a partially initialized op.
        with patch.object(
            fp8_sparse_attn_decode_op, "_load_flash_mla", return_value=True
        ) as load:
            op.bind_device(torch.device("cpu"))
            load.assert_called_once_with()


class SparseDecodeBackendSm120Test(unittest.TestCase):
    def setUp(self):
        for cached in (
            backend_selection._is_cuda_runtime,
            backend_selection._is_sm120_index,
            attention_backend._backend_for_cuda,
        ):
            cached.cache_clear()
            self.addCleanup(cached.cache_clear)
        runtime = patch.object(backend_selection, "is_cuda", return_value=True)
        runtime.start()
        self.addCleanup(runtime.stop)

    @patch(
        "rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata."
        "get_or_build_sched_meta"
    )
    def test_sm120_backend_does_not_build_flash_mla_metadata(self, build):
        device = torch.device("cuda:7")
        op = SparseAttnV4DecodeFp8Op(8, 512, 1.0)
        query = Mock(device=device)
        output = object()
        with (
            patch.object(
                torch.cuda,
                "current_device",
                side_effect=AssertionError("implicit device"),
            ),
            patch.object(
                backend_selection, "is_sm120", side_effect=lambda d: d == device
            ) as select,
            patch.object(fp8_sparse_attn_decode_op, "_load_flash_mla") as flash_mla,
            patch.object(
                sm120_sparse_mla, "_load_sm120_sparse_mla", return_value=object()
            ) as provider,
            patch.object(sm120_sparse_mla, "workspace") as workspace_fn,
            patch.object(
                sm120_sparse_mla, "forward_decode", return_value=output
            ) as forward,
        ):
            result = op.get_sched_meta(
                device,
                object(),
                batch_size=2,
                q_len=1,
                topk=64,
                extra_attn_type=None,
            )
            self.assertIsNone(result)
            for _ in range(2):
                self.assertIs(op.forward(query, None, None, None, result), output)
            select.assert_called_once_with(device)
            provider.assert_called_once_with()
            workspace_fn.assert_called_once_with(device, owner=ANY)
            owner = workspace_fn.call_args.kwargs["owner"]
            self.assertIsInstance(owner, sm120_sparse_mla.Sm120SparseMlaWorkspace)
            flash_mla.assert_not_called()
            build.assert_not_called()
            self.assertEqual(forward.call_count, 2)
            self.assertIs(forward.call_args.kwargs["q"], query)
            self.assertIs(forward.call_args.kwargs["workspace_owner"], owner)

    @patch(
        "rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata."
        "get_or_build_sched_meta"
    )
    def test_non_sm120_backend_builds_flash_mla_metadata(self, build):
        sentinel, metadata = object(), object()
        build.return_value = sentinel
        device = torch.device("cuda:3")
        op = SparseAttnV4DecodeFp8Op(8, 512, 1.0)
        with (
            patch.object(backend_selection, "is_sm120", return_value=False) as select,
            patch.object(
                fp8_sparse_attn_decode_op, "_load_flash_mla", return_value=True
            ),
            patch.object(sm120_sparse_mla, "_load_sm120_sparse_mla") as provider,
            patch.object(
                torch.cuda,
                "current_device",
                side_effect=AssertionError("implicit device"),
            ),
        ):
            result = op.get_sched_meta(
                device,
                metadata,
                batch_size=2,
                q_len=1,
                topk=64,
                extra_attn_type="HCA_KV",
            )
            self.assertIs(result, sentinel)
            select.assert_called_once_with(device)
            provider.assert_not_called()
        build.assert_called_once_with(
            metadata,
            batch_size=2,
            q_len=1,
            num_heads=8,
            topk=64,
            extra_attn_type="HCA_KV",
        )

    def test_missing_sm120_provider_fails_before_workspace_allocation(self):
        op = SparseAttnV4DecodeFp8Op(8, 512, 1.0)
        with (
            patch.object(backend_selection, "is_sm120", return_value=True),
            patch.object(sm120_sparse_mla, "_load_sm120_sparse_mla", return_value=None),
            patch.object(sm120_sparse_mla, "workspace") as workspace_fn,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "requires FlashInfer trtllm_batch_decode_sparse_mla_dsv4"
            ):
                op.bind_device(torch.device("cuda:2"))
            workspace_fn.assert_not_called()

    def test_workspace_failure_does_not_publish_binding(self):
        op = SparseAttnV4DecodeFp8Op(8, 512, 1.0)
        device = torch.device("cuda:2")
        with (
            patch.object(backend_selection, "is_sm120", return_value=True),
            patch.object(
                sm120_sparse_mla, "_load_sm120_sparse_mla", return_value=object()
            ),
            patch.object(
                sm120_sparse_mla,
                "workspace",
                side_effect=[RuntimeError("workspace"), None],
            ) as workspace_fn,
        ):
            with self.assertRaisesRegex(RuntimeError, "workspace"):
                op.bind_device(device)
            op.bind_device(device)
            self.assertEqual(workspace_fn.call_count, 2)
            workspace_fn.assert_called_with(device, owner=ANY)

    def test_workspace_ownership_isolated_for_standalone_ops_and_shared_for_model(self):
        device = torch.device("cuda:5")
        op_a = SparseAttnV4DecodeFp8Op(8, 512, 1.0)
        op_b = SparseAttnV4DecodeFp8Op(8, 512, 1.0)
        shared_ws = SparseAttentionWorkspace()
        op_m1 = SparseAttnV4DecodeFp8Op(8, 512, 1.0, attn_workspace=shared_ws)
        op_m2 = SparseAttnV4DecodeFp8Op(8, 512, 1.0, attn_workspace=shared_ws)
        with (
            patch.object(backend_selection, "is_sm120", return_value=True),
            patch.object(
                sm120_sparse_mla, "_load_sm120_sparse_mla", return_value=object()
            ),
            patch.object(sm120_sparse_mla, "workspace"),
        ):
            for op in (op_a, op_b, op_m1, op_m2):
                op.bind_device(device)
        self.assertIsNot(
            op_a._forward_impl.keywords["workspace_owner"],
            op_b._forward_impl.keywords["workspace_owner"],
        )
        expected_owner = shared_ws.get_or_create(sm120_sparse_mla.Sm120SparseMlaWorkspace)
        self.assertIsInstance(expected_owner, sm120_sparse_mla.Sm120SparseMlaWorkspace)
        self.assertIs(op_m1._forward_impl.keywords["workspace_owner"], expected_owner)
        self.assertIs(op_m2._forward_impl.keywords["workspace_owner"], expected_owner)

    def test_sparse_attention_workspace_concurrent_get_or_create(self):
        shared_ws = SparseAttentionWorkspace()
        created = []
        entered = threading.Event()
        release = threading.Event()

        def factory():
            entered.set()
            assert release.wait(timeout=5)
            obj = sm120_sparse_mla.Sm120SparseMlaWorkspace()
            created.append(obj)
            return obj

        results = [None, None]

        def worker(idx: int, wait_entry: bool) -> None:
            if wait_entry:
                assert entered.wait(timeout=5)
            results[idx] = shared_ws.get_or_create(factory)

        t0 = threading.Thread(target=worker, args=(0, False))
        t1 = threading.Thread(target=worker, args=(1, True))
        t0.start()
        t1.start()
        assert entered.wait(timeout=5)
        release.set()
        t0.join(timeout=5)
        t1.join(timeout=5)
        self.assertEqual(len(created), 1)
        self.assertIs(results[0], created[0])
        self.assertIs(results[1], created[0])


if __name__ == "__main__":
    unittest.main()
