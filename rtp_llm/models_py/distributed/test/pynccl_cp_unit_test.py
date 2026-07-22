import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.distributed import pynccl_cp


class PyncclCPUnitTest(unittest.TestCase):
    def test_all_gather_into_tensor_uses_torch_fallback_when_disabled(self):
        output = object()
        input_tensor = object()
        process_group = object()
        group = object()

        with patch.object(pynccl_cp, "_OPT_ENABLED", False), patch.object(
            pynccl_cp,
            "_resolve_process_group",
            return_value=process_group,
        ), patch.object(
            pynccl_cp.dist, "all_gather_into_tensor"
        ) as torch_all_gather:
            pynccl_cp.all_gather_into_tensor(
                output,
                input_tensor,
                group=group,
            )

        torch_all_gather.assert_called_once_with(
            output,
            input_tensor,
            group=process_group,
        )

    def test_all_gather_into_tensor_uses_direct_nccl_on_requested_stream(self):
        input_tensor = SimpleNamespace(
            device=torch.device("cuda", 1),
            dtype=torch.bfloat16,
            data_ptr=lambda: 0x1000,
            numel=lambda: 96,
        )
        output = SimpleNamespace(data_ptr=lambda: 0x2000)
        stream = SimpleNamespace(cuda_stream=0x3000)
        process_group = object()
        group = object()
        communicator = object()
        nccl_lib = MagicMock()

        with patch.object(pynccl_cp, "_OPT_ENABLED", True), patch.object(
            pynccl_cp, "_PYNCCL_VALIDATE", False
        ), patch.object(
            pynccl_cp,
            "_resolve_process_group",
            return_value=process_group,
        ), patch.object(
            pynccl_cp,
            "_require_comm",
            return_value=(nccl_lib, communicator),
        ):
            pynccl_cp.all_gather_into_tensor(
                output,
                input_tensor,
                group=group,
                stream=stream,
            )

        nccl_lib.all_gather.assert_called_once_with(
            0x1000,
            0x2000,
            96,
            torch.bfloat16,
            communicator,
            0x3000,
        )

    def test_all_gather_allocates_rank_major_output(self):
        input_tensor = torch.empty((3, 5), dtype=torch.float16)
        output = object()
        process_group = object()
        group = object()

        with patch.object(pynccl_cp, "_OPT_ENABLED", True), patch.object(
            pynccl_cp,
            "_resolve_process_group",
            return_value=process_group,
        ), patch.object(
            pynccl_cp,
            "_world_size",
            return_value=4,
        ), patch.object(
            pynccl_cp.torch,
            "empty",
            return_value=output,
        ) as empty, patch.object(
            pynccl_cp,
            "all_gather_into_tensor",
        ) as all_gather_into_tensor:
            result = pynccl_cp.all_gather(input_tensor, group=group)

        self.assertIs(result, output)
        empty.assert_called_once_with(
            (12, 5),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        all_gather_into_tensor.assert_called_once_with(
            output,
            input_tensor,
            group=process_group,
        )

    def test_all_gather_preserves_collective_torch_fallback(self):
        input_tensor = object()
        expected = object()
        group = object()

        with patch.object(pynccl_cp, "_OPT_ENABLED", False), patch.object(
            pynccl_cp,
            "_torch_all_gather",
            return_value=expected,
        ) as torch_all_gather:
            result = pynccl_cp.all_gather(input_tensor, group=group)

        self.assertIs(result, expected)
        torch_all_gather.assert_called_once_with(input_tensor, group=group)

    def test_destroy_releases_all_cached_communicators(self):
        first_lib = MagicMock()
        second_lib = MagicMock()
        first_comm = object()
        second_comm = object()
        first_window = object()
        second_window = object()

        with patch.dict(
            pynccl_cp._COMMS,
            {
                ("pg0", 0): (first_lib, first_comm),
                ("pg1", 1): (second_lib, second_comm),
            },
            clear=True,
        ), patch.dict(
            pynccl_cp._WORLD_SIZES,
            {"pg0": 2, "pg1": 4},
            clear=True,
        ), patch.dict(
            pynccl_cp._SYMM_WINDOWS,
            {
                ("pg0", 0, 0x1000): first_window,
                ("pg1", 1, 0x2000): second_window,
            },
            clear=True,
        ), patch.object(
            pynccl_cp,
            "_LIB",
            object(),
        ):
            pynccl_cp.destroy()
            self.assertEqual(pynccl_cp._COMMS, {})
            self.assertEqual(pynccl_cp._WORLD_SIZES, {})
            self.assertIsNone(pynccl_cp._LIB)

        first_lib.destroy_comm.assert_called_once_with(first_comm)
        second_lib.destroy_comm.assert_called_once_with(second_comm)
        first_lib.window_deregister.assert_called_once_with(first_comm, first_window)
        second_lib.window_deregister.assert_called_once_with(
            second_comm, second_window
        )

    def test_master_switch_controls_optimization_stack(self):
        with patch.object(pynccl_cp, "_OPT_ENABLED", True):
            self.assertTrue(pynccl_cp.enabled())

        with patch.object(pynccl_cp, "_OPT_ENABLED", False):
            self.assertFalse(pynccl_cp.enabled())

    def test_all_gather_prefers_symmetric_role_buffer(self):
        input_tensor = torch.empty((3, 5), dtype=torch.bfloat16)
        output = object()
        process_group = object()
        stream = object()

        with patch.object(pynccl_cp, "_OPT_ENABLED", True), patch.object(
            pynccl_cp, "_resolve_process_group", return_value=process_group
        ), patch.object(
            pynccl_cp, "_world_size", return_value=4
        ), patch.object(
            pynccl_cp, "_symm_output", return_value=output
        ) as symm_output, patch.object(
            pynccl_cp.torch.cuda, "current_stream", return_value=stream
        ), patch.object(
            pynccl_cp, "_symm_stream_allowed", return_value=True
        ), patch.object(
            pynccl_cp, "_pynccl_symm_all_gather"
        ) as symm_all_gather:
            result = pynccl_cp.all_gather(
                input_tensor,
                group="tp",
                role="mla_ckv",
            )

        self.assertIs(result, output)
        symm_output.assert_called_once_with(
            "mla_ckv",
            (12, 5),
            input_tensor.dtype,
            input_tensor.device,
            process_group,
        )
        symm_all_gather.assert_called_once_with(
            output,
            input_tensor,
            stream,
            process_group,
        )

    def test_symmetric_output_rejects_oversized_or_unlisted_roles(self):
        process_group = object()
        device = torch.device("cuda", 1)
        base = torch.empty(16, dtype=torch.bfloat16)
        base_key = (process_group, 1, "mla_ckv", torch.bfloat16)
        ptr_key = (process_group, 1, int(base.data_ptr()))

        with patch.object(pynccl_cp, "_OPT_ENABLED", True), patch.dict(
            pynccl_cp._SYMM_BASES,
            {base_key: base},
            clear=True,
        ), patch.dict(
            pynccl_cp._SYMM_CAPACITY_NBYTES,
            {ptr_key: base.numel() * base.element_size()},
            clear=True,
        ), patch.dict(
            pynccl_cp._SYMM_VIEWS,
            {},
            clear=True,
        ):
            view = pynccl_cp._symm_output(
                "mla_ckv", (4, 4), torch.bfloat16, device, process_group
            )
            oversized = pynccl_cp._symm_output(
                "mla_ckv", (5, 4), torch.bfloat16, device, process_group
            )
            unlisted = pynccl_cp._symm_output(
                "indexer_k_packed",
                (4, 4),
                torch.bfloat16,
                device,
                process_group,
            )

        self.assertEqual(tuple(view.shape), (4, 4))
        self.assertIsNone(oversized)
        self.assertIsNone(unlisted)

    def test_symmetric_windows_are_registered_during_init(self):
        process_group = object()
        device = torch.device("cuda", 1)
        communicator = object()
        nccl_lib = MagicMock()
        nccl_lib._has_symm = True
        real_empty = torch.empty
        created_bases = []

        def fake_empty(shape, *args, **kwargs):
            if shape == ():
                return real_empty(shape, *args, **kwargs)
            base = MagicMock()
            base.data_ptr.return_value = 0x1000 + 0x1000 * len(created_bases)
            created_bases.append(base)
            return base

        with patch.object(pynccl_cp, "_OPT_ENABLED", True), patch.object(
            pynccl_cp,
            "_require_comm",
            return_value=(nccl_lib, communicator),
        ), patch.object(
            pynccl_cp,
            "_symm_pool",
            return_value=object(),
        ), patch.object(
            pynccl_cp,
            "_symm_max_nbytes",
            return_value=64,
        ), patch.object(
            pynccl_cp.torch.cuda,
            "use_mem_pool",
            return_value=nullcontext(),
        ), patch.object(
            pynccl_cp.torch,
            "empty",
            side_effect=fake_empty,
        ), patch.dict(
            pynccl_cp._SYMM_BASES,
            {},
            clear=True,
        ), patch.dict(
            pynccl_cp._SYMM_WINDOWS,
            {},
            clear=True,
        ), patch.dict(
            pynccl_cp._SYMM_CAPACITY_NBYTES,
            {},
            clear=True,
        ), patch.object(
            pynccl_cp,
            "_SYMM_INIT_DONE",
            set(),
        ):
            pynccl_cp._init_symm_windows(process_group, device)
            self.assertEqual(len(pynccl_cp._SYMM_BASES), 4)
            self.assertIn((process_group, 1), pynccl_cp._SYMM_INIT_DONE)

        self.assertEqual(len(created_bases), 4)
        self.assertEqual(nccl_lib.window_register.call_count, 4)

    def test_fp8_is_mapped_to_byte_transport(self):
        fp8 = getattr(torch, "float8_e4m3fn", None)
        if fp8 is None:
            self.skipTest("torch build has no float8_e4m3fn")
        self.assertEqual(pynccl_cp._NCCL_DT[fp8], 1)


if __name__ == "__main__":
    unittest.main()
