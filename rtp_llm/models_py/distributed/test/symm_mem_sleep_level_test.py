import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rtp_llm.models_py.distributed import symm_mem


class SymmMemSleepLevelTest(unittest.TestCase):
    def setUp(self) -> None:
        symm_mem._symm_mem_comm = None

    def tearDown(self) -> None:
        symm_mem._symm_mem_comm = None

    def test_initializes_checkpoint_safe_fast_communicator(self) -> None:
        instance = MagicMock(disabled=False)
        group = MagicMock()
        with patch.object(
            symm_mem, "TorchSymmMemCommunicator", return_value=instance
        ) as communicator, patch("torch.cuda.current_device", return_value=0):
            self.assertIs(symm_mem.init_symm_mem_communicator(group), instance)
        communicator.assert_called_once_with(group, 0)

    def _construct_communicator(self, world_size, multicast_ptr):
        group = SimpleNamespace(group_name="test-group")
        handle = SimpleNamespace(multicast_ptr=multicast_ptr)
        fake_torch_symm_mem = SimpleNamespace(
            empty=MagicMock(return_value=object()),
            rendezvous=MagicMock(return_value=handle),
        )
        with patch.object(symm_mem, "torch_symm_mem_available", True), patch.object(
            symm_mem, "torch_symm_mem", fake_torch_symm_mem
        ), patch.object(
            symm_mem.dist, "get_world_size", return_value=world_size
        ), patch.object(
            symm_mem.torch.cuda, "set_device"
        ), patch.object(
            symm_mem.torch.cuda, "get_device_capability", return_value=(10, 0)
        ):
            communicator = symm_mem.TorchSymmMemCommunicator(group, 0)
        return communicator

    def test_world_two_two_shot_all_reduce_does_not_require_multicast(self) -> None:
        communicator = self._construct_communicator(world_size=2, multicast_ptr=0)

        self.assertFalse(communicator.disabled)
        self.assertFalse(communicator.has_multicast_support)
        self.assertTrue(
            communicator.should_torch_symm_mem_allreduce(
                symm_mem.torch.empty(4, dtype=symm_mem.torch.bfloat16)
            )
        )
        self.assertFalse(
            communicator.should_torch_symm_mem_allgather(
                symm_mem.torch.empty(4, dtype=symm_mem.torch.bfloat16)
            )
        )

    def test_multimem_all_reduce_is_disabled_without_multicast(self) -> None:
        communicator = self._construct_communicator(world_size=6, multicast_ptr=0)

        self.assertTrue(communicator.disabled)
        self.assertIsNone(communicator.handle)
        self.assertIsNone(communicator.buffer)

    def test_destroy_global_communicator_is_idempotent(self) -> None:
        instance = MagicMock(disabled=False)
        symm_mem._symm_mem_comm = instance

        symm_mem.destroy_symm_mem_communicator()
        symm_mem.destroy_symm_mem_communicator()

        instance.destroy.assert_called_once_with()
        self.assertIsNone(symm_mem.get_symm_mem_communicator())

    def test_three_init_destroy_cycles_replace_all_global_owners(self) -> None:
        instances = [MagicMock(disabled=False) for _ in range(3)]
        groups = [MagicMock() for _ in range(3)]

        with patch.object(
            symm_mem, "TorchSymmMemCommunicator", side_effect=instances
        ) as communicator_type, patch.object(
            symm_mem.torch.cuda, "current_device", return_value=0
        ), patch.object(
            symm_mem, "_destroy_torch_symm_mem_runtime_state"
        ) as destroy_runtime_state:
            for group, instance in zip(groups, instances):
                self.assertIs(symm_mem.init_symm_mem_communicator(group), instance)
                self.assertIs(symm_mem.get_symm_mem_communicator(), instance)

                symm_mem.destroy_symm_mem_communicator()
                symm_mem.destroy_symm_mem_communicator()

                instance.destroy.assert_called_once_with()
                self.assertIsNone(symm_mem.get_symm_mem_communicator())

        self.assertEqual(communicator_type.call_count, 3)
        self.assertEqual(destroy_runtime_state.call_count, 6)

    def test_destroy_releases_all_cuda_and_process_group_references(self) -> None:
        communicator = symm_mem.TorchSymmMemCommunicator.__new__(
            symm_mem.TorchSymmMemCommunicator
        )
        communicator.disabled = False
        communicator.handle = object()
        communicator.buffer = object()
        communicator.group = object()
        communicator.device = "cuda:0"

        with patch.object(
            symm_mem.torch.cuda, "is_available", return_value=True
        ), patch.object(symm_mem.torch.cuda, "synchronize") as synchronize:
            communicator.destroy()
            communicator.destroy()

        self.assertTrue(communicator.disabled)
        self.assertIsNone(communicator.handle)
        self.assertIsNone(communicator.buffer)
        self.assertIsNone(communicator.group)
        self.assertIsNone(communicator.device)
        self.assertEqual(synchronize.call_count, 1)
        synchronize.assert_called_with("cuda:0")

    def test_destroy_clears_all_torch_symm_mem_cuda_caches(self) -> None:
        class Pool:
            pass

        pool = Pool()
        pool_ref = weakref.ref(pool)
        fake_torch_symm_mem = SimpleNamespace(
            _group_name_to_workspace_tensor={"0": object()},
            _backend_streams={0: object()},
            _group_name_to_store={},
            _symm_mem_pools={"cuda:0": pool},
        )
        del pool

        with patch.object(symm_mem, "torch_symm_mem_available", True), patch.object(
            symm_mem, "torch_symm_mem", fake_torch_symm_mem, create=True
        ):
            symm_mem._destroy_torch_symm_mem_runtime_state()

        self.assertEqual(fake_torch_symm_mem._group_name_to_workspace_tensor, {})
        self.assertEqual(fake_torch_symm_mem._backend_streams, {})
        self.assertEqual(fake_torch_symm_mem._group_name_to_store, {})
        self.assertEqual(fake_torch_symm_mem._symm_mem_pools, {})
        self.assertIsNone(pool_ref())

    def test_destroy_rejects_legacy_group_info_that_native_torch_cannot_erase(
        self,
    ) -> None:
        workspace = {"0": object()}
        fake_torch_symm_mem = SimpleNamespace(
            _group_name_to_workspace_tensor=workspace,
            _backend_streams={},
            _group_name_to_store={"0": object()},
            _symm_mem_pools={},
        )
        with patch.object(symm_mem, "torch_symm_mem_available", True), patch.object(
            symm_mem, "torch_symm_mem", fake_torch_symm_mem, create=True
        ):
            with self.assertRaisesRegex(RuntimeError, "no native group-info"):
                symm_mem._destroy_torch_symm_mem_runtime_state()

        self.assertIs(fake_torch_symm_mem._group_name_to_workspace_tensor, workspace)
        self.assertEqual(len(workspace), 1)

    def test_destroy_runtime_state_is_compatible_with_older_torch(self) -> None:
        fake_torch_symm_mem = SimpleNamespace()
        with patch.object(symm_mem, "torch_symm_mem_available", True), patch.object(
            symm_mem, "torch_symm_mem", fake_torch_symm_mem, create=True
        ):
            symm_mem._destroy_torch_symm_mem_runtime_state()


if __name__ == "__main__":
    unittest.main()
