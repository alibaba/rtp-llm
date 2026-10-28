import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl import trt as op


class WorkspaceTest(unittest.TestCase):

    def setUp(self):
        op._g_trtllm_fmha_v2_workspace_pool.clear()

    def tearDown(self):
        op._g_trtllm_fmha_v2_workspace_pool.clear()

    def test_real_tbstars_shape_exceeds_error_allocation(self):
        self.assertGreaterEqual(
            op._trtllm_fmha_v2_workspace_size_bytes(1, 8192, 8), 524224 + 128
        )

    def test_grow_preserves_reuse_and_does_not_shrink(self):
        small = op._get_trtllm_fmha_v2_workspace('cpu')
        params = SimpleNamespace(batch_size=4, max_q_len=32768)
        large = op._ensure_trtllm_fmha_v2_workspace_size(small, params, 32)
        self.assertGreaterEqual(large.numel(), 4 * 32768 * 32 * 8 + 128)
        self.assertIs(op._g_trtllm_fmha_v2_workspace_pool[0], small)
        self.assertIs(op._ensure_trtllm_fmha_v2_workspace_size(large, params, 32), large)

    def _check_prepare_workspace(self, cls):
        instance = cls.__new__(cls)
        instance.head_num = instance.kv_head_num = 8
        instance.workspace_buffer = op._get_trtllm_fmha_v2_workspace('cpu')
        inputs = SimpleNamespace(
            input_lengths=torch.tensor([8192], dtype=torch.int32),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            cu_seqlens_device=torch.tensor([0, 8192], dtype=torch.int32),
            cu_kv_seqlens_device=torch.tensor([0, 8192], dtype=torch.int32),
            kv_cache_kernel_block_id_device=None,
        )
        params = instance.prepare(inputs)
        self.assertEqual(params.batch_size, 1)
        self.assertEqual(params.max_q_len, 8192)
        self.assertGreaterEqual(instance.workspace_buffer.numel(), 524224 + 128)
        workspace = instance.workspace_buffer
        inputs.input_lengths.fill_(16)
        inputs.cu_seqlens_device[1] = inputs.cu_kv_seqlens_device[1] = 16
        instance.prepare(inputs)
        self.assertIs(instance.workspace_buffer, workspace)

    def test_paged_prepare_expands_workspace_for_real_input(self):
        self._check_prepare_workspace(op.TRTLLMFMHAv2PagedPrefillOp)

    def test_nonpaged_prepare_expands_workspace_for_real_input(self):
        self._check_prepare_workspace(op.TRTLLMFMHAv2PrefillOp)

    def test_pool_keeps_largest_idle_buffer_and_resets_it(self):
        a = torch.ones(2048, dtype=torch.uint8)
        b = torch.ones(4096, dtype=torch.uint8)
        op._release_trtllm_fmha_v2_workspace(b)
        op._release_trtllm_fmha_v2_workspace(a)
        self.assertIs(op._get_trtllm_fmha_v2_workspace('cpu', 1800), b)
        self.assertEqual(b.count_nonzero().item(), 0)

    def test_pool_never_returns_undersized_buffer(self):
        a = torch.zeros(1024, dtype=torch.uint8)
        op._release_trtllm_fmha_v2_workspace(a)
        b = op._get_trtllm_fmha_v2_workspace('cpu', 2048)
        self.assertGreaterEqual(b.numel(), 2048)
        self.assertIsNot(a, b)

    def test_inference_context_allocation_can_be_reset_outside(self):
        with torch.inference_mode():
            a = op._get_trtllm_fmha_v2_workspace('cpu', 2048)
        self.assertFalse(a.is_inference())
        op._release_trtllm_fmha_v2_workspace(a)
        self.assertIs(op._get_trtllm_fmha_v2_workspace('cpu', 2048), a)

    def test_device_mismatch_is_not_reused(self):
        a = torch.empty(4096, device='meta', dtype=torch.uint8)
        op._release_trtllm_fmha_v2_workspace(a)
        b = op._get_trtllm_fmha_v2_workspace('cpu', 2048)
        self.assertEqual(b.device.type, 'cpu')
        self.assertIs(op._g_trtllm_fmha_v2_workspace_pool[0], a)

    def test_increasing_sizes_do_not_accumulate_idle_buffers(self):
        for size in (1024, 2048, 4096, 8192):
            buf = op._get_trtllm_fmha_v2_workspace('cpu', size)
            op._release_trtllm_fmha_v2_workspace(buf)
            self.assertEqual(len(op._g_trtllm_fmha_v2_workspace_pool), 1)
            self.assertIs(op._g_trtllm_fmha_v2_workspace_pool[0], buf)

    def test_distinct_borrowers_and_devices_keep_separate_storage(self):
        a = op._get_trtllm_fmha_v2_workspace('cpu', 2048)
        b = op._get_trtllm_fmha_v2_workspace('cpu', 4096)
        self.assertNotEqual(a.data_ptr(), b.data_ptr())
        other = torch.empty(8192, device='meta', dtype=torch.uint8)
        for buf in (a, b, other):
            op._release_trtllm_fmha_v2_workspace(buf)
        self.assertEqual(len(op._g_trtllm_fmha_v2_workspace_pool), 2)
        self.assertIs(op._get_trtllm_fmha_v2_workspace('cpu', 1024), b)
        self.assertIs(op._get_trtllm_fmha_v2_workspace('meta', 1024), other)


if __name__ == '__main__':
    unittest.main(verbosity=2)
