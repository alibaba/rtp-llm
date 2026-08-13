import importlib.util
import os
import unittest
from types import SimpleNamespace

import torch

try:
    from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
except ModuleNotFoundError:
    runner_so = os.environ.get("CUDA_GRAPH_TEST_RUNNER_SO")
    if not runner_so:
        raise
    spec = importlib.util.spec_from_file_location(
        "libtest_cuda_graph_runner", runner_so
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load CUDA Graph test runner from {runner_so}")
    runner_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner_module)
    CudaGraphRunner = runner_module.CudaGraphRunner
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import flashinfer_mla
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla import (
    MlaFlashInferDecodeOp,
)
from rtp_llm.ops.compute_ops import (
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
    rtp_llm_ops,
)
from rtp_llm.utils.model_weight import W


class _LayerKVCache:
    def __init__(self, kv_cache_base: torch.Tensor):
        self.kv_cache_base = kv_cache_base


class _FlashInferTargetVerifyModel:
    num_heads = 12
    kv_lora_rank = 512
    rope_dim = 64
    nope_dim = 128
    value_dim = 128
    page_size = 128
    max_seq_len = 131072
    layer_count = int(os.environ.get("FLASHINFER_GRAPH_TEST_LAYERS", "26"))

    def __init__(self) -> None:
        torch.manual_seed(41)
        self.planned_kv_lengths = []
        self.k_weight = (
            torch.randn(
                self.num_heads,
                self.nope_dim,
                self.kv_lora_rank,
                dtype=torch.float32,
                device="cuda",
            )
            * 0.02
        ).to(torch.bfloat16)
        self.v_weight = (
            torch.randn(
                self.num_heads,
                self.kv_lora_rank,
                self.value_dim,
                dtype=torch.float32,
                device="cuda",
            )
            * 0.02
        ).to(torch.bfloat16)
        max_pages = self.max_seq_len // self.page_size
        self.layer_cache = _LayerKVCache(
            (
                torch.randn(
                    max_pages,
                    self.page_size,
                    self.kv_lora_rank + self.rope_dim,
                    dtype=torch.float32,
                    device="cuda",
                )
                * 0.1
            ).to(torch.bfloat16)
        )

    def prepare_fmha_impl(self, inputs, is_cuda_graph):
        attention = inputs.attention_inputs
        batch_size = int(attention.input_lengths.numel())
        query_length = int(attention.total_tokens) // batch_size
        op = MlaFlashInferDecodeOp(
            self.num_heads,
            self.kv_lora_rank,
            self.rope_dim,
            self.nope_dim,
            self.page_size,
            1.0,
            True,
            False,
            [{W.mla_kc: self.k_weight, W.mla_vc: self.v_weight}] * self.layer_count,
            max_bs=batch_size,
            max_context_len=self.max_seq_len,
            num_tokens=batch_size * query_length,
            is_cuda_graph=is_cuda_graph,
        )
        params = rtp_llm_ops.FlashInferMlaAttnParams()
        sequence_lengths_host = attention.sequence_lengths_host
        if sequence_lengths_host is None:
            sequence_lengths_host = torch.empty(0, dtype=torch.int32)
        params.fill_params(
            attention.prefix_lengths_host,
            sequence_lengths_host,
            attention.input_lengths_host,
            attention.kv_cache_kernel_block_id_host,
            self.page_size,
            False,
        )
        op.plan(params)
        self.planned_kv_lengths.append(params.kvlen_h.tolist())
        metadata = SimpleNamespace(op=op, params=params)

        def prepare_cuda_graph(current_attention):
            current_sequence_lengths_host = current_attention.sequence_lengths_host
            if current_sequence_lengths_host is None:
                current_sequence_lengths_host = torch.empty(0, dtype=torch.int32)
            params.fill_params(
                current_attention.prefix_lengths_host,
                current_sequence_lengths_host,
                current_attention.input_lengths_host,
                current_attention.kv_cache_kernel_block_id_host,
                self.page_size,
                True,
            )
            op.plan(params)
            self.planned_kv_lengths.append(params.kvlen_h.tolist())

        metadata.prepare_cuda_graph = prepare_cuda_graph
        return metadata

    def reference_forward(self, inputs: PyModelInputs) -> torch.Tensor:
        """Run the same FlashInfer MLA attention eagerly for parity checking."""
        attention = inputs.attention_inputs
        query_length = int(attention.total_tokens) // int(
            attention.input_lengths.numel()
        )
        op = MlaFlashInferDecodeOp(
            self.num_heads,
            self.kv_lora_rank,
            self.rope_dim,
            self.nope_dim,
            self.page_size,
            1.0,
            True,
            False,
            [{W.mla_kc: self.k_weight, W.mla_vc: self.v_weight}],
            max_bs=int(attention.input_lengths.numel()),
            max_context_len=self.max_seq_len,
            num_tokens=int(attention.total_tokens),
            is_cuda_graph=False,
        )
        params = rtp_llm_ops.FlashInferMlaAttnParams()
        params.fill_params(
            attention.prefix_lengths_host,
            attention.sequence_lengths_host,
            attention.input_lengths_host,
            attention.kv_cache_kernel_block_id_host,
            self.page_size,
            False,
        )
        op.plan(params)
        hidden = inputs.input_hiddens.to(torch.bfloat16)
        q_nope = hidden[:, : self.num_heads * self.nope_dim].view(
            -1, self.num_heads, self.nope_dim
        )
        q_pe = hidden[
            :,
            self.num_heads
            * self.nope_dim : self.num_heads
            * (self.nope_dim + self.rope_dim),
        ].view(-1, self.num_heads, self.rope_dim)
        output = op.forward(q_nope, q_pe, self.layer_cache, 0).flatten(1)
        return torch.nn.functional.pad(
            output, (0, inputs.input_hiddens.size(1) - output.size(1))
        )

    def forward(self, inputs, fmha_impl=None):
        hidden = inputs.input_hiddens.to(torch.bfloat16)
        q_nope = hidden[:, : self.num_heads * self.nope_dim].view(
            -1, self.num_heads, self.nope_dim
        )
        q_pe = hidden[
            :,
            self.num_heads
            * self.nope_dim : self.num_heads
            * (self.nope_dim + self.rope_dim),
        ].view(-1, self.num_heads, self.rope_dim)
        for layer_id in range(self.layer_count):
            output = fmha_impl.op.forward(q_nope, q_pe, self.layer_cache, layer_id)
        # Keep the real FlashInfer output in the graph while preserving the
        # runner's fixed hidden width.
        output_flat = output.flatten(1)
        padded = torch.nn.functional.pad(
            output_flat, (0, inputs.input_hiddens.size(1) - output_flat.size(1))
        )
        return PyModelOutputs(padded)


class CudaGraphFlashInferTargetVerifyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(torch.cuda.is_available(), "test target requires CUDA")
        self.assertEqual(
            torch.cuda.get_device_capability(0)[0],
            10,
            "test target requires an SM100/SM103 GPU",
        )
        torch.cuda.set_device(0)
        flashinfer_mla.g_workspace_buffer = None

    def tearDown(self) -> None:
        flashinfer_mla.g_workspace_buffer = None

    @staticmethod
    def _build_replay_inputs(hidden_size: int) -> PyModelInputs:
        query_length = 4
        prefix_length = 64000
        max_kernel_pages = 131072 // 128
        inputs = PyModelInputs()
        attention = PyAttentionInputs()
        inputs.input_ids = torch.arange(query_length, dtype=torch.int32, device="cuda")
        inputs.input_hiddens = torch.randn(
            query_length, hidden_size, dtype=torch.float16, device="cuda"
        )
        attention.input_lengths = torch.tensor(
            [query_length], dtype=torch.int32, device="cuda"
        )
        attention.input_lengths_host = torch.tensor(
            [query_length], dtype=torch.int32
        ).pin_memory()
        attention.prefix_lengths = torch.tensor(
            [prefix_length], dtype=torch.int32, device="cuda"
        )
        attention.prefix_lengths_host = torch.tensor(
            [prefix_length], dtype=torch.int32
        ).pin_memory()
        attention.sequence_lengths = attention.prefix_lengths.clone()
        attention.sequence_lengths_host = attention.prefix_lengths_host.clone()
        attention.sequence_lengths_plus_1_d = attention.sequence_lengths + 1
        attention.decode_cu_seqlens_d = torch.tensor(
            [0, query_length], dtype=torch.int32, device="cuda"
        )
        block_table = torch.arange(
            max_kernel_pages, dtype=torch.int32, device="cuda"
        ).view(1, -1)
        attention.kv_cache_kernel_block_id_device = block_table
        attention.kv_cache_kernel_block_id_host = block_table.cpu()
        attention.kv_cache_block_id_device = block_table
        attention.kv_cache_block_id_host = attention.kv_cache_kernel_block_id_host
        attention.cu_seqlens = attention.decode_cu_seqlens_d
        attention.cu_kv_seqlens = attention.decode_cu_seqlens_d.clone()
        attention.padding_offset = torch.zeros(
            query_length, dtype=torch.int32, device="cuda"
        )
        attention.is_prefill = True
        attention.is_target_verify = True
        attention.total_tokens = query_length
        attention.context_total_kv_length = prefix_length + query_length
        inputs.attention_inputs = attention
        return inputs

    def test_k3_q_len4_first_runner_replay(self) -> None:
        model = _FlashInferTargetVerifyModel()
        runner = CudaGraphRunner()
        runner.init_decode(
            model,
            hidden_size=model.num_heads * (model.nope_dim + model.rope_dim),
            max_seq_len=model.max_seq_len,
            tokens_per_block=4096,
            kernel_tokens_per_block=model.page_size,
            decode_capture_batch_sizes=[1],
            num_tokens_per_bs=4,
            is_target_verify=True,
            max_context_batch_size=1,
        )
        # init_decode performs the first capture/replay validation. A second
        # replay verifies that the captured graph remains executable.
        replay_inputs = self._build_replay_inputs(
            model.num_heads * (model.nope_dim + model.rope_dim)
        )
        expected = model.reference_forward(replay_inputs)
        self.assertTrue(runner.canRun(replay_inputs))
        outputs = runner.forward(replay_inputs)
        torch.cuda.synchronize()
        self.assertTrue(torch.isfinite(outputs.hidden_states).all().item())
        self.assertIn([model.max_seq_len], model.planned_kv_lengths)
        self.assertEqual(model.planned_kv_lengths[-1], [64004])
        torch.testing.assert_close(
            outputs.hidden_states,
            expected.to(outputs.hidden_states.dtype),
            rtol=2e-2,
            atol=2e-2,
        )


if __name__ == "__main__":
    unittest.main()
