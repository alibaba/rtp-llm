import copy
import unittest

import torch

from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
from rtp_llm.ops.compute_ops import (
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
    get_typemeta,
)

GROUP_TAGS = ["full", "aux"]
HIDDEN_SIZE = 4
TOKENS_PER_BLOCK = 8


class GroupedBlockTableModel:
    """Small graph-safe model whose output exposes both tag-local block tables."""

    def __init__(self) -> None:
        self.recorders: dict[str, GroupPrepareRecorder] = {}
        self.capture_table_pointers: set[int] = set()

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        self.capture_table_pointers = {
            table.data_ptr()
            for tag in GROUP_TAGS
            for table in (
                inputs.attention_inputs[tag].kv_cache_kernel_block_id,
                inputs.attention_inputs[tag].kv_cache_kernel_block_id_device,
                inputs.attention_inputs[tag].kv_cache_block_id,
                inputs.attention_inputs[tag].kv_cache_block_id_device,
            )
        }
        self.recorders = {tag: GroupPrepareRecorder() for tag in GROUP_TAGS}
        return self.recorders

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        attention_inputs = inputs.attention_inputs
        full_inputs = attention_inputs["full"]
        aux_inputs = attention_inputs["aux"]
        signature = (
            full_inputs.kv_cache_kernel_block_id_device.sum()
            + 16 * aux_inputs.kv_cache_kernel_block_id_device.sum()
            + 256 * full_inputs.kv_cache_block_id_device.sum()
            + 4096 * aux_inputs.kv_cache_block_id_device.sum()
        ).to(inputs.input_hiddens.dtype)
        return PyModelOutputs(inputs.input_hiddens + signature)


class GroupPrepareRecorder:
    def __init__(self) -> None:
        self.host_physical: torch.Tensor | None = None

    def prepare_cuda_graph(self, inputs: PyAttentionInputs) -> None:
        self.host_physical = inputs.kv_cache_block_id.clone()


class GroupedSequenceLengthModel:
    """Expose the cumulative lengths used by a tagged captured graph."""

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        full_inputs = inputs.attention_inputs["full"]
        signature = torch.stack(
            (
                full_inputs.cu_seqlens_device[-1],
                full_inputs.cu_kv_seqlens_device[-1],
                full_inputs.input_lengths_device.sum(),
                full_inputs.prefix_lengths_device.sum(),
            )
        ).to(inputs.input_hiddens.dtype)
        return PyModelOutputs(inputs.input_hiddens + signature)


def _tag_attention_inputs(
    common: PyAttentionInputs,
    tags: list[str],
    kernel_values: dict[str, int],
    physical_values: dict[str, int],
    batch_size: int,
    kernel_block_count: int,
    physical_block_count: int,
) -> dict[str, PyAttentionInputs]:
    tagged = {}
    for tag in tags:
        tag_inputs = copy.copy(common)
        host_kernel_blocks = torch.full(
            (batch_size, kernel_block_count),
            kernel_values[tag],
            dtype=torch.int32,
        ).pin_memory()
        host_physical_blocks = torch.full(
            (batch_size, physical_block_count),
            physical_values[tag],
            dtype=torch.int32,
        ).pin_memory()
        tag_inputs.kv_cache_kernel_block_id = host_kernel_blocks
        tag_inputs.kv_cache_kernel_block_id_device = host_kernel_blocks.cuda()
        tag_inputs.kv_cache_block_id = host_physical_blocks
        tag_inputs.kv_cache_block_id_device = host_physical_blocks.cuda()
        tagged[tag] = tag_inputs
    return tagged


def _build_common_inputs(
    attention_inputs: PyAttentionInputs,
    tags: list[str],
    kernel_values: dict[str, int],
    physical_values: dict[str, int] | None,
    batch_size: int,
    token_count: int,
    kernel_block_count: int,
    physical_block_count: int,
) -> PyModelInputs:
    inputs = PyModelInputs()
    inputs.input_ids = torch.arange(token_count, dtype=torch.int32, device="cuda")
    inputs.input_hiddens = torch.zeros(
        (token_count, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
    )

    attention_inputs.dtype = get_typemeta(torch.zeros(1, dtype=torch.bfloat16))
    attention_inputs.padding_offset = torch.zeros(
        token_count, dtype=torch.int32, device="cuda"
    )
    attention_inputs.total_tokens = token_count
    attention_inputs.kv_cache_kernel_block_id = torch.zeros(
        (batch_size, kernel_block_count), dtype=torch.int32
    ).pin_memory()
    attention_inputs.kv_cache_kernel_block_id_device = (
        attention_inputs.kv_cache_kernel_block_id.cuda()
    )
    attention_inputs.kv_cache_block_id = torch.zeros(
        (batch_size, physical_block_count), dtype=torch.int32
    ).pin_memory()
    attention_inputs.kv_cache_block_id_device = (
        attention_inputs.kv_cache_block_id.cuda()
    )
    # Keep the request-level attention state alongside the tag-indexed block-table views.
    inputs.attention_inputs = attention_inputs
    inputs.attention_inputs = _tag_attention_inputs(
        attention_inputs,
        tags,
        kernel_values,
        physical_values or kernel_values,
        batch_size,
        kernel_block_count,
        physical_block_count,
    )
    return inputs


def _build_decode_inputs(
    tags: list[str],
    kernel_values: dict[str, int],
    physical_values: dict[str, int] | None = None,
    batch_size: int = 2,
    kernel_block_count: int = 1,
    physical_block_count: int = 1,
) -> PyModelInputs:
    attention_inputs = PyAttentionInputs()
    attention_inputs.is_prefill = False
    attention_inputs.is_target_verify = False
    attention_inputs.prefix_lengths = torch.empty(0, dtype=torch.int32).pin_memory()
    attention_inputs.input_lengths = torch.ones(
        batch_size, dtype=torch.int32
    ).pin_memory()
    attention_inputs.sequence_lengths = torch.ones(
        batch_size, dtype=torch.int32
    ).pin_memory()
    attention_inputs.sequence_lengths_plus_1_device = torch.full(
        (batch_size,), 2, dtype=torch.int32, device="cuda"
    )
    attention_inputs.decode_cu_seqlens_device = torch.arange(
        batch_size + 1, dtype=torch.int32, device="cuda"
    )
    attention_inputs.cu_seqlens = torch.zeros(
        batch_size + 1, dtype=torch.int32
    ).pin_memory()
    attention_inputs.cu_seqlens_device = attention_inputs.cu_seqlens.cuda()
    attention_inputs.cu_kv_seqlens_device = torch.zeros_like(
        attention_inputs.cu_seqlens_device
    )
    attention_inputs.context_total_kv_length = batch_size
    return _build_common_inputs(
        attention_inputs,
        tags,
        kernel_values,
        physical_values,
        batch_size=batch_size,
        token_count=batch_size,
        kernel_block_count=kernel_block_count,
        physical_block_count=physical_block_count,
    )


def _build_prefill_inputs(
    tags: list[str],
    kernel_values: dict[str, int],
    physical_values: dict[str, int] | None = None,
    seq_len: int = 4,
    kernel_block_count: int = 1,
    physical_block_count: int = 1,
) -> PyModelInputs:
    attention_inputs = PyAttentionInputs()
    attention_inputs.is_prefill = True
    attention_inputs.is_target_verify = False
    attention_inputs.input_lengths = torch.tensor(
        [seq_len], dtype=torch.int32
    ).pin_memory()
    attention_inputs.prefix_lengths = torch.zeros(1, dtype=torch.int32).pin_memory()
    attention_inputs.cu_seqlens = torch.tensor(
        [0, seq_len], dtype=torch.int32
    ).pin_memory()
    attention_inputs.cu_seqlens_device = attention_inputs.cu_seqlens.cuda()
    attention_inputs.cu_kv_seqlens_device = attention_inputs.cu_seqlens_device.clone()
    attention_inputs.context_total_kv_length = seq_len
    return _build_common_inputs(
        attention_inputs,
        tags,
        kernel_values,
        physical_values,
        batch_size=1,
        token_count=seq_len,
        kernel_block_count=kernel_block_count,
        physical_block_count=physical_block_count,
    )


def _build_target_verify_inputs(
    tags: list[str],
    values: dict[str, int],
    batch_size: int = 1,
    query_len: int = 5,
    prefix_len: int = 11,
    is_prefill: bool = True,
) -> PyModelInputs:
    token_count = batch_size * query_len

    attention_inputs = PyAttentionInputs()
    attention_inputs.is_prefill = is_prefill
    attention_inputs.is_target_verify = True
    attention_inputs.input_lengths = torch.full(
        (batch_size,), query_len, dtype=torch.int32
    ).pin_memory()
    attention_inputs.prefix_lengths = torch.full(
        (batch_size,), prefix_len, dtype=torch.int32
    ).pin_memory()
    attention_inputs.sequence_lengths = torch.empty(0, dtype=torch.int32).pin_memory()
    attention_inputs.sequence_lengths_plus_1_device = (
        attention_inputs.prefix_lengths.cuda() + 1
    )

    cu_q = torch.arange(0, token_count + 1, query_len, dtype=torch.int32).pin_memory()
    attention_inputs.cu_seqlens = cu_q
    attention_inputs.cu_seqlens_device = cu_q.cuda()
    attention_inputs.cu_kv_seqlens_device = torch.arange(
        0,
        batch_size * (query_len + prefix_len) + 1,
        query_len + prefix_len,
        dtype=torch.int32,
        device="cuda",
    )
    attention_inputs.decode_cu_seqlens = torch.arange(
        batch_size + 1, dtype=torch.int32
    ).pin_memory()
    attention_inputs.decode_cu_seqlens_device = (
        attention_inputs.decode_cu_seqlens.cuda()
    )

    attention_inputs.context_total_kv_length = batch_size * (query_len + prefix_len)

    if block_count is None:
        block_count = (
            prefix_len + query_len + TOKENS_PER_BLOCK - 1
        ) // TOKENS_PER_BLOCK
    return _build_common_inputs(
        attention_inputs,
        tags,
        values,
        None,
        batch_size=batch_size,
        token_count=token_count,
        kernel_block_count=block_count,
        physical_block_count=block_count,
    )


def _expected_signature(
    kernel_values: dict[str, int],
    physical_values: dict[str, int],
    batch_size: int,
    kernel_block_count: int,
    physical_block_count: int,
) -> int:
    return (
        batch_size * kernel_block_count * kernel_values["full"]
        + 16 * batch_size * kernel_block_count * kernel_values["aux"]
        + 256 * batch_size * physical_block_count * physical_values["full"]
        + 4096 * batch_size * physical_block_count * physical_values["aux"]
    )


class TestCudaGraphGroupedCache(unittest.TestCase):
    def _assert_replay_signature(
        self, runner: CudaGraphRunner, inputs: PyModelInputs, expected: int
    ) -> None:
        self.assertTrue(runner.canRun(inputs))
        output = runner.forward(inputs)
        torch.cuda.synchronize()
        expected_output = torch.full_like(output.hidden_states, expected)
        torch.testing.assert_close(output.hidden_states, expected_output)

    def test_decode_tag_validation_and_replay_updates(self) -> None:
        model = GroupedBlockTableModel()
        runner = CudaGraphRunner()
        runner.init_decode(
            model,
            HIDDEN_SIZE,
            4 * TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [2],
            GROUP_TAGS,
        )
        self.assertEqual(len(model.capture_table_pointers), 8)

        first_kernel = {"full": 2, "aux": 1}
        first_physical = {"full": 7, "aux": 3}
        first_inputs = _build_decode_inputs(
            GROUP_TAGS,
            first_kernel,
            first_physical,
            batch_size=2,
            kernel_block_count=3,
            physical_block_count=2,
        )
        full_inputs = first_inputs.attention_inputs["full"]
        aux_inputs = first_inputs.attention_inputs["aux"]
        pointers = {
            full_inputs.kv_cache_kernel_block_id.data_ptr(),
            full_inputs.kv_cache_kernel_block_id_device.data_ptr(),
            full_inputs.kv_cache_block_id.data_ptr(),
            full_inputs.kv_cache_block_id_device.data_ptr(),
            aux_inputs.kv_cache_kernel_block_id.data_ptr(),
            aux_inputs.kv_cache_kernel_block_id_device.data_ptr(),
            aux_inputs.kv_cache_block_id.data_ptr(),
            aux_inputs.kv_cache_block_id_device.data_ptr(),
        }
        self.assertEqual(len(pointers), 8)
        self._assert_replay_signature(
            runner,
            first_inputs,
            _expected_signature(first_kernel, first_physical, 2, 3, 2),
        )
        torch.testing.assert_close(
            model.recorders["full"].host_physical,
            torch.tensor([[7, 7, 0, 0], [7, 7, 0, 0]], dtype=torch.int32),
        )

        second_kernel = {"full": 5, "aux": 3}
        second_physical = {"full": 11, "aux": 13}
        self._assert_replay_signature(
            runner,
            _build_decode_inputs(
                GROUP_TAGS,
                second_kernel,
                second_physical,
                batch_size=1,
            ),
            _expected_signature(second_kernel, second_physical, 1, 1, 1),
        )
        torch.testing.assert_close(
            model.recorders["full"].host_physical,
            torch.tensor([[11, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.int32),
        )

        self.assertFalse(runner.canRun(_build_decode_inputs(["full"], {"full": 2})))
        self.assertFalse(
            runner.canRun(
                _build_decode_inputs(
                    ["full", "aux", "extra"],
                    {"full": 2, "aux": 1, "extra": 9},
                )
            )
        )
        self.assertFalse(
            runner.canRun(
                _build_decode_inputs(["full", "wrong"], {"full": 2, "wrong": 1})
            )
        )

    def test_prefill_grouped_capture_and_replay_updates(self) -> None:
        model = GroupedBlockTableModel()
        runner = CudaGraphRunner()
        runner.init_prefill(
            model,
            2,
            2 * TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [4],
            HIDDEN_SIZE,
            GROUP_TAGS,
        )

        first_kernel = {"full": 1, "aux": 2}
        first_physical = {"full": 3, "aux": 4}
        self._assert_replay_signature(
            runner,
            _build_prefill_inputs(
                GROUP_TAGS,
                first_kernel,
                first_physical,
                kernel_block_count=2,
                physical_block_count=2,
            ),
            _expected_signature(first_kernel, first_physical, 1, 2, 2),
        )
        torch.testing.assert_close(
            model.recorders["aux"].host_physical,
            torch.tensor([[4, 4], [0, 0]], dtype=torch.int32),
        )

        second_kernel = {"full": 4, "aux": 3}
        second_physical = {"full": 6, "aux": 5}
        self._assert_replay_signature(
            runner,
            _build_prefill_inputs(
                GROUP_TAGS, second_kernel, second_physical, seq_len=4
            ),
            _expected_signature(second_kernel, second_physical, 1, 1, 1),
        )
        torch.testing.assert_close(
            model.recorders["aux"].host_physical,
            torch.tensor([[5, 0], [0, 0]], dtype=torch.int32),
        )

    def test_grouped_block_table_validation_falls_back(self) -> None:
        runner = CudaGraphRunner()
        runner.init_decode(
            GroupedBlockTableModel(),
            HIDDEN_SIZE,
            2 * TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [2],
            GROUP_TAGS,
        )

        missing = _build_decode_inputs(GROUP_TAGS, {"full": 1, "aux": 2})
        runner.clearGroupPhysicalBlockTable(missing, "full", False)
        self.assertFalse(runner.canRun(missing))

        wrong_type = _build_decode_inputs(GROUP_TAGS, {"full": 1, "aux": 2})
        wrong_type.attention_inputs["full"].kv_cache_block_id_device = (
            wrong_type.attention_inputs["full"].kv_cache_block_id_device.to(torch.int64)
        )
        self.assertFalse(runner.canRun(wrong_type))

        wrong_dimension = _build_decode_inputs(GROUP_TAGS, {"full": 1, "aux": 2})
        wrong_dimension.attention_inputs["aux"].kv_cache_block_id = (
            wrong_dimension.attention_inputs["aux"].kv_cache_block_id.flatten()
        )
        self.assertFalse(runner.canRun(wrong_dimension))

        over_capacity = _build_decode_inputs(
            GROUP_TAGS,
            {"full": 1, "aux": 2},
            physical_block_count=3,
        )
        self.assertFalse(runner.canRun(over_capacity))

    def test_duplicate_capture_tag_is_rejected(self) -> None:
        runner = CudaGraphRunner()
        with self.assertRaisesRegex(
            RuntimeError, "duplicate CUDA graph KV cache tag=full"
        ):
            runner.init_decode(
                GroupedBlockTableModel(),
                HIDDEN_SIZE,
                TOKENS_PER_BLOCK,
                TOKENS_PER_BLOCK,
                TOKENS_PER_BLOCK,
                [1],
                ["full", "full"],
            )

    def test_target_verify_validates_exact_tag_set(self) -> None:
        runner = CudaGraphRunner()
        runner.init_decode(
            GroupedBlockTableModel(),
            HIDDEN_SIZE,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [2],
            GROUP_TAGS,
            True,
        )

        valid = _build_target_verify_inputs(
            GROUP_TAGS,
            {"full": 2, "aux": 1},
            batch_size=2,
            query_len=1,
            prefix_len=1,
        )
        self.assertTrue(runner.canRun(valid))

        missing = _build_target_verify_inputs(
            ["full"], {"full": 2}, batch_size=2, query_len=1, prefix_len=1
        )
        self.assertFalse(runner.canRun(missing))

        wrong = _build_target_verify_inputs(
            ["full", "wrong"],
            {"full": 2, "wrong": 1},
            batch_size=2,
            query_len=1,
            prefix_len=1,
        )
        self.assertFalse(runner.canRun(wrong))

        non_prefill = _build_target_verify_inputs(
            GROUP_TAGS,
            {"full": 2, "aux": 1},
            batch_size=2,
            query_len=1,
            prefix_len=1,
            is_prefill=False,
        )
        self.assertFalse(runner.canRun(non_prefill))

    def test_target_verify_clears_rounded_batch_sequence_lengths(self) -> None:
        query_len = 5
        prefix_len = 11
        runner = CudaGraphRunner()
        runner.init_decode(
            GroupedSequenceLengthModel(),
            HIDDEN_SIZE,
            64,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [4],
            GROUP_TAGS,
            True,
            query_len,
        )

        for batch_size in (1, 2, 4):
            with self.subTest(batch_size=batch_size):
                inputs = _build_target_verify_inputs(
                    GROUP_TAGS,
                    {"full": 2, "aux": 1},
                    batch_size=batch_size,
                    query_len=query_len,
                    prefix_len=prefix_len,
                )
                self.assertTrue(runner.canRun(inputs))
                self.assertEqual(runner.getCurrentRealGraphSize(), 4)

                output = runner.forward(inputs)
                torch.cuda.synchronize()
                total_query_length = batch_size * query_len
                total_kv_length = batch_size * (query_len + prefix_len)
                expected_signature = torch.tensor(
                    [
                        total_query_length,
                        total_kv_length,
                        total_query_length,
                        batch_size * prefix_len,
                    ],
                    dtype=output.hidden_states.dtype,
                    device=output.hidden_states.device,
                )
                torch.testing.assert_close(
                    output.hidden_states,
                    expected_signature.unsqueeze(0).expand_as(output.hidden_states),
                )


if __name__ == "__main__":
    unittest.main()
