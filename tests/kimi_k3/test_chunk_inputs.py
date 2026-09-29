"""CPU packing checks; native binding and CUDA execution require GPU validation."""

import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Attention(SimpleNamespace):
    def for_prefill_chunk(self, lengths, prefixes, device_lengths, device_prefixes):
        value = copy.copy(self)
        value.input_lengths = lengths
        value.prefix_lengths = prefixes
        value.input_lengths_device = device_lengths
        value.prefix_lengths_device = device_prefixes
        value.cache_store_writer = None
        value.cache_store_inputs = None
        return value


binding = ModuleType("rtp_llm.ops.compute_ops")
binding.PyModelInputs = SimpleNamespace
with patch.dict(sys.modules, {"rtp_llm.ops.compute_ops": binding}):
    packing = load(
        "k3_chunk_inputs_test", "rtp_llm/models_py/modules/kimi_k3/chunk_inputs.py"
    )
plans = load("k3_chunk_plans_test", "rtp_llm/models_py/modules/kimi_k3/chunk_plan.py")


@pytest.mark.parametrize("tail", [1, 7])
@pytest.mark.parametrize("tagged", [False, True])
def test_round_rebuild_preserves_real_rows_and_isolates_dummy_cache(tail, tagged):
    lengths = [8 + tail, 17]
    total = sum(lengths)
    ids = torch.arange(total, dtype=torch.int32)
    hidden = torch.arange(total * 2).reshape(total, 2)
    positions = torch.cat((torch.arange(lengths[0]), torch.arange(lengths[1]) + 4))
    table = torch.tensor([[11, 12, 13, 14, 15], [21, 22, 23, 24, 25]])
    source = Attention(
        kv_cache_block_id=table,
        kv_cache_block_id_device=table.clone(),
        kv_cache_kernel_block_id=table + 100,
        kv_cache_kernel_block_id_device=table + 100,
        cache_store_writer=object(),
        cache_store_inputs=object(),
    )
    other = copy.copy(source)
    other.kv_cache_kernel_block_id = table + 200
    other.kv_cache_kernel_block_id_device = table + 200
    inputs = SimpleNamespace(
        input_ids=ids,
        input_hiddens=hidden,
        combo_position_ids=positions,
        attention_inputs={"linear": source, "full": other} if tagged else source,
    )
    rounds = plans.plan_kimi_k3_chunk_rounds(
        lengths, [0, 4], chunk_budget=8, alignment_tokens=4
    )
    for plan in rounds:
        result = packing.build_chunk_inputs(inputs, plan, tp_size=8)
        real = torch.cat(
            [torch.arange(s.source_start, s.source_end) for s in plan.slices]
        )
        n = len(real)
        assert result.input_ids[:n].equal(ids[real])
        assert result.input_hiddens[:n].equal(hidden[real])
        assert result.combo_position_ids[:n].equal(positions[real])
        assert len(result.input_ids) % 8 == 0
        assert not result.input_ids[n:].count_nonzero()
        for tag, attn in (
            result.attention_inputs.items()
            if tagged
            else [("linear", result.attention_inputs)]
        ):
            assert attn.cache_store_writer is None
            assert attn.cache_store_inputs is None
            assert attn.sequence_lengths.numel() == 0
            assert attn.valid_token_mask.sum() == n
            assert attn.cu_seqlens[-1] == len(result.input_ids)
            assert attn.prefix_lengths[: len(plan.slices)].tolist() == [
                s.absolute_start for s in plan.slices
            ]
            for i, s in enumerate(plan.slices):
                assert attn.kv_cache_block_id[i].equal(table[s.original_batch_idx])
                offset = 100 if tag == "linear" else 200
                assert attn.kv_cache_kernel_block_id[i].equal(
                    table[s.original_batch_idx] + offset
                )
            if n % 8:
                assert attn.physical_request_count == len(plan.slices) + 1
                assert not attn.kv_cache_block_id[-1].count_nonzero()
                assert not attn.kv_cache_kernel_block_id[-1].count_nonzero()
    assert inputs.input_ids.equal(ids)
    assert source.kv_cache_block_id.equal(table)
    assert source.cache_store_writer is not None


class Output:
    def __init__(self, hidden, features):
        self.hidden_states = hidden
        self.mtp_target_hidden_states = features


@pytest.mark.parametrize("fail_round", [None, 2])
def test_chunk_controller_restores_outputs_and_publishes_only_after_success(fail_round):
    routing = ModuleType("rtp_llm.models_py.model_desc.block_map")
    routing.get_primary_attention_inputs = lambda inputs, cache: inputs.attention_inputs
    routing.select_attention_inputs_for_layer = (
        lambda inputs, cache, index: inputs.attention_inputs
    )
    ops = ModuleType("rtp_llm.ops.compute_ops")
    ops.PyModelOutputs = Output
    with patch.dict(
        sys.modules,
        {
            routing.__name__: routing,
            ops.__name__: ops,
            "rtp_llm.models_py.modules.kimi_k3.chunk_inputs": packing,
            "rtp_llm.models_py.modules.kimi_k3.chunk_plan": plans,
        },
    ):
        controller = load(
            "k3_chunk_controller_test",
            "rtp_llm/models_py/modules/kimi_k3/chunk_forward.py",
        )
    writer = Mock()
    table = torch.tensor([[11] * 6, [21] * 6])
    attention = Attention(
        is_cuda_graph=False,
        is_target_verify=False,
        is_mtp_draft_update=False,
        context_parallel_info=None,
        logical_request_count=2,
        input_lengths=torch.tensor([9, 17], dtype=torch.int32),
        prefix_lengths=torch.tensor([0, 4], dtype=torch.int32),
        kv_cache_block_id=table,
        kv_cache_block_id_device=table,
        kv_cache_kernel_block_id=table,
        kv_cache_kernel_block_id_device=table,
        cache_store_writer=writer,
        cache_store_inputs=object(),
    )
    ids = torch.arange(1, 27)
    inputs = SimpleNamespace(
        input_ids=ids,
        input_hiddens=None,
        combo_position_ids=ids,
        attention_inputs=attention,
        multimodal_inputs=SimpleNamespace(multimodal_features=[], mm_extra_input=[]),
    )
    calls = []
    state = {11: 0, 21: 100}
    processed = {11: 0, 21: 4}

    def forward(chunk, fmha):
        calls.append(chunk)
        if len(calls) == fail_round:
            raise RuntimeError("injected round failure")
        hidden = torch.zeros((chunk.input_ids.numel(), 1))
        offset = 0
        meta = chunk.attention_inputs
        for i in range(meta.logical_request_count):
            key = int(meta.kv_cache_block_id[i, 0])
            assert meta.prefix_lengths[i] == processed[key]
            length = int(meta.input_lengths[i])
            values = chunk.input_ids[offset : offset + length].cumsum(0) + state[key]
            hidden[offset : offset + length, 0] = values
            state[key] = int(values[-1])
            processed[key] += length
            offset += length
        return Output(hidden * 2, hidden)

    model = SimpleNamespace(
        kv_cache=SimpleNamespace(
            get_layer_cache=lambda index: SimpleNamespace(seq_size_per_block=4)
        ),
        layer_num=2,
        chunk_prefill_budget=8,
        tp_size=8,
        _forward_single=forward,
    )
    if fail_round:
        with pytest.raises(RuntimeError, match="injected"):
            controller.forward_prefill_chunks(model, inputs)
        writer.write.assert_not_called()
    else:
        output = controller.forward_prefill_chunks(model, inputs)
        expected = (
            torch.cat((ids[:9].cumsum(0), ids[9:].cumsum(0) + 100))
            .float()
            .reshape(-1, 1)
        )
        assert output.hidden_states.equal(expected * 2)
        assert output.mtp_target_hidden_states.equal(expected)
        assert writer.write.call_count == 2
        assert len(calls) > 1
