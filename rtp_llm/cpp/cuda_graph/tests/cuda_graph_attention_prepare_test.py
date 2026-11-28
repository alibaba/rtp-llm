"""Exercise model-owned preparation and the default external replay path."""

import unittest

import torch

from rtp_llm.cpp.cuda_graph.tests.cuda_graph_tagged_cache_test import (
    GROUP_TAGS,
    HIDDEN_SIZE,
    KERNEL_BLOCK_TABLE_WIDTH,
    CudaGraphRunner,
    _build_decode_inputs,
    _build_prefill_inputs,
    _build_target_verify_inputs,
)
from rtp_llm.ops.compute_ops import PyModelOutputs


class MetadataImpl:
    def __init__(self, inputs, tokens, read_host):
        self.calls = 0
        self.capture_calls = 0
        self.queries = tokens // inputs.input_lengths.numel()
        self.read_host = read_host
        self.host_sequence_lengths = inputs.sequence_lengths
        self.signature = torch.empty(
            (tokens, 1),
            dtype=torch.bfloat16,
            device=inputs.kv_cache_kernel_block_id_device.device,
        )
        self.prepare_cuda_graph(inputs)

    def prepare_cuda_graph(self, inputs):
        self.calls += 1
        self.capture_calls += int(torch.cuda.is_current_stream_capturing())
        lengths = (
            inputs.prefix_lengths_device
            if inputs.is_prefill
            else inputs.sequence_lengths_plus_1_device
        )
        values = inputs.kv_cache_kernel_block_id_device[:, 0] + lengths
        if self.read_host:
            if inputs.sequence_lengths.is_cuda:
                raise RuntimeError("host-dependent preparation lost its CPU mirror")
            values = values + torch.tensor(
                inputs.sequence_lengths.tolist(), device=values.device
            )
        self.signature.copy_(
            values.repeat_interleave(self.queries)[:, None].to(torch.bfloat16)
        )


class MetadataModel:
    def __init__(self, read_host=(False, False), *, prepare_in_forward=False):
        self.read_host = read_host
        if prepare_in_forward:
            self.cuda_graph_prepares_attention_metadata_in_forward = True
        self.probes = []

    def prepare_fmha_impl(
        self, inputs, is_cuda_graph=False, cuda_graph_selection_mode=None
    ):
        tokens = inputs.input_ids.numel()

        def create(attention, read_host):
            impl = MetadataImpl(attention, tokens, read_host)
            self.probes.append(impl)
            return impl

        groups = inputs.attention_inputs
        if isinstance(groups, dict):
            return {
                tag: create(groups[tag], read_host)
                for tag, read_host in zip(GROUP_TAGS, self.read_host)
            }
        return create(groups, self.read_host[0])

    def forward(self, inputs, fmha_impl=None):
        groups = inputs.attention_inputs
        primary = groups["full"] if isinstance(groups, dict) else groups
        if getattr(
            self, "cuda_graph_prepares_attention_metadata_in_forward", False
        ) and (
            not primary.is_prefill
            or primary.is_target_verify
            or primary.is_mtp_draft_update
        ):
            if isinstance(fmha_impl, dict):
                for tag, impl in fmha_impl.items():
                    impl.prepare_cuda_graph(groups[tag])
            else:
                fmha_impl.prepare_cuda_graph(groups)
        impls = fmha_impl.values() if isinstance(fmha_impl, dict) else (fmha_impl,)
        output = inputs.input_hiddens
        for factor, impl in zip((1, 16), impls):
            output = output + factor * impl.signature
        return PyModelOutputs(output)


def make_runner(model, role):
    runner = CudaGraphRunner()
    if role == "update":
        runner.init_mtp_draft_prefill(model, 2, HIDDEN_SIZE, 4, 1, request_lanes=True)
    else:
        runner.init_decode(
            model,
            HIDDEN_SIZE,
            128,
            KERNEL_BLOCK_TABLE_WIDTH,
            [1, 2],
            GROUP_TAGS,
            role == "verify",
            4 if role == "verify" else 1,
            0,
            2,
        )
    return runner


def make_inputs(role, batch, step):
    values = {"full": step + 1, "aux": step + 2}
    if role == "decode":
        inputs = _build_decode_inputs(GROUP_TAGS, values, batch_size=batch)
        for attention in inputs.attention_inputs.values():
            attention.sequence_lengths.fill_(step + 4)
            attention.sequence_lengths_plus_1_device.fill_(step + 5)
    else:
        inputs = _build_target_verify_inputs(
            GROUP_TAGS,
            values,
            batch_size=batch,
            query_len=4,
            prefix_len=step % 4 if role == "update" else step + 4,
        )
    if role == "update":
        attention = inputs.attention_inputs["full"]
        attention.is_target_verify = False
        attention.is_mtp_draft_update = True
        # The MTP test runner captures position_id_len_factor=1. Match the
        # native update's Q-wide per-request positions when replaying it.
        inputs.combo_position_ids = (
            attention.prefix_lengths.to(
                device=inputs.input_ids.device, non_blocking=True
            )[:, None]
            + torch.arange(4, dtype=torch.int32, device=inputs.input_ids.device)
        ).reshape(-1)
        inputs.attention_inputs = attention
    inputs.input_hiddens.fill_(step)
    return inputs


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class AttentionPrepareGraphTest(unittest.TestCase):
    def test_embedding_prefill_retains_external_preparation(self):
        model = MetadataModel(prepare_in_forward=True)
        runner = CudaGraphRunner()
        runner.init_prefill(
            model, 1, 8, KERNEL_BLOCK_TABLE_WIDTH, [4, 8], HIDDEN_SIZE, GROUP_TAGS
        )
        self.assertTrue(all(p.capture_calls == 0 for p in model.probes))
        for step, tokens in ((1, 4), (3, 8), (5, 4)):
            values = {"full": step + 1, "aux": step + 2}
            inputs = _build_prefill_inputs(GROUP_TAGS, values, tokens)
            self.assertTrue(runner.canRun(inputs))
            calls = sum(p.calls for p in model.probes)
            output = runner.forward(inputs).hidden_states
            self.assertEqual(sum(p.calls for p in model.probes), calls + 2)
            torch.testing.assert_close(
                output,
                torch.full_like(output, step + 1 + 16 * (step + 2)),
                rtol=0,
                atol=0,
            )

    def test_device_prepare_is_replayed_for_changing_metadata(self):
        for role in ("decode", "verify", "update"):
            with self.subTest(role=role):
                model = MetadataModel(prepare_in_forward=True)
                runner = make_runner(model, role)
                captured = [p for p in model.probes if p.capture_calls]
                self.assertTrue(captured)
                # Backend initialization, both eager warmups, then capture.
                self.assertTrue(all(p.calls == 4 for p in captured))
                calls = sum(p.calls for p in model.probes)
                if role == "decode":
                    host_lengths = [p.host_sequence_lengths.clone() for p in captured]
                for step, batch in ((1, 1), (2, 2), (4, 1), (7, 2)):
                    inputs = make_inputs(role, batch, step)
                    self.assertTrue(runner.canRun(inputs))
                    if step % 2 == 0:
                        self.assertTrue(runner.prepare(inputs))
                    output = runner.forward(inputs).hidden_states
                    length = (
                        step % 4
                        if role == "update"
                        else step + (5 if role == "decode" else 4)
                    )
                    signature = step + 1 + length
                    if role != "update":
                        signature += 16 * (step + 2 + length)
                    torch.testing.assert_close(
                        output,
                        torch.full_like(output, step + signature),
                        rtol=0,
                        atol=0,
                    )
                self.assertEqual(sum(p.calls for p in model.probes), calls)
                if role == "decode":
                    for probe, lengths in zip(captured, host_lengths):
                        self.assertTrue(
                            torch.equal(probe.host_sequence_lengths, lengths)
                        )

    def test_models_without_opt_in_keep_external_prepare(self):
        for read_host in (False, True):
            with self.subTest(read_host=read_host):
                model = MetadataModel(read_host=(False, read_host))
                runner = make_runner(model, "decode")
                self.assertTrue(all(p.capture_calls == 0 for p in model.probes))
                for step in (2, 5):
                    inputs = make_inputs("decode", 2, step)
                    self.assertTrue(runner.canRun(inputs))
                    calls = sum(p.calls for p in model.probes)
                    output = runner.forward(inputs).hidden_states
                    self.assertEqual(sum(p.calls for p in model.probes), calls + 2)
                    signature = step + 1 + step + 5
                    signature += 16 * (step + 2 + step + 5)
                    if read_host:
                        signature += 16 * (step + 4)
                    torch.testing.assert_close(
                        output,
                        torch.full_like(output, step + signature),
                        rtol=0,
                        atol=0,
                    )


if __name__ == "__main__":
    unittest.main()
