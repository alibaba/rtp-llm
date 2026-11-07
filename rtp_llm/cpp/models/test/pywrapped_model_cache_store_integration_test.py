import math
import unittest

import torch

from rtp_llm.cpp.models.test.libth_pywrapped_model_cache_store_integration_test import (
    PyModelInputs,
    PyModelOutputs,
    clear_comm_ops,
    register_comm_ops,
    run_scenario,
)


class CacheStoreForwardModel:
    """Test model that replaces attention math but keeps the real cache-store call."""

    def __init__(self) -> None:
        self.kv_cache = None
        self.forward_calls = 0
        self.micro_batch_calls = 0
        self.seen_input_lengths: list[list[int]] = []

    def initialize(self, resources) -> bool:
        self.kv_cache = resources.kv_cache
        return True

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        return None

    def _forward_one(self, inputs: PyModelInputs) -> PyModelOutputs:
        attention_inputs = inputs.attention_inputs
        first_inputs = (
            next(iter(attention_inputs.values()))
            if isinstance(attention_inputs, dict)
            else attention_inputs
        )
        self.seen_input_lengths.append(first_inputs.input_lengths.tolist())

        assert self.kv_cache is not None
        for layer_cache in self.kv_cache.get_layer_cache_groups(0):
            tag_inputs = (
                attention_inputs[layer_cache.tag]
                if isinstance(attention_inputs, dict)
                else attention_inputs
            )
            if (
                tag_inputs.cache_store_inputs is not None
                and tag_inputs.cache_store_writer is not None
            ):
                tag_inputs.cache_store_writer.write(
                    tag_inputs.cache_store_inputs, layer_cache
                )

        # Hidden rows carry the token id so a row-selection bug is visible in the
        # returned logits values, not only in shapes.
        hidden_states = inputs.input_ids.reshape(-1, 1).to(torch.float16)
        return PyModelOutputs(hidden_states)

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        self.forward_calls += 1
        return self._forward_one(inputs)

    def forward_micro_batch(self, inputs: list[PyModelInputs]) -> list[PyModelOutputs]:
        self.micro_batch_calls += 1
        return [self._forward_one(model_inputs) for model_inputs in inputs]


def _blocks_by_key(result: dict) -> dict[str, dict]:
    return {
        block["key"]: block
        for record in result["records"]
        for block in record["blocks"]
    }


def _record_for_request(result: dict, request_id: int) -> dict:
    matches = [
        record
        for record in result["records"]
        if record["request_id"] == str(request_id)
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one record for request {request_id}, got {len(matches)}"
        )
    return matches[0]


class PyWrappedModelCacheStoreIntegrationTest(unittest.TestCase):
    def test_multi_tag_uses_each_tag_local_physical_block_table(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "multi_tag")

        self.assertEqual(model.forward_calls, 1)
        self.assertEqual(len(result["records"]), 2)
        blocks = _blocks_by_key(result)

        full_blocks = {
            key: block for key, block in blocks.items() if "_tag_full" in key
        }
        linear_blocks = {
            key: block for key, block in blocks.items() if "_tag_linear" in key
        }
        self.assertEqual(len(full_blocks), 2)
        self.assertEqual(len(linear_blocks), 4)
        self.assertEqual(
            sorted(
                block["address"] - result["base_addresses"]["full"]
                for block in full_blocks.values()
            ),
            [16, 32],
        )
        self.assertEqual(
            sorted(
                block["address"] - result["base_addresses"]["linear"]
                for block in linear_blocks.values()
            ),
            [72, 96, 120, 144],
        )
        self.assertEqual({block["length"] for block in full_blocks.values()}, {16})
        self.assertEqual({block["length"] for block in linear_blocks.values()}, {24})

    def test_micro_batch_slices_request_metadata_with_block_rows(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "micro_batch")

        self.assertEqual(model.forward_calls, 0)
        self.assertEqual(model.micro_batch_calls, 1)
        self.assertEqual(model.seen_input_lengths, [[2, 4], [2]])
        self.assertEqual(len(result["records"]), 3)

        expected = {
            201: ([2101], [16]),
            202: ([2201, 2202], [32, 48]),
            203: ([2301], [64]),
        }
        base = result["base_addresses"]["default"]
        for request_id, (token_keys, offsets) in expected.items():
            record = _record_for_request(result, request_id)
            self.assertEqual(len(record["blocks"]), len(token_keys))
            self.assertEqual(
                sorted(block["address"] - base for block in record["blocks"]),
                offsets,
            )
            for token_key in token_keys:
                self.assertTrue(
                    any(
                        f"_token_id_str_{token_key}_" in block["key"]
                        for block in record["blocks"]
                    )
                )

    def test_micro_batch_scoring_rows_use_full_batch_reassembly(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "micro_batch_scoring")

        # need_all_logits must not disable layer micro-batching: the two sub-batches
        # run through forward_micro_batch, and post-layers receives the reassembled
        # full batch with the unsplit row indexes.
        self.assertEqual(model.forward_calls, 0)
        self.assertEqual(model.micro_batch_calls, 1)
        self.assertEqual(model.seen_input_lengths, [[2, 4], [2]])

        # tp==1 keeps the legacy contract: row metadata is ignored and the full
        # [tokens, vocab] window comes back with token-offset dispatch
        # (all_logits_lm_rows == -1). lm_output_indexes = [1, 5, 7] (last token of each
        # request), scoring rows [0, 2, 3] — both are inert here. With hidden = token
        # id and an all-ones head, each logits row is filled with its source token id
        # (tokens are iota starting at 1).
        all_logits = result["all_logits"]
        self.assertEqual(tuple(all_logits.shape), (8, 16))
        self.assertEqual(result["all_logits_lm_rows"], -1)
        torch.testing.assert_close(
            all_logits[:, 0].cpu(),
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            rtol=0,
            atol=0,
        )
        # The sampler-facing rows are the lm_output_indexes rows.
        torch.testing.assert_close(
            result["logits"][:, 0].cpu(), torch.tensor([2.0, 6.0, 8.0]), rtol=0, atol=0
        )

    def test_micro_batch_scoring_tp2_emits_sharded_partials(self) -> None:
        # tp>1 over mocked communication: the same scoring input must NOT produce a
        # [rows, vocab] all_logits at all — only the gathered lm rows and the merged
        # scoring partials (lse / target / top-k) in the [lm rows ++ scoring rows]
        # layout. The peer shard is emulated as local + 16 over a padded vocab of 32.
        gathered_shapes = []

        def gather(receivers, mode, senders, inplace):
            buffer = receivers[0]
            gathered_shapes.append(tuple(buffer.shape))
            local = buffer.view(2, buffer.size(0), -1)[0].clone()
            if buffer.size(0) > 1:
                # lm-row gather [3, 2*16]: peer = local + 16.
                buffer.view(2, buffer.size(0), -1).copy_(
                    torch.stack((local, local + 16))
                )
            else:
                # Scoring payload [1, 2P], P = rows*(2+2*kl): peer values +16; the
                # scenario's labels all live on rank0's shard so peer target = -inf.
                kl = 3
                rows = (buffer.size(1) // 2) // (2 + 2 * kl)
                p = local.reshape(-1)
                l0, t0, v0, i0 = p.split([rows, rows, rows * kl, rows * kl])
                r1 = torch.cat(
                    (
                        l0 + 16,
                        torch.full_like(t0, float("-inf")),
                        v0 + 16,
                        (i0.view(torch.int32).remainder(13) + 16).view(torch.float32),
                    )
                )
                buffer.view(2, -1).copy_(torch.stack((p, r1)))

        register_comm_ops(lambda *args: None, lambda *args: None, gather)
        try:
            model = CacheStoreForwardModel()
            result = run_scenario(model, "micro_batch_scoring_tp2")
        finally:
            clear_comm_ops()

        # Micro-batching still ran, and the reduction moved only compact payloads:
        # one [3, 32] lm-row gather and one payload all-gather covering the 3
        # scoring rows (shard_k = 3 -> P = 3*(2+2*3) = 24 -> buffer [1, 48]).
        self.assertEqual(model.micro_batch_calls, 1)
        self.assertEqual(gathered_shapes, [(3, 32), (1, 48)])

        self.assertIsNone(result["all_logits"])
        self.assertEqual(result["all_logits_lm_rows"], 3)

        # hidden = token id, all-ones head -> every local logit row is filled with the
        # row's source token id; the peer shard adds 16. lm rows = [2, 6, 8]: rank0's
        # half of the gathered row is v, rank1's half v+16.
        torch.testing.assert_close(
            result["logits"][:, 0].cpu(), torch.tensor([2.0, 6.0, 8.0]), rtol=0, atol=0
        )
        torch.testing.assert_close(
            result["logits"][:, 16].cpu(),
            torch.tensor([18.0, 22.0, 24.0]),
            rtol=0,
            atol=0,
        )
        lse = result["all_logits_lse"].cpu()
        # Row layout [lm(2,6,8) placeholder zeros; scoring(1,3,4) merged]. A scoring
        # row's local logits are constant v over 16 valid columns -> local lse =
        # v + ln16; the mock peer reports lse+16 -> merged = v + ln16 + ln(1+e^16).
        torch.testing.assert_close(lse[:3], torch.zeros(3), rtol=0, atol=0)
        for idx, v in enumerate([1.0, 3.0, 4.0]):
            self.assertAlmostEqual(
                float(lse[3 + idx]),
                v + math.log(16.0) + math.log1p(math.exp(16.0)),
                places=4,
            )
        # Labels {7,-1,4} are next-token ids < 16 -> all on rank0's shard.
        target = result["all_logits_target_logits"].cpu()
        torch.testing.assert_close(target[:3], torch.zeros(3), rtol=0, atol=0)
        torch.testing.assert_close(
            target[3:],
            torch.tensor([1.0, float("-inf"), 4.0]),
            rtol=0,
            atol=0,
        )
        # Merged top-k: peer values dominate; every scoring row's top-3 values equal
        # v+16 and the ids are rank1-valid (< 29) positions of equal-valued columns.
        topk_values = result["all_logits_topk_values"].cpu()
        topk_ids = result["all_logits_topk_ids"].cpu()
        self.assertEqual(tuple(topk_values.shape), (6, 3))
        torch.testing.assert_close(topk_values[:3], torch.zeros(3, 3), rtol=0, atol=0)
        for idx, v in enumerate([1.0, 3.0, 4.0]):
            torch.testing.assert_close(
                topk_values[3 + idx],
                torch.tensor([v + 16.0, v + 16.0, v + 16.0]),
                rtol=0,
                atol=0,
            )
            for k in range(3):
                self.assertTrue(16 <= int(topk_ids[3 + idx][k]) < 29)

    def test_context_parallel_publishes_original_lengths_not_local_chunk(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "cp_actual_lengths")

        # CP turns the six-token request into a four-token rank-local chunk for
        # attention, while CacheStore must still publish three two-token blocks.
        self.assertEqual(model.seen_input_lengths, [[4]])
        record = _record_for_request(result, 301)
        self.assertEqual(len(record["blocks"]), 3)
        base = result["base_addresses"]["default"]
        self.assertEqual(
            sorted(block["address"] - base for block in record["blocks"]),
            [16, 32, 48],
        )
        self.assertEqual(
            sorted(
                token_key
                for token_key in (3102, 3104, 3106)
                if any(
                    f"_token_id_str_{token_key}_" in block["key"]
                    for block in record["blocks"]
                )
            ),
            [3102, 3104, 3106],
        )

    def test_mtp_writer_uses_selected_sub_config_for_real_write(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "mtp_sub_config")

        record = _record_for_request(result, 401)
        self.assertEqual(len(record["blocks"]), 2)
        base = result["base_addresses"]["draft"]
        self.assertEqual(
            sorted(block["address"] - base for block in record["blocks"]),
            [32, 64],
        )
        self.assertEqual({block["length"] for block in record["blocks"]}, {32})
        self.assertTrue(
            all("model_id_7_" in block["key"] for block in record["blocks"])
        )
        self.assertTrue(all("_tag_draft" in block["key"] for block in record["blocks"]))


if __name__ == "__main__":
    unittest.main()
