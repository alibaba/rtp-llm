import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory import ModelFactory
from rtp_llm.model_factory_register import ModelDict, ensure_model_registered
from rtp_llm.models.qwen_3_dspark import (
    Qwen3DSpark,
    dspark_offset_d2t_to_absolute,
)
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen3_dspark_model import Qwen3DSparkModel
from rtp_llm.models_py.speculative.dspark_proposer_mixin import (
    graph_captured_greedy_markov_decode,
)
class Qwen3DSparkRegistrationTest(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available(),
        "DSpARK CUDA graph test requires CUDA",
    )
    def test_generic_markov_supports_batch_128_and_graph(self) -> None:
        # Exercise the production vocabulary and graph buckets without any
        # SM-specific kernel. One independent rank component per request
        # defines an exact seven-token chain and catches cross-row state leaks.
        vocab_size = 20_000
        target_state_base = vocab_size
        target_vocab_size = target_state_base + 128
        device = torch.device("cuda")

        d2t = torch.arange(vocab_size, dtype=torch.int64, device=device)
        d2t[:128].add_(target_state_base)
        markov_w1 = torch.zeros(
            (target_vocab_size, 256), dtype=torch.bfloat16, device=device
        )
        markov_w2 = torch.zeros(
            (vocab_size, 256), dtype=torch.bfloat16, device=device
        )
        rows = torch.arange(128, device=device)
        markov_w1[rows, rows] = 1
        markov_w1[target_state_base + rows, rows] = 1
        markov_w2[rows, rows] = 8

        for batch_size in (1, 2, 4, 8, 16, 24, 32, 64, 128):
            anchor = torch.arange(batch_size, dtype=torch.int32, device=device)
            base_logits = torch.zeros(
                (batch_size, 7, vocab_size),
                dtype=torch.bfloat16,
                device=device,
            )
            expected = (
                torch.arange(batch_size, dtype=torch.int32, device=device)
                .add(target_state_base)
                .view(batch_size, 1)
                .expand(batch_size, 7)
            )
            eager_output = graph_captured_greedy_markov_decode(
                base_logits, anchor, markov_w1, markov_w2, d2t
            )
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(eager_output, expected))

            base_logits.zero_()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = graph_captured_greedy_markov_decode(
                    base_logits, anchor, markov_w1, markov_w2, d2t
                )
            graph_output.fill_(-1)
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(graph_output, expected))

    def test_exact_markov_decode_searches_full_draft_vocab(self) -> None:
        # Base Top-2 contains ids 0 and 1, but the Markov correction promotes
        # id 3. This guards against reintroducing an approximate TopK shortcut.
        base_logits = torch.tensor(
            [[[10.0, 9.0, 0.0, -1.0], [7.0, 6.0, 5.0, 4.0]]]
        )
        markov_w1 = torch.zeros(10, 1)
        markov_w1[8, 0] = 1.0
        markov_w1[9, 0] = -1.0
        markov_w2 = torch.tensor([[0.0], [0.0], [0.0], [20.0]])
        d2t = torch.tensor([4, 5, 8, 9], dtype=torch.int64)

        proposals = graph_captured_greedy_markov_decode(
            base_logits, torch.tensor([8]), markov_w1, markov_w2, d2t
        )

        # Step 0 selects draft id 3 -> target id 9. The target id feeds W1 for
        # step 1, whose negative correction then restores draft id 0 -> id 4.
        self.assertEqual(proposals.dtype, torch.int32)
        self.assertEqual(proposals.tolist(), [[9, 4]])

    def test_speculators_d2t_offsets_are_normalized_to_absolute_ids(self) -> None:
        offsets = torch.tensor([1, 1, 2, -1], dtype=torch.int64)
        absolute = dspark_offset_d2t_to_absolute([offsets])
        self.assertEqual(absolute.tolist(), [1, 2, 4, 2])

    def test_reduced_vocab_mask_is_validated_against_input_embedding(self) -> None:
        target = ModelConfig()
        target.num_layers = 64
        draft = ModelConfig()
        draft.input_vocab_size = 248320
        draft.vocab_size = 20000
        draft.dspark_noise_token_id = 248127
        draft.dspark_target_layer_ids = [7, 20, 33, 46, 59]
        draft.dspark_markov_rank = 256
        draft.dspark_sample_from_anchor = False
        spec = SimpleNamespace(gen_num_per_cycle=7)

        ModelFactory._setup_dspark_configs(spec, target, draft)

        self.assertEqual(spec.sp_dspark_mask_token_id, 248127)
        self.assertEqual(target.capture_aux_hidden_layer_ids, [7, 20, 33, 46, 59])
        self.assertIsNone(draft.capture_aux_hidden_layer_ids)
        self.assertEqual(draft.dspark_target_layer_ids, [7, 20, 33, 46, 59])
        self.assertFalse(draft.enable_fp32_lm_head)

    def test_target_capture_layers_must_be_unique_and_ordered(self) -> None:
        target = ModelConfig()
        target.num_layers = 64
        draft = ModelConfig()
        draft.input_vocab_size = 248320
        draft.vocab_size = 20000
        draft.dspark_noise_token_id = 248127
        draft.dspark_target_layer_ids = [20, 7, 20]
        draft.dspark_markov_rank = 256
        spec = SimpleNamespace(gen_num_per_cycle=7)

        with self.assertRaisesRegex(ValueError, "unique and ordered"):
            ModelFactory._setup_dspark_configs(spec, target, draft)

    def test_aux_capture_materializes_fused_residual_boundary(self) -> None:
        capture = SimpleNamespace(
            config=SimpleNamespace(capture_aux_hidden_layer_ids=[1]),
            _mtp_aux_capture_layer_ids=(1,),
            _mtp_aux_capture_layer_id_set=frozenset((1,)),
            _mtp_target_hidden_states=None,
            _mtp_target_graph_buffer=None,
            _mtp_target_prompt_buffer=None,
            _mtp_aux_capture_buffer=None,
            _mtp_aux_capture_rows=0,
            _mtp_aux_capture_index=0,
        )
        hidden = torch.tensor([[1.0, 2.0]])
        residual = torch.tensor([[10.0, 20.0]])

        GptModelBase.begin_aux_hidden_capture(capture, hidden, False)
        GptModelBase.capture_aux_hidden(capture, 0, hidden, residual)
        GptModelBase.capture_aux_hidden(capture, 1, hidden, residual)
        GptModelBase.finish_aux_hidden_capture(capture)

        torch.testing.assert_close(
            capture._mtp_target_hidden_states, torch.tensor([[11.0, 22.0]])
        )
        self.assertEqual(
            capture._mtp_target_hidden_states.data_ptr(),
            capture._mtp_target_prompt_buffer.data_ptr(),
        )

    def test_target_verify_aux_capture_reuses_contiguous_graph_buffer(self) -> None:
        capture = SimpleNamespace(
            config=SimpleNamespace(capture_aux_hidden_layer_ids=[0, 1]),
            _mtp_aux_capture_layer_ids=(0, 1),
            _mtp_aux_capture_layer_id_set=frozenset((0, 1)),
            _mtp_target_hidden_states=None,
            _mtp_target_graph_buffer=None,
            _mtp_target_prompt_buffer=None,
            _mtp_aux_capture_buffer=None,
            _mtp_aux_capture_rows=0,
            _mtp_aux_capture_index=0,
        )
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        residual = torch.tensor([[10.0, 20.0], [30.0, 40.0]])

        GptModelBase.begin_aux_hidden_capture(capture, hidden, True)
        GptModelBase.capture_aux_hidden(capture, 0, hidden)
        GptModelBase.capture_aux_hidden(capture, 1, hidden, residual)
        GptModelBase.finish_aux_hidden_capture(capture)

        first_ptr = capture._mtp_target_graph_buffer.data_ptr()
        torch.testing.assert_close(
            capture._mtp_target_hidden_states,
            torch.tensor([[1.0, 2.0, 11.0, 22.0], [3.0, 4.0, 33.0, 44.0]]),
        )

        # Smaller graph keys reuse the max-batch warmup allocation and retain
        # its address, which is required by CUDA graph replay.
        GptModelBase.begin_aux_hidden_capture(capture, hidden[:1], True)
        GptModelBase.capture_aux_hidden(capture, 0, hidden[:1])
        GptModelBase.capture_aux_hidden(capture, 1, hidden[:1], residual[:1])
        GptModelBase.finish_aux_hidden_capture(capture)
        self.assertEqual(capture._mtp_target_graph_buffer.data_ptr(), first_ptr)
        self.assertEqual(tuple(capture._mtp_target_hidden_states.shape), (1, 4))

        # A long prompt must not replace the address retained by the captured
        # target-verify graphs.
        prompt_hidden = hidden.repeat(4, 1)
        GptModelBase.begin_aux_hidden_capture(capture, prompt_hidden, False)
        GptModelBase.capture_aux_hidden(capture, 0, prompt_hidden)
        GptModelBase.capture_aux_hidden(capture, 1, prompt_hidden)
        GptModelBase.finish_aux_hidden_capture(capture)
        self.assertEqual(capture._mtp_target_graph_buffer.data_ptr(), first_ptr)
        self.assertNotEqual(capture._mtp_target_prompt_buffer.data_ptr(), first_ptr)
        self.assertEqual(tuple(capture._mtp_target_hidden_states.shape), (8, 4))

    def test_commit_uses_kernel_block_ids_not_physical_cache_ids(self) -> None:
        kernel_ids = torch.tensor([[3, 4]], dtype=torch.int32)
        physical_ids = torch.tensor([[3003, 3004]], dtype=torch.int32)
        attention = SimpleNamespace(
            kv_cache_kernel_block_id_device=kernel_ids,
            kv_cache_kernel_block_id=torch.tensor([[5, 6]], dtype=torch.int32),
            kv_cache_block_id_device=physical_ids,
            kv_cache_block_id=torch.tensor([[5005, 5006]], dtype=torch.int32),
        )
        model = SimpleNamespace(
            embed_tokens=SimpleNamespace(weight=torch.empty(0)), kv_cache=None
        )
        model.dspark_attention_inputs = lambda inputs: inputs.attention_inputs

        selected = Qwen3DSparkModel._block_table(
            model, SimpleNamespace(attention_inputs=attention)
        )

        self.assertIs(selected, kernel_ids)

    def test_hybrid_tagged_inputs_select_draft_layer_group(self) -> None:
        full = SimpleNamespace()
        linear = SimpleNamespace()
        layer_cache = SimpleNamespace(tag="full")
        model = SimpleNamespace(
            kv_cache=SimpleNamespace(
                get_layer_cache_groups=lambda layer_idx: [layer_cache]
            )
        )
        inputs = SimpleNamespace(
            attention_inputs={"linear0": linear, "full": full}
        )

        selected = Qwen3DSparkModel.dspark_attention_inputs(model, inputs)

        self.assertIs(selected, full)

    def test_prompt_commit_publishes_canonical_single_cache_group(self) -> None:
        cache_store_inputs = SimpleNamespace()
        cache_store_writer = Mock()
        attention = SimpleNamespace(
            is_prefill=True,
            cache_store_inputs=cache_store_inputs,
            cache_store_writer=cache_store_writer,
        )
        layer_cache = SimpleNamespace(
            kv_cache_base=torch.zeros((1, 2, 1, 4, 1))
        )

        class CanonicalKVCache:
            def get_layer_cache(self, layer_idx: int):
                self.assert_layer(layer_idx)
                return layer_cache

            def get_layer_cache_groups(self, layer_idx: int):
                self.assert_layer(layer_idx)
                return [layer_cache]

            @staticmethod
            def assert_layer(layer_idx: int) -> None:
                if layer_idx != 0:
                    raise AssertionError(f"unexpected layer {layer_idx}")

        model = SimpleNamespace(
            layer_num=1,
            attn_configs=SimpleNamespace(
                kernel_tokens_per_block=4,
                size_per_head=1,
                kv_head_num=1,
            ),
            hidden_norm=lambda hidden: hidden,
            context_kv_projection=lambda hidden: torch.tensor(
                [[1.0, 2.0]], dtype=hidden.dtype
            ),
            context_k_norms=[lambda key: key],
            context_rope=SimpleNamespace(_apply_rope=lambda query, key, pos: None),
            kv_cache=CanonicalKVCache(),
            dspark_attention_inputs=lambda inputs: attention,
            _block_table=lambda inputs: torch.tensor([[0]], dtype=torch.int32),
        )

        Qwen3DSparkModel.commit_feature_rows(
            model,
            main_x=torch.ones((1, 1)),
            context_req_ids=torch.tensor([0]),
            context_positions=torch.tensor([0]),
            committed_ends=torch.tensor([1]),
            inputs=SimpleNamespace(),
        )

        cache_store_writer.write.assert_called_once_with(
            cache_store_inputs, layer_cache
        )

    def test_non_driver_tp_rank_skips_replicated_proposal_head(self) -> None:
        model = SimpleNamespace(
            _dspark_proposal_driver=False,
            _dspark_width=7,
            config=SimpleNamespace(dspark_sample_from_anchor=False),
            compute_draft_hidden_states=lambda hidden: hidden,
        )
        hidden = torch.arange(8 * 4, dtype=torch.bfloat16).reshape(8, 4)
        query_ids = torch.tensor(
            [[101, 201, 202, 203, 204, 205, 206, 207]], dtype=torch.int32
        )

        outputs = Qwen3DSparkModel.build_proposal_outputs(
            model, hidden, query_ids
        )

        self.assertEqual(outputs.hidden_states.data_ptr(), hidden.data_ptr())
        self.assertEqual(
            outputs.speculative_token_ids.tolist(),
            [[201, 202, 203, 204, 205, 206, 207]],
        )

    def test_config_wires_shared_dspark_contract(self) -> None:
        raw = {
            "architectures": ["Qwen3DSparkForCausalLM"],
            "transformer_layer_config": {
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_hidden_layers": 2,
                "vocab_size": 1024,
            },
            "draft_vocab_size": 1001,
            "aux_hidden_state_layer_ids": [0, 1],
            "mask_token_id": 1000,
            "block_size": 8,
            "markov_rank": 32,
        }
        with tempfile.TemporaryDirectory() as path:
            Path(path, "config.json").write_text(json.dumps(raw))
            config = Qwen3DSpark._create_config(path)

        self.assertFalse(config.attn_config.is_causal)
        self.assertTrue(config.dspark_sample_from_anchor)
        self.assertEqual(config.dspark_noise_token_id, 1000)
        self.assertEqual(config.dspark_target_layer_ids, [0, 1])
        self.assertEqual(config.dspark_markov_rank, 32)
        self.assertEqual(config.input_vocab_size, 1024)
        self.assertEqual(config.vocab_size, 1001)
        self.assertTrue(config.qk_norm)
        self.assertTrue(ensure_model_registered("qwen_3_dspark"))
        self.assertEqual(
            ModelDict.get_ft_model_type_by_config(raw), "qwen_3_dspark"
        )

    def test_speculators_checkpoint_uses_bonus_anchor_layout(self) -> None:
        raw = {
            "architectures": ["DSparkDraftModel"],
            "speculators_model_type": "dspark",
            "transformer_layer_config": {
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_hidden_layers": 2,
                "vocab_size": 1024,
            },
            "draft_vocab_size": 1001,
            "aux_hidden_state_layer_ids": [1, 2],
            "mask_token_id": 1000,
            "markov_rank": 32,
        }
        with tempfile.TemporaryDirectory() as path:
            Path(path, "config.json").write_text(json.dumps(raw))
            config = Qwen3DSpark._create_config(path)

        self.assertFalse(config.dspark_sample_from_anchor)
        self.assertEqual(config.dspark_target_layer_ids, [0, 1])


if __name__ == "__main__":
    unittest.main()
