"""Decode collective admission and PLE layout contracts (no GPU required)."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.modules.qwen4_exp.ple.ple import Qwen4ExpNGramEmbedding


class DecodeCommunicationTest(unittest.TestCase):
    def test_embedding_consumes_prepared_weights_without_preparing_again(self):
        from rtp_llm.models_py.modules.qwen4_exp.decode_communication import (
            DecodeVocabEmbedding,
        )

        weight = torch.ones(2, 4, dtype=torch.bfloat16)
        shard = SimpleNamespace(weight=weight, scales=None, vocab_size=4, start=0)
        reduce = lambda value, group: value
        before = dict(weight.__dict__)
        embedding = DecodeVocabEmbedding(shard, reduce)
        self.assertIs(embedding.shard, shard)
        self.assertIs(embedding.weight, weight)
        self.assertEqual(weight.__dict__, before)

    def test_preparation_publishes_both_owners_and_preserves_tied_head(self):
        from rtp_llm.models_py.modules.qwen4_exp import decode_vocab_weights as module
        from rtp_llm.models_py.modules.qwen4_exp.decode_communication import (
            DecodeVocabEmbedding,
        )
        from rtp_llm.models_py.modules.qwen4_exp.weight_views import PreparedWeightDict
        from rtp_llm.utils.model_weight import W

        source = torch.ones(4, 2, dtype=torch.bfloat16)
        converted = torch.ones(2, 4, dtype=torch.bfloat16)
        owners = {W.embedding: source, W.lm_head: source}
        compute = PreparedWeightDict(owners)
        with patch.object(
            module, "_repartition_vocab", return_value=SimpleNamespace(weight=converted)
        ):
            prepared = module.prepare_decode_vocab(
                source_weights=owners, compute_weights=compute, group=object()
            )
        embedding = DecodeVocabEmbedding(prepared, lambda x, group: x)
        self.assertIs(owners[W.embedding], embedding.weight)
        self.assertIs(compute[W.embedding], embedding.weight)
        self.assertIs(owners[W.lm_head], source)
        self.assertTrue(owners[W.embedding]._rtp_requires_restart_for_update)

    def test_draft_loader_preserves_logical_vocabulary_size(self):
        from rtp_llm.model_loader.loader import ModelLoader
        from rtp_llm.model_loader.model_weight_info import ModelWeights
        from rtp_llm.ops import TaskType
        from rtp_llm.utils.model_weight import W

        loader = ModelLoader.__new__(ModelLoader)
        loader._task_type = TaskType.LANGUAGE_MODEL
        loader._load_config = SimpleNamespace(num_layers=1, layer_range=[0])
        config = SimpleNamespace(
            embedding_size=0,
            vocab_size=129,
            normalize_lm_head_weight=False,
            logit_scale=1.0,
        )
        loader._weights_info = SimpleNamespace(
            model_config=config, create_dynamic_weights=lambda: []
        )
        weights = ModelWeights(0, "cpu", torch.bfloat16)
        shard = torch.zeros(65, 4, dtype=torch.bfloat16)
        shard._rtp_embedding_vocab_size = 129
        head = torch.ones(65, 4, dtype=torch.bfloat16)
        weights.global_weights.update({W.embedding: shard, W.lm_head: head})
        loader._load_dynamic_weights(weights, "cpu")
        self.assertEqual(config.embedding_size, 129)
        self.assertIs(weights.global_weights[W.lm_head], head)
        weights.global_weights[W.embedding] = torch.zeros(129, 2, dtype=torch.bfloat16)
        loader._load_dynamic_weights(weights, "cpu")
        self.assertEqual(config.embedding_size, 129)

    def test_global_update_rejects_restart_only_source_without_mutation(self):
        from rtp_llm.model_loader.model_weight_info import ModelWeights

        weights = ModelWeights(0, "cpu", torch.bfloat16)
        source = torch.ones(3, 4, dtype=torch.bfloat16)
        weights.set_global_weight("embedding", source)
        source._rtp_requires_restart_for_update = True
        with self.assertRaisesRegex(PermissionError, "restart"):
            weights.update_global_weight("embedding", torch.zeros_like(source))
        torch.testing.assert_close(source, torch.ones_like(source))
        source._rtp_requires_restart_for_update = False
        weights.update_global_weight("embedding", torch.zeros_like(source))
        torch.testing.assert_close(source, torch.zeros_like(source))

    def test_weight_manager_rejects_global_update_before_ipc_preparation(self):
        from rtp_llm.model_loader import weight_manager as module
        from rtp_llm.model_loader.model_weight_info import ModelWeights

        weights = ModelWeights(0, "cpu", torch.bfloat16)
        source = torch.ones(3, 4, dtype=torch.bfloat16)
        source._rtp_requires_restart_for_update = True
        weights.set_global_weight("embedding", source)
        manager = module.WeightManager.__new__(module.WeightManager)
        manager._weights = weights
        with patch.object(module.SharedMemIpcMeta, "decode") as decode:
            with self.assertRaisesRegex(PermissionError, "restart"):
                manager.update(
                    {
                        "name": "model.embed_tokens.weight",
                        "method": "shm",
                        "desc": "unused",
                    }
                )
            decode.assert_not_called()
        torch.testing.assert_close(source, torch.ones_like(source))

    def test_preparation_marks_original_global_scale_owner(self):
        from rtp_llm.models_py.modules.qwen4_exp import decode_vocab_weights as module
        from rtp_llm.utils.fp8_constants import FP8_STORAGE_SCALE_SUFFIX
        from rtp_llm.utils.model_weight import W

        weight = torch.ones(3, 4, dtype=torch.bfloat16)
        scale_owner = torch.tensor(0.5)
        source_weights = {
            W.embedding: weight,
            W.embedding + FP8_STORAGE_SCALE_SUFFIX: scale_owner,
        }
        with patch.object(
            module, "_repartition_vocab", return_value=SimpleNamespace(weight=weight)
        ):
            module.prepare_decode_vocab(
                source_weights=source_weights,
                compute_weights=dict(source_weights),
                group=object(),
            )
        self.assertTrue(weight._rtp_requires_restart_for_update)
        self.assertTrue(scale_owner._rtp_requires_restart_for_update)

    def test_draft_capture_registers_inputs_before_the_target_only_return(self):
        from rtp_llm.models_py.model_desc import qwen4_exp as model

        reduce = MagicMock()
        owner = SimpleNamespace(_decode_reduce=reduce)
        with patch.object(
            model,
            "get_primary_attention_inputs",
            return_value=SimpleNamespace(is_target_verify=False),
        ):
            model.Qwen4ExpModel.on_cuda_graph_capture(owner, 8, object())
        reduce.finish_graph_capture.assert_called_once_with()

    def test_ple_reduction_uses_packed_rows_and_restores_leading_dimensions(self):
        embedding = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
        torch.nn.Module.__init__(embedding)
        embedding.distributed_reduce = True
        embedding.tp_all_reduce = MagicMock(side_effect=lambda value, group: value + 1)
        value = torch.arange(3 * 4 * 16 * 160).reshape(3, 4, 16, 160).float()
        with patch.object(
            collective_torch, "all_reduce", side_effect=lambda x, **kw: x
        ):
            actual = embedding._reduce_rows(value)
        self.assertEqual(tuple(actual.shape), (3, 4, 2560))
        torch.testing.assert_close(actual, (value + 1).flatten(-2))
        self.assertEqual(embedding.tp_all_reduce.call_args.args[0].shape, (12, 2560))

    def test_all_integer_rows_admitted_without_widening_generic_dispatch(self):
        from rtp_llm.models_py.modules.qwen4_exp.decode_communication import (
            Qwen4DecodeReduce,
        )

        reduce = Qwen4DecodeReduce.__new__(Qwen4DecodeReduce)
        reduce.closed = False
        reduce.workspace = SimpleNamespace(all_reduce=MagicMock())
        for rows in range(1, 65):
            value = SimpleNamespace(
                is_cuda=True,
                dtype=torch.bfloat16,
                ndim=2,
                shape=(rows, 2560),
                is_contiguous=lambda: True,
                data_ptr=lambda: 0x1000,
            )
            reduce(value, collective_torch.Group.TP)
        self.assertEqual(reduce.workspace.all_reduce.call_count, 64)
        for rows, width, dtype, cuda, contiguous in (
            (0, 2560, torch.bfloat16, True, True),
            (65, 2560, torch.bfloat16, True, True),
            (8, 1280, torch.bfloat16, True, True),
            (8, 2560, torch.float32, True, True),
            (8, 2560, torch.bfloat16, False, True),
            (8, 2560, torch.bfloat16, True, False),
        ):
            value = SimpleNamespace(
                is_cuda=cuda,
                dtype=dtype,
                ndim=2,
                shape=(rows, width),
                is_contiguous=lambda: contiguous,
                data_ptr=lambda: 0x1000,
            )
            with patch.object(
                collective_torch, "all_reduce", return_value=value
            ) as fallback:
                self.assertIs(reduce(value, collective_torch.Group.TP), value)
                fallback.assert_called_once()

    def test_disabled_or_peer_ineligible_does_not_create_workspace(self):
        from rtp_llm.models_py.modules.qwen4_exp import decode_communication as module

        config = SimpleNamespace(hidden_size=2560)
        parallelism = SimpleNamespace(
            role_type="DECODE", tp_size=2, dp_size=1, local_world_size=2
        )
        communicator = SimpleNamespace(
            group=object(), disabled=False, cpp_small_enabled=True
        )
        for peers in ([("0", True), ("0", True)], [("1", True), ("1", False)]):
            with (
                patch.object(module, "get_all_reduce", return_value=communicator),
                patch.dict("os.environ", {"RTP_QWEN4_DECODE_CUSTOM_AR": peers[0][0]}),
                patch.object(
                    module.dist,
                    "all_gather_object",
                    side_effect=lambda out, value, group: out.__setitem__(
                        slice(None), peers
                    ),
                ),
                patch.object(module, "Qwen4DecodeReduce") as ctor,
            ):
                self.assertIsNone(module.prepare_decode_reduce(config, parallelism))
                ctor.assert_not_called()

    def test_peer_mode_mismatch_fails_before_collective_initialization(self):
        from rtp_llm.models_py.modules.qwen4_exp import decode_communication as module

        config = SimpleNamespace(hidden_size=2560)
        parallelism = SimpleNamespace(
            role_type="DECODE", tp_size=2, dp_size=1, local_world_size=2
        )
        communicator = SimpleNamespace(
            group=object(), disabled=False, cpp_small_enabled=True
        )
        with (
            patch.object(module, "get_all_reduce", return_value=communicator),
            patch.dict("os.environ", {"RTP_QWEN4_DECODE_CUSTOM_AR": "1"}),
            patch.object(
                module.dist,
                "all_gather_object",
                side_effect=lambda out, value, group: out.__setitem__(
                    slice(None), [("1", True), ("0", True)]
                ),
            ),
            patch.object(module, "Qwen4DecodeReduce") as ctor,
        ):
            with self.assertRaisesRegex(ValueError, "every TP rank"):
                module.prepare_decode_reduce(config, parallelism)
            ctor.assert_not_called()

    def test_prefill_role_never_initializes_decode_workspace(self):
        from rtp_llm.models_py.modules.qwen4_exp import decode_communication as module

        for role in ("PREFILL", "PDFUSION"):
            with patch.object(module.dist, "all_gather_object") as exchange:
                self.assertIsNone(
                    module.prepare_decode_reduce(
                        SimpleNamespace(hidden_size=2560),
                        SimpleNamespace(
                            role_type=role, tp_size=2, dp_size=1, local_world_size=2
                        ),
                    )
                )
                exchange.assert_not_called()


if __name__ == "__main__":
    unittest.main()
