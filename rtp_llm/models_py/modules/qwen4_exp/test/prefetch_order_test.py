import unittest
from collections.abc import Mapping
from types import MappingProxyType
from unittest.mock import patch

import torch

from rtp_llm.models_py.model_desc import qwen4_exp
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs


class DerivedModel(qwen4_exp.Qwen4ExpModel):
    pass


class StopAfterPreparation(Exception):
    pass


class PrefetchOrderTest(unittest.TestCase):
    @staticmethod
    def attention(*, prefill, graph):
        value = PyAttentionInputs()
        value.is_prefill = prefill
        value.is_cuda_graph = graph
        return value

    def check_order(
        self,
        attention,
        *,
        derived=False,
        packed=False,
        has_ple=True,
        delayed=False
    ):
        cls = DerivedModel if derived else qwen4_exp.Qwen4ExpModel
        model = cls.__new__(cls)
        torch.nn.Module.__init__(model)
        model.layers = torch.nn.ModuleList()
        model.ple_layers = torch.nn.ModuleDict(
            {"0": torch.nn.Identity()} if has_ple else {}
        )
        model.kv_cache = None
        order = []
        hidden = torch.zeros(1, 4)

        def embedding(_):
            order.append("embedding")
            return hidden

        def packed_embedding(_):
            return embedding(_) if packed else None

        def prefetch(_, actual_attention):
            # PyModelInputs converts native value structs and copies the
            # outer mapping; object identity is not its public contract.
            if isinstance(attention, Mapping):
                self.assertEqual(set(actual_attention), set(attention))
                pairs = [
                    (actual_attention[tag], value) for tag, value in attention.items()
                ]
            else:
                pairs = [(actual_attention, attention)]
            for actual, expected in pairs:
                self.assertIsInstance(actual, PyAttentionInputs)
                self.assertEqual(actual.is_prefill, expected.is_prefill)
                self.assertEqual(actual.is_cuda_graph, expected.is_cuda_graph)
            order.append("prefetch")
            return {}

        def stop(*_):
            raise StopAfterPreparation()

        model.word_embedding = embedding
        model._try_fp8_hyper_embedding = packed_embedding
        model._start_eager_ple_prefetch = prefetch
        model._build_attn_meta = stop
        inputs = PyModelInputs()
        inputs.attention_inputs = attention
        # Keep the native input adapter and tag selection real. Only the
        # downstream CUDA work is replaced, to observe forward call ordering.
        with patch.object(
            qwen4_exp, "_is_cuda_graph_forward", return_value=False
        ), self.assertRaises(
            StopAfterPreparation
        ):
            model._forward_with_ple_prefetch(inputs, object())
        self.assertEqual(
            order, ["embedding", "prefetch"] if delayed else ["prefetch", "embedding"]
        )

    def test_prefill_embedding_precedes_uva_only_in_the_explicit_target_domain(self):
        for prefill, graph, derived, packed in (
            (True, False, False, False),
            (True, False, False, True),
            (False, False, False, False),
            (True, True, False, False),
            (True, False, True, False),
        ):
            with self.subTest(
                prefill=prefill,
                graph=graph,
                derived=derived,
                packed=packed,
            ):
                self.check_order(
                    self.attention(prefill=prefill, graph=graph),
                    derived=derived,
                    packed=packed,
                    delayed=prefill and not graph and not derived,
                )

    def test_tagged_inputs_follow_the_ple_context_mode(self):
        for prefill, graph in ((True, False), (False, False), (True, True)):
            for packed in (False, True):
                for readonly_mapping in (False, True):
                    with self.subTest(
                        prefill=prefill,
                        graph=graph,
                        packed=packed,
                        readonly_mapping=readonly_mapping,
                    ):
                        groups = {
                            "full": self.attention(
                                prefill=not prefill, graph=not graph
                            ),
                            qwen4_exp.PLE_NGRAM_CTX_TAG: self.attention(
                                prefill=prefill, graph=graph
                            ),
                        }
                        if readonly_mapping:
                            groups = MappingProxyType(groups)
                        self.check_order(
                            groups, packed=packed, delayed=prefill and not graph
                        )

    def test_unrelated_models_do_not_require_a_ple_context_tag(self):
        for derived, has_ple in (
            (True, True),
            (False, False),
        ):
            with self.subTest(derived=derived, has_ple=has_ple):
                groups = {"full": self.attention(prefill=True, graph=False)}
                self.check_order(groups, derived=derived, has_ple=has_ple)


if __name__ == "__main__":
    unittest.main()
