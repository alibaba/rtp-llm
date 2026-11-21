"""Real HF Megatron-BERT vs packed NewLoader encoder (CPU attention backend)."""

import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from torch.nn import functional as F
from transformers import MegatronBertConfig, MegatronBertForSequenceClassification

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.megatron_bert.backbone import MegatronParameters
from rtp_llm.models_py.registry import register_model


@register_model("_megatron_native_independent_owner_test")
class CpuEncoder(MegatronParameters, RtpModule):
    def __init__(self, config, load):
        super().__init__()
        self.build_encoder(config, load)


class PackedSdpa:
    def __init__(self, lengths):
        self.lengths = lengths

    def forward(self, qkv, cache, i):
        def one(segment):
            q, k, v = [
                x.reshape(len(segment), 4, 4).transpose(0, 1)
                for x in segment.chunk(3, -1)
            ]
            return (
                F.scaled_dot_product_attention(q, k, v)
                .transpose(0, 1)
                .reshape(len(segment), -1)
            )

        return torch.cat([one(segment) for segment in qkv.split(self.lengths)])


def tp_fst_worker(rank, path, config, rendezvous):
    import torch.distributed as dist
    from safetensors import safe_open

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        load = NewLoaderConfig(
            device="cpu",
            load_method="fastsafetensors",
            compute_dtype=torch.float32,
            tp_size=2,
            tp_rank=rank,
        )
        loader = NewModelLoader(config, load, model_path=path)
        layer = loader.load().layers[1]
        assert loader._selected_load_method.value == "fastsafetensors"
        with safe_open(os.path.join(path, "model.safetensors"), framework="pt") as ref:
            expected = ref.get_tensor(
                "bert.encoder.layer.1.attention.self.query.weight"
            ).chunk(2, dim=0)[rank]
            torch.testing.assert_close(layer.query.weight, expected)
            expected = ref.get_tensor("bert.encoder.layer.1.output.dense.weight").chunk(
                2, dim=1
            )[rank]
            torch.testing.assert_close(layer.output.weight, expected)
    finally:
        dist.destroy_process_group()


class TestMegatron(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(316)
        cfg = MegatronBertConfig(
            hidden_size=16,
            intermediate_size=24,
            num_attention_heads=4,
            num_hidden_layers=2,
            vocab_size=31,
            max_position_embeddings=20,
            type_vocab_size=2,
            layer_norm_eps=1e-5,
            num_labels=3,
        )
        cls.ref = MegatronBertForSequenceClassification(cfg).eval()
        # Nontrivial norm affine parameters detect omitted/misplaced pre- and final norms.
        with torch.no_grad():
            for name, val in cls.ref.named_parameters():
                if ".ln." in name:
                    val.add_(torch.randn_like(val) * 0.14)
        cls.config = SimpleNamespace(
            model_type="_megatron_native_independent_owner_test",
            hidden_size=16,
            inter_size=24,
            num_layers=2,
            vocab_size=31,
            max_seq_len=20,
            type_vocab_size=2,
            layernorm_eps=1e-5,
            attn_config=SimpleNamespace(
                head_num=4, kv_head_num=4, size_per_head=4, is_causal=False
            ),
        )

    def load_case(self, load_method, values, *, tp=1, rank=0, custom=True):
        names = [
            "bert.pooler.dense.weight",
            "bert.pooler.dense.bias",
            "classifier.weight",
            "classifier.bias",
        ]
        cfg = NewLoaderConfig(
            device="cpu",
            load_method=load_method,
            compute_dtype=torch.float32,
            tp_size=tp,
            tp_rank=rank,
            custom_weight_mappings=(
                tuple(("__custom__." + n, n) for n in names) if custom else ()
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            save_file(values, os.path.join(tmp, "model.safetensors"))
            loader = NewModelLoader(self.config, cfg, model_path=tmp)
            model = loader.load()
        self.assertEqual(loader._selected_load_method.value, load_method)
        return model

    def assert_outputs(self, method):
        m = self.load_case(method, self.ref.state_dict())
        ids = torch.tensor([[3, 14, 6, 22], [19, 8, 12, 0]])
        types = torch.tensor([[0, 1, 1, 0], [1, 0, 1, 0]])
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
        active = mask.bool()
        pos = torch.arange(4).expand_as(ids)
        sdpa = PackedSdpa((4, 3))
        with torch.no_grad():
            hidden = m.encode_packed(
                ids[active],
                pos[active],
                types[active],
                1,
                lambda _: sdpa,
                lambda _: None,
            )
            expected = self.ref.bert(
                input_ids=ids, token_type_ids=types, attention_mask=mask
            )
            torch.testing.assert_close(
                hidden, expected.last_hidden_state[active], atol=1e-5, rtol=1e-5
            )
            params = {
                key.removeprefix("__custom__."): value
                for key, value in m.custom_weight_view().items()
            }
            pooled = F.linear(
                hidden[[0, 4]],
                params["bert.pooler.dense.weight"],
                params["bert.pooler.dense.bias"],
            ).tanh()
            torch.testing.assert_close(pooled, expected.pooler_output)
            logits = F.linear(
                pooled, params["classifier.weight"], params["classifier.bias"]
            )
            torch.testing.assert_close(
                logits,
                self.ref(
                    input_ids=ids, token_type_ids=types, attention_mask=mask
                ).logits,
            )

    def test_scratch_packed_encoder_and_task_head(self):
        self.assert_outputs("scratch")

    def test_fst_packed_encoder_and_task_head(self):
        self.assert_outputs("fastsafetensors")

    def test_tp2_bare_encoder_and_gamma_beta_aliases(self):
        sd = {
            k.replace(".ln.weight", ".ln.gamma").replace(".ln.bias", ".ln.beta"): v
            for k, v in self.ref.bert.state_dict().items()
        }
        for rank in (0, 1):
            model = self.load_case("scratch", sd, tp=2, rank=rank, custom=False)
            layer = model.layers[0]
            for target, src, dim in [
                (layer.query, "attention.self.query", 0),
                (layer.out, "attention.output.dense", 1),
                (layer.intermediate, "intermediate.dense", 0),
                (layer.output, "output.dense", 1),
            ]:
                base = "encoder.layer.0." + src
                torch.testing.assert_close(
                    target.weight, sd[base + ".weight"].chunk(2, dim)[rank]
                )
                expected = sd[base + ".bias"]
                torch.testing.assert_close(
                    target.bias, expected if dim == 1 else expected.chunk(2)[rank]
                )
            torch.testing.assert_close(model.final_norm.weight, sd["encoder.ln.gamma"])

    def test_reject_missing_final_norm_or_custom_weight(self):
        for missing in (
            "bert.encoder.ln.weight",
            "bert.encoder.layer.0.attention.ln.bias",
            "classifier.weight",
        ):
            sd = dict(self.ref.state_dict())
            sd.pop(missing)
            with self.subTest(missing=missing), self.assertRaisesRegex(
                (ValueError, RuntimeError), "missing|Missing|not loaded"
            ):
                self.load_case("scratch", sd)

    def test_tp2_native_fst_gloo(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_file(self.ref.state_dict(), os.path.join(tmp, "model.safetensors"))
            torch.multiprocessing.spawn(
                tp_fst_worker,
                args=(tmp, self.config, "file://" + os.path.join(tmp, "init")),
                nprocs=2,
            )


if __name__ == "__main__":
    unittest.main()
