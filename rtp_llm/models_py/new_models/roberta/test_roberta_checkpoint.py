"""Real RoBERTa checkpoints and CPU numerical reference; not platform golden."""

import json
import os
import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaModel

from rtp_llm.models_py.checkpoint_reader_test_support import checkpoint_readers
from rtp_llm.models_py.downstream_heads import handler_weight_mappings
from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.new_models.roberta.language import RobertaEncoderOwner
from rtp_llm.models_py.weight_mapper import discover_ckpt_files

HEAD_NAMES = (
    "classifier.dense.weight",
    "classifier.dense.bias",
    "classifier.out_proj.weight",
    "classifier.out_proj.bias",
)


def _hf_config():
    return RobertaConfig(
        vocab_size=23,
        hidden_size=8,
        intermediate_size=12,
        num_attention_heads=2,
        num_hidden_layers=2,
        max_position_embeddings=10,
        type_vocab_size=1,
        pad_token_id=1,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        layer_norm_eps=1e-5,
        num_labels=3,
    )


def _runtime_config(hf, directory):
    return SimpleNamespace(
        model_type="roberta",
        ckpt_path=directory,
        hidden_size=hf.hidden_size,
        inter_size=hf.intermediate_size,
        intermediate_size=hf.intermediate_size,
        num_layers=hf.num_hidden_layers,
        num_hidden_layers=hf.num_hidden_layers,
        vocab_size=hf.vocab_size,
        max_seq_len=hf.max_position_embeddings,
        max_position_embeddings=hf.max_position_embeddings,
        type_vocab_size=hf.type_vocab_size,
        layernorm_eps=hf.layer_norm_eps,
        layer_norm_eps=hf.layer_norm_eps,
        activation_type="gelu",
        layernorm_type="post_layernorm",
        norm_type="layernorm",
        has_pre_decoder_layernorm=True,
        has_positional_encoding=True,
        position_ids_style=1,
        special_tokens=SimpleNamespace(pad_token_id=hf.pad_token_id),
        pad_token_id=hf.pad_token_id,
        num_labels=hf.num_labels,
        attn_config=SimpleNamespace(
            head_num=hf.num_attention_heads,
            kv_head_num=hf.num_attention_heads,
            size_per_head=hf.hidden_size // hf.num_attention_heads,
            is_causal=False,
            need_rope_kv_cache=False,
        ),
        capture_aux_hidden_layer_ids=[],
        hidden_state_capture_layer_ids=[],
        is_mtp=False,
    )


def _load(directory, config, method, tp=1, rank=0, heads=False):
    parallel = SimpleNamespace(
        tp_size=tp,
        tp_rank=rank,
        dp_size=1,
        dp_rank=0,
        ep_size=1,
        ep_rank=0,
        pp_size=1,
        pp_rank=0,
        pp_stage_layer_counts=[],
        world_size=tp,
        world_rank=rank,
        local_world_size=tp,
        get_attn_tp_size=lambda: tp,
        get_attn_tp_rank=lambda: rank,
        get_ffn_tp_size=lambda: tp,
        get_ffn_tp_rank=lambda: rank,
    )
    load = NewLoaderConfig(
        device="cpu",
        compute_dtype=torch.float32,
        load_method=method,
        tp_size=tp,
        tp_rank=rank,
        parallelism_config=parallel,
        custom_weight_mappings=handler_weight_mappings(HEAD_NAMES) if heads else (),
    )
    loader = NewModelLoader(config, load, model_path=directory)
    model = loader.load()
    expected = (
        NewLoaderLoadMethod.SCRATCH
        if method == "scratch"
        else NewLoaderLoadMethod.FASTSAFETENSORS
    )
    if loader._selected_load_method != expected:
        raise AssertionError(
            f"Requested {method}, selected {loader._selected_load_method}"
        )
    return model


def _checkpoint(reference, directory, prefix="", legacy_norm=False):
    values = {}
    for name, value in reference.state_dict().items():
        if prefix and not name.startswith(("roberta.", "classifier.")):
            name = prefix + name
        if legacy_norm:
            name = name.replace(".LayerNorm.weight", ".LayerNorm.gamma")
            name = name.replace(".LayerNorm.bias", ".LayerNorm.beta")
        values[name] = value.detach().clone().contiguous()
    save_file(values, os.path.join(directory, "model.safetensors"))
    reference.config.to_json_file(os.path.join(directory, "config.json"))
    return values


def _packed_inputs(model, sequences, pad_id=1):
    # Exactly EmbeddingExecutor's packed position contract: every sequence
    # restarts at pad_token_id + 1. Padding is removed before the engine input.
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.int32)
    view = model.runtime_weight_view()
    return SimpleNamespace(
        input_ids=torch.tensor(
            [token for seq in sequences for token in seq], dtype=torch.int32
        ),
        attention_inputs=SimpleNamespace(input_lengths=lengths),
        bert_embedding_inputs=SimpleNamespace(
            combo_position_ids=torch.cat(
                [
                    torch.arange(len(seq), dtype=torch.int32) + pad_id + 1
                    for seq in sequences
                ]
            ),
            combo_tokens_type_ids=torch.zeros(
                sum(map(len, sequences)), dtype=torch.int32
            ),
            position_encoding=view["position_encoding.weight"],
            token_type_embedding=view["token_type_embedding.weight"],
            input_embedding_scalar=1.0,
        ),
    )


def _hf_inputs(sequences, pad_id=1):
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for index, seq in enumerate(sequences):
        ids[index, : len(seq)] = torch.tensor(seq)
        mask[index, : len(seq)] = 1
    return ids, mask


def _tp_checkpoint_worker(rank, directory, rendezvous, method):
    """Real CPU WORLD for TP checkpoint distribution, not GPU TP execution."""
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        hf = RobertaConfig.from_pretrained(directory)
        source = load_file(os.path.join(directory, "model.safetensors"))
        owner = _load(
            directory,
            _runtime_config(hf, directory),
            method,
            tp=2,
            rank=rank,
            heads=True,
        )
        view = owner.runtime_weight_view()
        for role, name in (
            ("embedding", "roberta.embeddings.word_embeddings.weight"),
            (
                "position_encoding.weight",
                "roberta.embeddings.position_embeddings.weight",
            ),
            (
                "token_type_embedding.weight",
                "roberta.embeddings.token_type_embeddings.weight",
            ),
        ):
            torch.testing.assert_close(view[role], source[name].chunk(2, dim=1)[rank])
        for name in HEAD_NAMES:
            torch.testing.assert_close(view["__custom__." + name], source[name])
        local = view["embedding"].contiguous()
        gathered = [torch.empty_like(local) for _ in range(2)]
        torch.distributed.all_gather(gathered, local)
        torch.testing.assert_close(
            torch.cat(gathered, dim=1),
            source["roberta.embeddings.word_embeddings.weight"],
        )
    finally:
        torch.distributed.destroy_process_group()


class RobertaNativeOutputTest(unittest.TestCase):
    @torch.inference_mode()
    def test_public_roberta_service_and_actual_handlers(self):
        from tokenizers import ByteLevelBPETokenizer

        from rtp_llm.config.py_config_modules import GenerateEnvConfig, RenderConfig
        from rtp_llm.models.bert import Roberta
        from rtp_llm.ops import TaskType
        from rtp_llm.utils.model_weight import W

        class CpuRoberta(Roberta):
            def _get_device_str(self):
                return "cpu"

        for task in (
            TaskType.DENSE_EMBEDDING,
            TaskType.SEQ_CLASSIFICATION,
            TaskType.RERANKER,
        ):
            with tempfile.TemporaryDirectory() as directory:
                tokenizer = ByteLevelBPETokenizer()
                tokenizer.train_from_iterator(
                    ["hello world", "tiny real roberta checkpoint"],
                    vocab_size=280,
                    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
                )
                tokenizer.save_model(directory)
                tokenizer.save(os.path.join(directory, "tokenizer.json"))
                with open(
                    os.path.join(directory, "tokenizer_config.json"), "w"
                ) as stream:
                    json.dump(
                        {"tokenizer_class": "RobertaTokenizer", "model_max_length": 10},
                        stream,
                    )
                hf = _hf_config()
                hf.vocab_size = tokenizer.get_vocab_size()
                torch.manual_seed(96)
                classifier = task != TaskType.DENSE_EMBEDDING
                reference = (
                    RobertaForSequenceClassification(hf)
                    if classifier
                    else RobertaModel(hf)
                ).eval()
                _checkpoint(reference, directory)
                sequences = [[0, 5, 6, 2], [0, 7, 2]]
                ids, mask = _hf_inputs(sequences)
                base = reference.roberta if classifier else reference
                batch_hidden = base(ids, attention_mask=mask).last_hidden_state
                expected_hidden = torch.cat(
                    [batch_hidden[i, : len(seq)] for i, seq in enumerate(sequences)]
                )
                expected = (
                    reference(ids, attention_mask=mask).logits
                    if classifier
                    else F.normalize(batch_hidden[:, 0], dim=-1)
                )
                for method in checkpoint_readers("scratch", "fastsafetensors"):
                    with self.subTest(task=task, method=method):
                        config = Roberta._create_config(directory)
                        config.model_type = "roberta"
                        config.tokenizer_path = directory
                        config.data_type = "fp32"
                        config.task_type = task
                        config.generate_env_config = GenerateEnvConfig()
                        config.render_config = RenderConfig()
                        config.max_seq_len = (
                            8  # deployment cap, not embedding table rows
                        )
                        parallel = SimpleNamespace(
                            world_size=1,
                            world_rank=0,
                            local_rank=0,
                            tp_size=1,
                            tp_rank=0,
                            ep_size=1,
                            ep_rank=0,
                            pp_size=1,
                            pp_rank=0,
                            dp_size=1,
                            dp_rank=0,
                            pp_stage_layer_counts=[],
                            get_attn_tp_size=lambda: 1,
                            get_attn_tp_rank=lambda: 0,
                            get_ffn_tp_size=lambda: 1,
                            get_ffn_tp_rank=lambda: 0,
                        )
                        service = CpuRoberta.from_config(
                            model_config=config,
                            parallelism_config=parallel,
                            kv_cache_config=SimpleNamespace(
                                multi_task_prompt="", multi_task_prompt_str=""
                            ),
                            hw_kernel_config=SimpleNamespace(
                                enable_cuda_graph=False, use_swizzleA=False
                            ),
                            moe_config=None,
                            fmha_config=None,
                            max_generate_batch_size=2,
                            load_method=method,
                            vit_config=None,
                            merge_lora=False,
                        )
                        self.assertEqual(config.position_ids_style, 1)
                        self.assertEqual(config.special_tokens.pad_token_id, 1)
                        self.assertIsInstance(service.py_model, RobertaEncoderOwner)
                        self.assertEqual(
                            service.model_weights_loader.loaded_method,
                            NewLoaderLoadMethod(method),
                        )
                        self.assertIs(
                            service.weight.get_global_weight(W.pre_decoder_ln_beta),
                            service.py_model.embeddings.LayerNorm.bias,
                        )
                        inputs = _packed_inputs(service.py_model, sequences)
                        hidden = service.py_model(inputs).hidden_states
                        torch.testing.assert_close(
                            hidden, expected_hidden, rtol=2e-5, atol=5e-6
                        )
                        actual = service.custom_module.handler.forward(
                            inputs.input_ids,
                            hidden,
                            inputs.attention_inputs.input_lengths,
                        )
                        torch.testing.assert_close(
                            actual, expected.to(actual.dtype), rtol=2e-5, atol=5e-6
                        )


class RobertaCheckpointTest(unittest.TestCase):
    @torch.inference_mode()
    def test_unindexed_torch_encoder_ignores_handler_sidecars(self):
        reference = RobertaModel(_hf_config()).eval()
        weights = {f"roberta.{k}": v for k, v in reference.state_dict().items()}
        sequence = [0, 5, 6, 2]
        expected = reference(torch.tensor([sequence])).last_hidden_state[0]
        for suffix in ("pt", "pth"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                reference.config.to_json_file(os.path.join(directory, "config.json"))
                for name, width in (("sparse_linear.pt", 1), ("colbert_linear.pt", 8)):
                    torch.save(
                        {"weight": torch.ones(width, 8), "bias": torch.zeros(width)},
                        os.path.join(directory, name),
                    )
                self.assertEqual(discover_ckpt_files(directory), [])
                encoder_file = os.path.join(directory, f"encoder.{suffix}")
                torch.save(weights, encoder_file)
                self.assertEqual(discover_ckpt_files(directory), [encoder_file])
                owner = _load(
                    directory, _runtime_config(reference.config, directory), "scratch"
                )
                actual = owner.forward_hidden(_packed_inputs(owner, [sequence]))
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=5e-6)

    @torch.inference_mode()
    def test_encoder_matches_transformers_with_offsets_and_padding(self):
        torch.manual_seed(41)
        reference = RobertaModel(_hf_config()).eval()
        # A loader must not zero or reinterpret pretrained padding rows.
        reference.embeddings.word_embeddings.weight[1].fill_(0.73)
        reference.embeddings.position_embeddings.weight[1].fill_(0.37)
        sequences = [[0, 5, 6, 2], [0, 7, 2]]
        ids, mask = _hf_inputs(sequences)
        expected_batch = reference(ids, attention_mask=mask).last_hidden_state
        expected = torch.cat(
            [expected_batch[i, : len(seq)] for i, seq in enumerate(sequences)]
        )
        for method in checkpoint_readers("scratch", "fastsafetensors"):
            for prefix in ("", "roberta."):
                for legacy_norm in (False, True):
                    with self.subTest(
                        method=method, prefix=prefix, legacy_norm=legacy_norm
                    ), tempfile.TemporaryDirectory() as directory:
                        _checkpoint(reference, directory, prefix, legacy_norm)
                        owner = _load(
                            directory,
                            _runtime_config(reference.config, directory),
                            method,
                        )
                        self.assertIsInstance(owner, RobertaEncoderOwner)
                        actual = owner.forward_hidden(_packed_inputs(owner, sequences))
                        torch.testing.assert_close(
                            actual, expected, rtol=2e-5, atol=5e-6
                        )
                        torch.testing.assert_close(
                            owner.runtime_weight_view()["embedding"][1],
                            reference.embeddings.word_embeddings.weight[1],
                        )
                        # Dense embedding uses the same first-token hidden vectors.
                        first = torch.tensor([0, len(sequences[0])])
                        torch.testing.assert_close(
                            F.normalize(actual[first], dim=-1),
                            F.normalize(expected[first], dim=-1),
                            rtol=2e-5,
                            atol=5e-6,
                        )

    @torch.inference_mode()
    def test_maximum_position_boundary_is_not_offset_twice(self):
        torch.manual_seed(52)
        reference = RobertaModel(_hf_config()).eval()
        sequence = [0, 3, 4, 5, 6, 7, 8, 2]  # positions 2..9 in a ten-row table
        for method in checkpoint_readers("scratch", "fastsafetensors"):
            with self.subTest(
                method=method
            ), tempfile.TemporaryDirectory() as directory:
                _checkpoint(reference, directory, "roberta.")
                owner = _load(
                    directory, _runtime_config(reference.config, directory), method
                )
                actual = owner.forward_hidden(_packed_inputs(owner, [sequence]))
                expected = reference(torch.tensor([sequence])).last_hidden_state[0]
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=5e-6)
                config = _runtime_config(reference.config, directory)
                config.max_seq_len += 1
                with self.assertRaisesRegex(
                    ValueError, "max_seq_len=11.*10-row position"
                ):
                    _load(directory, config, method)

    @torch.inference_mode()
    def test_classifier_reranker_projection_matches_transformers(self):
        torch.manual_seed(63)
        reference = RobertaForSequenceClassification(_hf_config()).eval()
        sequences = [[0, 5, 6, 2], [0, 7, 2]]
        ids, mask = _hf_inputs(sequences)
        expected = reference(ids, attention_mask=mask).logits
        for method in checkpoint_readers("scratch", "fastsafetensors"):
            with self.subTest(
                method=method
            ), tempfile.TemporaryDirectory() as directory:
                _checkpoint(reference, directory)
                owner = _load(
                    directory,
                    _runtime_config(reference.config, directory),
                    method,
                    heads=True,
                )
                hidden = owner.forward_hidden(_packed_inputs(owner, sequences))
                first = hidden[torch.tensor([0, len(sequences[0])])]
                exports = owner.runtime_weight_view()
                for name in HEAD_NAMES:
                    torch.testing.assert_close(
                        exports["__custom__." + name], reference.state_dict()[name]
                    )
                # Existing RobertaClassifierHandler formula, shared by reranker.
                dense = F.linear(
                    first,
                    exports["__custom__.classifier.dense.weight"],
                    exports["__custom__.classifier.dense.bias"],
                )
                actual = F.linear(
                    torch.tanh(dense),
                    exports["__custom__.classifier.out_proj.weight"],
                    exports["__custom__.classifier.out_proj.bias"],
                )
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=5e-6)

    def test_tp_checkpoint_hidden_columns_and_replicated_heads(self):
        torch.manual_seed(74)
        reference = RobertaForSequenceClassification(_hf_config()).eval()
        for method in checkpoint_readers("scratch", "fastsafetensors"):
            with self.subTest(
                method=method
            ), tempfile.TemporaryDirectory() as directory:
                _checkpoint(reference, directory)
                torch.multiprocessing.spawn(
                    _tp_checkpoint_worker,
                    args=(
                        directory,
                        os.path.join(directory, "gloo-rendezvous"),
                        method,
                    ),
                    nprocs=2,
                    join=True,
                )

    @torch.inference_mode()
    def test_sparse_colbert_sidecars_remain_handler_owned(self):
        torch.manual_seed(85)
        reference = RobertaModel(_hf_config()).eval()
        sequences = [[0, 5, 6, 2], [0, 7, 2]]
        ids, mask = _hf_inputs(sequences)
        expected_batch = reference(ids, attention_mask=mask).last_hidden_state
        expected = torch.cat(
            [expected_batch[i, : len(seq)] for i, seq in enumerate(sequences)]
        )
        sparse = {"weight": torch.randn(1, 8), "bias": torch.randn(1)}
        colbert = {"weight": torch.randn(8, 8), "bias": torch.randn(8)}
        for method in checkpoint_readers("scratch", "fastsafetensors"):
            with self.subTest(
                method=method
            ), tempfile.TemporaryDirectory() as directory:
                _checkpoint(reference, directory, "roberta.")
                torch.save(sparse, os.path.join(directory, "sparse_linear.pt"))
                torch.save(colbert, os.path.join(directory, "colbert_linear.pt"))
                owner = _load(
                    directory, _runtime_config(reference.config, directory), method
                )
                actual = owner.forward_hidden(_packed_inputs(owner, sequences))
                # Main handlers load these sidecars themselves. The encoder
                # loader must neither consume them nor modify their values.
                for filename, values in (
                    ("sparse_linear.pt", sparse),
                    ("colbert_linear.pt", colbert),
                ):
                    saved = torch.load(
                        os.path.join(directory, filename), weights_only=True
                    )
                    for key in values:
                        torch.testing.assert_close(saved[key], values[key])
                torch.testing.assert_close(
                    F.relu(F.linear(actual, **sparse)),
                    F.relu(F.linear(expected, **sparse)),
                    rtol=2e-5,
                    atol=5e-6,
                )
                offset = 0
                for seq in sequences:
                    keep = slice(offset + 1, offset + len(seq))  # ColBERT drops CLS
                    torch.testing.assert_close(
                        F.normalize(F.linear(actual[keep], **colbert), dim=-1),
                        F.normalize(F.linear(expected[keep], **colbert), dim=-1),
                        rtol=2e-5,
                        atol=5e-6,
                    )
                    offset += len(seq)

    def test_declared_classifier_cannot_succeed_with_missing_tensor(self):
        reference = RobertaForSequenceClassification(_hf_config()).eval()
        for method in checkpoint_readers("scratch", "fastsafetensors"):
            with self.subTest(
                method=method
            ), tempfile.TemporaryDirectory() as directory:
                values = _checkpoint(reference, directory)
                del values["classifier.out_proj.bias"]
                save_file(values, os.path.join(directory, "model.safetensors"))
                with self.assertRaisesRegex(
                    (ValueError, RuntimeError), "[Mm]issing|[Nn]ot.*loaded"
                ):
                    _load(
                        directory,
                        _runtime_config(reference.config, directory),
                        method,
                        heads=True,
                    )


if __name__ == "__main__":
    unittest.main()
