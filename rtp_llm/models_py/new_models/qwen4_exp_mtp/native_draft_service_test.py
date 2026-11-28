"""Production Qwen4 target/MTP routes and checkpoint ownership (CPU only)."""

import json
import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.models.qwen4_exp.qwen4_exp_mtp import Qwen4ExpMTP
from rtp_llm.models_py.checkpoint_reader_test_support import checkpoint_readers, has_native_fst, requires_native_fst
from rtp_llm.models_py.new_models.qwen4_exp.language import Qwen4ForCausalLM
from rtp_llm.models_py.new_models.qwen4_exp.test.test_native_service import (
    CheckpointOnlyQwen4,
)
from rtp_llm.models_py.new_models.qwen4_exp.test.test_precision_profiles import (
    source_checkpoint,
)
from rtp_llm.models_py.new_models.qwen4_exp_mtp.draft_service import (
    Qwen4ExpMTPForCausalLM,
)
from rtp_llm.models_py.new_models.qwen4_exp_mtp.draft_service import (
    Qwen4MTPDraftWeights,
)
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.borrowed_parameters import BorrowedParameter
from rtp_llm.models_py.new_models.qwen4_exp.target_weights import Qwen4TargetWeights
from rtp_llm.models.direct_checkpoint import load_owned_model, _runtime_view
from rtp_llm.config.qwen4_exp_config import resolve_qwen4_parallelism
from rtp_llm.models_py.new_models.qwen4_exp_mtp.vocabulary import (
    EMBED_SCALE,
    HEAD_SCALE,
)
from rtp_llm.models_py.registry import MODEL_REGISTRY, get_model_class
from rtp_llm.ops import HWKernelConfig, MoeConfig, ParallelismConfig


class CheckpointOnlyDraft(Qwen4ExpMTP):
    def load_tokenizer(self):
        pass

    def _get_device_str(self):
        return "cpu"

    def _init_custom_module(self):
        return None


def shared_checkpoint(directory, kind="nvfp4"):
    target_config, values = source_checkpoint(directory, kind)
    root = Path(directory)
    config_json = json.loads((root / "config.json").read_text())
    config_json["text_config"]["mtp"] = dict(
        hybrid=True,
        num_hidden_layers=1,
        layer_types=["full_attention"],
        rope_theta=10000,
        mtp_use_hidden_state_from_layer=None,
    )
    (root / "config.json").write_text(json.dumps(config_json))
    for name, tensor in tuple(values.items()):
        if name.startswith("model.language_model.layers.1."):
            subkey = name.removeprefix("model.language_model.layers.1.")
            for indexer in ("index_qk_proj", "q_layernorm", "k_layernorm"):
                if subkey == f"self_attn.{indexer}.weight":
                    subkey = f"self_attn.indexer.{indexer}.weight"
            values[f"mtp.layers.0.{subkey}"] = tensor.clone()
            if subkey == "self_attn.indexer.q_layernorm.weight":
                values[f"mtp.layers.0.{subkey}"] += 0.25
        if name.startswith("model.language_model.hyper_connection_mixer."):
            values["mtp." + name.removeprefix("model.language_model.")] = tensor.clone()
    for name, shape in (
        ("fc_embedding", (128, 128)),
        ("fc_hidden", (128, 128)),
        ("pre_fc_norm_embedding", (128,)),
        ("pre_fc_norm_hidden", (512,)),
    ):
        values[f"mtp.{name}.weight"] = (torch.randn(shape) * 0.1).bfloat16()
    save_file(values, str(root / "model.safetensors"))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {n: "model.safetensors" for n in values}})
    )
    draft_config = Qwen4ExpMTP._create_config_from_json(directory, config_json)
    draft_config.data_type = "bf16"
    draft_config.generate_env_config = None
    draft_config.enable_fp32_lm_head = False
    return target_config, draft_config, values


class DraftNativeServiceTest(unittest.TestCase):
    def test_ambiguous_quantized_ft_is_not_validated_as_an_hf_manifest(self):
        # Only the dispatch boundary: no claim that this header is a usable
        # quantized export or that its missing packed layout can be recovered.
        with tempfile.TemporaryDirectory() as archive:
            (Path(archive) / "model.safetensors.index.json").write_text(
                json.dumps({"is_ft_style_weight": True, "weight_map": {}})
            )
            for fp8_source in (False, True):
                config = SimpleNamespace(
                    ckpt_path=archive, is_mtp=True,
                    _qwen4_fp8_source=fp8_source, _qwen4_nvfp4_source=not fp8_source,
                )
                with self.assertRaisesRegex(ValueError, "original HF checkpoint"):
                    Qwen4ExpMTP._validate_checkpoint(config)

    def test_bf16_ft_writer_round_trip_with_owned_and_borrowed_vocab(self):
        self._check_ft_writer_round_trip(fp32_head=False)

    def test_fp32_ft_head_preserves_owned_and_borrowed_draft_vocab(self):
        self._check_ft_writer_round_trip(fp32_head=True)

    def _check_ft_writer_round_trip(self, fp32_head):
        from rtp_llm.tools.convert.ft_export_job import create_ft_export_job
        from rtp_llm.tools.convert.ft_export_metadata import generate_ft_index

        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as archive:
            tc, dc, values = shared_checkpoint(source, "bf16")
            tc.enable_fp32_lm_head = dc.enable_fp32_lm_head = fp32_head
            if fp32_head:
                # Input-only token IDs belong to the embedding, never the head.
                emb = "model.language_model.embed_tokens.weight"
                values[emb] = torch.cat((values[emb], values[emb][:3]))
                tc.input_vocab_size = dc.input_vocab_size = values[emb].shape[0]
                path = Path(source) / "config.json"
                raw = json.loads(path.read_text())
                raw["text_config"]["input_vocab_size"] = values[emb].shape[0]
                path.write_text(json.dumps(raw))
                save_file(values, str(Path(source) / "model.safetensors"))
            pc = ParallelismConfig()
            pc.world_size = pc.tp_size = pc.ep_size = 1
            pc.world_rank = pc.tp_rank = pc.ep_rank = pc.local_rank = 0
            hw = HWKernelConfig()
            hw.enable_cuda_graph = hw.use_swizzleA = False
            kwargs = dict(parallelism_config=pc, hw_kernel_config=hw,
                          kv_cache_config=KVCacheConfig(), fmha_config=None,
                          moe_config=MoeConfig(), load_method="scratch",
                          max_generate_batch_size=2, vit_config=None, merge_lora=False)
            descriptor = CheckpointOnlyDraft(model_config=dc, **kwargs)
            create_ft_export_job(descriptor).dump_weight_as_ft_style("cpu", archive)
            raw = (Path(source) / "config.json").read_text()
            (Path(archive) / "config.json").write_text(raw)
            generate_ft_index(archive, {"TP_SIZE": "1", "EP_SIZE": "1", "DP_SIZE": "1"})
            original = CheckpointOnlyDraft.from_config(model_config=dc, **kwargs)
            target = CheckpointOnlyQwen4.from_config(model_config=tc, **kwargs)
            self.addCleanup(original.py_model.close)
            self.addCleanup(target.py_model.close)
            # Give the live target distinct values: the second import must not
            # overwrite either loan with the valid but now stale archive copy.
            with torch.inference_mode():
                target.py_model.embed_tokens.weight.add_(1)
                target.py_model.lm_head.weight.add_(2)
            saved = {k: getattr(target.py_model, k).weight.clone() for k in ("embed_tokens", "lm_head")}
            for method in checkpoint_readers("scratch", "fastsafetensors"):
                for borrow in (False, True):
                    with self.subTest(method=method, borrow=borrow):
                        config = CheckpointOnlyDraft._create_config_from_json(archive, json.loads(raw))
                        config.data_type, config.generate_env_config = "bf16", None
                        config.enable_fp32_lm_head = fp32_head
                        if fp32_head:
                            config.input_vocab_size = tc.input_vocab_size
                        alias = (dict(weight_alias_owner=target, weight_alias_names=("embedding", "lm_head"))
                                 if borrow else {})
                        actual = CheckpointOnlyDraft.from_config(
                            model_config=config, **dict(kwargs, load_method=method), **alias)
                        self.addCleanup(actual.py_model.close)
                        self.assertEqual(actual.py_model.lm_head.weight.dtype,
                                         torch.float32 if fp32_head else torch.bfloat16)
                        expected = dict(original.py_model.named_parameters())
                        params = dict(actual.py_model.named_parameters())
                        self.assertEqual(params.keys(), expected.keys())
                        for name, param in params.items():
                            owner = name.partition(".")[0]
                            if borrow and name in ("embed_tokens.weight", "lm_head.weight"):
                                self.assertIs(param, getattr(target.py_model, owner).weight)
                                torch.testing.assert_close(param, saved[owner], rtol=0, atol=0)
                            else:
                                torch.testing.assert_close(param, expected[name], rtol=0, atol=0)

    def test_rank_local_mixed_roles_and_live_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _profiles_worker, args=(directory,), nprocs=2, join=True
            )

    @requires_native_fst
    def test_two_rank_fst_draft_keeps_local_target_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            shared_checkpoint(directory)
            torch.multiprocessing.spawn(
                _fst_worker, args=(directory,), nprocs=2, join=True
            )

    def test_bad_draft_anchor_does_not_change_published_target(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, vals = shared_checkpoint(directory)
            c = {"model_type": "_q4draft_tree", "ckpt_path": directory}
            load = NewLoaderConfig(
                device="cpu", compute_dtype=torch.bfloat16, load_method="scratch"
            )
            with patch.dict(
                MODEL_REGISTRY,
                {
                    "_q4target_tree": Qwen4TargetWeights,
                    "_q4draft_tree": Qwen4MTPDraftWeights,
                },
            ):
                target = NewModelLoader(
                    dict(c, model_type="_q4target_tree"), load, model_path=directory
                ).load()
                head = target.lm_head.weight
                before = head.clone()
                self.addCleanup(target.close)
                # Missing released nested indexer is not replaced by target QSA.
                del vals["mtp.layers.0.self_attn.indexer.q_layernorm.weight"]
                root = Path(directory)
                save_file(vals, str(root / "model.safetensors"))
                (root / "model.safetensors.index.json").write_text(
                    json.dumps(
                        {"weight_map": {key: "model.safetensors" for key in vals}}
                    )
                )
                with self.assertRaisesRegex(ValueError, "MTP anchor"):
                    NewModelLoader(
                        c,
                        load,
                        model_path=directory,
                        parameter_aliases={
                            name: BorrowedParameter(target, name)
                            for name in ("lm_head.weight", "embed_tokens.weight")
                        },
                    ).load()
                self.assertIs(head, target.lm_head.weight)
                torch.testing.assert_close(before, head, rtol=0, atol=0)

    def test_target_and_draft_keep_aliases_and_draft_state(self):
        for source in ("bf16", "nvfp4"):
            for method in checkpoint_readers("scratch", "fastsafetensors"):
                with self.subTest(source=source, method=method):
                    self._exercise(method, source)

    def _exercise(self, method, source):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"LOAD_METHOD": ""}),
        ):
            self.assertIs(get_model_class("qwen4_exp"), Qwen4ForCausalLM)
            self.assertIs(get_model_class("qwen4_exp_mtp"), Qwen4ExpMTPForCausalLM)
            tc, dc, values = shared_checkpoint(directory, source)
            pc = ParallelismConfig()
            pc.world_size = pc.tp_size = pc.ep_size = 1
            pc.world_rank = pc.tp_rank = pc.ep_rank = pc.local_rank = 0
            hw = HWKernelConfig()
            hw.enable_cuda_graph = False
            hw.use_swizzleA = False
            kwargs = dict(
                parallelism_config=pc,
                hw_kernel_config=hw,
                kv_cache_config=KVCacheConfig(),
                fmha_config=None,
                moe_config=MoeConfig(),
                load_method=method,
                max_generate_batch_size=2,
                vit_config=None,
                merge_lora=False,
            )
            target = CheckpointOnlyQwen4.from_config(model_config=tc, **kwargs)
            self.addCleanup(target.py_model.close)
            aliases = Qwen4ExpMTP.speculative_weight_alias_names(target, dc)
            self.assertEqual(aliases, ("embedding", "lm_head"))
            draft = CheckpointOnlyDraft.from_config(
                model_config=dc,
                weight_alias_owner=target,
                weight_alias_names=aliases,
                **kwargs,
            )
            self.addCleanup(draft.py_model.close)
            self.assertEqual(draft.model_weights_loader.loaded_method.value, method)
            self.assertIs(
                draft.py_model.embed_tokens.weight, target.py_model.embed_tokens.weight
            )
            self.assertIs(draft.py_model.lm_head.weight, target.py_model.lm_head.weight)
            runtime = draft.py_model._runtime
            draft.py_model.kv_cache = None
            self.assertIsNone(runtime.kv_cache)
            self.assertNotIn("kv_cache", draft.py_model.__dict__)
            self.assertEqual(len(runtime.layers), 1)
            self.assertEqual(len(runtime.ple_layers), 0)
            self.assertTrue(runtime.layers[0].self_attn.is_mtp_draft)
            self.assertIsNotNone(runtime.layers[0].self_attn.qsa_indexer)
            torch.testing.assert_close(
                draft.py_model.layers[0].self_attn.qsa_indexer.q_layernorm.weight,
                values["mtp.layers.0.self_attn.indexer.q_layernorm.weight"],
                rtol=0,
                atol=0,
            )
            self.assertEqual(len(draft.weight.weights), 1)
            self.assertIs(
                draft.weight.get_global_weight("lm_head"),
                target.py_model.lm_head.weight,
            )
            inputs = torch.randn(2, 128, dtype=torch.bfloat16)
            hidden = torch.randn(2, 512, dtype=torch.bfloat16)
            import torch.nn.functional as F

            def norm(x, name):
                f = x.float()
                return (
                    f
                    * torch.rsqrt(f.square().mean(-1, keepdim=True) + dc.layernorm_eps)
                    * (1 + values["mtp." + name + ".weight"].float())
                ).bfloat16()

            a = F.linear(
                norm(inputs, "pre_fc_norm_embedding"), values["mtp.fc_embedding.weight"]
            )
            b = F.linear(
                norm(hidden, "pre_fc_norm_hidden").view(-1, 128),
                values["mtp.fc_hidden.weight"],
            ).view(2, 4, 128)
            torch.testing.assert_close(
                runtime.mtp_input_projection(inputs, hidden),
                (a[:, None] + b).flatten(-2),
                rtol=0,
                atol=0,
            )
            ptrs = {n: p.data_ptr() for n, p in draft.py_model.named_parameters()}
            draft.py_model.process_weights_after_loading()
            self.assertEqual(
                ptrs, {n: p.data_ptr() for n, p in draft.py_model.named_parameters()}
            )


def _fst_worker(rank, directory):
    torch.set_num_threads(1)
    rendezvous = str(Path(directory) / "gloo_init")
    torch.distributed.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        with patch.dict(
            MODEL_REGISTRY,
            {
                "_q4target_tree": Qwen4TargetWeights,
                "_q4draft_tree": Qwen4MTPDraftWeights,
            },
        ):
            load = NewLoaderConfig(
                device="cpu",
                compute_dtype=torch.bfloat16,
                load_method="fastsafetensors",
                tp_size=2,
                tp_rank=rank,
                ep_size=2,
                ep_rank=rank,
            )

            def make(key, aliases=None):
                config = dict(
                    model_type=key, ckpt_path=directory, num_experts=2, moe_k=1
                )
                loader = NewModelLoader(
                    config, load, model_path=directory, parameter_aliases=aliases
                )
                result = loader.load()
                if loader.loaded_method.value != "fastsafetensors":
                    raise AssertionError(loader.loaded_method)
                return result

            target = make("_q4target_tree")
            try:
                loan = {
                    name: BorrowedParameter(target, name)
                    for name in ("embed_tokens.weight", "lm_head.weight")
                }
                draft = make("_q4draft_tree", loan)
                tc = unittest.TestCase()
                tc.assertIs(draft.embed_tokens.weight, target.embed_tokens.weight)
                tc.assertIs(draft.lm_head.weight, target.lm_head.weight)
                tc.assertEqual(draft.embed_tokens.weight.shape, (32, 64))
                tc.assertEqual(draft.lm_head.weight.shape, (16, 128))
                tc.assertEqual(draft.layers[0].mlp.experts.w13.shape, (1, 256, 64))
                tc.assertEqual(
                    draft.layers[0].mlp.experts.w13.unique().item(), rank + 1
                )
                tc.assertEqual(
                    draft.layers[0].self_attn.qkv_proj.weight.shape, (128, 128)
                )
            finally:
                target.close()
    finally:
        torch.distributed.destroy_process_group()


class _TargetStorage(Qwen4TargetWeights):
    # The production target's exact exports, without binding CUDA-only experts.
    runtime_weight_view = Qwen4ForCausalLM.runtime_weight_view


def _snapshot(owner):
    result = {}
    for name, value in owner.state_dict(keep_vars=True).items():
        result[name] = (
            value,
            value.data_ptr(),
            value.detach().reshape(-1).contiguous().view(torch.uint8).clone(),
        )
    return result


def _unchanged(owner, baseline):
    result = owner.state_dict(keep_vars=True)
    assert set(result) == set(baseline)
    for name, (value, ptr, expected) in baseline.items():
        actual = result[name]
        assert actual is value and actual.data_ptr() == ptr, name
        torch.testing.assert_close(
            actual.detach().reshape(-1).contiguous().view(torch.uint8),
            expected,
            rtol=0,
            atol=0,
            msg=name,
        )


def _profiles_worker(rank, directory):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo",
        init_method="file://" + str(Path(directory) / "profiles_gloo"),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        pc = ParallelismConfig()
        pc.world_size = pc.tp_size = 2
        pc.world_rank = pc.tp_rank = pc.local_rank = rank
        with (
            patch.dict(os.environ, {"LOAD_METHOD": ""}),
            patch.dict(
                MODEL_REGISTRY,
                {"qwen4_exp": _TargetStorage, "qwen4_exp_mtp": Qwen4MTPDraftWeights},
            ),
        ):
            for kind, modes in (
                ("bf16", ("inherit",)),
                ("fp8", ("inherit", "source_bf16", "source_bf16_shared_vocab")),
            ):
                for ep in (1, 2):
                    pc.ep_size, pc.ep_rank = ep, rank if ep == 2 else 0
                    with tempfile.TemporaryDirectory() as root:
                        tc, dc, vals = shared_checkpoint(root, kind)
                        pc_resolved = resolve_qwen4_parallelism(tc, pc)
                        for mode in modes:
                            if kind == "fp8":
                                dc.quantization = json.dumps(
                                    dict(
                                        method="modelopt_fp4",
                                        bits=4,
                                        group_size=16,
                                        is_quanted=False,
                                        model_recipe="qwen4_fp8_nvfp4",
                                        mtp_dense_storage=mode,
                                    )
                                )
                                Qwen4ExpMTP._configure_precision_recipe(dc)
                            drafts = []
                            for method in checkpoint_readers("scratch", "fastsafetensors"):
                                load = NewLoaderConfig.from_runtime(
                                    pc_resolved,
                                    device="cpu",
                                    compute_dtype=torch.bfloat16,
                                    load_method=method,
                                    moe_config=MoeConfig(),
                                    py_hw_kernel_config=HWKernelConfig(),
                                )
                                target = NewModelLoader(
                                    tc, load, model_path=root
                                ).load()
                                try:
                                    before = _snapshot(target)
                                    exports = target.runtime_weight_view()
                                    scaled = kind == "fp8" and mode != "source_bf16"
                                    aliases = (
                                        (
                                            "embedding",
                                            "lm_head",
                                            EMBED_SCALE,
                                            HEAD_SCALE,
                                        )
                                        if scaled
                                        else (
                                            ()
                                            if mode == "source_bf16"
                                            else ("embedding", "lm_head")
                                        )
                                    )
                                    if scaled:
                                        invalid = dict(exports)
                                        invalid[HEAD_SCALE] = exports[EMBED_SCALE]
                                        with unittest.TestCase().assertRaisesRegex(
                                            ValueError, "own scales"
                                        ):
                                            Qwen4MTPDraftWeights.capture_runtime_loan(
                                                target, tc, dc, invalid, "cpu"
                                            )
                                    source = SimpleNamespace(
                                        py_model=target,
                                        weight=_runtime_view(target, tc, "cpu"),
                                        model_config=tc,
                                    )
                                    service = SimpleNamespace(
                                        model_config=dc,
                                        _weight_alias_owner=source,
                                        _weight_alias_names=aliases,
                                        _get_device_str=lambda: "cpu",
                                        parallelism_config=pc_resolved,
                                        load_method=method,
                                        hw_kernel_config=HWKernelConfig(),
                                        moe_config=MoeConfig(),
                                        fmha_config=None,
                                        max_generate_batch_size=1,
                                        custom_module=None,
                                    )
                                    # Real service loader boundary, offline checkpoint owner only.
                                    # The mixed CUDA consumer is deliberately not published on CPU.
                                    load_owned_model(service, publish_runtime=False)
                                    draft = service.checkpoint_owner
                                    assert (
                                        service.model_weights_loader.loaded_method.value
                                        == method
                                    )
                                    if scaled:
                                        assert draft._vocab_loan is not None
                                        for role in aliases:
                                            assert (
                                                draft.runtime_weight_view()[role]
                                                is exports[role]
                                            ), role
                                        assert not draft.checkpoint_weight_name_filter()(
                                            "model.language_model.embed_tokens.weight"
                                        )
                                        assert (
                                            not draft.checkpoint_weight_name_filter()(
                                                "lm_head.weight"
                                            )
                                        )
                                        # Same shape but wrong rank: fail before preparing any target storage.
                                        wrong = NewLoaderConfig(
                                            device="cpu",
                                            compute_dtype=torch.bfloat16,
                                            load_method=method,
                                            tp_size=2,
                                            tp_rank=1 - rank,
                                            prepared_runtime_loan=draft._vocab_loan,
                                        )
                                        with unittest.TestCase().assertRaisesRegex(
                                            ValueError, "shard layout differs"
                                        ):
                                            Qwen4MTPDraftWeights(dc, wrong)
                                    elif mode == "source_bf16":
                                        assert (
                                            draft.embed_tokens.weight
                                            is not target.embed_tokens.weight
                                        )
                                        expect = vals[
                                            "model.language_model.embed_tokens.weight"
                                        ].chunk(2, dim=1)[rank]
                                        torch.testing.assert_close(
                                            draft.embed_tokens.weight,
                                            expect,
                                            rtol=0,
                                            atol=0,
                                        )
                                        assert draft.lm_head.storage_scale is None
                                    else:
                                        assert (
                                            draft.embed_tokens.weight
                                            is target.embed_tokens.weight
                                        )
                                    fc = draft.mtp_input_projection.fc_hidden
                                    assert (fc.weight.dtype == torch.float8_e4m3fn) == (
                                        kind == "fp8" and mode == "inherit"
                                    )
                                    _unchanged(target, before)
                                    drafts.append(
                                        {k: v[2] for k, v in _snapshot(draft).items()}
                                    )
                                finally:
                                    target.close()
                            if not has_native_fst():
                                continue  # Scratch ownership checks above still ran.
                            assert drafts[0].keys() == drafts[1].keys()
                            for key in drafts[0]:
                                torch.testing.assert_close(
                                    drafts[0][key],
                                    drafts[1][key],
                                    rtol=0,
                                    atol=0,
                                    msg=f"{kind}/{mode}/EP{ep}/rank{rank}/{key}",
                                )
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
