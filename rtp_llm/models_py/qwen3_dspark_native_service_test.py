"""Service construction with real target/draft checkpoint files and tokenizer.

Only the hardware placement is redirected to CPU for this regression. Both model
classes, alias declaration, loaders and publication path are the serving code.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from rtp_llm.model_factory import ModelFactory
from rtp_llm.model_factory_register import ModelDict
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models_py.model_loader import NewLoaderLoadMethod
from rtp_llm.models_py.new_models.qwen3 import Qwen3ForCausalLM
from rtp_llm.models_py.new_models.qwen3_dspark import Qwen3DSparkForCausalLM
from rtp_llm.models_py.new_models.qwen3_dspark.checkpoint_test import checkpoint
from rtp_llm.models_py.serving_route_checkpoint_test import service
from rtp_llm.ops import SpeculativeType


def _write_model(path, *, draft=False, vocab=8, loans=(), bad_order=False):
    path.mkdir()
    raw = dict(
        architectures=["Qwen3DSparkModel" if draft else "Qwen3ForCausalLM"],
        model_type="qwen3",
        hidden_size=4,
        intermediate_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        vocab_size=8,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        rms_norm_eps=1e-6,
    )
    data = checkpoint(draft)
    if draft:
        raw.update(
            mask_token_id=7,
            target_layer_ids=[0, 1],
            markov_rank=2,
            draft_vocab_size=vocab,
            dspark_shared_target_weights=list(loans),
        )
        if vocab == 8:
            data["markov_head.markov_w2.weight"] = torch.arange(16.0).reshape(8, 2)
            data["lm_head.weight"] = torch.arange(32.0).reshape(8, 4) + 313
            data["d2t"] = torch.zeros(8, dtype=torch.int64)
            if bad_order:
                data["d2t"][:2] = torch.tensor([1, -1])
        data["model.embed_tokens.weight"] = data["model.embed_tokens.weight"] + 101
        for role, name in (
            ("embedding", "model.embed_tokens.weight"),
            ("lm_head", "lm_head.weight"),
        ):
            if role in loans:
                del data[
                    name
                ]  # Explicitly omitted target-owned weight in the actual file.
    (path / "config.json").write_text(json.dumps(raw))
    tokenizer = Tokenizer(WordLevel({str(i): i for i in range(8)}, unk_token="0"))
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="0", eos_token="7"
    ).save_pretrained(path)
    save_file(data, str(path / "model.safetensors"))
    name = ModelDict.get_ft_model_type_by_config(raw)
    cls = ModelFactory.get_model_cls(name)
    config = cls.create_config(str(path))
    config.model_type = name
    config.is_mtp = draft
    config.data_type = "fp32"
    config.tokenizer_path = str(path)
    config.gen_num_per_cycle = 3
    return config


def _engine(root, method):
    p = service(root, method).parallelism_config
    p.local_world_size = 1
    p.pp_stage_layer_counts = []
    return SimpleNamespace(
        parallelism_config=p,
        runtime_config=SimpleNamespace(
            warm_up=False, model_warm_up=False, max_generate_batch_size=7, model_name=""
        ),
        hw_kernel_config=SimpleNamespace(enable_cuda_graph=False, use_swizzleA=False),
        kv_cache_config=SimpleNamespace(multi_task_prompt="", multi_task_prompt_str=""),
        fmha_config=None,
        moe_config=None,
        load_config=SimpleNamespace(
            load_method=method,
            force_cpu_load_weights=False,
            loader_recycle_handles=False,
            moe_pure_tp_preshard=False,
        ),
        sp_config=SimpleNamespace(type=SpeculativeType.DSPARK, gen_num_per_cycle=3),
    )


class DraftNativeOutputTest(unittest.TestCase):
    def test_target_exports_decoder_boundaries_for_draft(self):
        from rtp_llm.ops.compute_ops import PyAttentionInputs

        # All decoder layers and checkpoint parameters are real. The tiny CPU
        # attention stand-in supplies the four output channels of this example.
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            BaseModel, "_get_device_str", return_value="cpu"
        ):
            root = Path(tmp)
            target_cfg = _write_model(root / "target")
            draft_cfg = _write_model(root / "draft", draft=True, vocab=4)
            engine = _engine(tmp, "scratch")
            ModelFactory._setup_dspark_configs(engine.sp_config, target_cfg, draft_cfg)
            target = ModelFactory._create_model(target_cfg, engine).py_model
            ref = []
            handles = [
                layer.register_forward_hook(
                    lambda _, args, out: ref.append(out.detach().clone())
                )
                for layer in target.layers
            ]
            attn = SimpleNamespace(forward=lambda qkv, cache, idx: qkv[:, :4])
            prompt_ptr = None
            try:
                for verify in (False, True, True):
                    ref.clear()
                    meta = PyAttentionInputs()
                    meta.is_target_verify = verify
                    inputs = SimpleNamespace(
                        input_ids=torch.tensor([1, 2]),
                        capture_hidden_states=False,
                        attention_inputs=meta,
                        pp_intermediates={},
                    )
                    with torch.no_grad():
                        out = target(inputs, attn)
                    self.assertEqual(len(ref), 2)
                    torch.testing.assert_close(
                        out.mtp_target_hidden_states, torch.cat(ref, dim=-1)
                    )
                    self.assertEqual(
                        out.mtp_target_hidden_states.data_ptr(),
                        target.get_mtp_target_hidden_states(2).data_ptr(),
                    )
                    if not verify:
                        prompt_ptr = out.mtp_target_hidden_states.data_ptr()
                        verify_ptr = None
                    else:
                        self.assertNotEqual(
                            out.mtp_target_hidden_states.data_ptr(), prompt_ptr
                        )
                        if verify_ptr is not None:
                            self.assertEqual(
                                verify_ptr, out.mtp_target_hidden_states.data_ptr()
                            )
                        verify_ptr = out.mtp_target_hidden_states.data_ptr()
            finally:
                for handle in handles:
                    handle.remove()


class PublicDraftLoadingTest(unittest.TestCase):
    def test_factory_scratch_and_fst_keep_requested_storage(self):
        for method in ("scratch", "fastsafetensors"):
            for vocab, loans in (
                (4, ()),
                (4, ("embedding",)),
                (8, ("embedding", "lm_head")),
            ):
                with self.subTest(
                    method=method, vocab=vocab, loans=loans
                ), tempfile.TemporaryDirectory() as tmp, patch.object(
                    BaseModel, "_get_device_str", return_value="cpu"
                ):
                    path = Path(tmp)
                    tc = _write_model(path / "target")
                    dc = _write_model(
                        path / "draft", draft=True, vocab=vocab, loans=loans
                    )
                    engine = _engine(tmp, method)
                    ModelFactory._setup_dspark_configs(engine.sp_config, tc, dc)
                    target = ModelFactory._create_model(tc, engine)
                    proposal = ModelFactory.get_sp_model(
                        tc, dc, engine, target_model=target
                    )
                    draft = proposal.model
                    self.assertIsInstance(target.py_model, Qwen3ForCausalLM)
                    self.assertIsInstance(draft.py_model, Qwen3DSparkForCausalLM)
                    self.assertIs(draft.checkpoint_owner, draft.py_model)
                    self.assertIs(target.checkpoint_owner, target.py_model)
                    self.assertEqual(proposal.sp_type, SpeculativeType.DSPARK)
                    self.assertEqual(
                        draft.model_weights_loader.loaded_method,
                        NewLoaderLoadMethod(method),
                    )
                    self.assertEqual(
                        target.model_weights_loader.loaded_method,
                        NewLoaderLoadMethod(method),
                    )
                    self.assertEqual(set(draft._weight_alias_names), set(loans))
                    for role, name in (
                        ("embedding", "embed_tokens.weight"),
                        ("lm_head", "lm_head.weight"),
                    ):
                        src = target.weight.get_global_weight(role)
                        dst = draft.weight.get_global_weight(role)
                        self.assertIs(dst, draft.py_model.get_parameter(name))
                        self.assertEqual(dst is src, role in loans)
                        self.assertEqual(
                            dst.data_ptr() == src.data_ptr(), role in loans
                        )
                    self.assertEqual(
                        draft.py_model.d2t.tolist(),
                        [0, 2, 4, 6] if vocab == 4 else list(range(8)),
                    )
                    if "embedding" not in loans:
                        torch.testing.assert_close(
                            draft.py_model.embed_tokens.weight,
                            target.py_model.embed_tokens.weight + 101,
                        )
                    self.assertIs(
                        draft.weight.get_global_weight("dspark_markov_w1.weight"),
                        draft.py_model.markov_head.markov_w1.weight,
                    )

    def test_reordered_head_fails_without_modifying_target(self):
        for method in ("scratch", "fastsafetensors"):
            with self.subTest(
                method=method
            ), tempfile.TemporaryDirectory() as tmp, patch.object(
                BaseModel, "_get_device_str", return_value="cpu"
            ):
                path = Path(tmp)
                tc = _write_model(path / "target")
                dc = _write_model(
                    path / "draft",
                    draft=True,
                    vocab=8,
                    loans=("lm_head",),
                    bad_order=True,
                )
                engine = _engine(tmp, method)
                ModelFactory._setup_dspark_configs(engine.sp_config, tc, dc)
                target = ModelFactory._create_model(tc, engine)
                owned = target.py_model.lm_head.weight
                saved = owned.clone()
                with self.assertRaisesRegex(ValueError, "identity draft token order"):
                    ModelFactory.get_sp_model(tc, dc, engine, target_model=target)
                self.assertIs(owned, target.py_model.lm_head.weight)
                torch.testing.assert_close(owned, saved)


if __name__ == "__main__":
    unittest.main()
