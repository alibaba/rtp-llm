"""Real tiny draft/target checkpoint and loan protocol; no GPU execution."""

import gc
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.borrowed_parameters import (
    BorrowedParameter,
    bind_borrowed_parameters,
    exclude_borrowed_checkpoint_weights,
    validate_borrowed_parameters,
)
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.deepseek_v4.load_geometry import make_lm_head
from rtp_llm.models_py.new_models.deepseek_v41.backbone import OwnedV41Transformer
from rtp_llm.models_py.new_models.deepseek_v41.backbone_test import (
    args,
    checkpoint,
    tensor,
)
from rtp_llm.models_py.new_models.deepseek_v41.globals import OwnedV41Matrix
from rtp_llm.models_py.new_models.deepseek_v41_dspark.checkpoint import (
    V41DraftGlobalsMixin,
    V41DraftParameterMixin,
)
from rtp_llm.models_py.weight_mapper import get_all_weights


class _Backbone(V41DraftGlobalsMixin, OwnedV41Transformer):
    pass


class _CheckpointOwner(V41DraftParameterMixin, RtpModule):
    def __init__(self, rank=0, commit=False):
        RtpModule.__init__(self)
        a = args(rank, commit)
        a.compress_ratios = [0, 0, 0]
        a.v41_config = dict(
            kv_source_layer_ids=[], index_source_layer_ids=[], engram_layer_ids=[]
        )
        a.embedding_tp_size, a.embedding_tp_rank = 2, rank
        self._v4_args, self._commit_only_prefill = a, commit
        self.v4 = _Backbone(a, layer_prefix="mtp", global_prefix="mtp.2.")
        self._install_draft_auxiliary(64, (0, 1, 2), 8, 4, 2)


def values(commit=False):
    result = {}
    for name, value in checkpoint(commit, layer_prefix="mtp").items():
        if ".compressor." in name or ".indexer." in name:
            continue
        result["mtp.2.norm.weight" if name == "norm.weight" else name] = value
    result.update(
        {
            "embed.weight": tensor((8, 64), torch.bfloat16),
            "head.weight": tensor((8, 64), torch.bfloat16, seed=3),
            "mtp.0.main_proj.weight": tensor((64, 192), torch.float8_e4m3fn),
            "mtp.0.main_proj.scale": tensor((2, 6), torch.float8_e8m0fnu),
            "mtp.0.main_norm.weight": tensor((64,), torch.bfloat16),
            "mtp.2.confidence_head.proj.weight": tensor((1, 64), torch.bfloat16),
        }
    )
    if not commit:
        result["mtp.2.markov_head.embed.weight"] = tensor((8, 4), torch.bfloat16)
        result["mtp.2.markov_head.head.weight"] = tensor((8, 4), torch.bfloat16, seed=2)
    return result


class _TargetGlobals(RtpModule):
    def __init__(self, rank):
        super().__init__()
        self.embed = OwnedV41Matrix(
            8, 64, shard_dim=1, tp_size=2, tp_rank=rank, dtype=torch.bfloat16
        )
        self.head = make_lm_head(args(rank), torch.bfloat16)


def load_into(model, path, rank, native_fst=False):
    loader = NewModelLoader(
        SimpleNamespace(model_type="deepseek_v41_dspark"),
        NewLoaderConfig(device="cpu", tp_size=2, tp_rank=rank, ep_size=2, ep_rank=rank),
        model_path=str(path.parent),
    )
    include = exclude_borrowed_checkpoint_weights(
        model, model.checkpoint_weight_name_filter()
    )
    plan = {}
    resolver = loader._checkpoint_read_spec_resolver(model, include, None, plan)
    model._rtp_checkpoint_load_plan = plan
    iterator = (
        loader._fastsafetensors_weights_iterator(
            [str(path)], "cpu", include, None, resolver
        )
        if native_fst
        else get_all_weights(
            [str(path)], name_filter=include, read_spec_resolver=resolver
        )
    )
    model.load_weights(iterator)
    NewModelLoader._validate_loaded_weights(model)
    validate_borrowed_parameters(model, "cpu")


class DraftCheckpointTest(unittest.TestCase):
    def test_unknown_confidence_head_weights_still_fail_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            for name in (
                "mtp.2.confidence_head.proj.bias",
                "mtp.1.confidence_head.proj.weight",
            ):
                source = values()
                del source["mtp.2.confidence_head.proj.weight"]
                source[name] = torch.ones(1)
                save_file(source, str(path))
                for native_fst in (False, True):
                    with self.subTest(name=name, native_fst=native_fst):
                        with self.assertRaisesRegex(RuntimeError, "confidence_head"):
                            load_into(_CheckpointOwner(), path, 0, native_fst)

    def test_proposal_scratch_and_native_fst_every_parameter_tp_ep_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file(values(), str(path))
            for rank in (0, 1):
                scratch, fst = _CheckpointOwner(rank), _CheckpointOwner(rank)
                load_into(scratch, path, rank)
                load_into(fst, path, rank, native_fst=True)
                for name, value in scratch.named_parameters():
                    self.assertTrue(
                        torch.equal(
                            value.view(torch.uint8),
                            dict(fst.named_parameters())[name].view(torch.uint8),
                        ),
                        name,
                    )
                self.assertEqual(len(fst.v4.layers), 3)
                self.assertTrue(
                    all(layer.attn.compress_ratio == 0 for layer in fst.v4.layers)
                )
                self.assertTrue(all(layer.engram is None for layer in fst.v4.layers))
                self.assertIs(fst.main_proj, fst.auxiliary.main_proj)
                self.assertIs(
                    fst.runtime_weight_view()["dspark_markov_w1.weight"],
                    fst.auxiliary.markov_head.markov_w1.weight,
                )

    def test_commit_loans_exact_storage_survive_fst_gc_and_target_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file(values(True), str(path))
            for rank in (0, 1):
                target = _TargetGlobals(rank)
                target.load_weights(
                    get_all_weights(
                        [str(path)],
                        name_filter=lambda name: name
                        in ("embed.weight", "head.weight"),
                    )
                )
                draft = _CheckpointOwner(rank, True)
                self.assertTrue(draft.v4.embed.weight.is_meta)
                self.assertTrue(draft.v4.head.weight.is_meta)
                bind_borrowed_parameters(
                    draft,
                    {
                        "v4.embed.weight": BorrowedParameter(target, "embed.weight"),
                        "v4.head.weight": BorrowedParameter(target, "head.weight"),
                    },
                    "cpu",
                )
                before = target.head.weight.clone()
                NewModelLoader._prepare_model_for_fastsafetensors(draft, "cpu")
                load_into(draft, path, rank, native_fst=True)
                self.assertIs(draft.v4.embed.weight, target.embed.weight)
                self.assertIs(draft.v4.head.weight, target.head.weight)
                torch.testing.assert_close(target.head.weight, before)
                self.assertEqual(
                    set(draft.runtime_weight_view()), {"embedding", "lm_head"}
                )
                self.assertEqual(len(list(draft.named_parameters())), 14)
                with torch.no_grad():
                    target.head.weight.add_(1)
                torch.testing.assert_close(draft.v4.head.weight, before + 1)
                ref = weakref.ref(target)
                pointer = target.head.weight.data_ptr()
                del target
                gc.collect()
                self.assertIsNotNone(ref())
                self.assertEqual(draft.v4.head.weight.data_ptr(), pointer)
                validate_borrowed_parameters(draft, "cpu")

    def test_missing_projection_scale_does_not_mutate_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            source = values(True)
            target = _TargetGlobals(0)
            target.load_weights(
                {name: source[name] for name in ("embed.weight", "head.weight")}
            )
            before, pointer = target.head.weight.clone(), target.head.weight.data_ptr()
            draft = _CheckpointOwner(0, True)
            bind_borrowed_parameters(
                draft,
                {
                    "v4.embed.weight": BorrowedParameter(target, "embed.weight"),
                    "v4.head.weight": BorrowedParameter(target, "head.weight"),
                },
                "cpu",
            )
            del source["mtp.0.main_proj.scale"]
            save_file(source, str(path))
            with self.assertRaisesRegex(RuntimeError, "missing|required|Missing"):
                load_into(draft, path, 0)
            torch.testing.assert_close(target.head.weight, before)
            self.assertEqual(target.head.weight.data_ptr(), pointer)

    def test_commit_requires_both_loans_before_consuming_iterator(self):
        draft = _CheckpointOwner(0, True)

        def never_read():
            raise AssertionError("must reject before checkpoint read")
            yield

        with self.assertRaisesRegex(ValueError, "before dispatch"):
            draft.load_weights(never_read())


if __name__ == "__main__":
    unittest.main()
