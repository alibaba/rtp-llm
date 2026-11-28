import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.models_py.model_loader import NewLoaderConfig
from rtp_llm.models_py.new_models.deepseek_v4.load_geometry import (
    configure_lm_head,
    make_lm_head,
    validate_load_groups,
    validate_v4_static_placement,
)


class FlashLoadGeometryTest(unittest.TestCase):
    def test_pruned_head_reads_selected_rows_and_zero_pads_last_rank(self):
        model = dict(
            enable_output_vocab_pruning=True,
            output_vocab_ids=[1, 6, 8],
            output_vocab_padded_size=4,
        )
        source = torch.arange(20, dtype=torch.float32).view(10, 2)
        for rank, rows in enumerate(([1, 6], [8, None])):
            args = SimpleNamespace(vocab_size=10, dim=2, dp_size=1, ep_size=1)
            configure_lm_head(
                args, model, SimpleNamespace(lm_head_tp_size=2, lm_head_tp_rank=rank)
            )
            head = make_lm_head(args, torch.float32)
            spec = head.checkpoint_read_spec("weight", tuple(source.shape))
            head.load_weights({"weight": spec.materialize(source)})
            expected = torch.stack(
                [source[i] if i is not None else torch.zeros(2) for i in rows]
            )
            torch.testing.assert_close(head.weight, expected)

    def test_head_shards_full_and_preselected_rows_have_the_same_padding(self):
        for vocab, tp, dp, ep, rows in (
            (19, 1, 1, 1, 19),
            (19, 1, 2, 1, 24),
            (18, 2, 1, 1, 16),
            (19, 2, 1, 1, 16),
            (3, 2, 1, 2, 8),
        ):
            source = torch.arange(vocab * 4).reshape(vocab, 4).float() + 1
            padded = torch.nn.functional.pad(source, (0, 0, 0, rows * tp - vocab))
            for rank in range(tp):
                args = SimpleNamespace(
                    vocab_size=vocab,
                    dim=4,
                    dp_size=dp,
                    ep_size=ep,
                    lm_head_tp_size=tp,
                    lm_head_tp_rank=rank,
                )
                for presliced in (False, True):
                    with self.subTest(
                        vocab=vocab, tp=tp, rank=rank, presliced=presliced
                    ):
                        head = make_lm_head(args, torch.float32)
                        spec = head.checkpoint_read_spec("weight", tuple(source.shape))
                        value = (
                            spec.materialize(source) if spec and presliced else source
                        )
                        head.load_weights({"weight": value})
                        self.assertTrue(
                            torch.equal(head.weight, padded.chunk(tp)[rank])
                        )

    def config(self, **kwargs):
        return NewLoaderConfig(
            device="cpu",
            compute_dtype=torch.bfloat16,
            tp_size=2,
            tp_rank=1,
            **kwargs,
        )

    def test_cp2_replicates_attention_without_replicating_experts(self):
        native = SimpleNamespace(tp_size=1, tp_rank=0, ep_size=2, ep_rank=1)
        validate_load_groups(
            native, self.config(attn_tp_size=1, attn_tp_rank=0, ep_size=2, ep_rank=1)
        )

    def test_physical_tp_may_not_overwrite_cp_group(self):
        native = SimpleNamespace(tp_size=1, tp_rank=0, ep_size=2, ep_rank=1)
        with self.assertRaisesRegex(ValueError, "attention rank mismatch"):
            validate_load_groups(native, self.config(ep_size=2, ep_rank=1))

    def test_wrong_expert_rank_rejected_even_at_equal_world_size(self):
        native = SimpleNamespace(tp_size=2, tp_rank=1, ep_size=2, ep_rank=1)
        with self.assertRaisesRegex(ValueError, "expert rank mismatch"):
            validate_load_groups(native, self.config(ep_size=2, ep_rank=0))

    def test_nonidentity_and_redundancy_are_never_contiguous_ownership(self):
        native = SimpleNamespace(
            n_layers=2, n_routed_experts=4, n_physical_experts=4, commit_only=False
        )
        load = self.config(ep_size=2, ep_rank=1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "phy2log.json"
            model = SimpleNamespace(num_layers=2, phy2log_path=path)
            identity = [0, 1, 2, 3]
            path.write_text(json.dumps([identity, identity]))
            validate_v4_static_placement(native, model, load)
            path.write_text(json.dumps([identity, [2, 1, 0, 3]], indent=2))
            with self.assertRaisesRegex(ValueError, "layer 1 requires identity"):
                validate_v4_static_placement(native, model, load)
            native.n_physical_experts = 6
            with self.assertRaisesRegex(ValueError, "redundant experts"):
                validate_v4_static_placement(native, model, load)
            native.commit_only = True
            validate_v4_static_placement(native, model, load)


if __name__ == "__main__":
    unittest.main()
