"""CPU only: real PP/TP group membership, safetensors reads and stage-only drafts."""

import os
import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from rtp_llm.models_py.distributed.rank_layout import Group, RankLayout
from rtp_llm.models_py.layers.embedding import ParallelLMHead, VocabParallelEmbedding
from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewModelLoader,
    _ExpertRangeFilter,
)
from rtp_llm.models_py.parallel_checkpoint_test_support import (
    GlooCheckpointLoader,
    ParallelCheckpointOwner,
    runtime_config,
)


def _worker(rank, temp):
    dist.init_process_group(
        "gloo",
        init_method="file://" + os.path.join(temp, "rdzv"),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=45),
    )
    try:
        layout = RankLayout(4, 2, 1, 2)
        own = None
        for ranks in layout.groups(Group.STAGE):
            pg = dist.new_group(ranks, backend="gloo", timeout=timedelta(seconds=45))
            if rank in ranks:
                own = pg
        wrong_stage = None
        for ranks in layout.groups(Group.PP):
            lane = dist.new_group(ranks, backend="gloo", timeout=timedelta(seconds=45))
            if rank in ranks:
                wrong_stage = lane
        # An ordinary target runs on all four ranks; the draft only runs on the
        # tail. Its two other WORLD peers wait at a WORLD barrier, not in FST.
        for draft, experts in ((False, 0), (False, 2), (True, 2)):
            if not draft or rank >= 2:
                pc = runtime_config(rank, cp=draft, ffn_replicated=True)
                directory = os.path.join(temp, f"{pc.pp_rank}-{experts}")
                for method in ("scratch", "fastsafetensors"):
                    load = NewLoaderConfig.from_runtime(
                        pc,
                        device="cpu",
                        compute_dtype=torch.float32,
                        load_method=method,
                    )
                    loader = GlooCheckpointLoader(
                        SimpleNamespace(
                            model_type="_p0_main_parallel_runtime",
                            is_mtp=draft,
                            expert_num=experts,
                        ),
                        load,
                        model_path=directory,
                    )
                    loader.groups = {Group.TP: own, Group.STAGE: wrong_stage}
                    try:
                        loader._fastsafetensors_stage_ep_group()
                    except RuntimeError as exc:
                        assert "owning PP stage" in str(exc)
                    else:
                        raise AssertionError(
                            "accepted a different stage's checkpoint peers"
                        )
                    loader.groups[Group.STAGE] = own
                    if draft and method == "fastsafetensors":
                        paths = (
                            [os.path.join(directory, "model.safetensors")]
                            if pc.tp_rank == 0
                            else []
                        )
                        try:
                            loader._select_load_method(
                                load.load_method,
                                ParallelCheckpointOwner(loader.model_config, load),
                                paths,
                            )
                        except RuntimeError as exc:
                            assert "rank 1: checkpoint shard selection is empty" in str(
                                exc
                            )
                        else:
                            raise AssertionError(
                                "peer prerequisite failure was not propagated"
                            )
                    assert loader._fastsafetensors_stage_ep_group() is own
                    model = loader.load()
                    assert loader._selected_load_method.value == method
                    for i, name in enumerate(("attn", "ffn", "lm_head")):
                        expected = (
                            torch.arange(16, dtype=torch.float32).view(8, 2)
                            + 100 * pc.pp_rank
                            + 20 * i
                        )
                        leaf = getattr(model, name)
                        torch.testing.assert_close(
                            leaf.weight, expected.chunk(leaf.size)[leaf.rank]
                        )
                    if experts:
                        expected_expert = (
                            torch.arange(16, dtype=torch.float32).view(8, 2)
                            + 1000 * pc.pp_rank
                            + 300 * pc.ep_rank
                        )
                        torch.testing.assert_close(
                            model.experts[str(pc.ep_rank)].weight, expected_expert
                        )
            dist.barrier()
    finally:
        dist.destroy_process_group()


def _vocab_and_stacked_worker(rank, temp):
    dist.init_process_group(
        "gloo",
        init_method="file://" + os.path.join(temp, "rdzv"),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        loader = NewModelLoader(
            {"model_type": "_p0_main_parallel_runtime"},
            NewLoaderConfig(
                device="cpu", tp_size=2, tp_rank=rank, ep_size=2, ep_rank=rank
            ),
            model_path=temp,
        )
        embed = VocabParallelEmbedding(64, 2, 2, rank, torch.float32)
        ids = tuple(range(32))
        head = ParallelLMHead(64, 2, 2, rank, torch.float32, output_vocab_ids=ids)
        embed.tie_output_head(head)
        source = torch.arange(128, dtype=torch.float32).reshape(64, 2)
        # Only rank 1 needs extra embedding rows. Catch divergent plans before
        # a regression could strand either worker inside FST.
        spec = embed.checkpoint_read_spec("weight", tuple(source.shape))
        scatter = [None, None]
        dist.all_gather_object(scatter, spec.shard_dim)
        assert scatter[0] == scatter[1], scatter

        expected = {"weight"}
        for projection in ("down_proj", "gate_up_proj"):
            for expert in range(2 * rank, 2 * rank + 2):
                expected.add(f"experts.{expert}.{projection}.weight")
        seen = set()
        files = [os.path.join(temp, f"model-{i}.safetensors") for i in range(2)]
        for name, value in loader._fastsafetensors_weights_iterator(
            files,
            "cpu",
            None,
            _ExpertRangeFilter(4, 2, rank),
            embed.checkpoint_read_spec,
        ):
            assert name in expected and name not in seen, name
            seen.add(name)
            if name == "weight":
                embed.load_weights({name: value})
                torch.testing.assert_close(embed.weight, source.chunk(2)[rank])
                torch.testing.assert_close(
                    head.weight, source[16 * rank : 16 * rank + 16]
                )
            else:
                _, expert, projection, _ = name.split(".")
                offset = 100 if projection == "down_proj" else 200
                # Two distinct rows per rank, in two different physical files.
                torch.testing.assert_close(
                    value, torch.arange(3) + 3 * int(expert) + offset
                )
        assert seen == expected, (seen, expected)
    finally:
        dist.destroy_process_group()


class MainParallelRuntimeTest(unittest.TestCase):
    def test_pruned_tied_vocab_and_stacked_row_order_native_fst(self):
        with tempfile.TemporaryDirectory() as temp:
            experts = torch.arange(12).reshape(4, 3)
            save_file(
                {
                    "weight": torch.arange(128, dtype=torch.float32).reshape(64, 2),
                    "experts.down_proj": experts + 100,
                },
                os.path.join(temp, "model-0.safetensors"),
            )
            save_file(
                {"experts.gate_up_proj": experts + 200},
                os.path.join(temp, "model-1.safetensors"),
            )
            torch.multiprocessing.spawn(_vocab_and_stacked_worker, (temp,), nprocs=2)

    def test_runtime_role_coordinates(self):
        for rank in range(8):
            for cp, replicated in ((False, False), (True, False), (False, True)):
                pc = runtime_config(rank, dp=2, cp=cp, ffn_replicated=replicated)
                c = NewLoaderConfig.from_runtime(pc, device="cpu")
                self.assertIs(c.parallelism_config, pc)
                self.assertEqual(c.ep_rank, rank % 4)
                self.assertEqual((c.lm_head_tp_size, c.lm_head_tp_rank), (2, rank % 2))
                self.assertEqual(
                    (c.attn_tp_size, c.attn_tp_rank), (1, 0) if cp else (2, rank % 2)
                )
                self.assertEqual(
                    (c.ffn_tp_size, c.ffn_tp_rank),
                    (1, 0) if cp or replicated else (2, rank % 2),
                )

    def test_reject_stale_or_overridden_rank(self):
        pc = runtime_config(3)
        for field in ("tp_rank", "dp_rank", "pp_rank", "ep_rank"):
            saved = getattr(pc, field)
            setattr(pc, field, 1 - saved)
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "disagree"
            ):
                NewLoaderConfig.from_runtime(pc)
            setattr(pc, field, saved)
        with self.assertRaises(TypeError):
            NewLoaderConfig.from_runtime(pc, tp_rank=0)

    def test_stage_only_draft_actual_gloo_and_fst(self):
        with tempfile.TemporaryDirectory() as temp:
            for stage in range(2):
                for experts in (0, 2):
                    directory = os.path.join(temp, f"{stage}-{experts}")
                    os.mkdir(directory)
                    values = {
                        name
                        + ".weight": torch.arange(16, dtype=torch.float32).view(8, 2)
                        + stage * 100
                        + 20 * i
                        for i, name in enumerate(("attn", "ffn", "lm_head"))
                    }
                    values.update(
                        {
                            f"experts.{i}.weight": torch.arange(
                                16, dtype=torch.float32
                            ).view(8, 2)
                            + 1000 * stage
                            + 300 * i
                            for i in range(experts)
                        }
                    )
                    save_file(values, os.path.join(directory, "model.safetensors"))
            torch.multiprocessing.start_processes(
                _worker, (temp,), nprocs=4, start_method="spawn"
            )


if __name__ == "__main__":
    unittest.main()
