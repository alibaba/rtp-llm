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
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.read_spec import make_sharded_read_spec
from rtp_llm.models_py.registry import register_model


class _Runtime(SimpleNamespace):
    def get_attn_tp_size(self):
        return 1 if self.cp else self.tp_size

    def get_attn_tp_rank(self):
        return 0 if self.cp else self.tp_rank

    def get_ffn_tp_size(self):
        return 1 if self.ffn_replicated or self.cp else self.tp_size

    def get_ffn_tp_rank(self):
        return 0 if self.ffn_replicated or self.cp else self.tp_rank


def _config(rank, pp=2, tp=2, dp=1, ep=None, cp=False, ffn_replicated=False):
    world = pp * tp * dp
    layout = RankLayout(world, pp, dp, tp)
    coord = layout.coord_of(rank)
    ep = tp * dp if ep is None else ep
    return _Runtime(
        world_size=world,
        world_rank=rank,
        local_world_size=world,
        pp_size=pp,
        tp_size=tp,
        dp_size=dp,
        ep_size=ep,
        tp_rank=coord.tp,
        pp_rank=coord.pp,
        dp_rank=coord.dp,
        ep_rank=layout.ep_rank_of(rank, ep),
        cp=cp,
        ffn_replicated=ffn_replicated,
    )


class _Partition(RtpModule):
    def __init__(self, size, rank):
        super().__init__()
        self.size, self.rank = size, rank
        self.weight = torch.nn.Parameter(torch.empty(8 // size, 2))

    def checkpoint_read_spec(self, name, shape):
        return make_sharded_read_spec(name, shape, 0, self.size, self.rank)


@register_model("_p0_main_parallel_runtime")
class _Owner(RtpModule):
    def __init__(self, config, load):
        super().__init__()
        for name in ("attn", "ffn", "lm_head"):
            setattr(
                self,
                name,
                _Partition(
                    getattr(load, name + "_tp_size"), getattr(load, name + "_tp_rank")
                ),
            )
        if config.expert_num:
            self.experts = torch.nn.ModuleDict({str(load.ep_rank): _Partition(1, 0)})


class _GlooLoader(NewModelLoader):
    """Bind the actual Gloo groups allocated by this CPU launcher.

    The production launcher exposes its groups through collective_torch; no
    native config or CUDA process-group registry is synthesized in this test.
    """

    def _runtime_checkpoint_group(self, group):
        return self.groups[group]


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
                pc = _config(rank, cp=draft, ffn_replicated=True)
                directory = os.path.join(temp, f"{pc.pp_rank}-{experts}")
                for method in ("scratch", "fastsafetensors"):
                    load = NewLoaderConfig.from_runtime(
                        pc,
                        device="cpu",
                        compute_dtype=torch.float32,
                        load_method=method,
                    )
                    loader = _GlooLoader(
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
                                _Owner(loader.model_config, load),
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


class MainParallelRuntimeTest(unittest.TestCase):
    def test_runtime_role_coordinates(self):
        for rank in range(8):
            for cp, replicated in ((False, False), (True, False), (False, True)):
                pc = _config(rank, dp=2, cp=cp, ffn_replicated=replicated)
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
        pc = _config(3)
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
