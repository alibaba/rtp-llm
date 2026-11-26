"""Shared real-rank configuration and checkpoint owners for the Gloo loading tests."""

from types import SimpleNamespace

import torch

from rtp_llm.models_py.distributed.rank_layout import RankLayout
from rtp_llm.models_py.model_loader import NewModelLoader
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


def runtime_config(rank, pp=2, tp=2, dp=1, ep=None, cp=False, ffn_replicated=False):
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
class ParallelCheckpointOwner(RtpModule):
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


class GlooCheckpointLoader(NewModelLoader):
    """Bind the actual Gloo groups allocated by this CPU launcher.

    The production launcher exposes its groups through collective_torch; no
    native config or CUDA process-group registry is synthesized in this test.
    """

    def _runtime_checkpoint_group(self, group):
        return self.groups[group]
