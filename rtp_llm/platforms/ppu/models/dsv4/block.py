"""PPU HC normalization boundary."""

from rtp_llm.platforms.ppu.models.dsv4.impl.block import Block


class PpuBlock(Block):
    def _hc_pre_norm(self, hc, x, norm, *, tp_size, tp_rank, dbg_tag=None):
        return hc.pre_norm(x, norm, tp_size=tp_size, tp_rank=tp_rank, dbg_tag=dbg_tag)
