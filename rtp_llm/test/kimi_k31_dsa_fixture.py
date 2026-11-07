"""Shared K3.1 DSA decode fixtures (FP8 Indexer cache, BF16 MLA cache): procedural history, Page-RR geometry and a scattered page pool.

History values are a hash of (request seed, global position, feature, salt), so a reference can
regenerate any token's Indexer key or MLA latent on demand instead of holding a global copy: a
1M-token request needs no per-rank global tensor. The page pool places every request's owner pages
at random ids of a serving-sized cache, so slots and cache offsets reach the magnitudes a long-
running server sees (byte and element offsets beyond 2**31).
"""

import math

import torch

PAGE, KERNEL_PAGE = 1024, 128
INDEX_DIM, LATENT, ROPE = 128, 512, 64
INDEX_ENTRY = INDEX_DIM + 4  # FP8 key bytes, then one FP32 scale per token
INDEX_SALT, LATENT_SALT = 0x1D3, 0x7A5
_M32 = 0xFFFFFFFF


def _mix(x):
    """32-bit integer finalizer (lowbias32) on int64 tensors holding 32-bit values."""
    x = x & _M32
    x = x ^ (x >> 16)
    x = (x * 0x7FEB352D) & _M32
    x = x ^ (x >> 15)
    x = (x * 0x846CA68B) & _M32
    return x ^ (x >> 16)


def procedural(seed, positions, width, salt, scale, chunk=1 << 14):
    """Deterministic values [n, width] float32 of std `scale` for (seed, position, feature, salt)."""
    positions = positions.to(torch.int64)
    out = torch.empty(positions.numel(), width, dtype=torch.float32, device=positions.device)
    features = torch.arange(width, dtype=torch.int64, device=positions.device)[None, :]
    base = (seed * 0x9E3779B1 + salt * 0x85EBCA77) & _M32
    for start in range(0, positions.numel(), chunk):
        p = positions[start : start + chunk, None]
        x = _mix(_mix(p ^ base) + features * 0x632BE5AB)
        out[start : start + chunk] = (x.to(torch.float32) * (2.0 / 2**32) - 1.0) * (scale * math.sqrt(3))
    return out


def index_keys(seed, positions):
    """BF16-rounded Indexer keys [n, 128] (fp32) of a request's global positions."""
    return procedural(seed, positions, INDEX_DIM, INDEX_SALT, 0.1).bfloat16().float()


def fp8_index_keys(seed, positions):
    """The Indexer keys as the production writer stores them: (E4M3 values as fp32 [n, 128], scale [n]).

    indexer_k_quant_and_cache with "ue8m0": scale = 2**ceil(log2(max(amax, 1e-4) / 448)), value / scale
    converted to E4M3 (round to nearest even).
    """
    keys = index_keys(seed, positions)
    amax = keys.abs().amax(-1).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return (keys / scale[:, None]).to(torch.float8_e4m3fn).float(), scale


def latents(seed, positions):
    """BF16-rounded MLA latent+rope [n, 576] (fp32) of a request's global positions."""
    return procedural(seed, positions, LATENT + ROPE, LATENT_SALT, 0.1).bfloat16().float()


def local_to_global(local, tp, rank):
    """Global positions of this rank's Page-RR local positions."""
    return ((local // PAGE) * tp + rank) * PAGE + local % PAGE


def local_tokens(context, tp, rank):
    """How many of a request's first `context` positions this rank owns."""
    pages, tail = divmod(context, PAGE)
    full = pages // tp + (1 if rank < pages % tp else 0)
    return full * PAGE + (tail if pages % tp == rank else 0)


def owner_columns(context, tp, rank):
    return math.ceil(local_tokens(context, tp, rank) / KERNEL_PAGE)


def group_columns(contexts, tp):
    """Page-table width shared by every rank of the group: the widest owner share of the batch.

    Ranks must pass equal shapes: a rank-dependent width gives each rank its own Triton
    specializations, and a first-time module load (cuModuleLoadData) waits for the device's
    in-flight kernels, which can be collectives waiting for a peer stuck in its own load.
    """
    return max(max(owner_columns(c, tp, rank) for rank in range(tp)) for c in contexts)


def max_owner_columns(max_seq_len, tp):
    return math.ceil(math.ceil(max_seq_len / PAGE) / tp) * (PAGE // KERNEL_PAGE)


class PagePool:
    """This rank's serving-sized MLA latent and FP8 Indexer caches; page 0 stays reserved."""

    def __init__(self, pages, device, seed):
        self.pages = pages
        self.latent = torch.zeros(pages, KERNEL_PAGE, LATENT + ROPE, dtype=torch.bfloat16, device=device)
        self.index = torch.zeros(pages, KERNEL_PAGE, INDEX_ENTRY, dtype=torch.uint8, device=device)
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def allocate(self, columns):
        """Distinct random page ids for per-request column counts: a list of int32 tensors."""
        total = sum(columns)
        if total >= self.pages:
            raise ValueError(f"page pool of {self.pages} pages cannot hold {total}")
        ids = torch.randperm(self.pages - 1, generator=self.generator, device=self.latent.device)
        ids = (ids[:total] + 1).to(torch.int32)
        return list(torch.split(ids, columns))

    def write_history(self, seed, pages, context, tp, rank, chunk_pages=256):
        """Procedural keys and latents of a request's positions < context on its owner pages.

        Keys go through the production FP8 writer; latents are stored as BF16.
        """
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        for first in range(0, pages.numel(), chunk_pages):
            block = pages[first : first + chunk_pages].long()
            local = torch.arange(
                first * KERNEL_PAGE, (first + block.numel()) * KERNEL_PAGE, device=block.device
            )
            positions = local_to_global(local, tp, rank)
            live = (positions < context)[:, None]
            slots = (block[:, None] * KERNEL_PAGE + torch.arange(KERNEL_PAGE, device=block.device)).view(-1)
            live_slots = slots[live.view(-1)]
            if live_slots.numel():
                rtp_llm_ops.indexer_k_quant_and_cache(
                    index_keys(seed, positions[live.view(-1)]).bfloat16(),
                    self.index, live_slots, INDEX_DIM, "ue8m0",
                )
            values = latents(seed, positions) * live
            self.latent[block] = values.bfloat16().view(block.numel(), KERNEL_PAGE, LATENT + ROPE)

    def slot_to_global(self, requests):
        """Map from owner slot to (request, global position) for [(pages, tp, rank)] per request.

        Returns int64 [pages * 128] holding request * 2**32 + global position, or -1.
        """
        table = torch.full((self.pages * KERNEL_PAGE,), -1, dtype=torch.int64, device=self.latent.device)
        for request, (pages, tp, rank) in enumerate(requests):
            local = torch.arange(pages.numel() * KERNEL_PAGE, device=pages.device)
            slots = pages.long()[:, None] * KERNEL_PAGE + torch.arange(KERNEL_PAGE, device=pages.device)
            table[slots.reshape(-1)] = (request << 32) + local_to_global(local, tp, rank)
        return table
