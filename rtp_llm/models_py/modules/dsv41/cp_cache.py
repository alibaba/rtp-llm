"""Address arithmetic for DSV4.1 CP cache pages and rank-major packets."""

from __future__ import annotations

try:
    import triton.language as tl
    from triton import jit
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    jit = None


def round_robin_owner(position, owner_span: tl.constexpr, cp_size: tl.constexpr):
    """Return the rank owning the physical block that contains position."""
    return position // owner_span % cp_size


def round_robin_page(
    position, owner_span: tl.constexpr, page_span: tl.constexpr, cp_size: tl.constexpr
):
    """Map a logical position to its owner-local page column and page offset.

    The owner span is a whole number of pages. Both spans and the position
    must use the same unit, either raw tokens or compressed entries.
    """
    owner_block = position // owner_span
    column = owner_block // cp_size * (owner_span // page_span)
    column += position % owner_span // page_span
    return column, position % page_span


def ring_coordinates(position, page_span, ring_entries):
    """Return the request's logical page column and its cyclic row offset."""
    return position // page_span, position % ring_entries


def local_slice_offset(logical_offset, rank, slice_size):
    """Offsets in [0, slice_size) are writable by this rank."""
    return logical_offset - rank * slice_size


def rank_major_slice_offset(block, logical_offset, blocks, slice_size):
    """Address a byte or element in all-gathered equal-size block slices."""
    rank = logical_offset // slice_size
    return (rank * blocks + block) * slice_size + logical_offset % slice_size


def swa_prefix_start(prefix, window, *, maximum=max):
    """First cached row visible to the first new query."""
    return maximum(0, prefix - window + 1)


def page_interval(start, stop, page_span):
    """Page columns covering the half-open token interval [start, stop)."""
    return start // page_span, (stop + page_span - 1) // page_span


if jit is not None:
    _round_robin_owner_jit = jit(round_robin_owner)
    _round_robin_page_jit = jit(round_robin_page)
    ring_coordinates_jit = jit(ring_coordinates)
    local_slice_offset_jit = jit(local_slice_offset)
    rank_major_slice_offset_jit = jit(rank_major_slice_offset)

    @jit
    def round_robin_coordinates_jit(
        position, owner_span: tl.constexpr, page_span: tl.constexpr, cp_size: tl.constexpr
    ):
        owner = _round_robin_owner_jit(position, owner_span, cp_size)
        column, offset = _round_robin_page_jit(position, owner_span, page_span, cp_size)
        return owner, column, offset
