"""Length-bounded PageRR score restore for native K3.1 Decode selection."""

import torch
import triton
import triton.language as tl


@triton.jit
def _restore_visible(
    Received,
    QueryPositions,
    Complete,
    Ends,
    ROWS: tl.constexpr,
    SELECTION_ROWS: tl.constexpr,
    ROW_OFFSET: tl.constexpr,
    OWNER_WIDTH: tl.constexpr,
    SHARDS: tl.constexpr,
    PAGE: tl.constexpr,
    START: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    query_row = ROW_OFFSET + row
    query_position = tl.load(QueryPositions + query_row, query_row < ROWS, other=-1)
    capacity = OWNER_WIDTH * SHARDS
    length = tl.where(
        query_position >= 0,
        tl.minimum(tl.maximum(query_position + 1 - START * SHARDS, 0), capacity),
        0,
    )
    tl.store(Ends + row, length)
    lane = tl.arange(0, BLOCK)
    # Every owner chunk contains whole logical pages, so its union is the
    # contiguous global interval [START*SHARDS, (START+OWNER_WIDTH)*SHARDS).
    # A2A received owner-major rows; restore only the visible logical prefix.
    # The untouched capacity tail is safe only with the same Ends at TopK.
    for tile in tl.range(0, tl.cdiv(length, BLOCK)):
        column = tile * BLOCK + lane
        owner = column // PAGE % SHARDS
        owner_column = column // (PAGE * SHARDS) * PAGE + column % PAGE
        source = (
            owner.to(tl.int64) * SELECTION_ROWS + row
        ) * OWNER_WIDTH + owner_column
        score = tl.load(Received + source, column < length, other=-float("inf"))
        tl.store(Complete + row * capacity + column, score, column < length)


def restore_visible_decode_scores(
    received, query_positions, complete, ends, *, shard_rank, shards, page, start
):
    """Preserve fixed capacity/stride, write only each query's live prefix.

    The returned score columns are in true global-position order. Callers must
    pass Ends to a bounded selector; full-row readers cannot consume the tail.
    """
    rows, capacity = complete.shape
    if (
        received.dtype != torch.float32
        or complete.dtype != torch.float32
        or query_positions.dtype != torch.int32
        or ends.dtype != torch.int32
        or not all(
            t.is_cuda and t.is_contiguous()
            for t in (received, query_positions, complete, ends)
        )
        or ends.shape != (rows,)
        or capacity % shards
        or received.shape != (rows * shards, capacity // shards)
        or start % page
        or (capacity // shards) % page
    ):
        raise ValueError("Bounded Decode restore requires whole PageRR owner pages")
    if rows:
        _restore_visible[(rows,)](
            received,
            query_positions,
            complete,
            ends,
            query_positions.numel(),
            rows,
            shard_rank * rows,
            capacity // shards,
            shards,
            page,
            start,
            4096,
            num_warps=4,
        )
