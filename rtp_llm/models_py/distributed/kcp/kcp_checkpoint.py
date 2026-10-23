"""Stream-ordered checkpoint packing and direct cache publication."""
import os
import torch
import torch.distributed as dist
import triton
import triton.language as tl

from .kcp_bm_pipeline import gather_buffer


@triton.jit
def _pack_heads(SRC, DST, N: tl.constexpr, PAD: tl.constexpr,
                WORLD: tl.constexpr, STATE: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    count = WORLD * PAD * STATE
    slot = (x // STATE) % PAD
    owner = x // (STATE * PAD)
    value = tl.load(SRC + (slot * WORLD + owner) * STATE + x % STATE,
                    (x < count) & (slot < N), 0)
    tl.store(DST + x, value, x < count)


@triton.jit
def _unpack_heads(SRC, DST, INDEX, COUNT: tl.constexpr,
                  STATE: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    physical = tl.load(INDEX + x // STATE, x < COUNT * STATE, 0).to(tl.int64)
    value = tl.load(SRC + physical * STATE + x % STATE, x < COUNT * STATE, 0)
    tl.store(DST + x, value, x < COUNT * STATE)


def return_checkpoints(local, index, output, padded, group):
    """CP token/full-head states -> native packed page/TP-head states."""
    if os.environ.get('GLM53_KCP_PEER_CHECKPOINT', '0') == '1':
        from .kcp_peer_checkpoint import return_checkpoints as peer_return
        return peer_return(local, index, output, padded, group)
    world = dist.get_world_size(group)
    _, count, heads, key_dim, value_dim = local.shape
    if heads % world or output.numel() != index.numel() * (heads // world) * key_dim * value_dim:
        raise ValueError('Checkpoint layout does not match the caller buffer')
    shape = (padded, heads // world, key_dim, value_dim)
    outgoing = gather_buffer(group, shape, local.dtype, local.device, 'checkpoint_send')
    incoming = gather_buffer(group, shape, local.dtype, local.device, 'checkpoint_recv')
    state = (heads // world) * key_dim * value_dim
    _pack_heads[(triton.cdiv(outgoing.numel(), 1024),)](
        local, outgoing, N=count, PAD=padded, WORLD=world, STATE=state, BLOCK=1024)
    dist.all_to_all_single(incoming, outgoing, group=group)
    _unpack_heads[(triton.cdiv(output.numel(), 1024),)](
        incoming, output, index, COUNT=index.numel(), STATE=state, BLOCK=1024)


@triton.jit
def _pack_conv(SRC, DST, N: tl.constexpr, PAD: tl.constexpr,
               WORLD: tl.constexpr, HISTORY: tl.constexpr,
               WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    per_segment = HISTORY * 3 * WIDTH
    count = WORLD * PAD * per_segment
    channel = (x // WIDTH) % 3
    history = (x // (3 * WIDTH)) % HISTORY
    slot = (x // per_segment) % PAD
    owner = x // (PAD * per_segment)
    source = ((slot * HISTORY + history) * 3 + channel) * WORLD * WIDTH + owner * WIDTH + x % WIDTH
    value = tl.load(SRC + source, (x < count) & (slot < N), 0)
    tl.store(DST + x, value, x < count)


def exchange_conv_states(tails, padded, local_width, group):
    world = dist.get_world_size(group)
    count, history, _ = tails.shape
    shape = (padded, history, 3, local_width)
    outgoing = gather_buffer(group, shape, tails.dtype, tails.device, 'conv_publish_send')
    incoming = gather_buffer(group, shape, tails.dtype, tails.device, 'conv_publish_recv')
    _pack_conv[(triton.cdiv(outgoing.numel(), 512),)](
        tails, outgoing, N=count, PAD=padded, WORLD=world,
        HISTORY=history, WIDTH=local_width, BLOCK=512)
    dist.all_to_all_single(incoming, outgoing, group=group)
    return incoming


@triton.jit
def _store_conv(SRC, PHYSICAL, CHECKPOINT, SEQUENCE, PAGE, BLOCK_MAP, CACHE,
                BM_ROW, BM_COL, CACHE_BLOCK, CACHE_ROW, CACHE_COL,
                MAX_PAGES, BLOCKS, HISTORY: tl.constexpr, WIDTH: tl.constexpr,
                BLOCK: tl.constexpr):
    entry = tl.program_id(0).to(tl.int64)
    x = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    seq = tl.load(SEQUENCE + entry).to(tl.int64)
    page = tl.load(PAGE + entry).to(tl.int64)
    page_ok = (page >= 0) & (page < MAX_PAGES)
    block = tl.load(BLOCK_MAP + seq * BM_ROW + page * BM_COL, page_ok, -1).to(tl.int64)
    # Preserve the old conv publication's >=0 contract. SSM publication below
    # uses its existing >0 sentinel contract in store_kda_recurrent_checkpoints.
    valid = page_ok & (block >= 0) & (block < BLOCKS) & (x < HISTORY * WIDTH)
    checkpoint = tl.load(CHECKPOINT + entry).to(tl.int64)
    physical = tl.load(PHYSICAL + checkpoint).to(tl.int64)
    value = tl.load(SRC + physical * HISTORY * WIDTH + x, valid, 0)
    tl.store(CACHE + block * CACHE_BLOCK + (x // WIDTH) * CACHE_ROW + (x % WIDTH) * CACHE_COL,
             value, valid)


def publish_conv_states(incoming, physical, metadata, block_map, cache):
    """Gather/boolean-index/index_copy in one masked device-side scatter."""
    count = metadata.store_checkpoint_indices.numel()
    if not count:
        return
    _, history, width = cache.shape
    _store_conv[(count, triton.cdiv(history * width, 256))](
        incoming, physical, metadata.store_checkpoint_indices,
        metadata.store_sequence_indices, metadata.store_page_indices,
        block_map, cache, *block_map.stride(), *cache.stride(),
        block_map.shape[1], cache.shape[0], HISTORY=history, WIDTH=width, BLOCK=256)
