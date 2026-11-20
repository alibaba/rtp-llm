# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent warp-specialized fused replay verify (KdaFusedWS): KdaFused's math
(deferred RecoverSSM: the replay of the pending accepted tokens, the checkpoint
write and the chunked verify) on KdaWS's schedule.

One CTA per SM walks every `ctas`-th (request, head) item (strided, see kernel):

- two **prep warpgroups** (thread c = channel c; warpgroup w takes items w, w + 2,
  ...) stream each item in (projection rows and conv window by cp.async, the
  item's old records into their staging), run the convolution, SiLU, norms, F_B
  gate projection and decays, and fill one of four operand blocks: the chunk
  vectors Gamma_t k_t and Gamma_t q_t, the replay keys kn_t P_t and D_n, the old
  corrections, v, beta, the output gate and the q x q pair sums;
- the **scan group** (8 warps) takes the item's state page (bulk copy) into
  registers (thread (g, l): rows g + 32 r, keys 4 (l + 8 c) + [0, 4)), replays and
  stores the checkpoint, forms the 2q row dot products, solves each row's q x q
  system (thread = row), writes the corrections and runs the gated RMSNorm.

Synchronization as KdaWS (state_full / state_empty / ops_full[4] / done[4])."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from .primitives import (
    acquire_tensor_map,
    alloc,
    at,
    commit_bulk_group,
    copy_g2s,
    div_full,
    fence_async_shared,
    fma,
    fma2,
    load_bulk,
    load_shared_f32x4,
    load_shared_u32x2,
    load_tma_tile,
    load_weight_bf16x4,
    mma_into,
    mul_rn,
    reciprocal,
    scatter_reduce,
    set_max_registers,
    sqrt_approx,
    store_bulk,
    store_tma_tile,
    sync_threads,
    triton_exp,
    triton_sigmoid,
    wait,
    wait_bulk_reads,
    wait_bulk_writes,
    zeros,
)

K = 128
GROUP_THREADS = 256
PREP_THREADS = 128
THREADS = GROUP_THREADS + 2 * PREP_THREADS
ROWS = 4  # state rows per scan thread: g + 32 r
LANES = 8  # scan threads per row
CHUNKS = 4  # 4-key chunks per thread and row
SCAN_REGISTERS, PREP_REGISTERS = 168, 88
PAGE_BYTES = K * K * 4
COPY_BYTES = 16384
TMA_TILE_DIM = 32
TMA_TILE_ELEMENTS = TMA_TILE_DIM * TMA_TILE_DIM
TMA_VALUE_STRIPES = K // TMA_TILE_DIM
SMEM_LIMIT = 227 * 1024
META = 8  # live at 0, head/token/n/record at 2..5, padded to 8


# Extracted from vLLM d07b7db9785564549d163937b5d5d9c04c76d626.
# vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/kda/fused.py

def pair_count(tokens):
    """A (s < t) and B (s <= t) pair products: tokens^2."""
    return tokens * tokens


def a_index(t, s):
    return t * (t - 1) // 2 + s


def b_index(tokens, t, s):
    return tokens * (tokens - 1) // 2 + t * (t + 1) // 2 + s

def operand_bytes(tokens, rq):
    """One operand block: chunk vectors (2q, the decays and later the outputs
    alias them), replay keys (rq) and D_n, old corrections (rq), v (FP32), the
    output gate (BF16), beta, pair sums, meta."""
    p2 = 1 << (pair_count(tokens) - 1).bit_length()
    floats = (2 * tokens + rq + 1 + rq + tokens) * K + 8 + p2
    return floats * 4 + tokens * K * 2 + META * 4


def staging_bytes(tokens, rq, window, conv_size):
    rows = 5 * tokens * K * 2  # x (3 parts), gate, F_A
    conv = 3 * K * window * conv_size
    records = rq * 2 * K * 2 + rq * K * 4  # old key / gate, old corrections
    return rows + conv + records


def shared_bytes(tokens, rq, window, conv_size):
    staging = staging_bytes(tokens, rq, window, conv_size)
    sums = 2 * 64 * 4 * 4 + 2 * 8 * 4 * 4
    dots = K * (2 * tokens + 1) * 4
    blocks = 4 * operand_bytes(tokens, rq)
    return PAGE_BYTES + blocks + 2 * staging + sums + dots + 12 * 8 + 1024


class KdaFusedWS:
    """KdaFused's operands and semantics on `ctas` persistent CTAs."""

    registers = (SCAN_REGISTERS, PREP_REGISTERS)

    def __init__(self, heads, tokens, window, ctas, *, state_tma=False):
        assert 2 <= tokens <= 8
        self.state_tma = state_tma
        self.heads, self.tokens = heads, tokens
        self.window, self.ctas = window, ctas

    @cute.jit
    def __call__(self, *args):
        self.kernel(args[:-1]).launch(
            grid=(self.ctas, 1, 1),
            block=(THREADS, 1, 1),
            stream=args[-1],
            use_pdl=True,
            min_blocks_per_mp=1,
        )

    # ---------------------------------------------------------------- helpers
    @cute.jit
    def item_info(self, args, item):
        """(head, request, state slot, record slot, pending n, window offset)."""
        slots, record_rows, pending = args[7], args[19], args[20]
        RQ: cutlass.Constexpr = args[17].shape[2]
        head, request = item % self.heads, item // self.heads
        slot = Int32(slots[request, 0])
        rec = Int32(record_rows[request])
        n = Int32(0)
        if rec >= 0:
            n = Int32(pending[rec])
        if n > RQ:
            n = Int32(RQ)
        if n < 0:
            n = Int32(0)
        offset = Int32(0)
        if n > 0:
            offset = n - 1
        return head, request, slot, rec, n, offset

    @cute.jit
    def stage_conv(self, args, convs, p, head, conv_slot):
        conv_state = args[2]
        dim: cutlass.Constexpr = self.heads * K
        size: cutlass.Constexpr = conv_state.element_type.width // 8
        per: cutlass.Constexpr = 16 // size
        # The launch encodes zero as padding and record ID + 1 as live.
        page = conv_state[Int64(conv_slot - 1), None, None]
        part_granules: cutlass.Constexpr = K * self.window // per
        for part in cutlass.range_constexpr(3):
            base = at(page, part * dim + head * K, 0)
            for j in cutlass.range_constexpr(-(-part_granules // PREP_THREADS)):
                k = p + PREP_THREADS * j
                if k < part_granules:
                    copy_g2s(at(convs, part, per * k), base + per * k)

    @cute.jit
    def stage_rows(self, args, xs, gs, fas, p, head, request):
        x, f_a, gate = args[0], args[3], args[10]
        T: cutlass.Constexpr = self.tokens
        dim: cutlass.Constexpr = self.heads * K
        token = request * T
        k, kind = p % 16, p // 16
        if kind < 5:
            for t in cutlass.range_constexpr(T):
                if kind < 3:
                    source = at(x, token + t, kind * dim + head * K + 8 * k)
                    copy_g2s(at(xs, t, kind, 8 * k), source)
                elif kind == 3:
                    copy_g2s(at(gs, t, 8 * k), at(gate, token + t, head * K + 8 * k))
                else:
                    copy_g2s(at(fas, t, 8 * k), at(f_a, token + t, 8 * k))

    @cute.jit
    def stage_records(self, args, kgs, us, p, head, rec):
        """cp.async an item's old records: key / raw gate [RQ, 2K] (BF16) and the
        corrections [RQ, K] (FP32), 32 granules per token each."""
        corrections, key_gates = args[17], args[18]
        RQ: cutlass.Constexpr = corrections.shape[2]
        for j in cutlass.range_constexpr(-(-(RQ * 32) // PREP_THREADS)):
            g = p + PREP_THREADS * j
            if g < RQ * 32:
                t, k = g // 32, g % 32
                copy_g2s(at(kgs, t, 8 * k), at(key_gates, rec, head, t, 8 * k))
                copy_g2s(at(us, t, 4 * k), at(corrections, rec, head, t, 4 * k))

    @cute.jit
    def stage(self, args, staging, p, item):
        convs, xs, gs, fas, _, _, kgs, us = staging
        head, request, slot, rec, _, _ = self.item_info(args, item)
        if slot > 0:
            self.stage_conv(args, convs, p, head, slot)
            self.stage_rows(args, xs, gs, fas, p, head, request)
            if rec >= 0:
                self.stage_records(args, kgs, us, p, head, rec)

    @cute.jit
    def block_sums(self, values, sums, p, count: cutlass.Constexpr):
        lane, warp = p % 32, p // 32
        total, selected = scatter_reduce(values, lane, count)
        if lane % (32 // count) == 0:
            sums[selected, warp] = total

    @staticmethod
    def quad_sum(r):
        return (r[0] + r[2]) + (r[1] + r[3])

    @cute.jit
    def silu(self, value):
        silu = div_full(value, Float32(1) + triton_exp(-value))
        return Float32(BFloat16(silu))

    @cute.jit
    def decay(self, a, raw_g, dt, lower_bound):
        """`a` = exp(A_log)."""
        gate = lower_bound * triton_sigmoid(a * (raw_g + dt))
        return triton_exp(gate)

    # ----------------------------------------------------------------- kernel
    @cute.kernel
    def kernel(self, args):
        conv_state, output, corrections = args[2], args[9], args[17]
        T: cutlass.Constexpr = self.tokens
        H: cutlass.Constexpr = self.heads
        RQ: cutlass.Constexpr = corrections.shape[2]
        P2: cutlass.Constexpr = 1 << (pair_count(T) - 1).bit_length()
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        requests: cutlass.Constexpr = output.shape[0] // T
        total_items: cutlass.Constexpr = H * requests
        # CTA c walks items c, c + ctas, ... (strided: CTAs running side by side
        # take neighbouring items, so different heads and state pages, instead of
        # all taking item j of equal contiguous ranges: with ranges a multiple of
        # the head count every CTA reads the same head's F_B rows and the same
        # page offset at once).
        first, stride = Int32(cta), Int32(self.ctas)
        count = (Int32(total_items) - first + stride - 1) // stride

        smem = cutlass.utils.SmemAllocator()
        num_pages: cutlass.Constexpr = 2 if self.state_tma else 1
        pages = alloc(smem, Float32, num_pages * K * K, align=1024)
        conv_type: cutlass.Constexpr = conv_state.element_type
        xv_s = alloc(
            smem, Float32, (4, 2 * T, K)
        )  # decays -> Gamma k, Gamma q; outputs
        kp_s = alloc(smem, Float32, (4, RQ, K))
        dn_s = alloc(smem, Float32, (4, K))
        u_s = alloc(smem, Float32, (4, RQ, K))
        v_s = alloc(smem, Float32, (4, T, K))
        g_s = alloc(smem, BFloat16, (4, T, K))
        beta_s = alloc(smem, Float32, (4, 8))
        pr_s = alloc(smem, Float32, (4, P2))
        meta = alloc(smem, Int32, (4, META))
        convs = alloc(smem, conv_type, (2, 3, K * self.window), align=16)
        xs = alloc(smem, BFloat16, (2, T, 3, K), align=16)
        gs = alloc(smem, BFloat16, (2, T, K), align=16)
        fas = alloc(smem, BFloat16, (2, T, K), align=16)
        kgs = alloc(smem, BFloat16, (2, RQ, 2 * K), align=16)
        us = alloc(smem, Float32, (2, RQ, K), align=16)
        sums = alloc(smem, Float32, (2, 32, 4))
        psums = alloc(smem, Float32, (2, 64, 4))
        out_sums = alloc(smem, Float32, (2, 8, 4))
        dots = alloc(smem, Float32, (K, 2 * T + 1))
        bars = alloc(smem, cutlass.Uint64, 12, align=8).iterator
        state_full, state_empty, ops_full, done = bars, bars + 2, bars + 4, bars + 8

        if tid == 0:
            for page_index in cutlass.range_constexpr(num_pages):
                cute.arch.mbarrier_init(state_full + page_index, 1)
                cute.arch.mbarrier_init(state_empty + page_index, GROUP_THREADS)
            for blk in cutlass.range_constexpr(4):
                cute.arch.mbarrier_init(ops_full + blk, PREP_THREADS)
                cute.arch.mbarrier_init(done + blk, GROUP_THREADS)
            cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        ops = (xv_s, kp_s, dn_s, u_s, v_s, g_s, beta_s, pr_s, meta)

        if tid >= GROUP_THREADS:
            set_max_registers(self.registers[1], increase=False)
            w = (tid - GROUP_THREADS) // PREP_THREADS
            p = (tid - GROUP_THREADS) % PREP_THREADS
            staging = (
                convs[w, None, None], xs[w, None, None, None], gs[w, None, None],
                fas[w, None, None], sums[w, None, None], psums[w, None, None],
                kgs[w, None, None], us[w, None, None],
            )  # fmt: skip
            self.prep(args, w, p, first, stride, count, staging, ops, ops_full, done)
        else:
            set_max_registers(self.registers[0], increase=True)
            self.scan(args, tid, first, stride, count, pages, ops, dots, out_sums,
                      state_full, state_empty, ops_full, done)  # fmt: skip

    @cute.jit
    def transfer_tma_page(self, page, tensor_map, slot, head, barrier, load: cutlass.Constexpr):
        # A slab is 32 FP32 values by 32 key groups. Consecutive V stripes
        # with a fixed key residue are placed next to one another in shared.
        for tile in cutlass.range_constexpr(K * K // TMA_TILE_ELEMENTS):
            key_residue = tile // TMA_VALUE_STRIPES
            value_start = (tile % TMA_VALUE_STRIPES) * TMA_TILE_DIM
            coord = (value_start, key_residue, 0, head, slot)
            tile_pointer = page.iterator + tile * TMA_TILE_ELEMENTS
            if cutlass.const_expr(load):
                load_tma_tile(tile_pointer, tensor_map, coord, barrier)
            else:
                store_tma_tile(tile_pointer, tensor_map, coord)
        if cutlass.const_expr(not load):
            commit_bulk_group()

    @cute.jit
    def load_page(self, args, page, item, state_full):
        head, request = item % self.heads, item // self.heads
        slot = args[16][request, 0]
        if slot > 0:
            cute.arch.mbarrier_arrive_and_expect_tx(state_full, PAGE_BYTES)
            if cutlass.const_expr(self.state_tma):
                self.transfer_tma_page(page, args[22].iterator, slot, head, state_full, True)
            else:
                source = args[8][Int64(slot), None, None, None]
                for i in cutlass.range_constexpr(PAGE_BYTES // COPY_BYTES):
                    rows: cutlass.Constexpr = COPY_BYTES // (K * 4)
                    load_bulk(
                        page.iterator + i * rows * K,
                        at(source, head, i * rows, 0),
                        COPY_BYTES,
                        state_full,
                    )
        else:
            cute.arch.mbarrier_arrive_and_expect_tx(state_full, 0)

    @cute.jit
    def save_page(self, state, page, slot, head):
        rows: cutlass.Constexpr = COPY_BYTES // (K * 4)
        for i in cutlass.range_constexpr(PAGE_BYTES // COPY_BYTES):
            store_bulk(
                at(state, Int64(slot), head, i * rows, 0),
                page.iterator + i * rows * K,
                COPY_BYTES,
            )

    @cute.jit
    def state_page_index(self, key, value):
        """Shared offset of a checkpoint element at global [K,V]."""
        if cutlass.const_expr(self.state_tma):
            key_group = key // 4
            tile = (key % 4) * TMA_VALUE_STRIPES + value // TMA_TILE_DIM
            swizzled_value = (value % TMA_TILE_DIM) ^ ((key_group & 7) * 4)
            return tile * TMA_TILE_ELEMENTS + key_group * TMA_TILE_DIM + swizzled_value
        else:
            return key * K + value

    @cute.jit
    def checkpoint_page(self, pages, index):
        return cute.make_tensor(
            pages.iterator + index * (K * K),
            cute.make_layout(K * K),
        )

    # -------------------------------------------------------------- scan role
    @cute.jit
    def scan(self, args, t_g, first, stride, count, pages, ops, dots, out_sums, state_full,  # noqa: E501
             state_empty, ops_full, done):  # fmt: skip
        xv_s, kp_s, dn_s, u_s, v_s, g_s, beta_s, pr_s, meta = ops
        state, corrections = args[8], args[17]
        T: cutlass.Constexpr = self.tokens
        RQ: cutlass.Constexpr = corrections.shape[2]
        lane_in, base_row = t_g % LANES, t_g // LANES
        nw = args[11][t_g % K]
        if cutlass.const_expr(self.state_tma):
            if t_g == 0:
                acquire_tensor_map(args[22].iterator)
        initial_page = self.checkpoint_page(pages, Int32(0))
        if t_g == 0:
            self.load_page(args, initial_page, first, state_full)
        for j in cutlass.range(count, unroll=1):
            b = 2 * (j % 2) + (j // 2) % 2  # operand block
            following = first + (j + 1) * stride
            page_id = Int32(0)
            phase = j % 2
            if cutlass.const_expr(self.state_tma):
                page_id = j % 2
                phase = (j // 2) % 2
            page = self.checkpoint_page(pages, page_id)
            wait(state_full + page_id, phase)
            history = cute.make_rmem_tensor((ROWS, CHUNKS, 4), Float32)
            item = first + j * stride
            request = item // self.heads
            source_slot = args[16][request, 0]
            for r in cutlass.range_constexpr(ROWS):
                for c in cutlass.range_constexpr(CHUNKS):
                    row = base_row + 32 * r
                    key = 4 * (lane_in + LANES * c)
                    for e in cutlass.range_constexpr(4):
                        history[r, c, e] = Float32(0)
                        if source_slot > 0:
                            history[r, c, e] = page[self.state_page_index(key + e, row)]
            if cutlass.const_expr(self.state_tma):
                if j + 1 < count:
                    next_page_id = 1 - page_id
                    next_page = self.checkpoint_page(pages, next_page_id)
                    if t_g == 0:
                        if j > 0:
                            # The previous iteration's writer owns next_page
                            # until its bulk stores finish reading shared memory.
                            wait_bulk_reads()
                        self.load_page(args, next_page, following, state_full + next_page_id)
            wait(ops_full + b, (j // 4) % 2)
            live = meta[b, 0] != 0
            head, token, n, rec = Int32(0), Int32(0), Int32(0), Int32(-1)
            write_slot = Int32(0)
            write_checkpoint = Int32(0)
            if live:
                head, token, n = meta[b, 2], meta[b, 3], meta[b, 4]
                rec = meta[b, 5]
                write_slot = args[21][request, 0]
                write_checkpoint = Int32((n > 0) | (source_slot != write_slot))
                # Replay accepted tokens and publish the checkpoint when it changes or moves.
                u = cute.make_rmem_tensor((ROWS, RQ), Float32)
                for r in cutlass.range_constexpr(ROWS):
                    for t in cutlass.range_constexpr(RQ):
                        u[r, t] = u_s[b, t, base_row + 32 * r]
                for c in cutlass.range_constexpr(CHUNKS):
                    key = 4 * (lane_in + LANES * c)
                    d4 = load_shared_f32x4(at(dn_s, b, key))
                    final = zeros((ROWS, 4), Float32)
                    for t in cutlass.range_constexpr(RQ):
                        k4 = load_shared_f32x4(at(kp_s, b, t, key))
                        for r in cutlass.range_constexpr(ROWS):
                            for pp in cutlass.range_constexpr(2):
                                pair = fma2(
                                    (u[r, t], u[r, t]),
                                    (k4[2 * pp], k4[2 * pp + 1]),
                                    (final[r, 2 * pp], final[r, 2 * pp + 1]),
                                )
                                final[r, 2 * pp], final[r, 2 * pp + 1] = pair
                    for r in cutlass.range_constexpr(ROWS):
                        for pp in cutlass.range_constexpr(2):
                            pair = fma2(
                                (history[r, c, 2 * pp], history[r, c, 2 * pp + 1]),
                                (d4[2 * pp], d4[2 * pp + 1]),
                                (final[r, 2 * pp], final[r, 2 * pp + 1]),
                            )
                            history[r, c, 2 * pp], history[r, c, 2 * pp + 1] = pair
                        if write_checkpoint != 0:
                            row = base_row + 32 * r
                            for e in cutlass.range_constexpr(4):
                                page[self.state_page_index(key + e, row)] = history[r, c, e]
            # Publish this page before the asynchronous bulk writer reads it.
            if write_checkpoint != 0:
                fence_async_shared()
            cute.arch.mbarrier_arrive(state_empty + page_id)
            if t_g == 0:
                wait(state_empty + page_id, phase)
                if write_checkpoint != 0:
                    if cutlass.const_expr(self.state_tma):
                        self.transfer_tma_page(page, args[22].iterator, write_slot, head, None, False)
                    else:
                        self.save_page(state, page, write_slot, head)
                        wait_bulk_reads()
                if cutlass.const_expr(not self.state_tma):
                    if j + 1 < count:
                        self.load_page(args, page, following, state_full)
            if live:
                # The 2q dot products of each row, two rows at a time.
                for half in cutlass.range_constexpr(2):
                    acc = zeros((2, 2 * T, 2), Float32)
                    for c in cutlass.range_constexpr(CHUNKS):
                        key = 4 * (lane_in + LANES * c)
                        for jj in cutlass.range_constexpr(2 * T):
                            x4 = load_shared_f32x4(at(xv_s, b, jj, key))
                            for rr in cutlass.range_constexpr(2):
                                r = 2 * half + rr
                                for pp in cutlass.range_constexpr(2):
                                    pair = fma2(
                                        (
                                            history[r, c, 2 * pp],
                                            history[r, c, 2 * pp + 1],
                                        ),
                                        (x4[2 * pp], x4[2 * pp + 1]),
                                        (acc[rr, jj, 0], acc[rr, jj, 1]),
                                    )
                                    acc[rr, jj, 0], acc[rr, jj, 1] = pair
                    for rr in cutlass.range_constexpr(2):
                        red = [Float32(0) for _ in range(16)]
                        for jj in cutlass.range_constexpr(2 * T):
                            red[jj] = acc[rr, jj, 0] + acc[rr, jj, 1]
                        selected = Int32(0)
                        size = 16
                        for offset in (4, 2, 1):
                            size //= 2
                            upper = (lane_in & offset) != 0
                            for i in cutlass.range_constexpr(size):
                                send = red[i] if upper else red[i + size]
                                keep = red[i + size] if upper else red[i]
                                red[i] = keep + cute.arch.shuffle_sync_bfly(
                                    send, offset
                                )
                            if upper:
                                selected += size
                        row = base_row + 32 * (2 * half + rr)
                        for i in cutlass.range_constexpr(2):
                            if selected + i < 2 * T:
                                dots[row, selected + i] = red[i]
                sync_threads(3, GROUP_THREADS)
                # Per row (thread t_g < 128): the q x q substitution and the outputs
                # (into the block's chunk vectors, read by now).
                if t_g < K:
                    us_new = zeros(T, Float32)
                    for t in cutlass.range_constexpr(T):
                        total = dots[t_g, t]
                        for s in cutlass.range_constexpr(t):
                            total = fma(pr_s[b, a_index(t, s)], us_new[s], total)
                        us_new[t] = (v_s[b, t, t_g] - total) * beta_s[b, t]
                    for t in cutlass.range_constexpr(T):
                        o = dots[t_g, T + t]
                        for s in cutlass.range_constexpr(t + 1):
                            o = fma(pr_s[b, b_index(T, t, s)], us_new[s], o)
                        xv_s[b, t, t_g] = Float32(BFloat16(o))
                        if rec >= 0:
                            corrections[rec, head, t, t_g] = us_new[t]
                self.epilogue(args, t_g, b, head, token, xv_s, g_s,
                              out_sums[j % 2, None, None], nw)  # fmt: skip
            cute.arch.mbarrier_arrive(done + b)
        if cutlass.const_expr(self.state_tma):
            # The other lanes have no bulk groups. Use a common exit path
            # while waiting for the leader's checkpoint writes to finish.
            wait_bulk_writes()
        else:
            if t_g == 0:
                wait_bulk_writes()

    # -------------------------------------------------------------- prep role
    @cute.jit
    def prep(self, args, w, p, first, stride, count, staging, ops, ops_full, done):
        weights, conv_state = args[1], args[2]
        beta, a_log, dt_bias = args[4:7]
        output = args[9]
        lower_bound, scale = args[12:14]
        f_b = args[15]
        key_gates = args[18]
        xv_s, kp_s, dn_s, u_s, v_s, g_s, beta_s, pr_s, meta = ops
        convs, xs, gs, fas, sums, psums, kgs, us = staging
        T: cutlass.Constexpr = self.tokens
        H: cutlass.Constexpr = self.heads
        RQ: cutlass.Constexpr = args[17].shape[2]
        NP: cutlass.Constexpr = pair_count(T)
        P2: cutlass.Constexpr = 1 << (NP - 1).bit_length()
        NS: cutlass.Constexpr = 1 << (2 * T + RQ - 1).bit_length()  # norms
        dim: cutlass.Constexpr = H * K
        lane, warp = p % 32, p // 32
        group_id, quarter = lane >> 2, lane & 3
        c = p
        bar = 1 + w
        conv_type: cutlass.Constexpr = conv_state.element_type

        start = first + w * stride
        if w >= count:
            start = first
        if w < count:
            self.stage(args, staging, p, start)
        cute.arch.cp_async_commit_group()

        for j in cutlass.range(w, count, 2, unroll=1):
            item = first + j * stride
            head, request, slot, rec, n, offset = self.item_info(args, item)
            token = request * T
            blk = 2 * w + (j // 2) % 2
            cute.arch.cp_async_wait_group(0)
            sync_threads(bar, PREP_THREADS)
            if j >= 4:
                wait(done + blk, ((j - 4) // 4) % 2)
            if slot > 0:
                wr = cute.make_rmem_tensor((8, 4), Uint32)
                for chunk in cutlass.range_constexpr(8):
                    feature = head * K + 32 * warp + group_id
                    col = 16 * chunk + 4 * quarter
                    low = load_weight_bf16x4(at(f_b, feature, col))
                    high = load_weight_bf16x4(at(f_b, feature + 8, col))
                    wr[chunk, 0], wr[chunk, 2] = low
                    wr[chunk, 1], wr[chunk, 3] = high
                beta_value = Float32(0)
                if p < T:
                    beta_value = Float32(beta[token + p, head])
                a = triton_exp(Float32(a_log[head]))

                # Convolution + SiLU of q, k, v (thread c: channel c of each part).
                values = cute.make_rmem_tensor((3, T), Float32)
                conv_page = conv_state[Int64(slot - 1), None, None]
                for part in cutlass.range_constexpr(3):
                    taps = cute.make_rmem_tensor(3, conv_type)
                    for t in cutlass.range_constexpr(3):
                        taps[t] = convs[part, c * self.window + offset + t]
                    coeff = [Float32(weights[part, i, head * K + c]) for i in range(4)]
                    inputs = [xs[t, part, c] for t in range(T)]
                    for t in cutlass.range_constexpr(T):
                        accum = Float32(0)
                        for i in cutlass.range_constexpr(4):
                            tap = t + i
                            term = taps[tap] if tap < 3 else inputs[tap - 3]
                            accum = fma(Float32(term), coeff[i], accum)
                        values[part, t] = self.silu(accum)
                    row = c * self.window
                    convs[part, row] = conv_type(taps[1])
                    convs[part, row + 1] = conv_type(taps[2])
                    for t in cutlass.range_constexpr(T):
                        convs[part, row + 2 + t] = conv_type(inputs[t])
                fence_async_shared()
                for t in cutlass.range_constexpr(T):
                    v_s[blk, t, c] = values[2, t]
                    g_s[blk, t, c] = gs[t, c]
                # The old records (staged) are read; the new key records go out.
                old_keys = cute.make_rmem_tensor(RQ, Float32)
                old_gates = cute.make_rmem_tensor(RQ, Float32)
                for t in cutlass.range_constexpr(RQ):
                    live_t = t < n
                    old_keys[t] = Float32(0)
                    old_gates[t] = Float32(0)
                    u_old = Float32(0)
                    if live_t:
                        old_keys[t] = Float32(kgs[t, c])
                        old_gates[t] = Float32(kgs[t, K + c])
                        u_old = us[t, c]
                    u_s[blk, t, c] = u_old
                if rec >= 0:
                    for t in cutlass.range_constexpr(T):
                        key_gates[rec, head, t, c] = BFloat16(values[1, t])

                # q, k and old key norms (one reduction).
                squares = [Float32(0) for _ in range(NS)]
                for t in cutlass.range_constexpr(T):
                    squares[2 * t] = values[0, t] * values[0, t]
                    squares[2 * t + 1] = values[1, t] * values[1, t]
                for t in cutlass.range_constexpr(RQ):
                    squares[2 * T + t] = old_keys[t] * old_keys[t]
                self.block_sums(squares, sums, p, NS)

                # Gate projection: the decays (into the chunk vector rows) and raw
                # gates (records).
                for m in cutlass.range_constexpr(2):
                    if cutlass.const_expr(m == 1):
                        for chunk in cutlass.range_constexpr(8):
                            feature = head * K + 32 * warp + 16 + group_id
                            col = 16 * chunk + 4 * quarter
                            low = load_weight_bf16x4(at(f_b, feature, col))
                            high = load_weight_bf16x4(at(f_b, feature + 8, col))
                            wr[chunk, 0], wr[chunk, 2] = low
                            wr[chunk, 1], wr[chunk, 3] = high
                    acc = zeros(4, Float32)
                    for chunk in cutlass.range_constexpr(8):
                        wa = tuple(wr[chunk, e] for e in range(4))
                        ab = load_shared_u32x2(
                            at(fas, group_id % T, 16 * chunk + 4 * quarter)
                        )
                        mma_into(acc, wa, ab)
                    for e in cutlass.range_constexpr(4):
                        t = 2 * quarter + e % 2
                        feature = 32 * warp + 16 * m + group_id + 8 * (e // 2)
                        raw_g = Float32(BFloat16(acc[e]))
                        if t < T:
                            dt = dt_bias[head * K + feature]
                            xv_s[blk, t, feature] = self.decay(
                                a, raw_g, dt, lower_bound
                            )
                            if rec >= 0:
                                key_gates[rec, head, t, K + feature] = BFloat16(raw_g)
                if p < T:
                    beta_s[blk, p] = triton_sigmoid(beta_value)
                dt_c = Float32(dt_bias[head * K + c])
                old_decays = cute.make_rmem_tensor(RQ, Float32)
                for t in cutlass.range_constexpr(RQ):
                    old_decays[t] = self.decay(a, old_gates[t], dt_c, lower_bound)
                sync_threads(bar, PREP_THREADS)  # norm sums, decays
                if p == 0:
                    size: cutlass.Constexpr = K * self.window * conv_type.width // 8
                    for part in cutlass.range_constexpr(3):
                        channel0 = part * dim + head * K
                        store_bulk(
                            at(conv_page, channel0, 0), at(convs, part, 0), size
                        )

                # Normalized q, k; the chunk vectors (over this channel's decays).
                ds = cute.make_rmem_tensor(T, Float32)
                for t in cutlass.range_constexpr(T):
                    ds[t] = xv_s[blk, t, c]
                qs = cute.make_rmem_tensor(T, Float32)
                ks = cute.make_rmem_tensor(T, Float32)
                gamma = Float32(1)
                for t in cutlass.range_constexpr(T):
                    qn = self.quad_sum(sums[2 * t, None]) + 1.0e-6
                    kn = self.quad_sum(sums[2 * t + 1, None]) + 1.0e-6
                    fq = reciprocal(sqrt_approx(qn))
                    fk = reciprocal(sqrt_approx(kn))
                    qs[t] = mul_rn(mul_rn(values[0, t], fq), scale)
                    ks[t] = mul_rn(values[1, t], fk)
                    gamma = gamma * ds[t]
                    xv_s[blk, t, c] = gamma * ks[t]
                    xv_s[blk, T + t, c] = gamma * qs[t]
                # The replay keys (zero past n) and D_n.
                fd = Float32(1)
                for r in cutlass.range_constexpr(RQ):
                    t = RQ - 1 - r
                    weighted = Float32(0)
                    if t < n:
                        total = self.quad_sum(sums[2 * T + t, None])
                        kn_old = old_keys[t] * cute.math.rsqrt(total + Float32(1e-6))
                        weighted = kn_old * fd
                        fd = fd * old_decays[t]
                    kp_s[blk, t, c] = weighted
                dn_s[blk, c] = fd
                # The pair products, 32 at a time.
                for g in cutlass.range_constexpr(-(-P2 // 32)):
                    size_g: cutlass.Constexpr = min(32, P2)
                    products = [Float32(0) for _ in range(size_g)]
                    for s in cutlass.range_constexpr(T):
                        p_st = Float32(1)
                        for t in cutlass.range_constexpr(s, T):
                            if cutlass.const_expr(t > s):
                                p_st = p_st * ds[t]
                            weighted = ks[s] * p_st
                            if cutlass.const_expr(t > s):
                                idx = a_index(t, s)
                                if cutlass.const_expr(32 * g <= idx < 32 * g + size_g):
                                    products[idx - 32 * g] = weighted * ks[t]
                            idx = b_index(T, t, s)
                            if cutlass.const_expr(32 * g <= idx < 32 * g + size_g):
                                products[idx - 32 * g] = weighted * qs[t]
                    total, selected = scatter_reduce(products, lane, size_g)
                    if lane % (32 // size_g) == 0:
                        psums[32 * g + selected, warp] = total
                sync_threads(bar, PREP_THREADS)
                if p < P2:
                    pr_s[blk, p] = self.quad_sum(psums[p, None])
                if p == 0:
                    meta[blk, 0] = 1
                    meta[blk, 2] = head
                    meta[blk, 3] = token
                    meta[blk, 4] = n
                    meta[blk, 5] = rec
            else:
                if p == 0:
                    meta[blk, 0] = 0
                for t in cutlass.range_constexpr(T):
                    output[token + t, head * K + c] = BFloat16(0)
            cute.arch.mbarrier_arrive(ops_full + blk)

            if p == 0:
                wait_bulk_reads()
            sync_threads(bar, PREP_THREADS)
            if j + 2 < count:
                self.stage(args, staging, p, item + 2 * stride)
            cute.arch.cp_async_commit_group()

        # Bulk groups belong to the issuing thread. Only p == 0 has
        # writes to drain. Keep this wait on the common prep exit path.
        wait_bulk_writes()

    @cute.jit
    def epilogue(self, args, t_g, b, head, token, o_s, g_s, out_sums, nw):
        output, eps = args[9], args[14]
        T: cutlass.Constexpr = self.tokens
        lane, warp = t_g % 32, t_g // 32
        c, half = t_g % K, t_g // K
        sync_threads(3, GROUP_THREADS)
        for t in cutlass.range_constexpr(T):
            if t % 2 == half:
                o = o_s[b, t, c]
                total = cute.arch.warp_reduction_sum(o * o)
                if lane == 0:
                    out_sums[t, warp % 4] = total
        sync_threads(3, GROUP_THREADS)
        for t in cutlass.range_constexpr(T):
            if t % 2 == half:
                sumsq = self.quad_sum(out_sums[t, None])
                variance = sumsq * (1.0 / 128) + eps
                rstd = reciprocal(sqrt_approx(variance))
                sigmoid_gate = triton_sigmoid(Float32(g_s[b, t, c]))
                result = o_s[b, t, c] * rstd * nw * sigmoid_gate
                output[token + t, head * K + c] = BFloat16(result)
