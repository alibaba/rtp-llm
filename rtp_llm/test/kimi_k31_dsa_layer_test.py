"""K3.1 DSA decode layer under a simulated decode server: every launch checked against fp32.

The unit is one KimiK31DSA layer built from the serving config (kimi_k31_dsa_layer_fixture). A
decode server is simulated for each speculative width Q (one query is a plain decode):
requests join with a random history length (log-uniform up to max_seq_len, short ones, and the
edges of top-k, kernel pages, Page-RR pages and stripes), advance by the accepted 1..Q tokens per
step (rejected drafts are rewritten on the next step), and leave. Every step picks a sampled graph
bucket (B*Q <= 512), sets the live batch inside (previous bucket, bucket] so dummy rows are
exercised, and replays the captured layer (a few steps run eagerly instead). After every launch:

- selection: the Indexer's global positions are an exact top-k of fp32 scores computed from the
  path's own E4M3 queries, head gates and cached keys (count min(2048, causal length), unique,
  causal; set equal outside a narrow tie band at the k-th score);
- sparse MLA: the attention output equals an fp32 sparse attention over the path's selected
  positions (all-head partials on each owner, merged by LSE, V-up per head);
- cache: the current tokens are stored at the slots of their owner pages and nothing else
  changed (per-page fingerprints; page 0 excluded, dummy rows write there);
- the layer output is finite, and equal across arms when several are given.

Run on eight GPUs (one DCP8 group):
  torchrun --no-python --nproc_per_node=8 bazel-bin/rtp_llm/test/kimi_k31_dsa_layer_test
      --config-json <K3.1 checkpoint>/config.json --queries '[4]' --steps 5
"""

import argparse
import faulthandler
import json
import math
import os
import random
import sys
import time

import torch
import torch.distributed as dist

try:
    from rtp_llm.test import kimi_k31_dsa_layer_fixture as fx
except ImportError:  # run as a plain script next to the fixture (no Bazel runfiles)
    import kimi_k31_dsa_layer_fixture as fx

TOPK, INDEX_HEADS, LATENT, ROPE, NOPE, V_HEAD = 2048, 32, 512, 64, 128, 128
# Switch environment per arm (dsa_dcp_options); switches are read when the layer and its
# workspace owner are built, so arms coexist in one process.
ARMS = {"dev": {}, "mega_topk": {"DSA_DCP_TOPK": "mega_topk"}}


# Row counts (requests x Q) at MegaTopk's launch-shape switches on 148 SMs: clusters of 16/8/4/2 CTAs
# fit up to about 9/18/37/74 rows, two CTAs per SM above 148 rows, and the 512-row maximum.
ROW_TARGETS = (8, 32, 74, 148, 256, 512)


def buckets_for(queries, limit=fx.MAX_TOKENS):
    """Graph buckets whose row counts land near the kernel switch points, plus a single request."""
    largest = limit // queries
    return sorted({1} | {min(largest, max(1, round(rows / queries))) for rows in ROW_TARGETS})


# ---------------------------------------------------------------- simulated server


class Server:
    """Request membership and lengths; identical on every rank (one seeded RNG for decisions)."""

    def __init__(self, queries, pools, *, max_seq_len, tp, rank, seed, budget=None):
        self.queries, self.pools, self.max_seq_len = queries, pools, max_seq_len
        self.tp, self.rank = tp, rank
        self.rng = random.Random(seed)
        self.pages = torch.Generator().manual_seed(seed * 31 + rank)
        # Pages this server may hold on every rank (servers sharing the pools split them).
        self.budget = budget or min(pool.cache.shape[0] for pool in pools.values()) - 1
        self.used = 0
        self.active = []
        self.next_seed = seed * 100_000

    def _columns(self, capacity):
        return max(fx.owner_columns(capacity, self.tp, r) for r in range(self.tp))

    def _length(self):
        q, top = self.queries, self.max_seq_len
        draw = self.rng.random()
        if draw < 0.25:
            edges = [q, q + 1, TOPK - 1, TOPK, TOPK + 1, top]
            for unit in (fx.KERNEL_PAGE, fx.PAGE, fx.PAGE * self.tp):
                k = self.rng.randint(1, max(1, min(64, top // unit)))
                edges += [k * unit - 1, k * unit, k * unit + 1]
            return min(top, max(q, self.rng.choice(edges)))
        if draw < 0.45:
            return self.rng.randint(q, 4096)
        return min(top, max(q, int(math.exp(self.rng.uniform(math.log(q), math.log(top))))))

    def _join(self, length, lifetime):
        capacity = min(self.max_seq_len, length + self.queries * lifetime)
        columns = self._columns(capacity)
        if self.used + columns > self.budget:
            return False
        request = fx.Request(self.next_seed, length - self.queries, capacity)
        self.next_seed += 1
        if not request.allocate(self.pools, self.tp, self.rank, self.pages):
            raise RuntimeError("page pool accounting diverged from the shared budget")
        request.columns, request.fresh = columns, True
        self.used += columns
        self.active.append(request)
        return True

    def _leave(self, request):
        request.release(self.pools)
        self.used -= request.columns
        self.active.remove(request)

    def step(self, batch):
        """Advance the live requests by their accepted tokens, then match the batch size."""
        for request in list(self.active):
            request.fresh = False
            request.prefix += self.rng.randint(1, self.queries)
            if request.prefix + self.queries > request.capacity:
                self._leave(request)
        while len(self.active) > batch:
            self._leave(self.rng.choice(self.active))
        longest = max((r.prefix + self.queries for r in self.active), default=0)
        while len(self.active) < batch:
            # Keep one request at the global maximum whenever the pool allows it.
            if longest < self.max_seq_len - 64 * self.queries and self._join(self.max_seq_len, 1):
                longest = self.max_seq_len
                continue
            if not self._join(self._length(), self.rng.randint(1, 8)):
                if not self._join(self.rng.randint(self.queries, 4096), 1):
                    raise RuntimeError("page pool cannot hold the requested batch")
        self.rng.shuffle(self.active)
        return list(self.active)


# ---------------------------------------------------------------- checks


def index_entries(cache, slots):
    """(E4M3 keys as fp32 [n, 128], FP32 scales [n]) at slots of the block-planar FP8 cache."""
    flat = cache.view(cache.shape[0], -1)
    pages, offsets = slots // fx.KERNEL_PAGE, slots % fx.KERNEL_PAGE
    columns = torch.arange(128, device=slots.device)
    keys = flat[pages[:, None], offsets[:, None] * 128 + columns]
    scale_bytes = flat[pages[:, None], 128 * 128 + offsets[:, None] * 4 + columns[:4]]
    return keys.view(torch.float8_e4m3fn).float(), scale_bytes.contiguous().view(torch.float32)[:, 0]


def owned_slots(request, tag, pool, positions, tp, rank):
    """Slots of the positions this rank owns (others -1)."""
    pages = request.pages[tag]
    owner, local = fx.global_owner(positions.long(), tp)
    mine = (owner == rank) & (positions >= 0)
    if not pages.numel():  # a short request may own no page on this rank
        return torch.full_like(local, -1)
    slots = pool.slots(pages, torch.where(mine, local, 0))
    return torch.where(mine, slots, -1)


def check_cache(step, recorded, pools, snapshot, tp, rank):
    """Current tokens at their owner slots; every other byte of the pools unchanged."""
    failures = []
    expected = {"full": [], "indexer_kv": []}
    for index, request in enumerate(step.requests):
        rows = slice(index * step.queries, (index + 1) * step.queries)
        positions = torch.arange(request.prefix, request.prefix + step.queries, device="cuda")
        for tag in fx.TAGS:
            slots = owned_slots(request, tag, pools[tag], positions, tp, rank)
            mine = slots >= 0
            if mine.any():
                expected[tag].append((slots[mine], torch.nonzero(mine).flatten() + rows.start))
    full = pools["full"].cache.view(-1, LATENT + ROPE)
    latent = torch.cat((recorded["latent"], recorded["suffix"]), dim=-1)
    index_pool = pools["indexer_kv"].cache
    for tag, entries in expected.items():
        if not entries:
            continue
        slots = torch.cat([s for s, _ in entries])
        rows = torch.cat([r for _, r in entries])
        if tag == "full":
            if not torch.equal(full[slots], latent[rows]):
                failures.append("latent cache entries differ from the layer's latent/suffix")
        else:
            scratch = torch.zeros(1 + rows.numel() // fx.KERNEL_PAGE + 1, fx.KERNEL_PAGE, fx.INDEX_ENTRY,
                                  dtype=torch.uint8, device="cuda")
            scratch_slots = fx.KERNEL_PAGE + torch.arange(rows.numel(), device="cuda")
            fx.rtp_llm_ops.indexer_k_quant_and_cache(
                recorded["index_key"][rows].contiguous(), scratch, scratch_slots, 128, "ue8m0"
            )
            want, got = index_entries(scratch, scratch_slots), index_entries(index_pool, slots)
            if not (torch.equal(want[0], got[0]) and torch.equal(want[1], got[1])):
                failures.append("Indexer cache entries differ from the layer's quantized keys")
    for tag in fx.TAGS:
        before, touched, saved = snapshot[tag]
        after = pools[tag].checksum()
        changed = torch.nonzero((after != before).any(1)).flatten().tolist()
        allowed = set(touched.tolist()) | {0}
        stray = [page for page in changed if page not in allowed]
        if stray:
            failures.append(f"{tag}: {len(stray)} pages outside the current tokens changed")
        # Inside the touched pages only the current entries may change.
        if touched.numel():
            cache = pools[tag].cache
            restored = saved.clone()
            for slots, _ in expected[tag]:
                for slot in slots.tolist():
                    page, offset = divmod(slot, fx.KERNEL_PAGE)
                    where = int(torch.nonzero(touched == page)[0])
                    if tag == "full":
                        restored[where, offset] = cache[page, offset]
                    else:
                        flat, live = restored[where].view(-1), cache[page].view(-1)
                        flat[offset * 128 : offset * 128 + 128] = live[offset * 128 : offset * 128 + 128]
                        base = 128 * 128 + offset * 4
                        flat[base : base + 4] = live[base : base + 4]
            if not torch.equal(restored, cache[touched.long()]):
                failures.append(f"{tag}: non-current entries of the current pages changed")
    return failures


def take_snapshot(step, pools, tp, rank):
    snapshot = {}
    for tag in fx.TAGS:
        pages = set()
        for request in step.requests:
            positions = torch.arange(request.prefix, request.prefix + step.queries, device="cuda")
            slots = owned_slots(request, tag, pools[tag], positions, tp, rank)
            pages.update((slots[slots >= 0] // fx.KERNEL_PAGE).tolist())
        touched = torch.tensor(sorted(pages), dtype=torch.int64, device="cuda")
        snapshot[tag] = (pools[tag].checksum(), touched, pools[tag].cache[touched].clone())
    return snapshot


def global_selection(recorded, rows, tp, rank):
    """Each row's selected global positions [rows, TOPK] (-1 padded), and format problems.

    The DCP path gives every rank the whole selection; an owner-local selector gives each rank
    the positions it owns. Either way the group's union is the row's selection.
    """
    local = recorded["selected"][:rows].long().contiguous()
    gathered = [torch.empty_like(local) for _ in range(tp)]
    dist.all_gather(gathered, local)
    if all(torch.equal(g, gathered[0]) for g in gathered[1:]):
        return gathered[0], []
    problems = []
    owner, _ = fx.global_owner(local.clamp(min=0), tp)
    if ((local >= 0) & (owner != rank)).any():
        problems.append("owner-local selection lists positions of another owner")
    merged = torch.cat(gathered, dim=1).sort(dim=1, descending=True).values
    if (merged[:, TOPK:] >= 0).any():
        problems.append("owner-local selections hold more than top-k positions")
    return merged[:, :TOPK].contiguous(), problems


def reference_scores(step, recorded, pools, tp, rank, selected, chunk=1 << 16):
    """Per live row: this rank's top fp32 scores, and the scores of the selected positions it owns.

    FP8 paged MQA: sum over heads of gate * relu(q_e4m3 . k_e4m3) * key scale.
    Returns (values [rows, TOPK], scores of `selected` owned here, 0 elsewhere).
    """
    q, rows = step.queries, step.rows
    values = torch.full((rows, TOPK), -float("inf"), device="cuda")
    chosen = torch.zeros(rows, TOPK, device="cuda")
    pool = pools["indexer_kv"]
    for index, request in enumerate(step.requests):
        first = index * q
        query = recorded["index_query"][first : first + q].float()
        gates = recorded["index_weights"][first : first + q, :, 0].float()
        causal = request.prefix + 1 + torch.arange(q, device="cuda")
        count = fx.local_tokens(request.prefix + q, tp, rank)
        if count == 0:
            continue
        score = torch.empty(q, count, device="cuda")
        for start in range(0, count, chunk):
            local = torch.arange(start, min(start + chunk, count), device="cuda")
            keys, scales = index_entries(pool.cache, pool.slots(request.pages["indexer_kv"], local))
            logits = torch.relu(torch.einsum("qhd,kd->qhk", query, keys)) * scales
            score[:, start : start + local.numel()] = (logits * gates[:, :, None]).sum(1)
        global_positions = fx.local_to_global(torch.arange(count, device="cuda"), tp, rank)
        score.masked_fill_(global_positions[None, :] >= causal[:, None], -float("inf"))
        k = min(TOPK, count)
        top = score.topk(k, dim=1)
        values[first : first + q, :k] = top.values
        mine = selected[first : first + q]
        owner, local = fx.global_owner(mine.clamp(min=0), tp)
        hit = (mine >= 0) & (owner == rank) & (local < count)
        gathered = score.gather(1, torch.where(hit, local, 0))
        chosen[first : first + q] = torch.where(hit, gathered, 0.0)
    return values, chosen


def check_selection(step, recorded, pools, tp, rank, tie=1e-4):
    q, rows = step.queries, step.rows
    selected, failures = global_selection(recorded, rows, tp, rank)
    values, chosen = reference_scores(step, recorded, pools, tp, rank, selected)
    gathered_values = [torch.empty_like(values) for _ in range(tp)]
    dist.all_gather(gathered_values, values)
    dist.all_reduce(chosen)
    candidates = torch.cat(gathered_values, dim=1)
    causal = torch.cat([r.prefix + 1 + torch.arange(q, device="cuda") for r in step.requests])
    expected_count = causal.clamp(max=TOPK)
    valid = selected >= 0
    count_bad = valid.sum(1) != expected_count
    causal_bad = (valid & (selected >= causal[:, None])).any(1)
    ordered = torch.where(valid, selected, -1 - torch.arange(TOPK, device="cuda")[None, :]).sort(1).values
    duplicate = (ordered[:, 1:] == ordered[:, :-1]).any(1)
    kth = candidates.topk(TOPK, dim=1).values.gather(1, (expected_count - 1)[:, None])[:, 0]
    band = tie * kth.abs() + 1e-6 * candidates.clamp(min=0).nan_to_num(0, 0, 0).amax(1) + 1e-30
    low = (valid & (chosen < (kth - band)[:, None])).any(1)
    above = (candidates > (kth + band)[:, None]).sum(1)
    above_chosen = (valid & (chosen > (kth + band)[:, None])).sum(1)
    missing = above_chosen != above
    swaps = (valid & ((chosen - kth[:, None]).abs() <= band[:, None])).sum(1)
    for name, bad in (("count", count_bad), ("causality", causal_bad), ("duplicates", duplicate),
                      ("below k-th score", low), ("missing above k-th score", missing)):
        if bad.any():
            failures.append(f"selection {name}: {int(bad.sum())} rows, first {int(torch.nonzero(bad)[0])}")
    if low.any() and os.environ.get("K31_LAYER_TEST_DEBUG"):
        row = int(torch.nonzero(low)[0])
        wrong = selected[row][valid[row] & (chosen[row] < kth[row] - band[row])][:6]
        owner, local = fx.global_owner(wrong, tp)
        failures.append(f"row {row} causal {int(causal[row])} kth {float(kth[row]):.4g}: wrong positions "
                        f"{wrong.tolist()} owners {owner.tolist()} local {local.tolist()} "
                        f"scores {chosen[row][valid[row] & (chosen[row] < kth[row] - band[row])][:6].tolist()}")
    return failures, {"tie_band_entries": int(swaps.sum())}


def check_attention(step, recorded, pools, weights, tp, rank, softmax_scale, chunk=32):
    """fp32 sparse MLA over the path's selections: owner partials, LSE merge, local heads."""
    rows, heads = step.rows, weights.key_up.shape[0]
    local_heads = heads // tp
    query = recorded["sfa_query"][: step.padded_rows]
    gathered = [torch.empty_like(query) for _ in range(tp)]
    dist.all_gather(gathered, query.contiguous())
    query = torch.stack(gathered, 1).flatten(1, 2)[:rows].float()  # [rows, heads, 192]
    absorbed = torch.einsum("rhd,hdl->rhl", query[..., :NOPE], weights.key_up.float())
    query = torch.cat((absorbed, query[..., NOPE:]), dim=-1)
    selected = recorded["selected"][:rows].long()
    latent = pools["full"].cache.view(-1, LATENT + ROPE)
    request_of = torch.arange(len(step.requests), device="cuda").repeat_interleave(step.queries)
    partial = torch.zeros(rows, heads, V_HEAD, device="cuda")
    lse = torch.full((rows, heads), -float("inf"), device="cuda")
    for first in range(0, rows, chunk):
        last = min(first + chunk, rows)
        slots = torch.full((last - first, TOPK), -1, dtype=torch.int64, device="cuda")
        for row in range(first, last):
            request = step.requests[int(request_of[row])]
            slots[row - first] = owned_slots(request, "full", pools["full"], selected[row], tp, rank)
        mine = slots >= 0
        kv = latent[slots.clamp(min=0)].float()  # [n, TOPK, 576]
        logits = torch.einsum("rhd,rkd->rhk", query[first:last], kv) * softmax_scale
        logits.masked_fill_(~mine[:, None, :], -float("inf"))
        peak = logits.amax(-1, keepdim=True)
        safe = torch.where(torch.isfinite(peak), peak, 0)
        probs = torch.exp(logits - safe)
        total = probs.sum(-1, keepdim=True)
        out = torch.einsum("rhk,rkl->rhl", probs, kv[..., :LATENT]) / total.clamp(min=1e-30)
        partial[first:last] = torch.einsum("rhl,hlv->rhv", out, weights.value_up.float())
        lse[first:last] = torch.where(total[..., 0] > 0, safe[..., 0] + total[..., 0].log(), -float("inf"))
    parts = [torch.empty_like(partial) for _ in range(tp)]
    lses = [torch.empty_like(lse) for _ in range(tp)]
    dist.all_gather(parts, partial)
    dist.all_gather(lses, lse)
    parts, lses = torch.stack(parts), torch.stack(lses)
    merged = torch.logsumexp(lses, 0)
    scale = torch.exp(lses - torch.where(torch.isfinite(merged), merged, 0)).nan_to_num(0)
    reference = (parts * scale[..., None]).sum(0)[:, rank * local_heads : (rank + 1) * local_heads]
    output = recorded["sfa_output"].reshape(-1, local_heads, V_HEAD)[:rows].float()
    error = (output - reference).abs().amax((1, 2))
    magnitude = reference.abs().amax((1, 2)).clamp(min=1e-6)
    relative = error / magnitude
    failures = []
    if not torch.isfinite(output).all():
        failures.append("attention output not finite")
    return failures, {"attn_row_rel_max": float(relative.max()) if rows else 0.0,
                      "attn_rel_l2": float((output - reference).norm() / reference.norm().clamp(min=1e-30))}


# ---------------------------------------------------------------- driver


class Step:
    def __init__(self, requests, queries, padded):
        self.requests, self.queries, self.padded = requests, queries, padded
        self.rows, self.padded_rows = len(requests) * queries, padded * queries


class Arm:
    def __init__(self, name, config, parallel, weights, layer_index, buckets, queries):
        unit = fx.sp_unit(parallel.tp_size, queries)
        saved = {key: os.environ.get(key) for key in ARMS[name]}
        os.environ.update(ARMS[name])
        try:
            self.layer = fx.build_layer(config, parallel, weights, layer_index)
            self.owner = fx.build_owner(
                config, parallel, [(layer_index, weights)],
                max_requests=max(math.ceil(b / unit) * unit for b in buckets), queries=queries,
            )
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        self.name = name
        self.weights_by_layer = [None] * config.num_layers
        self.weights_by_layer[layer_index] = weights
        self.recorder = fx.Recorder(self.layer)
        self.slots = {}


def run_step(arm, step, config, parallel, pools, hidden, *, graph, bucket, max_seq_len, pool_handle):
    tp, rank = parallel.tp_size, parallel.tp_rank
    if graph:
        slot = arm.slots.get(bucket)
        if slot is None:
            slot = arm.slots[bucket] = fx.GraphSlot(
                config, parallel, [arm.layer], arm.weights_by_layer, arm.owner, arm.recorder, pools,
                queries=step.queries, bucket=bucket, max_seq_len=max_seq_len, pool_handle=pool_handle,
            )
        slot.load(step.requests, hidden)
        output = slot.replay()[0]
        arm.softmax_scale = slot.fmha["full"].fmha_impl.scale
        return output, slot.recorded, slot.fmha
    inputs = fx.attention_inputs(step.requests, step.queries, step.padded, graph_width=None,
                                 tp=tp, rank=rank, device="cuda")
    fmha = fx.build_fmha(config, parallel, arm.weights_by_layer, inputs, arm.owner, arm.recorder, graph=False)
    caches = {tag: pools[tag].layer_cache(arm.layer.layer_idx) for tag in fx.TAGS}
    output = arm.layer(hidden, fmha, caches, inputs["full"], None)
    arm.softmax_scale = fmha["full"].fmha_impl.scale
    return output, dict(arm.recorder.last), fmha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", required=True)
    # Q classes by the sequence-parallel padding unit 8 / gcd(8, Q), plain decode and the widest Q.
    parser.add_argument("--queries", default="[1, 3, 4, 6, 8, 16]", help="speculative widths Q to simulate")
    parser.add_argument("--buckets", default=None, help="graph buckets (default: buckets_for)")
    parser.add_argument("--steps", type=int, default=5, help="steps per Q and bucket")
    parser.add_argument("--eager-fraction", type=float, default=0.1)
    parser.add_argument("--arms", default='["dev"]')
    parser.add_argument("--max-seq-len", type=int, default=1 << 20)
    parser.add_argument("--pool-pages", type=int, default=96 * 1024, help="kernel pages per cache group per rank")
    parser.add_argument("--attn-tolerance", type=float, default=2e-2, help="max row error / row max")
    parser.add_argument("--seed", type=int, default=3101)
    parser.add_argument("--step-timeout", type=int, default=1800)
    args = parser.parse_args()
    parallel = fx.init_distributed(int(os.environ["WORLD_SIZE"]))
    tp, rank = parallel.tp_size, parallel.tp_rank
    faulthandler.enable(file=sys.stderr, all_threads=True)
    failures = 0
    for queries in json.loads(args.queries):
        buckets = json.loads(args.buckets) if args.buckets else buckets_for(queries)
        buckets = [b for b in buckets if b * queries <= fx.MAX_TOKENS]
        config = fx.load_config(args.config_json, max_seq_len=args.max_seq_len, queries=queries)
        layer_index = config.k3_runtime_config.dsa.layers[0]
        weights, global_weights = fx.layer_weights(config, parallel, seed=args.seed)
        pools = {tag: fx.Pool(tag, args.pool_pages, "cuda") for tag in fx.TAGS}
        arms = [Arm(name, config, parallel, weights, layer_index, buckets, queries) for name in json.loads(args.arms)]
        server = Server(queries, pools, max_seq_len=args.max_seq_len, tp=tp, rank=rank, seed=args.seed + queries)
        pool_handle = torch.cuda.graph_pool_handle()
        decisions = random.Random(args.seed * 7 + queries)
        hidden_gen = torch.Generator(device="cuda").manual_seed(args.seed * 13 + rank)
        schedule = [b for b in buckets for _ in range(args.steps)]
        decisions.shuffle(schedule)
        for number, bucket in enumerate(schedule):
            faulthandler.dump_traceback_later(args.step_timeout, exit=True)
            index = buckets.index(bucket)
            low = buckets[index - 1] + 1 if index else 1
            batch = decisions.randint(low, bucket)
            graph = decisions.random() >= args.eager_fraction
            requests = server.step(batch)
            for request in requests:
                if request.fresh:
                    request.write_history(pools, tp, rank)
            unit = fx.sp_unit(tp, queries)
            padded = math.ceil((bucket if graph else batch) / unit) * unit
            step = Step(requests, queries, padded)
            hidden = torch.randn(step.padded_rows // tp, config.hidden_size, generator=hidden_gen,
                                 device="cuda").bfloat16()
            live = torch.arange(rank * hidden.shape[0], (rank + 1) * hidden.shape[0], device="cuda") < step.rows
            hidden *= live[:, None]
            outputs = []
            for arm in arms:
                snapshot = take_snapshot(step, pools, tp, rank)
                started = time.time()
                output, recorded, fmha = run_step(arm, step, config, parallel, pools, hidden, graph=graph,
                                                  bucket=bucket, max_seq_len=args.max_seq_len,
                                                  pool_handle=pool_handle)
                torch.cuda.synchronize()
                elapsed = time.time() - started
                problems = check_cache(step, recorded, pools, snapshot, tp, rank)
                selection, stats = check_selection(step, recorded, pools, tp, rank)
                attention, attn_stats = check_attention(step, recorded, pools, global_weights, tp, rank,
                                                        arm.softmax_scale)
                problems += selection + attention
                if attn_stats["attn_row_rel_max"] > args.attn_tolerance:
                    problems.append(f"attention row error {attn_stats['attn_row_rel_max']:.3g}")
                rows_local = output[: hidden.shape[0]][live]
                if not torch.isfinite(rows_local).all():
                    problems.append("layer output not finite")
                outputs.append(output.clone())
                failed = torch.tensor([len(problems)], device="cuda")
                dist.all_reduce(failed)
                failures += int(failed.item() > 0)
                lengths = [r.prefix + queries for r in requests]
                record = {"q": queries, "step": number, "arm": arm.name, "graph": graph, "bucket": bucket,
                          "batch": batch, "min_len": min(lengths), "max_len": max(lengths),
                          "seconds": round(elapsed, 3), "passed": failed.item() == 0,
                          "kv_blocks": [len(fmha["indexer_kv"].page_rr_params.indexer_execution.blocks),
                                        len(fmha["full"].page_rr_params.sfa_execution.blocks)],
                          # The selection path that actually ran (MegaTopk falls back when its logits do not fit).
                          "mega_topk": fmha["indexer_kv"].page_rr_params.indexer_execution.mega_topk,
                          # Indexer phase: planned bytes and the other phases' resident bytes (MiB).
                          "indexer_mib": [
                              fmha["indexer_kv"].page_rr_params.indexer_execution.required_bytes >> 20,
                              arm.owner.other_resident_bytes(
                                  fmha["indexer_kv"].page_rr_params.indexer_execution.workspace) >> 20],
                          **stats, **attn_stats, "problems": problems[:4]}
                if rank == 0 or problems:
                    print("STEP " + json.dumps(record), flush=True)
            for other in outputs[1:]:
                if not torch.equal(other, outputs[0]):
                    difference = (other.float() - outputs[0].float()).abs().max().item()
                    print(f"ARM-DIFF q{queries} step {number} rank {rank}: {difference:.3g}", flush=True)
        del arms, pools, server
        torch.cuda.empty_cache()
    faulthandler.cancel_dump_traceback_later()
    if failures and rank == 0:
        print(f"{failures} launches failed their checks", flush=True)
    dist.barrier()
    # NCCL teardown with captured graphs alive can hang in destroy_process_group; the run is over.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1 if failures else 0)


if __name__ == "__main__":
    main()
