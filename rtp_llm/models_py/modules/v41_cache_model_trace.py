"""Fixed Aone V4.1 cache fixture. Records logical ownership and digests, not payloads."""

import contextvars
import hashlib
import json
import os
import re
from pathlib import Path

_ACTIVE = contextvars.ContextVar("v41_cache_fixture", default=None)
TARGET = "//rtp_llm/test/smoke:deepseek_v41_cache_rounds_sm100_arm"
PHASES = ("cold", "fill", "hit")
SLOTS = (1, 127, 255, 257, 383, 511, 513, 639, 781)
DRAFT_POS = (510, 511, 512, 655, 656, 767, 781, 782)
ENABLED = os.getenv("RTP_V41_CACHE_TRACE") == "1" and os.getenv("TEST_TARGET") == TARGET


def active():
    return _ACTIVE.get()


def run_fixed(prefix, forward, *args):
    # C++ verified singleton unsplit IDs, length, prefix, model role, and excluded
    # warmup before reaching here. Never enable tracing with the marker alone.
    if os.getenv("RTP_V41_CACHE_TRACE") != "1" or os.getenv("TEST_TARGET") != TARGET:
        return forward(*args)
    root = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
    try:
        marker = (root / "v41_cache_trace_stage").read_text()
    except FileNotFoundError:
        return forward(*args)
    match = re.fullmatch(r"([0-9a-f]{32}) (cold|fill|hit)\n", marker)
    if not match or prefix != (512 if match[2] == "hit" else 0):
        raise ValueError("Unexpected fixed V4.1 cache fixture phase/prefix")
    token = _ACTIVE.set(Trace(root, match[1], match[2], prefix))
    try:
        return forward(*args)
    finally:
        _ACTIVE.reset(token)


class Trace:
    def __init__(self, root, run, phase, prefix):
        self.root, self.run, self.phase, self.prefix = root, run, phase, prefix

    def emit(self, rank, stage, pos, value):
        import torch

        if value.ndim < 2:
            value = value.reshape(-1, 1)
        if len(value) != len(pos) or len(set(pos)) != len(pos):
            raise ValueError(f"Invalid fixture rows at {stage}")
        # Hash rank-local payload in the reported position order; layout checks
        # precede value comparisons. No physical address, block id or token ID.
        raw = value.detach().contiguous().view(torch.uint8).cpu().numpy()
        data = {
            "rank": int(rank),
            "stage": stage,
            "positions": pos,
            "row_shape": list(value.shape[1:]),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "samples": {
                str(i): hashlib.sha256(row).hexdigest()
                for i, row in zip(pos, raw)
                if i in (*SLOTS, 0, 512, 782)
            },
        }
        path = self.root / f"v41_cache_model_{self.run}_{self.phase}_{int(rank)}.jsonl"
        with path.open("a", encoding="ascii") as f:
            f.write(json.dumps(data, separators=(",", ":")) + "\n")

    def hidden(self, stage, cp, value):
        if cp is None or int(cp.cp_size) != 2 or cp.local_is_real.numel() != len(value):
            raise ValueError(f"Unmappable V4.1 fixture CP activation: {stage}")
        keep = cp.local_is_real.reshape(-1).bool()
        pos = cp.global_positions.reshape(-1)[keep].tolist()
        self.emit(cp.cp_rank, stage, pos, value[keep])

    def pool(self, point, attn, cp):
        """Look at the exact physical rows addressed by the production slot map."""
        import torch

        positions = [
            i for i in SLOTS if point != "reused_before_write" or i < self.prefix
        ]
        for suffix, region in (
            ("main", attn._global_region()),
            ("index", "indexer_kv"),
        ):
            t = attn._source_pool(region)
            if t is None or t.ndim != 3 or t.dtype != torch.uint8:
                raise ValueError(f"Unbound V4.1 packed {suffix} pool")
            p = torch.tensor(positions, dtype=torch.long, device=t.device)
            slots = attn._slots(region, p, torch.zeros_like(p))
            if slots is None:
                raise ValueError(f"Missing V4.1 {suffix} slot map")
            mask = slots >= 0
            owned = slots[mask].long()
            # Block-major views can have row tail padding; do not flatten them.
            self.emit(
                cp.cp_rank,
                f"layer_2.{point}.{suffix}",
                p[mask].tolist(),
                t[owned // t.size(1), owned % t.size(1)],
            )

    def draft_written(self, lid, attn, pool, entries, cp, positions, req, slots):
        # After the production writer, before P publishes to D. Select only
        # rows the real writer retained in this request's ring generation.
        if cp is None or int(cp.cp_size) != 2 or int(lid) not in (0, 1, 2):
            raise ValueError("Unexpected V4.1 draft CP layers")
        sliced = attn._swa_cp_byte_sliced()
        raw = attn._pool_raw_u8("swa_kv") if sliced else pool
        rows = {}
        for p in DRAFT_POS:
            live = (positions == p) & (req == 0) & (slots >= 0)
            selected = slots[live]
            if selected.numel() > 1:
                raise ValueError("duplicate logical draft writer")
            if selected.numel():
                rows[str(p)] = chunks(
                    raw, int(selected.item()), entries, int(cp.cp_rank), sliced
                )
        out = {
            "layer": int(lid),
            "rank": int(cp.cp_rank),
            "rows": rows,
            "sliced": bool(sliced),
        }
        file = (
            self.root
            / f"v41_cache_model_{self.run}_{self.phase}_{int(cp.cp_rank)}_draft_write.jsonl"
        )
        with file.open("a", encoding="ascii") as f:
            f.write(json.dumps(out, separators=(",", ":")) + "\n")

    def readback(self, attn, cp):
        import torch

        # Native _produce_global publishes one tuple per request. Entries are
        # in logical compressed order, irrespective of page placement.
        [(g, k)] = attn._shared_attention["global"][attn.layer_id]
        rows = list(SLOTS)
        start = torch.tensor(rows, device=g.device, dtype=torch.long) // 2
        if int(attn.compress_ratio) != 2 or len(g) != 783 // 2 or len(k) != len(g):
            raise ValueError("Unexpected V4.1 layer-2 readback geometry")
        self.emit(cp.cp_rank, "layer_2.readback.main", rows, g.index_select(0, start))
        if isinstance(k, torch.Tensor):
            self.emit(
                cp.cp_rank, "layer_2.readback.index", rows, k.index_select(0, start)
            )
        else:
            self.emit(
                cp.cp_rank,
                "layer_2.readback.index_quant",
                rows,
                k.quant.index_select(0, start),
            )
            self.emit(
                cp.cp_rank,
                "layer_2.readback.index_scale",
                rows,
                k.scale.index_select(0, start),
            )


def chunks(raw, slot, entries, rank, sliced):
    """Interleaved 512B payload + 16B scales; merge both P byte slices by offset."""
    import torch

    if slot < 0 or raw.dtype != torch.uint8 or raw.ndim not in (2, 3):
        raise ValueError("Invalid V4.1 SWA byte geometry")
    block, offset = divmod(slot, int(entries))
    if sliced:
        if raw.ndim != 2 or raw.shape[1] % 16:
            raise ValueError("Unsupported CP byte split")
        row = raw[block]
        begin = rank * row.numel()
        end = begin + row.numel()
    else:
        if raw.ndim != 3 or raw.shape[2] != 528:
            raise ValueError("Not the V4.1 528B codec")
        row = raw[block, offset]
        begin = offset * 528
        end = begin + 528
    start = max(begin, offset * 528)
    stop = min(end, (offset + 1) * 528)
    if start >= stop:
        return {}
    payload = row[start - begin : stop - begin].contiguous().cpu().numpy()
    return {
        str((start - offset * 528) // 16 + i): hashlib.sha256(v).hexdigest()
        for i, v in enumerate(payload.reshape(-1, 16))
    }


def remember(model, lid, prefix, indices, topk, pool):
    # Tensor references only: do not allocate tensors, transfer data, or add
    # kernels to the graph. Read after replay from the exact captured operands.
    import torch

    cache = getattr(model, "_v41_fixture_proposal_refs", None)
    if cache is None:
        cache = {}
        model._v41_fixture_proposal_refs = cache
    path = "graph" if torch.cuda.is_current_stream_capturing() else "eager"
    cache[path, int(prefix.numel()), int(lid)] = (prefix, indices, topk, pool)


def first_proposal(model, inputs, path):
    # Native caller verified the stream's full immutable prompt before TP
    # sync and excluded fake/warmup. Called AFTER actual eager/graph execution.
    import torch

    if not ENABLED:
        return
    root = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
    try:
        label = (root / "v41_cache_trace_stage").read_text()
    except FileNotFoundError:
        return
    match = re.fullmatch(r"([0-9a-f]{32}) (cold|fill|hit)\n", label)
    if not match:
        raise ValueError("Invalid model fixture marker")
    file = root / f"v41_cache_model_{match[1]}_{match[2]}_draft_read.json"
    if file.exists():
        return
    if path not in ("eager", "graph"):
        raise ValueError("Unexpected D execution path")
    from rtp_llm.models_py.modules.dsv41.kv_cache_utils import (
        as_tagged_attention_inputs,
    )

    static = as_tagged_attention_inputs(inputs.attention_inputs)
    src = getattr(static, "prefix_lengths_device", None)
    if src is None or src.numel() != 1:
        raise ValueError("Expected captured single D proposal")
    actual = int(src.reshape(-1)[0].item())
    if actual != 783:
        raise ValueError("Missing initial 783-prefix D proposal")
    by_layer = {}
    for lid in range(3):
        prefix, indices, topk, pool = model._v41_fixture_proposal_refs[path, 1, lid]
        if int(prefix.reshape(-1)[0].item()) != actual or int(pool.shape[2]) != 528:
            raise ValueError("Mismatched active model or SWA capture")
        # These are the global indices actually fed to the FlashMLA op.
        n = int(topk.flatten()[0].item())
        first = indices.reshape(-1, indices.size(-1))[0, :n]
        win = min(int(model._v4_args.window_size), actual)
        if n != win + int(model._gen_num_per_cycle) or win != 128:
            raise ValueError("Unexpected active proposal span")
        # All the queries attend to the same ordered committed window.
        if not torch.equal(
            indices.reshape(-1, indices.size(-1))[:, :n],
            first.expand(indices.numel() // indices.size(-1), n),
        ):
            raise ValueError("Nonmatching noncausal query slot layout")
        rows = {}
        for p in DRAFT_POS:
            if actual - win <= p < actual:
                slot = int(first[p - (actual - win)].item())
                if slot < 0:
                    raise ValueError("Unallocated committed KV on real D proposal")
                rows[str(p)] = chunks(pool, slot, pool.shape[1], 0, False)
        by_layer[str(lid)] = rows
    with file.open("x", encoding="ascii") as f:
        json.dump(
            {
                "path": path,
                "prefix": actual,
                "visible_start": actual - win,
                "layers": by_layer,
            },
            f,
            separators=(",", ":"),
        )


def _order(stage):
    if stage == "embedding":
        return (0, 0)
    if stage == "target_export":
        return (100, 0)
    if stage == "draft_import":
        return (101, 0)
    if re.fullmatch(r"layer_([0-9]|[1-3][0-9])", stage):
        return (1 + int(stage[6:]), 90)
    sub = (
        "reused_before_write.main",
        "reused_before_write.index",
        "after_write.main",
        "after_write.index",
        "readback.main",
        "readback.index",
        "readback.index_quant",
        "readback.index_scale",
    )
    if stage.startswith("layer_2.") and stage[8:] in sub:
        return (3, sub.index(stage[8:]))
    raise ValueError("Unrecognized V4.1 stage")


def _check(run):
    if run.get("schema") != 1 or not re.fullmatch(
        "[a-f0-9]{32}", run.get("run_id", "")
    ):
        raise ValueError("Invalid model trace envelope")
    phases = run["phases"]
    if set(phases) != set(PHASES):
        raise ValueError("Incomplete V4.1 phase set")
    for phase in PHASES:
        prefix = 512 if phase == "hit" else 0
        doc = phases[phase]
        if doc["prefix"] != prefix or set(doc["ranks"]) != {"0", "1"}:
            raise ValueError("Unexpected fixed fixture rank/prefix")
        ranks = doc["ranks"]
        _draft_summary(doc["draft_swa"])
        expected = {"embedding", "target_export", "draft_import"}
        expected.update(f"layer_{n}" for n in range(40))
        expected.update(
            {
                "layer_2.after_write.main",
                "layer_2.after_write.index",
                "layer_2.readback.main",
            }
        )
        if phase == "hit":
            expected.update(
                {
                    "layer_2.reused_before_write.main",
                    "layer_2.reused_before_write.index",
                }
            )
        extra = set(ranks["0"]) - expected
        if extra not in (
            {"layer_2.readback.index"},
            {"layer_2.readback.index_quant", "layer_2.readback.index_scale"},
        ):
            raise ValueError(f"Incomplete V4.1 {phase} indexer read")
        expected |= extra
        for rank, stages in ranks.items():
            if set(stages) != expected:
                raise ValueError(
                    f"Missing/unexpected V4.1 {phase} rank {rank} stages: {sorted(expected.symmetric_difference(stages))}"
                )
            for key, row in stages.items():
                p = row["positions"]
                if (
                    row["rank"] != int(rank)
                    or row["stage"] != key
                    or not isinstance(p, list)
                    or any(type(i) is not int or not 0 <= i < 783 for i in p)
                    or len(set(p)) != len(p)
                    or not re.fullmatch("[a-f0-9]{64}", row["sha256"])
                    or not row["row_shape"]
                    or any(type(n) is not int or n <= 0 for n in row["row_shape"])
                    or not isinstance(row["dtype"], str)
                    or set(row["samples"])
                    != {str(i) for i in p if i in (*SLOTS, 0, 512, 782)}
                    or any(
                        not re.fullmatch("[a-f0-9]{64}", digest)
                        for digest in row["samples"].values()
                    )
                ):
                    raise ValueError(f"Malformed {phase} rank {rank} {key} data")
        for stage in expected:
            p = [ranks[r][stage]["positions"] for r in ("0", "1")]
            if ".readback." in stage:
                if any(x != list(SLOTS) for x in p):
                    raise ValueError("Incomplete gathered cache reads")
            elif ".reused_before_write." in stage or ".after_write." in stage:
                wanted = [k for k in SLOTS if ".after_write." in stage or k < prefix]
                if sorted(p[0] + p[1]) != wanted:
                    raise ValueError(f"Cache row ownership mismatch: {phase}/{stage}")
            elif stage.startswith("layer_") and int(stage[6:]) >= 20:
                # CED may compact and redistribute valid rows after layer 20.
                allp = p[0] + p[1]
                if (
                    not allp
                    or 782 not in allp
                    or len(set(allp)) != len(allp)
                    or min(allp) < prefix
                ):
                    raise ValueError(f"Incomplete CED layout at {phase}/{stage}")
            elif sorted(p[0] + p[1]) != list(range(prefix, 783)):
                raise ValueError(f"Missing/duplicate logical token: {phase}/{stage}")
        for r in ranks:
            for key in ("target_export", "draft_import"):
                row = ranks[r][key]
                if row["dtype"] != "torch.bfloat16" or row["row_shape"] != [3 * 5120]:
                    raise ValueError("Invalid V4.1 DSpARK features")


def _draft_summary(d):
    read = d["read"]
    if (
        read["path"] not in ("eager", "graph")
        or read["prefix"] != 783
        or read["visible_start"] != 655
        or set(read["layers"]) != {"0", "1", "2"}
    ):
        raise ValueError("Incomplete first D proposal")
    writers = d["write"]
    if set(writers) != {"0", "1"}:
        raise ValueError("Missing P commit rank")
    expected = {str(p) for p in DRAFT_POS if p >= 655}
    chunks_all = {str(i) for i in range(33)}
    summary = {}
    for lid in ("0", "1", "2"):
        cached = {}
        for rank in ("0", "1"):
            if set(writers[rank]) != {"0", "1", "2"}:
                raise ValueError("Missing P commit layer")
            rec = writers[rank][lid]
            if (
                rec.get("rank") != int(rank)
                or rec.get("layer") != int(lid)
                or type(rec.get("sliced")) is not bool
            ):
                raise ValueError("Invalid P commit identity")
            for p, parts in rec["rows"].items():
                if p not in {str(x) for x in DRAFT_POS} or not set(parts) <= chunks_all:
                    raise ValueError("Unexpected P logical position/byte chunk")
                dest = cached.setdefault(p, {})
                for i, h in parts.items():
                    if not re.fullmatch("[0-9a-f]{64}", h) or (
                        i in dest and dest[i] != h
                    ):
                        raise ValueError("Inconsistent P byte slices")
                    dest[i] = h
        seen = read["layers"][lid]
        if set(seen) != expected:
            raise ValueError("Missing ordered D visible positions")
        for p, parts in seen.items():
            if (
                set(parts) != chunks_all
                or set(cached.get(p, {})) != chunks_all
                or any(not re.fullmatch("[0-9a-f]{64}", h) for h in parts.values())
            ):
                raise ValueError("Incomplete P/D 528-byte logical entry")
        summary[lid] = {
            "p_written": sorted(map(int, cached)),
            "d_read": sorted(map(int, seen)),
            "p_to_d_different_positions": sorted(
                int(p) for p in seen if seen[p] != cached[p]
            ),
            "p_canonical": cached,
        }
    return summary


def finalize(root, run_id):
    root = Path(root)
    if not re.fullmatch("[a-f0-9]{32}", run_id):
        raise ValueError("Invalid run id")
    phases = {}
    for phase in PHASES:
        ranks = {}
        for rank in ("0", "1"):
            path = root / f"v41_cache_model_{run_id}_{phase}_{rank}.jsonl"
            if not path.is_file():
                raise ValueError(f"Missing model trace: {phase}/{rank}")
            stages = {}
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row["stage"] in stages:
                    raise ValueError(f"Duplicate {phase}/{rank}/{row['stage']}")
                stages[row["stage"]] = row
            ranks[rank] = stages
        writers = {}
        for rank in ("0", "1"):
            f = root / f"v41_cache_model_{run_id}_{phase}_{rank}_draft_write.jsonl"
            if not f.is_file():
                raise ValueError(f"Missing {phase}/{rank} P draft cache writes")
            rs = [json.loads(l) for l in f.read_text().splitlines()]
            if len(rs) != 3:
                raise ValueError(f"Incomplete/duplicate {phase}/{rank} P draft write")
            writers[rank] = {str(v["layer"]): v for v in rs}
        f = root / f"v41_cache_model_{run_id}_{phase}_draft_read.json"
        if not f.is_file():
            raise ValueError(f"Missing {phase} first D proposal")
        phases[phase] = {
            "prefix": 512 if phase == "hit" else 0,
            "ranks": ranks,
            "draft_swa": {"write": writers, "read": json.loads(f.read_text())},
        }
    out = {"schema": 1, "run_id": run_id, "phases": phases}
    _check(out)
    (root / "v41_cache_model.json").write_text(
        json.dumps(out, separators=(",", ":")) + "\n"
    )
    return out


def within_run(doc):
    phases = doc["phases"]
    handoff = {}
    for phase in PHASES:
        ranks = phases[phase]["ranks"]
        same = all(
            all(
                stages["target_export"][k] == stages["draft_import"][k]
                for k in ("positions", "dtype", "row_shape", "sha256")
            )
            for stages in ranks.values()
        )
        handoff[phase] = same
    pool = {}
    for region in ("main", "index"):

        def samples(phase, point):
            out = {}
            for stages in phases[phase]["ranks"].values():
                out.update(stages[f"layer_2.{point}.{region}"]["samples"])
            return out

        fill = samples("fill", "after_write")
        hit = samples("hit", "reused_before_write")
        pool[region] = [int(i) for i in hit if hit[i] != fill[i]]
    p_to_d = {}
    for phase in PHASES:
        p_to_d[phase] = {
            k: {a: v for a, v in summary.items() if a != "p_canonical"}
            for k, summary in _draft_summary(phases[phase]["draft_swa"]).items()
        }
    return {
        "target_equals_draft_import": handoff,
        "fill_to_hit_different_cache_positions": pool,
        "draft_swa_p_to_d": p_to_d,
    }


def compare(old, new):
    for v in (old, new):
        _check(v)
    result = {}
    for phase in PHASES:
        o, n = (v["phases"][phase]["ranks"] for v in (old, new))
        diff = None
        for key in sorted(o["0"].keys() | n["0"].keys(), key=_order):
            for rank in ("0", "1"):
                if key not in o[rank] or key not in n[rank]:
                    diff = {
                        "status": "incomparable_representation",
                        "stage": key,
                        "rank": int(rank),
                    }
                    break
                a, b = o[rank][key], n[rank][key]
                if any(a[k] != b[k] for k in ("positions", "dtype", "row_shape")):
                    diff = {
                        "status": "incomparable_layout",
                        "stage": key,
                        "rank": int(rank),
                    }
                    break
            if diff:
                break
            for rank in ("0", "1"):
                a, b = o[rank][key], n[rank][key]
                if a["sha256"] != b["sha256"]:
                    diff = {
                        "status": "first_observed_difference",
                        "stage": key,
                        "rank": int(rank),
                        "different_sample_positions": [
                            int(i)
                            for i in a["samples"]
                            if a["samples"][i] != b["samples"][i]
                        ],
                    }
                    break
            if diff:
                break
        if not diff:
            od, nd = (v["phases"][phase]["draft_swa"] for v in (old, new))
            osum, nsum = _draft_summary(od), _draft_summary(nd)
            for lid in ("0", "1", "2"):
                a, b = osum[lid]["p_canonical"], nsum[lid]["p_canonical"]
                if set(a) != set(b):
                    diff = {
                        "status": "incomparable_layout",
                        "stage": f"p_draft_write_{lid}",
                    }
                elif a != b:
                    diff = {
                        "status": "first_observed_difference",
                        "stage": f"p_draft_write_{lid}",
                        "different_sample_positions": sorted(
                            int(i) for i in a if a[i] != b[i]
                        ),
                    }
                if diff:
                    break
            if not diff:
                for lid in ("0", "1", "2"):
                    a, b = od["read"]["layers"][lid], nd["read"]["layers"][lid]
                    if od["read"]["path"] != nd["read"]["path"]:
                        diff = {
                            "status": "incomparable_execution",
                            "stage": f"d_draft_read_{lid}",
                        }
                    elif a != b:
                        diff = {
                            "status": "first_observed_difference",
                            "stage": f"d_draft_read_{lid}",
                            "different_sample_positions": sorted(
                                int(i) for i in a if a[i] != b[i]
                            ),
                        }
                    if diff:
                        break
        result[phase] = diff or {"status": "identical_at_observed_boundaries"}
    return {
        "schema": 1,
        "by_phase": result,
        "within_run": {"old": within_run(old), "new": within_run(new)},
    }
