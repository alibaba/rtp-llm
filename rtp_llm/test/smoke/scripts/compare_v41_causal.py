"""Compare public per-rank causal reports. Never opens the private relay."""

import argparse
import json
import re
from pathlib import Path

CORE = {
    "input_ids",
    "hash_ids",
    "token_mask",
    "incoming.observed",
    "previous_mix.observed",
    "engram.in",
    "engram_wkv.in",
    "engram_wkv.out",
    "engram.out",
    "attn_hc.readout",
    "attn_hc.post",
    "attn_hc.comb",
    "attn_norm.in",
    "attn_norm.out",
    "attention.in",
    "rope",
    "qr",
    "new_kv",
    "swa_tail.observed",
    "swa_tail.used",
    "q_rope",
    "flash.indices",
    "flash.lens",
    "flash.out",
    "o_proj",
    "attention.out",
    "moe.in",
    "moe.out",
    "outgoing",
    "next_mix",
}
FIXED = ("incoming.used", "previous_mix.used", "swa_tail.used")
CELLS = {
    "replay_nn": (False, False),
    "replay_on": (True, False),
    "replay_no": (False, True),
    "replay": (True, True),
}
PROMPT = "d570954fff2202dfde49e873ece0fc1adcb2df1c6a9095cb98fe6601ac4645c4"


def read(root, role, mode, rank):
    d = json.loads((root / f"v41_causal_{mode}_r{rank}.json").read_text())
    if (
        d.get("schema"),
        d.get("family"),
        d.get("role"),
        d.get("rank"),
        d.get("mode"),
    ) != (1, "v41", role, rank, mode) or not re.fullmatch("[a-f0-9]{32}", d["nonce"]):
        raise ValueError("invalid V4.1 causal manifest")
    pos = d["positions"]
    if (
        not pos
        or len(set(pos)) != len(pos)
        or not all(type(n) is int and 512 <= n < 783 for n in pos)
    ):
        raise ValueError("invalid V4.1 positions")
    seq = [key for key, value in d["events"]]
    required = CORE.union(FIXED) if mode in CELLS else CORE
    if not required.issubset(seq) or len(set(seq)) != len(seq):
        raise ValueError(
            "missing or ambiguous stage: " + ",".join(sorted(required.difference(seq)))
        )
    if not d["weights"] or d["skipped"]:
        raise ValueError("runtime weights incomplete: " + ",".join(d["skipped"]))
    cp, g, o = d["cp"], d["swa_geometry"], d["outcome"]
    live = cp["live_rows"]
    if (
        not re.fullmatch("[a-f0-9]{32}", d["request_id"])
        or cp["size"] != 2
        or cp["rank"] != rank
        or len(live) != len(pos)
        or not 1 <= cp["local_rows"] <= 1024
        or live != sorted(set(live))
        or any(type(i) is not int or not 0 <= i < cp["local_rows"] for i in live)
    ):
        raise ValueError("invalid physical/logical CP correspondence")
    if (
        g["logical_start"] != 385
        or len(g["slots"]) != 127
        or not 0 < g["entries"] <= 65536
    ):
        raise ValueError("invalid SWA slot extent")
    pages = set()
    for logical, (page, offset) in enumerate(g["slots"], 385):
        if [page, offset] == [-1, -1]:
            continue
        if page not in pages and page != len(pages) or offset != logical % g["entries"]:
            raise ValueError("invalid SWA page/ring correspondence")
        pages.add(page)
    if (
        o["input_digest"] != PROMPT
        or o["reuse"] != 512
        or type(o["decode_reuse"]) is not int
        or not 0 <= o["decode_reuse"] <= 783
        or not 2 <= o["output_count"] <= 24
        or not re.fullmatch("[0-9a-f]{64}", o["output_hash"])
        or type(o["matches_golden"]) is not bool
        or not 1 <= len(o["accepts"]) <= o["output_count"] - 1
        or any(type(n) is not int or not 0 <= n <= 3 for n in o["accepts"])
        or sum(o["accepts"]) > o["output_count"] - 1
    ):
        raise ValueError("invalid linked HTTP/DSpARK outcome")
    checks = [v for _, v in d["events"]] + list(d["weights"].values())
    for v in checks:
        if (
            not re.fullmatch("[0-9a-f]{64}", v["sha256"])
            or not v["dtype"]
            or not isinstance(v["shape"], list)
        ):
            raise ValueError("invalid V4.1 fingerprint")
        if "rows" in v and (
            len(v["rows"]) != len(pos)
            or not all(re.fullmatch("[0-9a-f]{64}", i) for i in v["rows"])
        ):
            raise ValueError("invalid V4.1 row fingerprints")
    return d


def compare_gate(old, new):
    records = [r.get("engram_gate", {}) for r in (old, new)]
    if not all(r.get("complete") for r in records):
        return {"status": "incomplete"}
    a, b = records
    same = {
        name: a["outcomes"][name] == b["outcomes"][name]
        for name in ("actual", "dense_native", "torch_ref")
    }
    pos = old.get("positions")
    first = {}
    if pos is not None and pos == new.get("positions"):
        for name in same:
            x, y = (r["outcome_rows"][name] for r in records)
            if len(x) != len(y) or len(x) != len(pos):
                return {"status": "incomplete", "reason": "gate row count mismatch"}
            first[name] = min((p for p, i, j in zip(pos, x, y) if i != j), default=None)
    return dict(
        status="compared",
        operands_equal=a["operands"] == b["operands"],
        eps_equal=a["eps"] == b["eps"],
        outcomes_equal=same,
        first_logical_difference=first,
        entry_strided_operands=[r["entry_strided_operands"] for r in records],
        pre_fix_paths=[r["pre_fix_path"] for r in records],
        actual_path=[r["actual_path"] for r in records],
        actual_matches_native=[
            r["outcomes"]["actual"] == r["outcomes"]["dense_native"] for r in records
        ],
        actual_matches_reference=[
            r["outcomes"]["actual"] == r["outcomes"]["torch_ref"] for r in records
        ],
        cubin_sha256=[
            r["launches"].get("actual", {}).get("binary_sha256") for r in records
        ],
    )


def compare(old, new):
    reports = [{"rank": r} for r in range(2)]
    nonces = set()
    runs, outcomes, anchors, layer_out = {}, {}, {}, {}
    for mode in ("natural", *CELLS):
        positions = []
        for rank in (0, 1):
            a, b = (
                read(p, role, mode, rank)
                for p, role in ((old, "oldfst"), (new, "newfst"))
            )
            nonces.update((a["nonce"], b["nonce"]))
            if (
                a["positions"] != b["positions"]
                or a["cp"] != b["cp"]
                or a["swa_geometry"] != b["swa_geometry"]
                or a["weights"].keys() != b["weights"].keys()
            ):
                raise ValueError("different CP or weight layout")
            positions.extend(a["positions"])
            weights = [k for k in a["weights"] if a["weights"][k] != b["weights"][k]]
            left, right = dict(a["events"]), dict(b["events"])
            for doc, events in ((a, left), (b, right)):
                role = doc["role"]
                key = role, mode
                if rank == 0:
                    runs[key], outcomes[key] = doc["request_id"], doc["outcome"]
                    if (
                        sum(v == runs[key] for (r, _m), v in runs.items() if r == role)
                        != 1
                    ):
                        raise ValueError("reused causal request identity")
                elif runs[key] != doc["request_id"] or outcomes[key] != doc["outcome"]:
                    raise ValueError(
                        "two CP reports are not from the same HTTP request"
                    )
                layer_out[role, mode, rank] = events["outgoing"]["sha256"]
                if mode == "natural":
                    anchors[role, rank] = doc, events
                    continue
                base, source = anchors[role, rank]
                if (
                    doc["positions"] != base["positions"]
                    or doc["cp"] != base["cp"]
                    or doc["swa_geometry"] != base["swa_geometry"]
                    or doc["weights"] != base["weights"]
                ):
                    raise ValueError(
                        "control CP/cache layout or weights changed between cells"
                    )
                donor = anchors["oldfst", rank][1]
                for name, use_old in zip(
                    ("incoming", "previous_mix", "swa_tail"), (True, *CELLS[mode])
                ):
                    target = name + ".observed"
                    if events[target] != source[target]:
                        raise ValueError(
                            "natural operands drifted before intervention: " + name
                        )
                    if events[name + ".used"] != (donor if use_old else source)[target]:
                        raise ValueError("expected crossed operand not shared: " + name)
                for name in (
                    "input_ids",
                    "hash_ids",
                    "token_mask",
                    "rope",
                    "flash.indices",
                    "flash.lens",
                ):
                    if events[name] != source[name]:
                        raise ValueError(
                            "uncontrolled execution metadata changed: " + name
                        )
            first = None
            skip = (
                {
                    "incoming.observed",
                    "previous_mix.observed",
                    "swa_tail.observed",
                    *FIXED,
                }
                if mode in CELLS
                else set()
            )
            for (ka, va), (kb, vb) in zip(a["events"], b["events"]):
                if ka != kb:
                    first = {"execution_path_old": ka, "execution_path_new": kb}
                    break
                if va != vb and ka not in skip:
                    rows = []
                    if "rows" in va and "rows" in vb:
                        rows = [
                            p
                            for p, x, y in zip(a["positions"], va["rows"], vb["rows"])
                            if x != y
                        ]
                    first = {
                        "stage": ka,
                        "first_logical_position": min(rows) if rows else None,
                    }
                    break
            if first is None and len(left) != len(right):
                raise ValueError("unmatched trailing V4.1 stages")
            reports[rank][mode] = {
                "engram_gate": compare_gate(a, b),
                "first_observed_difference": first,
                "differing_native_weights": weights,
                "causal_interpretation": (
                    "weight_or_address_difference_do_not_infer_kernel"
                    if weights
                    else (
                        "different_execution_paths"
                        if first and "execution_path_old" in first
                        else (
                            "first_observed_boundary_not_kernel_root"
                            if first
                            else "no_difference_at_recorded_boundaries"
                        )
                    )
                ),
            }
        if sorted(positions) != list(range(512, 783)):
            raise ValueError("incomplete CP coverage")
    if len(nonces) != 1:
        raise ValueError("reports do not belong to the same paired invocation")
    controls = {}
    for role in ("oldfst", "newfst"):
        controls[role] = {m: outcomes[role, m] for m in ("natural", *CELLS)}
    old_stable = all(
        outcomes["oldfst", m] == outcomes["oldfst", "replay_nn"]
        and all(
            layer_out["oldfst", m, r] == layer_out["oldfst", "replay_nn", r]
            for r in (0, 1)
        )
        for m in CELLS
    )
    contrasts = {}
    incomplete = (
        []
        if old_stable
        else ["OldFST four-cell identical-input control not reproducible"]
    )
    for name, source, m1, m2 in (
        ("mix_at_natural_swa", "previous_mix", "replay_nn", "replay_on"),
        ("mix_at_old_swa", "previous_mix", "replay_no", "replay"),
        ("swa_at_natural_mix", "swa_tail", "replay_nn", "replay_no"),
        ("swa_at_old_mix", "swa_tail", "replay_on", "replay"),
    ):
        actual_delta = any(
            anchors["oldfst", r][1][source + ".observed"]
            != anchors["newfst", r][1][source + ".observed"]
            for r in (0, 1)
        )
        a, b = outcomes["newfst", m1], outcomes["newfst", m2]
        if a["decode_reuse"] != b["decode_reuse"]:
            incomplete.append(name + ": decode cache coverage differs")
        contrasts[name] = dict(
            cells=[m1, m2],
            operand_changed=actual_delta,
            layer1_changed=any(
                layer_out["newfst", m1, r] != layer_out["newfst", m2, r] for r in (0, 1)
            ),
            output_changed=a["output_hash"] != b["output_hash"],
            accepted=[sum(a["accepts"]), sum(b["accepts"])],
        )
        if not actual_delta and (a != b or contrasts[name]["layer1_changed"]):
            incomplete.append(name + ": different outcome without an operand contrast")
    return dict(
        status="incomplete" if incomplete else "compared",
        incomplete_reasons=incomplete,
        family="v41",
        nonce=next(iter(nonces)),
        ranks=reports,
        outcomes=controls,
        new_factor_contrasts=contrasts,
        interpretation="Observed boundaries; absence of a difference is not proof for other layers.",
    )


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("old", type=Path)
    cli.add_argument("new", type=Path)
    args = cli.parse_args()
    try:
        result = compare(args.old, args.new)
    except (OSError, KeyError, ValueError, TypeError) as error:
        print(json.dumps(dict(status="incomplete", family="v41", reason=str(error))))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "compared" else 2


if __name__ == "__main__":
    raise SystemExit(main())
