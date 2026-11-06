"""Read the realized K3.1 cache geometry before issuing smoke requests."""

import json
import re

from math import ceil

LONG_INPUT_TOKENS = 1_048_576
MAX_OUTPUT_TOKENS = 256
MTP_RESERVE_TOKENS = 4


def fastsafetensors_proof(logs):
    evidence = {}
    marker = re.compile(r"load method: ([\w-]+), finally choose load method: ([\w-]+)")
    for rank in range(8):
        path = logs / f"main_{rank}.log"
        selected = []
        with path.open(errors="replace") as stream:
            for lineno, line in enumerate(stream, 1):
                match = marker.search(line)
                if match:
                    if match.groups() != ("fastsafetensors", "fastsafetensors"):
                        raise ValueError(f"{path}:{lineno}: unexpected checkpoint loader")
                    selected.append(lineno)
        if len(selected) < 2:
            raise ValueError(f"{path}: missing FastSafetensors evidence for target and draft")
        evidence[str(rank)] = selected
    return evidence


def cache_geometry(engine_log, role, local_tp, *, long_case):
    text = engine_log.read_text(errors="replace")
    cp_rows = re.findall(
        r"CP sharded KV cache enabled: cp_rank=(\d+), cp_size=(\d+), "
        r"block_size=(\d+), virtual_block_size=(\d+)", text,
    )
    if set(tuple(map(int, row)) for row in cp_rows) != {
        (rank, local_tp, 1024, 1024 * local_tp) for rank in range(local_tp)
    }:
        raise ValueError(f"{role}: unexpected PageRR shard geometry")

    headers = list(re.finditer(
        r"\[RANK (\d+)\][^\n]*cache manager initialized with config CacheConfig\{\n",
        text,
    ))
    # Rank log records can interleave before the closing brace. Read the
    # leading topology fields only, before the next rank's header.
    layouts = {
        header[1]: text[header.end():headers[i + 1].start() if i + 1 < len(headers) else None]
        for i, header in enumerate(headers)
    }
    if set(layouts) != {str(rank) for rank in range(8)}:
        raise ValueError(f"{role}: missing finalized cache layout on some ranks")

    def field(body, key):
        match = re.search(rf"^  {re.escape(key)}=(.*)$", body, re.MULTILINE)
        if match is None:
            raise ValueError(f"{role}: missing finalized cache field {key}")
        return match[1]

    capacity = {}
    for rank, body in layouts.items():
        if (field(body, "layer_num"), field(body, "layer_all_num")) != ("93", "94"):
            raise ValueError(f"{role} rank {rank}: expected the full target and native MTP")
        tags = [tag.strip() for tag in field(body, "group_tags").strip("[]").split(",")]
        blocks = [int(value) for value in field(body, "group_block_nums").split(",")]
        groups = json.loads(field(body, "layers_by_group"))
        if (len(tags) != len(blocks) or len(tags) != len(groups)
                or set(tags) != {"full", "indexer_kv", "linear"}
                or 93 not in groups[tags.index("full")]):
            raise ValueError(f"{role} rank {rank}: target/draft full cache group is missing")
        if "kernel_seq_size_per_block=128" not in body:
            raise ValueError(f"{role} rank {rank}: expected 128-token kernel pages")
        if role == "prefill" and field(body, "linear_step") != "8":
            raise ValueError(f"{role} rank {rank}: expected eight-page KDA checkpoints")
        usable = {tag: count - 1 for tag, count in zip(tags, blocks)}
        if long_case:
            max_sequence = LONG_INPUT_TOKENS + MAX_OUTPUT_TOKENS
            # PageRR rounds up to a whole local page before the full-cache
            # manager adds the speculative reserve to that local length.
            paged_minimum = ceil(max_sequence / (1024 * local_tp)) + ceil(MTP_RESERVE_TOKENS / 1024)
            # Prefill retains one checkpoint per stripe, up to two tail
            # states and the destination allocated before dropping old tails.
            # Decode's replay pool only needs a live tail and its replacement.
            linear_minimum = (
                max_sequence // 8192 + 2 + 1 + (MTP_RESERVE_TOKENS - 1)
                if role == "prefill" else 3
            )
            if (min(usable["full"], usable["indexer_kv"]) < paged_minimum
                    or usable["linear"] < linear_minimum):
                raise ValueError(f"{role} rank {rank}: 1M cache capacity is insufficient: {usable}")
        elif role == "prefill" and usable["full"] != 95:
            raise ValueError(f"{role} rank {rank}: expected 95 usable pages for cache pressure")
        capacity[rank] = usable

    return dict(
        runtime_verified=True, runtime_layers=93, page_size=1024, kernel_page_size=128,
        source_shards=8, local_shards=local_tp, checkpoint_span=8192, cache_key_stride=8192,
        usable_pages=capacity, draft_layer_shared_full_group=True, role=role,
    )
