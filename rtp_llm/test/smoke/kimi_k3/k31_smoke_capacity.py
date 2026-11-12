"""Read the realized K3.1 cache geometry before issuing smoke requests."""

import json
import re

from math import ceil

LONG_INPUT_TOKENS = 1_048_576
MAX_OUTPUT_TOKENS = 256
MTP_RESERVE_TOKENS = 4


def long_prefill_linear_minimum(page_size=1024):
    # LINEAR allocation keeps physical-page slots. Canonical CP storage keys
    # span eight pages, but do not reduce the allocator's physical inventory.
    return (ceil((LONG_INPUT_TOKENS + MAX_OUTPUT_TOKENS) / page_size)
            + 2 + 1 + (MTP_RESERVE_TOKENS - 1))


def device_pool_usable_blocks(text, tag):
    legacy = re.findall(rf"kvc raw pool\[DEVICE/{re.escape(tag)}\]:[^\n]*?\btotal=(\d+)", text)
    physical = re.findall(
        rf"DeviceBlockPool\.cc:[^\n]*init success: pool_name={re.escape(tag)} "
        r"[^\n]*\bphysical_block_count=(\d+)", text,
    )
    # BlockTreeCache includes its reserved block zero in the physical count.
    usable = [int(value) for value in legacy] + [int(value) - 1 for value in physical]
    if not usable:
        return None
    if min(usable) <= 0 or len(set(usable)) != 1:
        raise ValueError(f"Device/{tag}: inconsistent usable pool capacity")
    return usable[0]


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


def cache_geometry(engine_log, role, local_tp, *, long_case, last_n_checkpoints=None,
                   last_n_page_tokens=None, calibration=False, prefill_tail_blocks=None,
                   prefill_test_block_num=None):
    if prefill_tail_blocks is not None and (
        role != "prefill" or type(prefill_tail_blocks) is not int
        or not 1 <= prefill_tail_blocks <= 5 or last_n_checkpoints is not None
    ):
        raise ValueError("Prefill retention must be N=1..5 outside the dedicated Last-N profile")
    if type(calibration) is not bool or calibration and last_n_checkpoints != 5:
        raise ValueError('calibration requires the explicit Last-5 CP geometry')
    if last_n_checkpoints is not None and (
        type(last_n_checkpoints) is not int or not 1 <= last_n_checkpoints <= 5 or local_tp != 8
    ):
        raise ValueError("Last-N geometry requires N=1..5 and eight local shards")
    if last_n_page_tokens is not None and (last_n_checkpoints is None or last_n_page_tokens not in (64, 128)):
        raise ValueError("Last-N page size requires an explicit Last-N smoke and 64 or 128 tokens")
    if prefill_test_block_num is not None and (
        type(prefill_test_block_num) is not int or prefill_test_block_num < 2
    ):
        raise ValueError("Prefill test_block_num must be an integer of at least 2")
    kernel_tokens = last_n_page_tokens or 128
    page_tokens = kernel_tokens if last_n_checkpoints is not None else 1024
    checkpoint_tokens = page_tokens * local_tp
    text = engine_log.read_text(errors="replace")
    cp_rows = re.findall(
        r"CP sharded KV cache enabled: cp_rank=(\d+), cp_size=(\d+), "
        r"block_size=(\d+), virtual_block_size=(\d+)", text,
    )
    if set(tuple(map(int, row)) for row in cp_rows) != {
        (rank, local_tp, page_tokens, checkpoint_tokens) for rank in range(local_tp)
    }:
        raise ValueError(f"{role}: unexpected PageRR shard geometry")

    headers = list(re.finditer(
        r"\[RANK (\d+)\][^\n]*cache manager initialized with config CacheConfig\{\n",
        text,
    ))
    # Rank log records can interleave; a block ends at CacheConfig's own
    # column-0 closing brace, not at the next rank header.
    layouts = {}
    for header in headers:
        close = re.search(r"^}\n", text[header.end():], re.MULTILINE)
        layouts[header[1]] = text[header.end():header.end() + close.end()] if close else text[header.end():]
    if set(layouts) != {str(rank) for rank in range(8)}:
        raise ValueError(f"{role}: missing finalized cache layout on some ranks")

    def field(body, key):
        """Shared (CacheConfig-level) field printed at two-space indent."""
        match = re.search(rf"^  {re.escape(key)}=(.*)$", body, re.MULTILINE)
        if match is None:
            raise ValueError(f"{role}: missing finalized cache field {key}")
        return match[1]

    def main_module(body):
        """Main module block: `  main_config:` followed by a CacheModuleConfig."""
        match = re.search(
            r"^  main_config:\n  CacheModuleConfig\{\n(.*?)^  \}\n", body,
            re.MULTILINE | re.DOTALL,
        )
        if match is None:
            raise ValueError(f"{role}: missing finalized main module cache layout")
        return match[1]

    def module_field(main, key):
        """CacheModuleConfig field printed at four-space indent."""
        match = re.search(rf"^    {re.escape(key)}=(.*)$", main, re.MULTILINE)
        if match is None:
            raise ValueError(f"{role}: missing finalized cache field {key}")
        return match[1]

    def layer_all_num(body, main):
        main_layers = int(module_field(main, "layer_num"))
        modules = int(field(body, "mtp_sub_configs.size()"))
        mtp_layers = [int(value) for value in re.findall(r"^      layer_num=(\d+)$", body, re.MULTILINE)]
        if len(mtp_layers) != modules:
            raise ValueError(f"{role}: MTP module layer counts were not fully logged")
        return str(main_layers + sum(mtp_layers))

    capacity = {}
    for rank, body in layouts.items():
        physical_linear_pages = False
        main = main_module(body)
        if (module_field(main, "layer_num"), layer_all_num(body, main)) != ("93", "94"):
            raise ValueError(f"{role} rank {rank}: expected the full target and native MTP")
        tags = [tag.strip() for tag in module_field(main, "group_tags").strip("[]").split(",")]
        blocks = [int(value) for value in module_field(main, "group_block_nums").split(",")]
        groups = json.loads(module_field(main, "layers_by_group"))
        if (len(tags) != len(blocks) or len(tags) != len(groups)
                or set(tags) != {"full", "indexer_kv", "linear"}
                or 92 not in groups[tags.index("full")]):
            raise ValueError(f"{role} rank {rank}: target/draft full cache group is missing")
        if f"kernel_seq_size_per_block={kernel_tokens}\n" not in body:
            raise ValueError(f"{role} rank {rank}: expected {kernel_tokens}-token kernel pages")
        if last_n_checkpoints is not None:
            expected_n = last_n_checkpoints if role == "prefill" and not calibration else 2147483647
            if (module_field(main, "reuse_cache_tail_blocks") != str(expected_n)
                    or field(body, "seq_size_per_block") != str(page_tokens)
                    or field(body, "local_kv_cache_sharded") != "true"):
                raise ValueError(f"{role} rank {rank}: Last-N runtime configuration differs")
            linear = re.search(r"LinearKVCacheSpec\{([^}]+)\}", main)
            if linear is None or re.search(
                rf"^      cache_key_token_stride={page_tokens}$", linear[1], re.MULTILINE
            ) is None:
                raise ValueError(f"{role} rank {rank}: physical LINEAR checkpoint stride differs")
        elif role == "prefill":
            legacy_step = re.search(r"^  linear_step=(.*)$", body, re.MULTILINE)
            if legacy_step is not None:
                valid_step = legacy_step[1] == "8"
            else:
                # BlockTreeCache publishes CP key strides on its cloned
                # topology; CacheConfig logs the physical per-shard spec.
                # The CP rows above prove the corresponding virtual span.
                linear = re.search(r"LinearKVCacheSpec\{([^}]+)\}", main)
                valid_step = (
                    field(body, "seq_size_per_block") == "1024"
                    and field(body, "local_kv_cache_sharded") == "true"
                    and local_tp == 8
                    and linear is not None
                    and re.search(r"^      cache_key_token_stride=1024$", linear[1], re.MULTILINE)
                    is not None
                )
                physical_linear_pages = True
            if not valid_step:
                raise ValueError(f"{role} rank {rank}: expected eight-page KDA checkpoints")
        if prefill_tail_blocks is not None and module_field(main, "reuse_cache_tail_blocks") != str(prefill_tail_blocks):
            raise ValueError(f"{role} rank {rank}: Prefill checkpoint retention differs")
        usable = {tag: count - 1 for tag, count in zip(tags, blocks)}
        if last_n_checkpoints is not None:
            # The 304-slot profile includes an actual cold 1M request and a
            # no-reuse 10I+128 precision reference. Reserve the next page too.
            full_minimum = 16 if calibration else ceil(LONG_INPUT_TOKENS / checkpoint_tokens) + 2
            cold_reference_minimum = ceil((10 * checkpoint_tokens + 128) / page_tokens) + 4
            linear_minimum = (128 if calibration else cold_reference_minimum) if role == 'prefill' else 3
            if (min(usable["full"], usable["indexer_kv"]) < full_minimum
                    or usable["linear"] < linear_minimum):
                raise ValueError(f"{role} rank {rank}: Last-N smoke capacity is insufficient: {usable}")
        elif long_case:
            max_sequence = LONG_INPUT_TOKENS + MAX_OUTPUT_TOKENS
            # PageRR rounds up to a whole local page before the full-cache
            # manager adds the speculative reserve to that local length.
            paged_minimum = ceil(max_sequence / (1024 * local_tp)) + ceil(MTP_RESERVE_TOKENS / 1024)
            # BlockTreeCache allocates physical LINEAR slots; the legacy
            # allocator retains one checkpoint per stripe. Also reserve tail
            # states and the destination allocated before dropping old tails.
            # Decode's replay pool only needs a live tail and its replacement.
            linear_minimum = 3
            if role == "prefill":
                if prefill_tail_blocks is not None:
                    linear_minimum = prefill_tail_blocks + 2
                elif physical_linear_pages:
                    linear_minimum = long_prefill_linear_minimum(int(field(body, "seq_size_per_block")))
                else:
                    linear_minimum = max_sequence // 8192 + 2 + 1 + (MTP_RESERVE_TOKENS - 1)
            if (min(usable["full"], usable["indexer_kv"]) < paged_minimum
                    or usable["linear"] < linear_minimum):
                raise ValueError(f"{role} rank {rank}: 1M cache capacity is insufficient: {usable}")
        elif role == "prefill":
            # The launch profile sets physical blocks, including reserved block zero.
            expected = (prefill_test_block_num if prefill_test_block_num is not None else 96) - 1
            if usable["full"] != expected:
                raise ValueError(
                    f"{role} rank {rank}: expected {expected} usable pages for cache pressure, "
                    f"got {usable['full']}"
                )
        capacity[rank] = usable

    return dict(
        runtime_verified=True, runtime_layers=93, page_size=page_tokens, kernel_page_size=kernel_tokens,
        source_shards=8, local_shards=local_tp, checkpoint_span=checkpoint_tokens, cache_key_stride=checkpoint_tokens,
        usable_pages=capacity, draft_layer_shared_full_group=True, role=role,
        **({"reuse_cache_tail_blocks": prefill_tail_blocks} if prefill_tail_blocks is not None else {}),
        **({"reuse_cache_tail_blocks": last_n_checkpoints if role == "prefill" and not calibration else 2147483647,
            "last_n_calibration": calibration}
           if last_n_checkpoints is not None else {}),
    )
