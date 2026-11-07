"""Check borrowed Prefill state ownership using native execution evidence."""


def audit_c7_miss(events, seed_request_ids, probe_request_ids, *, group_tag="linear"):
    root = events[0]
    seeds = [row for row in root if row.get('event') == 'linear_cache_publish_candidates'
             and row.get('request_id') in seed_request_ids and row.get('group_tag') == group_tag
             and [item['tokens'] for item in row['checkpoints']] == list(range(8192, 12289, 1024))]
    if not seeds:
        raise ValueError('C7 negative control lacks a seed with only C8..C12 published')
    proof = []
    for rid in probe_request_ids:
        rounds = sorted((row for row in root if row.get('event') == 'linear_prefill_round_completed'
                         and row.get('request_id') == rid and row.get('group_tag') == group_tag),
                        key=lambda row: row['time_ns'])
        if (not rounds or rounds[0]['prefix_tokens'] != 0 or rounds[0]['input_block'] != 0
                or not any(seed['time_ns'] <= rounds[0]['time_ns'] for seed in seeds)
                or any(row.get('event') == 'linear_cache_matched' and row.get('request_id') == rid
                       and row.get('group_tag') == group_tag and row.get('checkpoint_tokens', 0) > 0
                       for row in root)):
            raise ValueError('C7 negative probe did not actually recompute from an empty LINEAR state')
        proof.append(dict(request_id=rid, reused_tokens=0, first_input_block=0))
    if not proof:
        raise ValueError('C7 negative control has no actual probe identities')
    return proof


def audit_checkpoint_reuse(events, seed_request_ids, probe_request_ids, source_tier,
                           *, checkpoint_tokens=10240, group_tag="linear"):
    if source_tier not in {"DEVICE", "HOST", "DISK"} or not seed_request_ids or not probe_request_ids:
        raise ValueError("tier checkpoint proof needs explicit seed/probe identities")
    root = events[0]
    seeds = [row for row in root if row.get("event") == "linear_cache_publish_candidates"
             and row.get("request_id") in seed_request_ids and row.get("group_tag") == group_tag
             and [item["tokens"] for item in row["checkpoints"]] == list(range(8192, 12289, 1024))]
    if not seeds:
        raise ValueError("seed did not publish the actual Last-5 C8..C12 checkpoints")
    seed_keys = {item["cache_key"] for seed in seeds for item in seed["checkpoints"]
                 if item["tokens"] == checkpoint_tokens and "cache_key" in item}
    if not seed_keys:
        raise ValueError("seed checkpoint lacks its actual content cache key")
    proofs = []
    for request_id in probe_request_ids:
        rounds = sorted((row for row in root if row.get("event") == "linear_prefill_round_completed"
                         and row.get("request_id") == request_id and row.get("group_tag") == group_tag),
                        key=lambda row: row["time_ns"])
        if not rounds or rounds[0]["prefix_tokens"] != checkpoint_tokens:
            raise ValueError("probe did not actually continue from the requested LINEAR checkpoint")
        first = rounds[0]
        matches = [row for row in root if row.get("event") == "linear_cache_matched"
                   and row.get("request_id") == request_id and row.get("group_tag") == group_tag
                   and row.get("checkpoint_tokens") == checkpoint_tokens
                   and row.get("cache_key") in seed_keys
                   and row.get("source_tier") == source_tier
                   and any(seed["time_ns"] <= row["time_ns"] for seed in seeds)
                   and row["time_ns"] <= first["time_ns"]]
        if not matches:
            raise ValueError("no actual LINEAR match from the required source tier")
        if source_tier == "DEVICE":
            if not any(row["source_block"] == first["input_block"] for row in matches):
                raise ValueError("Device match was not the state consumed by the first chunk")
            if not any(item['block'] == first['input_block'] for seed in seeds for item in seed['checkpoints']
                       if item['tokens'] == checkpoint_tokens):
                raise ValueError("immediate Device input was not the state published by the seed")
        else:
            loaded = [row for row in root if row.get("event") == "linear_cache_loaded"
                      and row.get("request_id") == request_id and row.get("group_tag") == group_tag
                      and row.get("checkpoint_tokens") == checkpoint_tokens
                      and row.get("source_tier") == source_tier
                      and row.get("device_block") == first["input_block"]
                      and any(match["time_ns"] <= row["time_ns"] for match in matches)
                      and row["time_ns"] <= first["time_ns"]]
            if not loaded:
                raise ValueError("matched lower-tier state has no completed load into the consumed block")
            writes = [row for row in root if row.get('event') == 'linear_cache_transfer_completed'
                      and row.get('group_tag') == group_tag and row.get('target_tier') == source_tier
                      and row.get('crc_enabled') is True
                      and row.get('payload_bytes', 0) > 0
                      and any(row.get('cache_key') == match.get('cache_key')
                              and row.get('target_block') == match.get('source_block')
                              and any(seed['time_ns'] <= row['time_ns'] for seed in seeds)
                              and row['time_ns'] <= match['time_ns'] for match in matches)]
            if not writes:
                raise ValueError('lower-tier checkpoint has no completed write into the matched source block')
        proofs.append(dict(request_id=request_id, checkpoint_tokens=checkpoint_tokens,
                           source_tier=source_tier, input_block=first["input_block"],
                           cache_key=matches[0]['cache_key']))
    return proofs


def audit_prefill_states(events, *, tail_blocks=5, page_tokens=128,
                         checkpoint_tokens=1024, max_chunk_tokens=65536,
                         require_one_million=True, root_rank=0):
    if not 1 <= tail_blocks <= 5 or checkpoint_tokens % page_tokens:
        raise ValueError("unsupported Last-N audit geometry")
    root = events[root_rank]
    allocations = {}
    rounds = {}
    publications = {}
    for row in root:
        event = row.get("event")
        if event not in {"linear_prefill_allocated", "linear_prefill_round_completed",
                         "linear_cache_publish_candidates"}:
            continue
        key = row["request_id"], row["group_tag"], row.get("batch", 0)
        target = (allocations if event == "linear_prefill_allocated" else
                  rounds if event == "linear_prefill_round_completed" else publications)
        target.setdefault(key, []).append(row)
    if not allocations or set(rounds) - set(allocations) or set(publications) - set(allocations):
        raise ValueError("native Last-N allocation/round evidence is missing")

    proofs, million = [], False
    for key, history in rounds.items():
        history = sorted(history, key=lambda row: row["time_ns"])
        # A failed load can change the reuse boundary before any forward.
        ready = [row for row in allocations[key] if row["time_ns"] <= history[0]["time_ns"]]
        if not ready:
            raise ValueError(f"{key}: no allocation before execution")
        allocation = max(ready, key=lambda row: row["time_ns"])
        prompt, reused = allocation["prompt_tokens"], allocation["initial_reuse_tokens"]
        if (allocation["configured_tail_blocks"] != tail_blocks
                or allocation["page_tokens"] != page_tokens
                or allocation["checkpoint_tokens"] != checkpoint_tokens
                or not 0 <= reused < prompt or reused % checkpoint_tokens):
            raise ValueError(f"{key}: native Last-N geometry differs")
        complete = prompt // checkpoint_tokens
        selected = {index * checkpoint_tokens
                    for index in range(max(1, complete - tail_blocks + 1), complete + 1)}
        expected_new = len({point for point in selected if point > reused})
        separate_tail = bool(prompt % checkpoint_tokens)
        expected_new += separate_tail
        if (allocation["has_separate_tail"] != separate_tail
                or allocation["planned_new_blocks"] != expected_new
                or allocation["borrowed_block"] <= 0
                or allocation["owned_state_blocks"] < expected_new
                or (reused == 0 and allocation["owned_state_blocks"] != expected_new)):
            raise ValueError(f"{key}: cold working-state allocation is not bounded by N plus tail")

        frontier, previous_output = reused, None
        previous_time = allocation["time_ns"]
        frozen, written, alias_rounds = {}, {}, 0
        for row in history:
            end, writes = row["end_tokens"], row["writes"]
            if (row["prefix_tokens"] != frontier or not frontier < end <= prompt
                    or end - frontier > max_chunk_tokens
                    or (end != prompt and end % page_tokens)):
                raise ValueError(f"{key}: chunk boundaries are discontinuous or invalid")
            if ((previous_output is not None and row["input_block"] != previous_output)
                    or (frontier == 0 and row["input_block"] != 0)
                    or (frontier > 0 and row["input_block"] <= 0)):
                raise ValueError(f"{key}: next chunk did not read the actual previous output")
            expected_writes = sorted(point for point in selected if frontier < point <= end)
            if end not in selected:
                expected_writes.append(end)
            blocks = [write["block"] for write in writes]
            if ([write["tokens"] for write in writes] != expected_writes
                    or len(blocks) != len(set(blocks)) or any(block <= 0 for block in blocks)
                    or not writes or row["output_block"] != blocks[-1]):
                raise ValueError(f"{key}: execution has missing points or concurrent duplicate writers")
            if row["input_block"] > 0 and row["input_block"] in blocks:
                alias_rounds += 1
                for rank in range(8):
                    if not any(
                        snapshot.get("event") == "linear_prefill_conv_snapshot_submitted"
                        and snapshot.get("world_rank") == rank
                        and previous_time <= snapshot["time_ns"] <= row["time_ns"]
                        and any(item == dict(prefix_tokens=frontier, query_tokens=end-frontier,
                                             input_block=row["input_block"])
                                for item in snapshot["rows"])
                        for snapshot in events[rank]
                    ):
                        raise ValueError(f"{key} rank{rank}: aliased conv input has no snapshot submission")
            for write in writes:
                point, block = write["tokens"], write["block"]
                if block in frozen:
                    raise ValueError(f"{key}: a finalized checkpoint block was overwritten")
                if write["retained"] != (point in selected):
                    raise ValueError(f"{key}: temporary state marked reusable")
                if point in selected:
                    frozen[block] = point
                    written[point] = block, row["time_ns"]
                elif block != allocation["borrowed_block"]:
                    raise ValueError(f"{key}: temporary output allocated another working block")
            frontier, previous_output, previous_time = end, row["output_block"], row["time_ns"]

        published = set()
        for publication in publications.get(key, ()):
            if publication["prompt_tokens"] != prompt or publication["target_tier"] not in {"DEVICE", "HOST", "DISK"}:
                raise ValueError(f"{key}: publication geometry or tier differs")
            checkpoints = publication["checkpoints"]
            points = [item["tokens"] for item in checkpoints]
            if len(points) > tail_blocks or len(points) != len(set(points)):
                raise ValueError(f"{key}: publication exceeds Last-N or duplicates a point")
            for item in checkpoints:
                point = item["tokens"]
                if (point not in written or item["block"] != written[point][0]
                        or publication["time_ns"] < written[point][1]):
                    raise ValueError(f"{key}: publication exposes an uninitialized or temporary state")
                published.add(point)
        is_million = prompt == 1_048_576 and reused == 0 and frontier == prompt
        if is_million:
            if published != selected or len(history) < 16:
                raise ValueError("1M cold request lacks final Last-N publication or actual chunk execution")
            million = True
        proofs.append(dict(request_id=key[0], group_tag=key[1], prompt_tokens=prompt,
                           initial_reuse_tokens=reused, completed_tokens=frontier,
                           new_state_blocks=expected_new, rounds=len(history),
                           aliased_rounds=alias_rounds, published_checkpoints=sorted(published)))
    if require_one_million and not million:
        raise ValueError("no actual cold 1M/64K Last-N request")
    return dict(tail_blocks=tail_blocks, page_tokens=page_tokens,
                checkpoint_tokens=checkpoint_tokens, max_chunk_tokens=max_chunk_tokens,
                cold_one_million_verified=million, requests=proofs)
