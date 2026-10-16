"""CPU-only admission checks for an explicit, topology-independent token budget.

Cache-block padding is conservative for packed CP execution when the block is
a multiple of the execution alignment. This bounds token capacity, not total
GPU memory (weights, KV pools, communication and JIT have separate budgets).
"""


def aligned(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def positive_tokens(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def grid_workspace_tokens(payload):
    metadata = payload.get("generator", {})
    if metadata.get("workspace_policy") not in (None, "none"):
        raise ValueError(
            "workspace_policy is no longer supported; regenerate with --workspace-tokens"
        )
    value = metadata.get("workspace_tokens")
    return None if value is None else positive_tokens(value, "workspace_tokens")


def grid_token_budget(payload):
    workspace = grid_workspace_tokens(payload)
    value = (
        payload.get("generator", {})
        .get("parameters", {})
        .get("max_batch_tokens", workspace)
    )
    return positive_tokens(value, "max_batch_tokens")


def workspace_capacity(tokens, block):
    positive_tokens(tokens, "workspace_tokens")
    positive_tokens(block, "cache_alignment")
    capacity = tokens // block * block
    if not capacity:
        raise ValueError("workspace_tokens cannot hold one cache block")
    return capacity


def validate_token_budget(
    cases, *, workspace_tokens, commit_tail, block, output_tokens=1, token_budget=None
):
    """Validate packed sums for measurements, concurrent seeds and the probe.

    Reserve output before rounding each request to a cache block. Unlike a
    batch-times-longest rectangle, this admits a long request with short peers.
    Summary fields are never trusted.
    """
    capacity = workspace_capacity(workspace_tokens, block)
    if output_tokens != 1 or commit_tail <= 0 or commit_tail % block:
        raise ValueError("token budget requires aligned seed tail and decode=1")
    token_budget = (
        workspace_tokens
        if token_budget is None
        else positive_tokens(token_budget, "token_budget")
    )
    stats = []

    def phase(case_id, name, lengths):
        count = sum(n for n, _ in lengths)
        total = sum(n * length for n, length in lengths)
        padded = sum(
            n * aligned(length + output_tokens, block) for n, length in lengths
        )
        if total > token_budget or padded > capacity:
            raise ValueError(
                f"case {case_id} {name}: input sum={total} (budget={token_budget}), "
                f"padded token sum with output reserve={padded} "
                f"(workspace={capacity}); shorten requests or reduce batch"
            )
        stats.append(
            dict(
                case_id=case_id,
                phase=name,
                requests=count,
                input_tokens=total,
                workspace_tokens=padded,
            )
        )

    phase("probe", "probe", [(1, max(2 * block, block + commit_tail))])
    for index, case in enumerate(cases):
        case_id = case.get("case_id", index)
        batch = positive_tokens(case.get("batch_size", 1), "batch_size")
        policy = case.get("prefix_policy", "independent")
        if policy not in ("independent", "shared_by_group"):
            raise ValueError(f"case {case_id}: invalid prefix policy")
        groups = case.get("request_groups")
        if groups is None:
            groups = [
                dict(
                    count=batch,
                    input_len=case["input_len"],
                    cache_len=case.get("cache_len", 0),
                )
            ]
        measure, seeds = [], []
        for group in groups:
            count = positive_tokens(group["count"], "count")
            length = positive_tokens(group["input_len"], "input_len")
            cached = group.get("cache_len", 0)
            if type(cached) is not int or not 0 <= cached < length:
                raise ValueError(f"case {case_id}: invalid request group")
            measure.append((count, length))
            if cached:
                if cached % commit_tail or cached + commit_tail > length:
                    raise ValueError(
                        f"case {case_id}: cache alignment/seed tail exceeds input"
                    )
                seeds.append(
                    (1 if policy == "shared_by_group" else count, cached + commit_tail)
                )
        if sum(n for n, _ in measure) != batch:
            raise ValueError(f"case {case_id}: group counts must sum to batch_size")
        phase(case_id, "measure", measure)
        phase(case_id, "seed", seeds)
    return stats
