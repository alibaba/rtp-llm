"""Protocol agreement using an already bootstrapped CPU TCPStore."""

from datetime import timedelta


def verify_store_protocol(store, *, namespace, rank, ranks, digest, timeout_s):
    """Namespace must identify this model generation within the existing store.

    Callers choose a homogeneous TP/EP/CP group, not unrelated PP stages or PD
    roles. No process group is created here and no GPU operation is performed.
    """
    ranks = tuple(ranks)
    if not namespace or rank not in ranks or len(set(ranks)) != len(ranks):
        raise ValueError("A unique execution group and model namespace are required")
    if timeout_s <= 0:
        raise ValueError("Protocol verification requires a positive timeout")
    prefix = f"module_dispatch/{namespace}/"
    own_key = prefix + str(rank)
    # Reusing a namespace could observe stale peers after model reload. Refuse
    # that lifecycle error; the worker supplies a new generation for a reload.
    if store.add(own_key + "/claim", 1) != 1:
        raise RuntimeError(f"Protocol namespace already used: {namespace}, rank {rank}")
    store.set(own_key, digest)
    keys = [prefix + str(peer) for peer in ranks]
    store.wait(keys, timedelta(seconds=timeout_s))
    mismatches = {peer: store.get(key).decode() for peer, key in zip(ranks, keys)}
    if any(value != digest for value in mismatches.values()):
        raise RuntimeError(
            f"Module protocol disagreement for {namespace}: {mismatches}"
        )
    return True
