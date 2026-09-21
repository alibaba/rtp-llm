"""Profile-aware rebalance traffic and legacy per-owner counter anchors."""

from ..contracts import StageHandler, StageOutput
from . import elastic_lifecycle as life


def batch_validate(params, plan):
    p = life._params(params, plan, {"method"}, {"method"})
    if p["method"] not in ("FetchResponse", "GenerateStreamCall"):
        raise ValueError("unsupported rebalance consumer")
    return p


def batch(ctx, params, deadline):
    return life.bounded_batch(ctx, 50, deadline, expected_method=params["method"])


def anchor_validate(params, plan):
    p = life._params(params, plan, {"old", "new", "engine"}, {"old", "new", "engine"})
    for key in ("old", "new"):
        plan.reference(p[key], "snapshot")
    life._name(p["engine"], plan)
    return p


def anchor(ctx, params, deadline):
    deadline.check()
    old, new = (ctx.resource(params[k], "snapshot") for k in ("old", "new"))
    name = ctx.resolve(params["engine"])
    if set(old["counts"]) != {"prefill-0", "prefill-1"} or name in old["counts"]:
        raise ValueError("rebalance requires the two original Prefill owners")
    if name not in new["counts"]:
        raise ValueError("missing newcomer counter anchor")
    counts = dict(old["counts"], **{name: new["counts"][name]})
    if any(type(v) is not int or v < 0 for v in counts.values()):
        raise ValueError("invalid rebalance counter anchor")
    handle = ctx.register_resource(
        "snapshot",
        dict(counts=counts, old_anchor=old, newcomer_anchor=new),
        historical=True,
    )
    return StageOutput(output=dict(before=handle))


HANDLERS = [
    StageHandler(
        "elastic_rebalance_batch",
        batch_validate,
        batch,
        {"result": "snapshot"},
        checks=frozenset({"complete", "no_errors", "protocol"}),
    ),
    StageHandler(
        "elastic_rebalance_anchor", anchor_validate, anchor, {"before": "snapshot"}
    ),
]
