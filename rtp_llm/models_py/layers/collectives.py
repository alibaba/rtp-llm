"""Lazy bridge to the existing TP collective implementation.

Keep CPU/TP=1 layer construction independent of native inference bindings.
This does not create process groups or change collective runtime semantics.
"""

from rtp_llm.models_py.distributed.rank_layout import Group


def all_reduce(tensor, *, group):
    from rtp_llm.models_py.distributed.collective_torch import all_reduce as impl

    return impl(tensor, group=group)


def all_gather(tensor, *, group):
    from rtp_llm.models_py.distributed.collective_torch import all_gather as impl

    return impl(tensor, group=group)
