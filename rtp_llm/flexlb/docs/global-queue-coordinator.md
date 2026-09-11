# Global queue scheduling

QUEUE ordering determines the order of placement attempts, not strict arrival-order
publication to a worker. An unsuccessful request waits; later requests may try their
own resource requirements, including on the same worker. DIRECT is unchanged.

The coordinator has three responsibilities:

1. Scan a bounded slice of the ordered queue, merging awakened requests by their
   original FIFO/priority order.
2. Plan routes in parallel and publish them in order. If an earlier request becomes
   ready during planning, discard the uncommitted suffix and scan it again.
3. Remove completed requests or register a failed attempt with `PlacementWaitQueue`.

`PlacementWaitQueue` is the only waiting policy. Both exact-worker and selector-domain
failures use the same state transitions:

| Trigger | Effect |
| --- | --- |
| Placement fails | Park on the reported capacity key, unless its version advanced during planning |
| Relevant capacity/topology event | Grant one retry opportunity to that domain |
| Decision loop scans | Transfer at most one request per ready domain, within the planner limit |
| Retry succeeds or leaves the queue | Return the opportunity so remaining capacity can be used |
| Retry fails | Consume the opportunity; wait for a newer event |

The wait source is a notification key, not a route constraint. Every retry runs normal
fleet selection. A selector miss does not block unrelated requests in that group.
Exact keys match role/address across worker generations and group changes. Group events
do not wake all exact-worker waiters. Capacity version checks and park publication share
the coordinator lock with event handling, preventing lost wakeups.

This retains bounded overload behavior: with 10,000 waiters and one released Decode
slot, the contract test observes one successful retry and one confirming miss, not a
replay of the backlog. No successor-capacity probes, paused-successor references,
per-request conflict history, endpoint exclusions, or separate planning-capacity ledger
are needed. Routing keeps its cheap prefill-capacity filter; endpoint admission remains
the authority for resource acquisition and preemption.

Validation includes FIFO/priority ordering, stale plans, deadline/cancellation races,
one-event multi-slot progress, one-slot overload, request-specific selector/admission
failures, topology replacement, and fleet-wide retry routing.
