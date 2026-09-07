# Quota blocked-phase topology sampling correction

The fixed `5affc1c6de64fc8a1bf0d74dc3789d035c9a7307` Java run of
`master_dispatch_quota::single_prefill_ttl::batch-window` ended in ERROR at
`blocked`, with `KeyError: 'PREFILL'`. That original result remains in the runtime
ledger; this correction has no post-fix Java PASS claim.

The old `master_quota_block` stops the only Prefill, waits three seconds, then
issues ten requests concurrently with a twelve-second stream budget and tests
that at least half fail. It does not require Master topology samples while
issuing those requests. The generic candidate batch adapter added unconditional
sampling and indexed both role rows, so the intentionally absent Prefill could
prevent the blocked-request and subsequent TTL predicates from running.

`master_request_batch.sample_topology` is now a strict boolean, defaulting to
true. Only quota's `blocked` stage explicitly sets it false. The adapter still
records and joins all request workers; its empty topology list means no topology
was sampled, not that any worker count was zero. Coldstart cannot disable
sampling, and every existing default caller still requires real role/count
observations. A post-batch sampling window cannot be requested with sampling
turned off. Missing required topology remains ERROR.

The ten requests, concurrency ten, 2048 input/two output/one RID-derived cache
key, twelve-second bound, success-rate <= 0.5 gate, scheduler-ledger TTL window
of 95 seconds, and twenty serial recovery requests remain unchanged. This patch
addresses the added sampling precondition only; it does not claim all other
Master gates or failure classifications are identical to the old callable.

Focused tests compile the actual quota YAML through the strict loader and
default catalog, then execute its batch adapter with actual executor workers
and recorded client evidence while replacing external request/HTTP transport.
Five successes satisfy the unchanged threshold; six fail it. With topology
sampling disabled no Master topology call occurs and all ten records survive.
Coldstart and default callers still error on a missing Prefill row. These tests
are local models, not Java execution or paired legacy replacement acceptance.
