# Full-shrink construction and ownership

The `kv_full_shrink` variant is a literal-control-flow migration candidate, not
proof that the legacy performance construction works as described. Real Java
acceptance remains pending. The mock and production scheduler are separate owners.

## Actual shared performance owner

At this candidate's Java base, `JavaMockEngineCluster.startEngine` passes the
same `performance` object to every `FastRpcService` (lines 315-318); the service
constructor assigns it directly (1045), and `getPerformance()` returns it (4774).
`DynamicEngineManager.addEngine` uses that same object (171-175).
`MockControlServer.handleSetPerf` calls `service.getPerformance()` and changes its
Decode override (378-385). Therefore addressing one worker selects a service but
does not give that service a separate performance model. Per-worker KV pools,
request state and drain ownership are still separate; this finding must not be
expanded into a claim that those resources are shared too.

| Ordered legacy operation | Actual shared Decode scale after operation |
| --- | --- |
| Start private 2P/2D fault environment; baseline flow | 1 |
| Set decode-0=60, then decode-1=60 | 60 for both |
| Fill real 24-block pools; remove decode-0 with 60s drain | 60 for survivor |
| Collect terminals; account50; add new Decode; steady flow | New Decode also uses the same 60 override |
| Set decode-1=1000 | 1000 for every service sharing the model |
| Set newcomer=60 | 60 for both remaining Decodes, overwriting the previous 1000 |
| Fill; remove decode-1 with 5s drain; collect | 60, not a victim-only 1000 tail |
| Restore newcomer=1; recovery20 | 1 |

The candidate preserves this sequence. Its tests model one shared field and check
the full ordered write list. The 5s timeout branch still requires `drained=false`,
5000..10000ms `drain_ms`, at least one 8510 retirement with the exact generation
message, and <=40s visible terminals. These checks can fail; no finding conversion
or threshold adjustment is permitted. Even a passing modeled trace does not prove
that the shared-Java construction reliably holds victim work past 5s.

## Proposed separate construction correction

A possible test-only adaptation is to set shared scale60, then shared1000, fill,
remove the victim with the existing5s cap, and only after removal restore the
survivor's scale60 before collecting. `scheduleNextDecodeStepLocked` reads the
current `performance.decodeStepDelayMs` for each subsequent step (3631), so this
would change future survivor steps; an already scheduled long step is not
retroactively shortened. The existing40s client cap must still be measured.

This is a global slowdown followed by recovery, not independent per-worker
slowdown. It must be a separately identified construction-correction variant and
fixed commit, reviewed independently, with no implicit legacy coverage mapping.
It is not part of this literal candidate. A real per-worker performance isolation
change would instead belong to the shared mock implementation owner and needs
its own tests; this task does not change Java.
