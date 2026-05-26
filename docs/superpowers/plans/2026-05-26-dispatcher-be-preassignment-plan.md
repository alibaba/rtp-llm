# Dispatcher BE Pre-Assignment Implementation Plan

**Date**: 2026-05-26
**Branch**: `feature/master-batch-schedule`
**Scope**: Dispatcher (Java) only. FE Python is a separate PR.
**Memory**: `project_dispatcher_be_preassignment.md`

## Goal

Let dispatcher pre-resolve N BE targets via master's `/rtp_llm/batch_schedule` and stamp each chunk body with `pre_assigned_be={server_ip,http_port,grpc_port}`. FE that recognizes the field skips its own `/rtp_llm/schedule` round-trip; FE that doesn't recognize the field silently ignores it (pydantic `extra="ignore"`) and behaves exactly as today. Both worlds work side by side.

## Non-Goals

- FE Python changes (separate PR)
- BE failure semantics on FE side (FE owns the fallback path: pre-assigned BE dead → FE retries via local `/schedule`)
- Auto-failover at dispatcher level (dispatcher does not subscribe to BE health; rare event handled at FE)

## Design Decisions (locked in)

| # | Decision | Rationale |
|---|---|---|
| 1 | `pre_assigned_be` at chunk-body top-level, sibling of `prompt_batch` / `generate_config` | Routing hint, not a generation param; clean separation |
| 2 | Field shape = exact `BatchScheduleTarget` JSON: `{server_ip, http_port, grpc_port}` | Zero re-mapping for FE; reuse existing master DTO |
| 3 | Opt-in via `DISPATCH_CONFIG.preAssignBe=false` (default) + `DISPATCH_PRE_ASSIGN_BE=true` env override | Ship dispatcher-side code path safely (no behavior change off); flip when FE is ready |
| 4 | Old FE without recognition → unknown field ignored (pydantic), goes through original `/schedule` path | Backward compat; no FE prerequisite for this PR to merge |
| 5 | Master `/batch_schedule` call fails → silently log WARN, don't stamp field, fall through to original behavior | Rare event; never block real traffic |

## File Changes

### New files

- `flexlb-api/src/main/java/org/flexlb/dispatcher/BatchScheduleClient.java`
  Interface: `Mono<List<BatchScheduleTarget>> requestTargets(int count)`
- `flexlb-api/src/main/java/org/flexlb/dispatcher/WebClientBatchScheduleClient.java`
  Implementation: HTTP POST to `<master>/rtp_llm/batch_schedule` via WebClient. Master URL = `http://localhost:<master-port>` (same JVM today; future-proof for cross-process).
- `flexlb-api/src/test/java/org/flexlb/dispatcher/WebClientBatchScheduleClientTest.java`
  Unit tests: success / 5xx / non-2xx / connect refused / count=0.

### Modified files

- `flexlb-api/src/main/java/org/flexlb/dispatcher/DispatchConfig.java`
  Add field `boolean preAssignBe = false`. Existing `EnvConfigOverrides` reflection picks it up automatically as `DISPATCH_PRE_ASSIGN_BE`.

- `flexlb-api/src/main/java/org/flexlb/dispatcher/GenericBatchHandler.java`
  Constructor takes `BatchScheduleClient batchScheduleClient` and `boolean preAssignBe`. After `splitChunks(arr)`, before fanout: if `preAssignBe`, call `batchScheduleClient.requestTargets(chunks.size()).onErrorReturn(emptyList())` (silent degrade). For each chunk-body, if targets[i] exists, set top-level `pre_assigned_be` to `BatchScheduleTarget` JSON (use ObjectMapper to serialize, then `obj.set("pre_assigned_be", node)`).

- `flexlb-api/src/main/java/org/flexlb/dispatcher/DispatcherConfiguration.java`
  Wire `WebClientBatchScheduleClient` (always created when dispatcher enabled — uses master's local URL). Pass `client` + `cfg.isPreAssignBe()` to `GenericBatchHandler`. Add `preAssignBe` to the boot WARN line so operators see it.

### Test additions

- `flexlb-api/src/test/java/org/flexlb/dispatcher/DispatchConfigTest.java`
  Add cases: default `preAssignBe=false`, env override flips it, JSON override flips it.
- `flexlb-api/src/test/java/org/flexlb/dispatcher/GenericBatchHandlerTest.java`
  3 new cases:
  1. `preAssignBe=true` + client returns N targets → every chunk body has `pre_assigned_be={server_ip,http_port,grpc_port}` matching the i-th target
  2. `preAssignBe=true` + client returns empty list (failure) → no chunk has `pre_assigned_be`, request still succeeds via fanout
  3. `preAssignBe=false` (default) → no chunk has `pre_assigned_be`, client never called
- `flexlb-api/src/test/java/org/flexlb/dispatcher/DispatcherE2ETest.java`
  1 new case: `preAssignBeStampsTargetsOnEachChunk` — mock master `/rtp_llm/batch_schedule` returns 3 targets, dispatcher splits 9-prompt batch into 3 chunks of 3, verify each chunk's recorded request body has the corresponding `pre_assigned_be` block.

## Implementation Order (TDD-friendly)

1. **DispatchConfig field** + DispatchConfigTest 3 cases (smallest unit, no deps)
2. **BatchScheduleClient interface + impl** + WebClientBatchScheduleClientTest (HTTP + JSON; mock server)
3. **GenericBatchHandler integration** + GenericBatchHandlerTest 3 new cases (mock `BatchScheduleClient`)
4. **DispatcherConfiguration wiring** + DispatcherConfigurationTest minor update (constructor signature)
5. **DispatcherE2ETest** new case (full stack)
6. Build + run all dispatcher tests
7. Commit + push

Each step keeps the tree green. Committing after step 6 (single commit) is fine because Step 1-5 individually can't ship — `BatchScheduleClient` exists but unused, etc.

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Master `/batch_schedule` slow → adds latency to every batch | Async (Mono) + fail-fast timeout (reuse `cfg.getBatchTimeoutMs()` cap); on timeout fall through to no-pre-assignment |
| `preAssignBe=true` deployed before FE supports it | Backward compat by design (decision #4); worst case = wasted `/batch_schedule` call. Tested explicitly in case where FE mock doesn't read the field |
| Master JVM lookup hardcoded localhost | Future-proof: keep master URL configurable via `DispatchConfig.masterUrl` (default `http://localhost:7001`) so cross-process deploy works without code change. **DEFERRED**: not in this PR; current same-JVM deploy uses localhost; cross-process needs separate work to run master+dispatcher in different processes anyway |
| Single point of failure: dispatcher → master `/batch_schedule` fails → all chunks lose pre-assignment | Acceptable: degrade is silent and FE still chooses BE itself. No request loss |
| `pre_assigned_be` stamped on a chunk whose target dies in the gap | FE-side fallback (out of scope); rare event |

## Acceptance Criteria

- [ ] All existing dispatcher tests still green (107 tests)
- [ ] 3 new GenericBatchHandlerTest cases green
- [ ] WebClientBatchScheduleClientTest green
- [ ] DispatcherE2ETest new case green
- [ ] `mvnw -pl flexlb-api -am compile` clean
- [ ] `git diff` shows: 2 new files + 4 modified files + 4 test additions
- [ ] Commit message follows conventional commits (`feat(dispatcher): pre-resolve BE targets via batch_schedule when DISPATCH_PRE_ASSIGN_BE=true`)
- [ ] Push to `feature/master-batch-schedule` with `--force-with-lease`

## Out of Scope (next PR)

- FE Python `pre_assigned_be` recognition + fallback
- Configurable master URL (`DispatchConfig.masterUrl`)
- Cross-process dispatcher↔master deployment
- BE health subscription on dispatcher side

---

## Revision 2026-05-27 — Reuse `generate_config.role_addrs`, drop `pre_assigned_be`

The original wire-shape decision (`pre_assigned_be` as a new top-level chunk-body field) is **superseded**. Code review surfaced that FE already supports the exact mechanism we need — through `generate_config.role_addrs` — and has done so for a long time (PD-disagg's prefill→decode handoff uses the same field).

### Evidence FE already supports it (no FE change needed)

`rtp_llm/server/backend_rpc_server_visitor.py:217-226`:

```python
role_addrs_specified = bool(input.generate_config.role_addrs)
master_addr = self.host_service.get_master_addr()
...
if not role_addrs_specified and master_addr and not input_token_batched:
    master_route_result = await self.get_master_route_addrs(input)   # /schedule call
```

If `role_addrs` is non-empty in the request, FE skips master entirely. The struct (`rtp_llm/config/generate_config.py:19-126`) is `List[RoleAddr]` where each `RoleAddr = {role, ip, http_port, grpc_port}`.

C++ side (`PrefillRpcServer.cc:127-132`) also honors per-request role addrs verbatim — this is not a new code path.

### Why the original `pre_assigned_be` design was wrong

- Decision #1 picked a brand-new top-level field. pydantic `extra="ignore"` on `GenerateInput` silently drops unknown top-level fields → field never reaches `generate_config.role_addrs` → FE still calls `/schedule` even with `DISPATCH_PRE_ASSIGN_BE=true`.
- Decision #4's "backward compat" argument is technically true but masks a deeper issue: it requires a future FE patch to lift the new field into `role_addrs`. That patch is the only reason the env flag must default off.
- Net effect: today's `LocalBatchScheduleClient` is in-place but **functionally a no-op** when `preAssignBe=true` (master gets called, target gets resolved, target gets stamped, FE drops it on the floor). The "Stage 3" gate is artificial.

### Master also has the role

`DefaultRouter.batchSchedule` (lines 142-154) is **single-role-only by design** — it explicitly rejects multi-role deployments (PD-disagg goes through `/schedule` per request). At line 154 the role is already pinned:

```java
RoleType roleType = roleTypes.get(0);   // ← known here
...
List<BatchScheduleTarget> targets = batchLoadBalancer.selectBatch(count, roleType, null);
return BatchScheduleResponse.success(targets);   // ← but dropped on the way out
```

So one batch_schedule response carries N targets all of the same role. Role belongs on `BatchScheduleResponse`, not on each `BatchScheduleTarget`.

### Revised design (replaces decisions #1–#4)

| # | New decision |
|---|---|
| 1 | Stamp into `chunk.generate_config.role_addrs` (existing FE-recognized field) instead of new top-level `pre_assigned_be` |
| 2 | Wire shape per addr: `{role, ip, http_port, grpc_port}` (matches `rtp_llm.config.generate_config.RoleAddr` exactly) |
| 3 | `DISPATCH_PRE_ASSIGN_BE` default **true** — no FE prerequisite remains |
| 4 | Old field `pre_assigned_be` removed (Stage 2 code never functioned end-to-end; nothing to deprecate) |
| 5 | Failure path unchanged: master call fails → empty list → no stamping → silent fallback |

### File changes (delta on top of merged Stage 2)

- `flexlb-common/.../BatchScheduleResponse.java` — add `RoleType role` field (Jackson `@JsonProperty("role")`).
- `flexlb-sync/.../DefaultRouter.java:169` — `resp.setRole(roleType)` before returning.
- `flexlb-api/.../BatchScheduleClient.java` — return type from `Mono<List<BatchScheduleTarget>>` to `Mono<BatchScheduleResult>` where `BatchScheduleResult = {role, targets}`. (Or pass role through a sibling channel — interface choice; keep one trip-and-drop wrapper.)
- `flexlb-api/.../LocalBatchScheduleClient.java` — propagate `resp.getRole()` into the wrapper; failure path returns `{null, []}`.
- `flexlb-api/.../GenericBatchHandler.java::stampPreAssignedBe` — rewrite to:
  ```java
  ObjectNode gc = (ObjectNode) chunkBody.path("generate_config");
  if (gc.isMissingNode()) gc = chunkBody.putObject("generate_config");
  ArrayNode roleAddrs = gc.withArray("role_addrs");
  ObjectNode addr = roleAddrs.addObject();
  addr.put("role", result.getRole().name());
  addr.put("ip", target.getServerIp());
  addr.put("http_port", target.getHttpPort());
  addr.put("grpc_port", target.getGrpcPort());
  ```
  (preserves any `role_addrs` the user explicitly set — appends instead of overwrites)
- `flexlb-api/.../DispatchConfig.java` — `preAssignBe = true` default; remove the "wait for FE recognition" doc paragraph; keep the silent-fallback paragraph.
- Tests: `GenericBatchHandlerTest` 3 new cases need to read `chunk.generate_config.role_addrs[0]` instead of top-level `pre_assigned_be`. `DispatcherE2ETest` similarly. `LocalBatchScheduleClientTest` updated to assert role propagation. `BatchScheduleResponse` JSON snapshot tests updated.

### Why this is one commit, not two-stage rollout

The Stage 2 plan's two-stage rollout (dispatcher first, FE second, then flip env) was driven entirely by the wrong wire-shape choice. With `role_addrs`:

- Old FE = today's FE. Already supports `role_addrs`. Already in production handling PD-disagg through this mechanism.
- Default-on is safe: dispatcher fails to reach master → empty list → no stamping → FE behaves exactly as today.
- Default-on is the right default: every batch request that *does* pre-resolve avoids one master round-trip + one race-prone independent decision per chunk.

### Migration notes

The whole `feature/master-batch-schedule` branch (including Stage 2 commit `d07e2cdf1`) is unreleased, so there is **no** rolling-deploy version-skew to handle: dispatcher and master ship together, every BE target the dispatcher receives carries a non-null role by construction. No defensive null-role check, no conditional fallback per-target — if a non-null role ever appears in production it's a master-side bug to surface (NPE), not silently work around.

- Stage 2 commit `d07e2cdf1` introduced `pre_assigned_be` chunk-body field — this revision removes it. Search for `FIELD_PRE_ASSIGNED_BE`, `pre_assigned_be` in tests/docs.
- `BatchScheduleTarget` gains a per-target `role` field (not response-level) so the data shape mirrors Python `RoleAddr` 1:1 and remains correct under any future multi-role batch_schedule with no schema change. `BatchScheduleResponse` is unchanged.
- Internal `RoleAddr` enum names (`PDFUSION` / `PREFILL` / `DECODE` / `VIT`) must match between Java `RoleType.name()` and Python `RoleType` enum — verify before merge.

### Master /batch_schedule strategy decoupling (added in this revision)

A second issue surfaced during implementation: today `/batch_schedule` queries the same per-role load balancer registry that `/schedule` uses (`getStrategyForRoleType`). Default strategies (SHORTEST_TTFT / WEIGHTED_CACHE) don't implement `BatchLoadBalancer`, so `/batch_schedule` rejects with "strategy does not support batch" until the operator switches the role to `ROUND_ROBIN` — which then makes `/schedule` lose its smart routing for that role. Either-or is a usability cliff that nullifies the dispatcher pre-assignment we're enabling here.

**Fix:** `FlexlbConfig.batchLoadBalanceStrategy` (default `ROUND_ROBIN`) and a new `getBatchStrategyForRoleType()` resolver. `DefaultRouter` maintains a parallel `batchLoadBalancerMap` populated from this. `batchSchedule` consults the batch map instead of the regular map. `/schedule`'s strategy selection is unchanged.

**Failure mode:** if an operator explicitly sets a non-batch-capable `batchLoadBalanceStrategy`, `/batch_schedule` rejects loudly with "batchStrategy for role X does not support batch_schedule" — never a silent fallback.

**Why per-role override fields aren't added yet:** all four roles share the same default (RR is the only batch-capable strategy today). When a future batch-aware strategy makes role-specific tuning meaningful, add `decodeBatchLoadBalanceStrategy` etc. mirroring `decodeLoadBalanceStrategy`'s precedent.

### Implementation summary (delivered)

Files touched:

- **flexlb-common**
  - `BatchScheduleTarget.java`: +role field, +3-arg backward-compat ctor
  - `FlexlbConfig.java`: +batchLoadBalanceStrategy field, +getBatchStrategyForRoleType
- **flexlb-sync**
  - `DefaultRouter.java`: +batchLoadBalancerMap, +per-role boot WARN echoing both strategies, batchSchedule queries batch map, error message names "batchStrategy"
  - `RoundRobinLoadBalancer.java`: buildTarget takes RoleType param, sets it on the target
- **flexlb-api**
  - `DispatchConfig.java`: preAssignBe default flipped to true, doc rewritten to explain "FE already supports role_addrs natively"
  - `GenericBatchHandler.java`: stampPreAssignedBe rewritten to append into generate_config.role_addrs, removed FIELD_PRE_ASSIGNED_BE constant, preserves user-supplied role_addrs entries
- Tests updated/added: DispatchConfigTest (default flip), DispatcherE2ETest (verifyChunkHasRoleAddr), GenericBatchHandlerTest (role_addrs assertions + new preAssignBePreservesUserSuppliedRoleAddrs case), DefaultRouterTest (per-target role assertion + batchStrategy decoupling test + replaceBatchLoadBalancer helper)

Test result: 380/380 across all four flexlb modules.
