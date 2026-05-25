# FlexLB Transparent Dispatcher Implementation Plan

> **设计上下文先读 [`docs/dispatcher.md`](../../dispatcher.md)** — 那是 dispatcher 的 source of truth(架构、决策、代码地图、状态)。这份文件是**执行用的 TDD 任务清单**(V1-V12 Step-by-Step),不重复设计讨论。两边冲突以 `docs/dispatcher.md` 为准。
>
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a transparent, in-JVM dispatcher to FlexLB that fans a large `prompt_batch` out across a pool of frontend (FE) workers in K-sized sub-batches and merges the responses, so clients see the same FE API while the work is parallelized.

**Architecture:** A new `org.flexlb.dispatcher` package inside `flexlb-api`, running in the **same JVM** as the Master and **sharing the Master's 7001 listener** (same-port default — see Task 10). The dispatcher mirrors the FE's HTTP surface as Spring routes ordered AFTER the Master's: the batch path (`/batch_infer`) is split → fanned out → merged; every other unmatched path is blind-forwarded to one FE; the Master's `/rtp_llm/*` always match first (`@Order`). Since same-JVM already shares CPU/heap/GC, a separate listener would only isolate the event loop — kept as an optional variant (Task 10-alt). v1 is **non-streaming** only. An optional phase pre-assigns BE targets via a **master-aware** batch-schedule call (in-process on master, forward on slave) so FE workers skip the per-request Master round-trip.

**Tech Stack:** Java 21, Spring Boot 2.7.18 (WebFlux), Reactor Netty, Jackson, JUnit 5 + Mockito 5.20.0. Built with `./mvnw` from `rtp_llm/flexlb`.

---

## 2026-05-25 — Stage 2 Architecture Update

> **Why this exists:** review of Stage 1's shipped code surfaced three holes the original plan didn't anticipate. Stage 2 fixes them with a table-driven multi-endpoint design and replaces v1 Tasks 14, 15, 15b, 16, 17 (Phase 4 + Phase 5). v1 Phases 1–3 + the ServiceDiscovery follow-up stay in main as-is and get **incrementally refactored** by the V-series tasks below.

### What's changing (and why)

| Aspect | v1 (shipped) | v2 (this update) | Why |
|---|---|---|---|
| Route mounting | Catch-all on shared 7001 + `@Order(0)` defended Master routes | `POST /dispatcher/<path>` prefix; no catch-all, no `@Order` tricks | v1 silently shadowed FE's `/health` (Master's `/health` won). v2 prefix isolates dispatcher's namespace; clients opt in explicitly. |
| Batch endpoints handled | Only `POST /batch_infer` | `/batch_infer`, `/` (when `prompt` is a list), `/v1/batch/chat/completions`, `/v1/embeddings` (and the 6 embedding variants); pluggable via spec table | v1's hand-rolled `BatchSplitter` only knew `prompt_batch`. Real FE has 9+ batch-shaped paths; refusing them all is the same as not having a dispatcher for those flows. |
| Failure semantics | All-success path padded with placeholders for failed chunks (`/batch_infer` only); other endpoints would inherit a "any sub-batch fails → 500" by default | **Unified partial-failure** for every batch endpoint: failed positions kept in-place with a per-endpoint failure-shape factory; top-level `_partial_failure` metadata; HTTP 200 unless **all** chunks failed | Fanning out 500 items across 5 FEs makes partial failure 5× more likely than the single-FE baseline. All-or-nothing semantics make the dispatcher **less** available than no dispatcher. |
| Passthrough timeout | `.timeout(3000ms)` on the whole exchange — any long-running stream (SSE chat completion, typical 30–120 s) cut at 3 s | Stream-friendly: connect + idle timeouts (small); total-duration cap (large, default 600 s); no per-exchange wall-clock | v1's "non-streaming only" caveat means any client that sets `stream=true` against `/dispatcher/v1/chat/completions` silently breaks. |
| Client compatibility | "Transparent" — clients hit existing FE paths via Master's port | Explicit — clients use `/dispatcher/<original_path>` to opt in; direct-to-FE keeps working | Transparency was the wrong goal: it created the `/health` collision and gives no way to A/B between dispatcher and direct. Opt-in URL = intent on the wire. |

### What is NOT changing

- Same JVM, same port (7001), `ServiceDiscovery`-backed FE pool (Stage 1 follow-up, commits `58b087264` / `972ebc4b2` / `3456bcd7f`)
- `BatchScheduleCoordinator` + master/slave forwarding logic (in production, unaffected)
- Dedicated `dispatcher-fe` `ConnectionProvider`, 16 MiB per-response cap (`WebClientFeClient`)
- The FE pool fanout pattern (chunk → POST to one FE per chunk → merge)
- `request_id` trace continuity gap (`project_frontend_request_id_overwrite.md`) — still deferred
- **BE pre-assignment (v1 Phase 4)** — **deferred indefinitely**. It saves the per-request FE→Master round-trip, but v2's table-driven spec leaves a place to add it later (per-endpoint, gated). v1 Tasks 13b/14/15/15b are superseded; do not implement them now. Re-open only if a real benchmark shows `/schedule` round-trip is the bottleneck.

### File Structure additions (v2)

All new files under `flexlb-api/src/main/java/org/flexlb/dispatcher/`:

- `BatchEndpointSpec.java` — value type: `path`, `requestArrayField`, `responseArrayField`, `FailedItemFactory`, `PostMerger` (optional).
- `FailedItemFactory.java` — `JsonNode build(int absoluteIndex, String reason, ObjectMapper mapper)`. Three built-ins as static instances: `NULL`, `OPENAI_ERROR`, `EMBEDDING_NULL`.
- `PostMerger.java` — `void apply(ObjectNode mergedBody, List<SubBatchResult> subs, List<Integer> failedIndices, ObjectMapper mapper)`. One impl: `EmbeddingPostMerger`.
- `BatchEndpointRegistry.java` — `@Configuration` that exposes `List<BatchEndpointSpec>` as a bean (hard-coded for now; extension knob later if needed).
- `PartialFailureMerger.java` — generic merger; replaces v1's `ResponseMerger` for the batch path.
- `EmbeddingPostMerger.java` — rewrites `data[i].index` to absolute offset; sums `usage.prompt_tokens` / `usage.total_tokens` from successful sub-bodies.
- `GenericBatchHandler.java` — one handler for all batch specs; replaces `DispatchHandler.handleBatch`.

Files modified:
- `BatchSplitter.java` — adds `splitArray(ArrayNode, int, ObjectMapper)` returning `List<ArrayNode>`; old `split(List<String>, int)` removed (only used by handleBatch, going away).
- `DispatchRouter.java` — register `POST /dispatcher/<path>` per spec from registry; catch-all `path("/dispatcher/**")` → passthrough.
- `WebClientPassthroughClient.java` — strip leading `/dispatcher` before composing the target URL; drop per-exchange `.timeout()`.
- `DispatchHandler.java` — keep `handlePassthrough` only; delete `handleBatch` (moved to `GenericBatchHandler`).
- `DispatcherConfiguration.java` — wire `BatchEndpointRegistry` + `GenericBatchHandler` + pass spec list to `DispatchRouter`.
- `DispatchConfig.java` — split timeouts: `feConnectTimeoutMs` (default 2000), `feResponseTimeoutMs` (default 5000) for batch chunks, `feMaxStreamDurationMs` (default 600000 = 10 min) for passthrough total-duration cap. Old `feRequestTimeoutMs` removed (or aliased to `feResponseTimeoutMs` for compat with deployed env JSON).

Tests:
- New: `BatchEndpointSpecTest`, `FailedItemFactoryTest`, `BatchEndpointRegistryTest`, `PartialFailureMergerTest`, `EmbeddingPostMergerTest`, `GenericBatchHandlerTest`, `StreamingPassthroughTest`, `DispatcherE2ETest`.
- Updated: `BatchSplitterTest` (covers `splitArray`), `DispatchRouterTest` (prefix scoping), `WebClientPassthroughClientTest` (prefix-strip), `DispatcherConfigurationTest` (new bean graph), `DispatchConfigTest` (new timeout fields).
- Deleted: `SamePortPrecedenceTest` (no catch-all → no precedence to defend), `DispatchHandlerTest` (replaced by `GenericBatchHandlerTest` + leaner passthrough check), `ResponseMergerTest` (covered by `PartialFailureMergerTest`).

### Stage 2 Task Map

| Order | Task | What ships | Depends on |
|---|---|---|---|
| V1 | Adopt `/dispatcher/` prefix | Routes scoped, prefix-strip in passthrough | none |
| V2 | `BatchEndpointSpec` + `FailedItemFactory` + `PostMerger` interfaces | Value types + 3 built-in failure factories | none |
| V3 | Generalize `BatchSplitter` to `ArrayNode` | `splitArray()` | none |
| V4 | `PartialFailureMerger` | Generic merge with partial-failure metadata | V2, V3 |
| V5 | `EmbeddingPostMerger` | Index renumbering + usage summation | V2, V4 |
| V6 | `BatchEndpointRegistry` (3 verified specs + spike for `/`) | 4-row spec table; `/` shape verified | V2, V5 |
| V7 | `GenericBatchHandler` | One handler dispatching by spec | V3, V4, V6 |
| V8 | Rewire `DispatchRouter` + `DispatcherConfiguration` from registry | Multi-endpoint live | V1, V7 |
| V9 | Stream-friendly passthrough timeouts | SSE works through `/dispatcher/v1/chat/completions` | V1 |
| V10 | End-to-end test (4 endpoints, induced partial failure) | One test class covers all | V8, V9 |
| V11 | Docs — `DISPATCH_CONFIG` + endpoint table + partial-failed contract + migration | `rtp_llm/flexlb/CLAUDE.md` + `README.md` updated | V8, V9 |
| V12 | Housekeeping — mark v1 Tasks 14/15/15b/16/17 superseded; delete dead files | Clean repo | all above |

---

## Stage 2 Tasks

### Task V1: Adopt `/dispatcher/` prefix

> Why first: this is the smallest behavior shift that fixes the `/health` shadow bug **and** unlocks all later tasks (V8 will register multiple `/dispatcher/<path>` routes). After this task, dispatcher does the same thing it did before, just under a different URL.

**Files:**
- Modify: `flexlb-api/src/main/java/org/flexlb/dispatcher/DispatchRouter.java`
- Modify: `flexlb-api/src/main/java/org/flexlb/dispatcher/WebClientPassthroughClient.java`
- Modify: `flexlb-api/src/test/java/org/flexlb/dispatcher/DispatchRouterTest.java`
- Modify: `flexlb-api/src/test/java/org/flexlb/dispatcher/WebClientPassthroughClientTest.java`
- Delete: `flexlb-api/src/test/java/org/flexlb/dispatcher/SamePortPrecedenceTest.java`

- [ ] **Step 1: Write the failing tests**

In `DispatchRouterTest.java` add (replacing/augmenting existing routing assertions):

```java
@Test
void nonDispatcherPathsAreNotMatched() {
    DispatchHandler handler = mock(DispatchHandler.class);
    var routes = new DispatchRouter(handler).routes();
    WebTestClient client = WebTestClient.bindToRouterFunction(routes).build();

    client.post().uri("/batch_infer").bodyValue("{}").exchange().expectStatus().isNotFound();
    client.get().uri("/health").exchange().expectStatus().isNotFound();
    client.post().uri("/chat/completions").bodyValue("{}").exchange().expectStatus().isNotFound();
    verifyNoInteractions(handler);
}

@Test
void dispatcherBatchInferGoesToHandleBatch() {
    DispatchHandler handler = mock(DispatchHandler.class);
    when(handler.handleBatch(any()))
        .thenReturn(ServerResponse.ok().bodyValue("batch-handled").build());
    var client = WebTestClient.bindToRouterFunction(new DispatchRouter(handler).routes()).build();

    client.post().uri("/dispatcher/batch_infer").bodyValue("{}").exchange()
        .expectStatus().isOk()
        .expectBody(String.class).isEqualTo("batch-handled");
    verify(handler).handleBatch(any());
}

@Test
void dispatcherAnyOtherPathGoesToPassthrough() {
    DispatchHandler handler = mock(DispatchHandler.class);
    when(handler.handlePassthrough(any()))
        .thenReturn(ServerResponse.ok().bodyValue("pass").build());
    var client = WebTestClient.bindToRouterFunction(new DispatchRouter(handler).routes()).build();

    client.get().uri("/dispatcher/v1/models").exchange()
        .expectStatus().isOk()
        .expectBody(String.class).isEqualTo("pass");
    verify(handler).handlePassthrough(any());
}
```

In `WebClientPassthroughClientTest.java`, change the test that sends to `/worker_status` to send to `/dispatcher/worker_status`, and assert the **MockWebServer** received the request at `/worker_status` (without the prefix):

```java
@Test
void forwardsToFeStrippingDispatcherPrefix() throws Exception {
    mockFe.enqueue(new MockResponse().setBody("ok").setResponseCode(200));
    var pool = new FePool(() -> List.of("http://" + mockFe.getHostName() + ":" + mockFe.getPort()));
    var passthrough = new WebClientPassthroughClient(WebClient.create(), pool, 5000);

    var request = MockServerRequest.builder()
        .method(HttpMethod.GET)
        .uri(URI.create("http://localhost/dispatcher/worker_status?role=PREFILL"))
        .build();
    StepVerifier.create(passthrough.forward(request))
        .assertNext(resp -> assertEquals(HttpStatus.OK, resp.statusCode()))
        .verifyComplete();

    RecordedRequest recorded = mockFe.takeRequest(1, TimeUnit.SECONDS);
    assertEquals("/worker_status?role=PREFILL", recorded.getPath());  // prefix stripped
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `./mvnw test -pl flexlb-api -am -Dtest='DispatchRouterTest,WebClientPassthroughClientTest' -P-internal`
Expected: FAIL — `nonDispatcherPathsAreNotMatched` because today everything matches the catch-all; passthrough test fails because `/dispatcher` is forwarded unchanged.

- [ ] **Step 3: Implement the prefix**

Rewrite `DispatchRouter.routes()`:

```java
public RouterFunction<ServerResponse> routes() {
    return route()
            .POST("/dispatcher/batch_infer", handler::handleBatch)
            .route(RequestPredicates.path("/dispatcher/**"), handler::handlePassthrough)
            .build();
}
```

In `WebClientPassthroughClient.forward()`, change the URL build to strip the prefix:

```java
URI src = request.uri();
String fePath = src.getRawPath().startsWith("/dispatcher/")
        ? src.getRawPath().substring("/dispatcher".length())  // keeps the leading "/"
        : src.getRawPath();
String pathAndQuery = src.getRawQuery() == null ? fePath : fePath + "?" + src.getRawQuery();
URI target = URI.create(feBaseUrl + pathAndQuery);
```

Delete `SamePortPrecedenceTest.java`: its catch-all assumption is gone; its "Master wins over dispatcher" guarantee is now satisfied by **path disjointness**, asserted in `DispatchRouterTest.nonDispatcherPathsAreNotMatched`.

> Note on `@Order(0)` annotations on `HttpLoadBalanceServer.loadBalancePrefill()`, `HealthCheckServer.healthCheck()`, `AppStateHookServer.appStateHook()`: they become **no-op** but stay (removing them is a separate harmless cleanup that touches Master code — keep this task scoped to dispatcher).

- [ ] **Step 4: Run to verify it passes**

Run: `./mvnw test -pl flexlb-api -am -Dtest='org.flexlb.dispatcher.*' -P-internal`
Expected: PASS — all remaining dispatcher tests green.

- [ ] **Step 5: Commit**

```bash
git rm flexlb-api/src/test/java/org/flexlb/dispatcher/SamePortPrecedenceTest.java
git add flexlb-api/src/main/java/org/flexlb/dispatcher/DispatchRouter.java \
        flexlb-api/src/main/java/org/flexlb/dispatcher/WebClientPassthroughClient.java \
        flexlb-api/src/test/java/org/flexlb/dispatcher/DispatchRouterTest.java \
        flexlb-api/src/test/java/org/flexlb/dispatcher/WebClientPassthroughClientTest.java
git commit -m "refactor(dispatcher): scope routes under /dispatcher/* prefix, drop catch-all"
```

---

### Task V2: BatchEndpointSpec + FailedItemFactory + PostMerger interfaces

> Pure types; no I/O. Establishes the contract used by V3–V8.

**Files:**
- Create: `flexlb-api/src/main/java/org/flexlb/dispatcher/BatchEndpointSpec.java`
- Create: `flexlb-api/src/main/java/org/flexlb/dispatcher/FailedItemFactory.java`
- Create: `flexlb-api/src/main/java/org/flexlb/dispatcher/PostMerger.java`
- Create: `flexlb-api/src/test/java/org/flexlb/dispatcher/FailedItemFactoryTest.java`

- [ ] **Step 1: Write the failing tests**

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class FailedItemFactoryTest {
    private final ObjectMapper mapper = new ObjectMapper();

    @Test
    void nullFactoryReturnsJsonNull() {
        JsonNode n = FailedItemFactory.NULL.build(7, "fe_timeout", mapper);
        assertTrue(n.isNull());
    }

    @Test
    void openAiErrorFactoryBuildsErrorObject() {
        JsonNode n = FailedItemFactory.OPENAI_ERROR.build(7, "fe_timeout", mapper);
        assertEquals(7, n.get("index").asInt());
        assertEquals("fe_timeout", n.get("error").get("message").asText());
        assertEquals("dispatcher_sub_batch_failed", n.get("error").get("code").asText());
    }

    @Test
    void embeddingNullFactoryBuildsDataItemWithNullEmbedding() {
        JsonNode n = FailedItemFactory.EMBEDDING_NULL.build(7, "fe_timeout", mapper);
        assertEquals(7, n.get("index").asInt());
        assertTrue(n.get("embedding").isNull());
        assertEquals("fe_timeout", n.get("error").asText());
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `./mvnw test -pl flexlb-api -am -Dtest=FailedItemFactoryTest -P-internal`
Expected: FAIL — types don't exist yet.

- [ ] **Step 3: Write the types**

`FailedItemFactory.java`:

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

@FunctionalInterface
public interface FailedItemFactory {
    JsonNode build(int absoluteIndex, String reason, ObjectMapper mapper);

    FailedItemFactory NULL = (idx, reason, mapper) -> mapper.nullNode();

    FailedItemFactory OPENAI_ERROR = (idx, reason, mapper) -> {
        ObjectNode err = mapper.createObjectNode();
        err.put("code", "dispatcher_sub_batch_failed");
        err.put("message", reason);
        ObjectNode item = mapper.createObjectNode();
        item.put("index", idx);
        item.set("error", err);
        return item;
    };

    FailedItemFactory EMBEDDING_NULL = (idx, reason, mapper) -> {
        ObjectNode item = mapper.createObjectNode();
        item.put("index", idx);
        item.set("embedding", mapper.nullNode());
        item.put("error", reason);
        return item;
    };
}
```

`PostMerger.java`:

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.List;

@FunctionalInterface
public interface PostMerger {
    /** Called once after PartialFailureMerger has stitched the array; aggregates cross-chunk fields. */
    void apply(ObjectNode mergedBody, List<SubBatchResult> subs, List<Integer> failedIndices, ObjectMapper mapper);
}
```

`BatchEndpointSpec.java`:

```java
package org.flexlb.dispatcher;

import lombok.Value;

import javax.annotation.Nullable;

@Value
public class BatchEndpointSpec {
    String path;                           // e.g. "/batch_infer"
    String requestArrayField;              // e.g. "prompt_batch"
    String responseArrayField;             // e.g. "response_batch"
    FailedItemFactory failedItemFactory;
    @Nullable PostMerger postMerger;       // null for endpoints with no cross-chunk aggregation
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `./mvnw test -pl flexlb-api -am -Dtest=FailedItemFactoryTest -P-internal`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add flexlb-api/src/main/java/org/flexlb/dispatcher/BatchEndpointSpec.java \
        flexlb-api/src/main/java/org/flexlb/dispatcher/FailedItemFactory.java \
        flexlb-api/src/main/java/org/flexlb/dispatcher/PostMerger.java \
        flexlb-api/src/test/java/org/flexlb/dispatcher/FailedItemFactoryTest.java
git commit -m "feat(dispatcher): BatchEndpointSpec + FailedItemFactory + PostMerger SPI"
```

---

### Task V3: Generalize `BatchSplitter` to `ArrayNode`

> Today's `BatchSplitter.split(List<String>, int)` is only used by the soon-to-be-deleted `handleBatch`. Generalize once; the new generic handler will use `splitArray`. The existing `chunkCount(int, int)` stays as-is.

**Files:**
- Modify: `flexlb-api/src/main/java/org/flexlb/dispatcher/BatchSplitter.java`
- Modify: `flexlb-api/src/test/java/org/flexlb/dispatcher/BatchSplitterTest.java`

- [ ] **Step 1: Write the failing test**

In `BatchSplitterTest.java` add:

```java
@Test
void splitArrayKeepsItemOrderAndShape() {
    ObjectMapper m = new ObjectMapper();
    ArrayNode arr = m.createArrayNode();
    for (int i = 0; i < 7; i++) {
        ObjectNode o = m.createObjectNode();
        o.put("i", i);
        arr.add(o);
    }
    List<ArrayNode> chunks = BatchSplitter.splitArray(arr, 3, m);
    assertEquals(3, chunks.size());
    assertEquals(3, chunks.get(0).size());
    assertEquals(3, chunks.get(1).size());
    assertEquals(1, chunks.get(2).size());
    assertEquals(0, chunks.get(0).get(0).get("i").asInt());
    assertEquals(6, chunks.get(2).get(0).get("i").asInt());
}

@Test
void splitArrayEmptyReturnsEmpty() {
    ObjectMapper m = new ObjectMapper();
    List<ArrayNode> chunks = BatchSplitter.splitArray(m.createArrayNode(), 5, m);
    assertTrue(chunks.isEmpty());
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `./mvnw test -pl flexlb-api -am -Dtest=BatchSplitterTest -P-internal`
Expected: FAIL — `splitArray` doesn't exist.

- [ ] **Step 3: Implement**

```java
public static List<ArrayNode> splitArray(ArrayNode arr, int chunkSize, ObjectMapper mapper) {
    int n = arr.size();
    if (n == 0) return List.of();
    int chunks = (n + chunkSize - 1) / chunkSize;
    List<ArrayNode> out = new ArrayList<>(chunks);
    for (int c = 0; c < chunks; c++) {
        ArrayNode chunk = mapper.createArrayNode();
        int start = c * chunkSize;
        int end = Math.min(start + chunkSize, n);
        for (int i = start; i < end; i++) chunk.add(arr.get(i));
        out.add(chunk);
    }
    return out;
}
```

Delete the old `split(List<String>, int)` once V7 removes its caller (or in this task if no other call sites remain — verify with `grep -r "BatchSplitter.split(" flexlb-api/src/main/java`).

- [ ] **Step 4: Run to verify it passes**

Run: `./mvnw test -pl flexlb-api -am -Dtest=BatchSplitterTest -P-internal`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add flexlb-api/src/main/java/org/flexlb/dispatcher/BatchSplitter.java \
        flexlb-api/src/test/java/org/flexlb/dispatcher/BatchSplitterTest.java
git commit -m "feat(dispatcher): BatchSplitter.splitArray for generic JSON-array batches"
```

---

### Task V4: `PartialFailureMerger`

> Replaces v1's `ResponseMerger`. One generic merger for every batch endpoint; failure mode comes from the spec. Bodies of successful sub-batches are JSON dicts; we use the **first successful sub's body** as the envelope template, replace its `responseArrayField` with the merged array, then optionally invoke `spec.postMerger` for cross-chunk aggregation (e.g. embedding `usage` sum).

**Files:**
- Create: `flexlb-api/src/main/java/org/flexlb/dispatcher/PartialFailureMerger.java`
- Create: `flexlb-api/src/test/java/org/flexlb/dispatcher/PartialFailureMergerTest.java`
- Modify: `flexlb-api/src/main/java/org/flexlb/dispatcher/MergedResponse.java` — add `failedIndices` field + getter (extends, doesn't replace, existing `succeededChunks`/`totalChunks`).
- Modify: `flexlb-api/src/main/java/org/flexlb/dispatcher/SubBatchResult.java` — add `String reason` (nullable; populated on failure) and `int startIndex` (absolute offset of this chunk's first item in the full batch) — needed so failure factories know the absolute index.

- [ ] **Step 1: Write the failing tests**

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class PartialFailureMergerTest {
    private final ObjectMapper m = new ObjectMapper();
    private final BatchEndpointSpec batchInferSpec = new BatchEndpointSpec(
            "/batch_infer", "prompt_batch", "response_batch", FailedItemFactory.NULL, null);

    @Test
    void allSuccessMergesArraysNoPartialField() {
        ObjectNode body1 = m.createObjectNode();
        ArrayNode arr1 = body1.putArray("response_batch");
        arr1.add(textNode("a0")); arr1.add(textNode("a1"));
        ObjectNode body2 = m.createObjectNode();
        ArrayNode arr2 = body2.putArray("response_batch");
        arr2.add(textNode("a2"));

        var subs = List.of(
                SubBatchResult.ok(body1, 2, 0),
                SubBatchResult.ok(body2, 1, 2));
        MergedResponse merged = PartialFailureMerger.merge(subs, batchInferSpec, m);

        assertFalse(merged.allFailed());
        ArrayNode out = (ArrayNode) merged.body().get("response_batch");
        assertEquals(3, out.size());
        assertEquals("a0", out.get(0).get("v").asText());
        assertEquals("a2", out.get(2).get("v").asText());
        assertNull(merged.body().get("_partial_failure"));
    }

    @Test
    void partialFailurePadsFailedChunkAndAddsMetadata() {
        ObjectNode body1 = m.createObjectNode();
        body1.putArray("response_batch").add(textNode("a0")).add(textNode("a1"));

        var subs = List.of(
                SubBatchResult.ok(body1, 2, 0),
                SubBatchResult.failed(2, 2, "fe_timeout"));
        MergedResponse merged = PartialFailureMerger.merge(subs, batchInferSpec, m);

        assertFalse(merged.allFailed());
        ArrayNode out = (ArrayNode) merged.body().get("response_batch");
        assertEquals(4, out.size());
        assertTrue(out.get(2).isNull());
        assertTrue(out.get(3).isNull());
        ObjectNode pf = (ObjectNode) merged.body().get("_partial_failure");
        assertEquals(2, pf.get("failed_count").asInt());
        assertEquals(4, pf.get("total_count").asInt());
        ArrayNode fi = (ArrayNode) pf.get("failed_indices");
        assertEquals(2, fi.get(0).asInt());
        assertEquals(3, fi.get(1).asInt());
    }

    @Test
    void allFailedFlagsAllFailedNoEnvelopeRequired() {
        var subs = List.of(
                SubBatchResult.failed(2, 0, "fe_down"),
                SubBatchResult.failed(2, 2, "fe_down"));
        MergedResponse merged = PartialFailureMerger.merge(subs, batchInferSpec, m);

        assertTrue(merged.allFailed());
    }

    @Test
    void openAiSpecPadsFailedItemsWithErrorObjects() {
        BatchEndpointSpec spec = new BatchEndpointSpec(
                "/v1/batch/chat/completions", "requests", "responses",
                FailedItemFactory.OPENAI_ERROR, null);
        ObjectNode body1 = m.createObjectNode();
        ArrayNode r = body1.putArray("responses");
        ObjectNode ok = m.createObjectNode(); ok.put("id", "ok0");
        r.add(ok);
        var subs = List.of(
                SubBatchResult.ok(body1, 1, 0),
                SubBatchResult.failed(1, 1, "fe_5xx"));
        MergedResponse merged = PartialFailureMerger.merge(subs, spec, m);

        ArrayNode out = (ArrayNode) merged.body().get("responses");
        assertEquals(2, out.size());
        assertEquals("ok0", out.get(0).get("id").asText());
        assertEquals(1, out.get(1).get("index").asInt());
        assertEquals("fe_5xx", out.get(1).get("error").get("message").asText());
    }

    private ObjectNode textNode(String v) { ObjectNode n = m.createObjectNode(); n.put("v", v); return n; }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `./mvnw test -pl flexlb-api -am -Dtest=PartialFailureMergerTest -P-internal`
Expected: FAIL — `PartialFailureMerger` and the new `SubBatchResult` constructors don't exist.

- [ ] **Step 3: Implement**

Update `SubBatchResult.java` (preserve existing constructors as deprecated or delete depending on caller list; `FanoutService` is the only producer and gets rewritten in V7):

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import lombok.Value;

import javax.annotation.Nullable;

@Value
public class SubBatchResult {
    boolean success;
    int chunkSize;          // number of items in this sub-batch
    int startIndex;         // absolute index of this chunk's first item (0-based)
    @Nullable JsonNode body;
    @Nullable String reason;

    public static SubBatchResult ok(JsonNode body, int chunkSize, int startIndex) {
        return new SubBatchResult(true, chunkSize, startIndex, body, null);
    }
    public static SubBatchResult failed(int chunkSize, int startIndex, String reason) {
        return new SubBatchResult(false, chunkSize, startIndex, null, reason);
    }
}
```

Update `MergedResponse.java`:

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.Value;

import java.util.List;

@Value
public class MergedResponse {
    ObjectNode body;
    int succeededChunks;
    int totalChunks;
    List<Integer> failedIndices;

    public boolean allFailed() { return succeededChunks == 0; }
}
```

`PartialFailureMerger.java`:

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.ArrayList;
import java.util.List;

public final class PartialFailureMerger {
    private PartialFailureMerger() {}

    public static MergedResponse merge(List<SubBatchResult> subs, BatchEndpointSpec spec, ObjectMapper mapper) {
        ObjectNode envelope = null;
        int totalItems = 0;
        int succeededChunks = 0;
        for (SubBatchResult s : subs) {
            totalItems += s.getChunkSize();
            if (s.isSuccess()) {
                succeededChunks++;
                if (envelope == null && s.getBody() instanceof ObjectNode on) {
                    envelope = on.deepCopy();
                    envelope.set(spec.getResponseArrayField(), mapper.createArrayNode());
                }
            }
        }
        if (succeededChunks == 0) {
            return new MergedResponse(mapper.createObjectNode(), 0, subs.size(), allIndices(totalItems));
        }
        ArrayNode merged = (ArrayNode) envelope.get(spec.getResponseArrayField());
        List<Integer> failedIndices = new ArrayList<>();
        for (SubBatchResult s : subs) {
            if (s.isSuccess()) {
                JsonNode subArr = s.getBody().get(spec.getResponseArrayField());
                if (subArr != null && subArr.isArray()) subArr.forEach(merged::add);
            } else {
                for (int i = 0; i < s.getChunkSize(); i++) {
                    int abs = s.getStartIndex() + i;
                    merged.add(spec.getFailedItemFactory().build(abs, s.getReason(), mapper));
                    failedIndices.add(abs);
                }
            }
        }
        if (!failedIndices.isEmpty()) {
            ObjectNode pf = envelope.putObject("_partial_failure");
            pf.put("failed_count", failedIndices.size());
            pf.put("total_count", totalItems);
            ArrayNode fi = pf.putArray("failed_indices");
            failedIndices.forEach(fi::add);
        }
        if (spec.getPostMerger() != null) {
            spec.getPostMerger().apply(envelope, subs, failedIndices, mapper);
        }
        return new MergedResponse(envelope, succeededChunks, subs.size(), failedIndices);
    }

    private static List<Integer> allIndices(int n) {
        List<Integer> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) out.add(i);
        return out;
    }
}
```

Delete `ResponseMerger.java` and `ResponseMergerTest.java` (their logic is now in `PartialFailureMerger`).

- [ ] **Step 4: Run to verify it passes**

Run: `./mvnw test -pl flexlb-api -am -Dtest='PartialFailureMergerTest,BatchSplitterTest' -P-internal`
Expected: PASS. Also confirm the package compiles: `./mvnw compile -pl flexlb-api -am -P-internal`. (Compile may flag `FanoutService` / `DispatchHandler` callers of the old `SubBatchResult.ok(body, n)` / `ResponseMerger` — leave broken until V7 rewires them; do **not** patch around it here. Run `./mvnw test -Dtest=PartialFailureMergerTest -DfailIfNoTests=false` if the broader test target won't compile.)

> **Compile-break warning:** this task deletes `ResponseMerger` while `FanoutService` and `DispatchHandler.handleBatch` still reference it. The package won't fully compile until V7 lands. If you need a green tree intermediate, keep `ResponseMerger.java` as a thin deprecated shim that delegates to `PartialFailureMerger.merge(..., batchInferSpec, mapper)` for the `/batch_infer`-only callers, and delete it in V7.

- [ ] **Step 5: Commit**

```bash
git add flexlb-api/src/main/java/org/flexlb/dispatcher/PartialFailureMerger.java \
        flexlb-api/src/main/java/org/flexlb/dispatcher/MergedResponse.java \
        flexlb-api/src/main/java/org/flexlb/dispatcher/SubBatchResult.java \
        flexlb-api/src/test/java/org/flexlb/dispatcher/PartialFailureMergerTest.java
# If using the shim: git add ResponseMerger.java; otherwise: git rm ResponseMerger.java ResponseMergerTest.java
git commit -m "feat(dispatcher): PartialFailureMerger with per-spec failure shaping + metadata"
```

---

### Task V5: `EmbeddingPostMerger`

> Cross-chunk aggregation for OpenAI embedding-shaped responses: renumber `data[i].index` to absolute offset, sum `usage.prompt_tokens` and `usage.total_tokens` across all successful sub-bodies. Failed positions (already placed by `PartialFailureMerger` with `EMBEDDING_NULL`) keep their factory-assigned index untouched.

**Files:**
- Create: `flexlb-api/src/main/java/org/flexlb/dispatcher/EmbeddingPostMerger.java`
- Create: `flexlb-api/src/test/java/org/flexlb/dispatcher/EmbeddingPostMergerTest.java`

- [ ] **Step 1: Write the failing test**

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class EmbeddingPostMergerTest {
    private final ObjectMapper m = new ObjectMapper();

    @Test
    void renumbersIndicesAndSumsUsage() {
        ObjectNode merged = m.createObjectNode();
        ArrayNode data = merged.putArray("data");
        data.add(itemWithIndex(0));  // chunk 0 item 0 (sub-relative was 0)
        data.add(itemWithIndex(1));  // chunk 0 item 1 (sub-relative was 1)
        data.add(itemWithIndex(0));  // chunk 1 item 0 (sub-relative was 0)
        data.add(itemWithIndex(1));  // chunk 1 item 1
        ObjectNode usage = merged.putObject("usage");
        usage.put("prompt_tokens", 5);
        usage.put("total_tokens", 5);

        ObjectNode sub0 = m.createObjectNode();
        sub0.putObject("usage").put("prompt_tokens", 5).put("total_tokens", 5);
        ObjectNode sub1 = m.createObjectNode();
        sub1.putObject("usage").put("prompt_tokens", 7).put("total_tokens", 7);
        var subs = List.of(
                SubBatchResult.ok(sub0, 2, 0),
                SubBatchResult.ok(sub1, 2, 2));

        new EmbeddingPostMerger().apply(merged, subs, List.of(), m);

        ArrayNode out = (ArrayNode) merged.get("data");
        assertEquals(0, out.get(0).get("index").asInt());
        assertEquals(1, out.get(1).get("index").asInt());
        assertEquals(2, out.get(2).get("index").asInt());
        assertEquals(3, out.get(3).get("index").asInt());
        assertEquals(12, merged.get("usage").get("prompt_tokens").asInt());
        assertEquals(12, merged.get("usage").get("total_tokens").asInt());
    }

    private ObjectNode itemWithIndex(int i) {
        ObjectNode o = m.createObjectNode();
        o.put("index", i);
        ArrayNode emb = o.putArray("embedding");
        emb.add(0.1).add(0.2);
        return o;
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `./mvnw test -pl flexlb-api -am -Dtest=EmbeddingPostMergerTest -P-internal`
Expected: FAIL — `EmbeddingPostMerger` doesn't exist.

- [ ] **Step 3: Implement**

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.List;

public class EmbeddingPostMerger implements PostMerger {
    @Override
    public void apply(ObjectNode mergedBody, List<SubBatchResult> subs, List<Integer> failedIndices, ObjectMapper mapper) {
        ArrayNode data = (ArrayNode) mergedBody.get("data");
        if (data != null) {
            for (int i = 0; i < data.size(); i++) {
                JsonNode item = data.get(i);
                if (item instanceof ObjectNode on) on.put("index", i);
            }
        }
        long promptTokens = 0;
        long totalTokens = 0;
        for (SubBatchResult s : subs) {
            if (!s.isSuccess() || s.getBody() == null) continue;
            JsonNode usage = s.getBody().get("usage");
            if (usage == null) continue;
            promptTokens += usage.path("prompt_tokens").asLong(0);
            totalTokens  += usage.path("total_tokens").asLong(0);
        }
        if (mergedBody.has("usage") || promptTokens > 0 || totalTokens > 0) {
            ObjectNode u = mergedBody.has("usage") ? (ObjectNode) mergedBody.get("usage") : mergedBody.putObject("usage");
            u.put("prompt_tokens", promptTokens);
            u.put("total_tokens", totalTokens);
        }
    }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `./mvnw test -pl flexlb-api -am -Dtest=EmbeddingPostMergerTest -P-internal`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add flexlb-api/src/main/java/org/flexlb/dispatcher/EmbeddingPostMerger.java \
        flexlb-api/src/test/java/org/flexlb/dispatcher/EmbeddingPostMergerTest.java
git commit -m "feat(dispatcher): EmbeddingPostMerger renumbers indices + sums usage"
```

---

### Task V6: `BatchEndpointRegistry` (3 verified specs + spike for `/`)

> Hard-codes the initial spec table. **Verify `/`'s batch response shape first** — the codebase says `len(request.input_texts) > 1` triggers batch mode (`frontend_worker.py:244-249`) but the response wire field is not obvious from a static read. If verification shows `/`'s batch response uses `response_batch` (same as `/batch_infer`), include it; otherwise add a 5th spec with the actual field or drop `/` for now.

**Files:**
- Create: `flexlb-api/src/main/java/org/flexlb/dispatcher/BatchEndpointRegistry.java`
- Create: `flexlb-api/src/test/java/org/flexlb/dispatcher/BatchEndpointRegistryTest.java`

- [ ] **Step 1: Verify the `/` batch response shape**

Spin up a local FE (any small open-source model is fine; see `rtp_llm/flexlb/CLAUDE.md` "Run Application" or use `internal_source/.cursor/skills/test-execution/pre_build_check.sh`-prepared env) and POST:

```bash
curl -s http://127.0.0.1:<fe_port>/ \
  -H 'content-type: application/json' \
  -d '{"prompt": ["你好", "在吗", "吃了吗"]}' | jq 'keys'
```

Record the top-level key holding the array of results in this task's commit message. If it's `response_batch`, the spec below is correct as-written. If it's something else (e.g. `responses`), change `responseArrayField` accordingly. If the local environment can't be brought up in a reasonable time, **skip `/` from the registry** and add a TODO comment — V12 will revisit.

- [ ] **Step 2: Write the failing test**

```java
package org.flexlb.dispatcher;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class BatchEndpointRegistryTest {
    @Test
    void registryContainsAllFourSpecsKeyedByPath() {
        List<BatchEndpointSpec> specs = new BatchEndpointRegistry().batchSpecs();
        Map<String, BatchEndpointSpec> byPath = specs.stream()
                .collect(java.util.stream.Collectors.toMap(BatchEndpointSpec::getPath, s -> s));

        assertEquals("prompt_batch", byPath.get("/batch_infer").getRequestArrayField());
        assertEquals("response_batch", byPath.get("/batch_infer").getResponseArrayField());

        // Adjust the next two lines to match Step 1's findings; the spec below assumes `/`'s
        // batch response wire field is `response_batch` (same as /batch_infer). If verification
        // showed a different field, change here and in BatchEndpointRegistry.batchSpecs().
        assertEquals("prompt", byPath.get("/").getRequestArrayField());
        assertEquals("response_batch", byPath.get("/").getResponseArrayField());

        assertEquals("requests", byPath.get("/v1/batch/chat/completions").getRequestArrayField());
        assertEquals("responses", byPath.get("/v1/batch/chat/completions").getResponseArrayField());

        assertEquals("input", byPath.get("/v1/embeddings").getRequestArrayField());
        assertEquals("data", byPath.get("/v1/embeddings").getResponseArrayField());
        assertTrue(byPath.get("/v1/embeddings").getPostMerger() instanceof EmbeddingPostMerger);
    }
}
```

- [ ] **Step 3: Run to verify it fails**

Run: `./mvnw test -pl flexlb-api -am -Dtest=BatchEndpointRegistryTest -P-internal`
Expected: FAIL — registry doesn't exist.

- [ ] **Step 4: Implement**

```java
package org.flexlb.dispatcher;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.List;

@Configuration
public class BatchEndpointRegistry {

    @Bean
    public List<BatchEndpointSpec> batchSpecs() {
        EmbeddingPostMerger embedding = new EmbeddingPostMerger();
        return List.of(
                new BatchEndpointSpec("/batch_infer",
                        "prompt_batch", "response_batch", FailedItemFactory.NULL, null),
                // `/` raw inference, batch mode triggered when `prompt` is a list.
                // Wire field assumed to match /batch_infer; verify per Step 1.
                new BatchEndpointSpec("/",
                        "prompt", "response_batch", FailedItemFactory.NULL, null),
                new BatchEndpointSpec("/v1/batch/chat/completions",
                        "requests", "responses", FailedItemFactory.OPENAI_ERROR, null),
                new BatchEndpointSpec("/v1/embeddings",
                        "input", "data", FailedItemFactory.EMBEDDING_NULL, embedding)
        );
    }
}
```

> Embedding variants (`/v1/embeddings/dense|sparse|colbert|similarity`, `/v1/reranker`, `/v1/classifier`) are **not** added in V6. They share the embedding response shape but have different request fields (e.g. reranker uses `documents`, classifier uses … TBD). Add them after V10 lands and each variant's wire format is verified. Each addition is one row in `batchSpecs()`.

- [ ] **Step 5: Run to verify it passes + Commit**

Run: `./mvnw test -pl flexlb-api -am -Dtest=BatchEndpointRegistryTest -P-internal`
Expected: PASS.

```bash
git add flexlb-api/src/main/java/org/flexlb/dispatcher/BatchEndpointRegistry.java \
        flexlb-api/src/test/java/org/flexlb/dispatcher/BatchEndpointRegistryTest.java
git commit -m "feat(dispatcher): BatchEndpointRegistry with batch_infer + / + OpenAI batch chat + embeddings"
```

---

### Task V7: `GenericBatchHandler`

> One handler for every batch spec. Logic: parse JSON body → look up the spec → if `requestArrayField` is missing/empty/not-array OR array length ≤ `subBatchSize` then forward as a single passthrough (no value-add from splitting one chunk); otherwise split with `BatchSplitter.splitArray`, fan out each chunk (replacing only `requestArrayField`, preserving everything else like `generate_config`, `model`, etc.), then `PartialFailureMerger.merge`. HTTP 200 on partial success, 500 only on `MergedResponse.allFailed()`.

**Files:**
- Create: `flexlb-api/src/main/java/org/flexlb/dispatcher/GenericBatchHandler.java`
- Create: `flexlb-api/src/test/java/org/flexlb/dispatcher/GenericBatchHandlerTest.java`
- Modify: `flexlb-api/src/main/java/org/flexlb/dispatcher/FanoutService.java` — generalize from `dispatch(List<String> prompts, JsonNode gc)` to `dispatchChunks(String fePath, List<ObjectNode> chunkBodies)`. Each chunk body is already the **full** FE request (envelope + array field replaced).
- Modify: `flexlb-api/src/main/java/org/flexlb/dispatcher/FeClient.java` / `WebClientFeClient.java` — generalize `postBatch(feUrl, body)` to `post(feUrl, fePath, body)` so the same client posts to any FE path.
- Modify: `flexlb-api/src/main/java/org/flexlb/dispatcher/DispatchHandler.java` — delete `handleBatch` (and its tests). Keep `handlePassthrough`.

- [ ] **Step 1: Write the failing tests**

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpStatus;
import org.springframework.mock.web.reactive.function.server.MockServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.net.URI;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class GenericBatchHandlerTest {
    private final ObjectMapper mapper = new ObjectMapper();
    private final BatchEndpointSpec spec = new BatchEndpointSpec(
            "/batch_infer", "prompt_batch", "response_batch", FailedItemFactory.NULL, null);

    @Test
    void singleChunkPassesThroughWithoutSplit() {
        FanoutService fanout = mock(FanoutService.class);
        PassthroughClient passthrough = mock(PassthroughClient.class);
        when(passthrough.forward(any())).thenReturn(
                Mono.just(ServerResponse.ok().bodyValue("p").build().block()));

        var handler = new GenericBatchHandler(fanout, passthrough, mapper, /*K=*/5);
        ObjectNode body = mapper.createObjectNode();
        body.putArray("prompt_batch").add("a").add("b");

        var req = MockServerRequest.builder()
                .method(org.springframework.http.HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/batch_infer"))
                .body(Mono.just(body));
        StepVerifier.create(handler.handle(req, spec)).expectNextCount(1).verifyComplete();
        verify(passthrough).forward(any());
        verifyNoInteractions(fanout);
    }

    @Test
    void multiChunkSplitsAndMergesReturning200() {
        FanoutService fanout = mock(FanoutService.class);
        ObjectNode subBody = mapper.createObjectNode();
        subBody.putArray("response_batch").add("r0").add("r1").add("r2");
        when(fanout.dispatchChunks(eq("/batch_infer"), anyList())).thenAnswer(inv -> {
            List<ObjectNode> bodies = inv.getArgument(1);
            List<SubBatchResult> subs = new ArrayList<>();
            int start = 0;
            for (ObjectNode b : bodies) {
                int sz = b.get("prompt_batch").size();
                subs.add(SubBatchResult.ok(subBody, sz, start));
                start += sz;
            }
            return Mono.just(PartialFailureMerger.merge(subs, spec, mapper));
        });

        var handler = new GenericBatchHandler(fanout, mock(PassthroughClient.class), mapper, /*K=*/2);
        ObjectNode body = mapper.createObjectNode();
        body.putArray("prompt_batch").add("a").add("b").add("c");

        var req = MockServerRequest.builder()
                .method(org.springframework.http.HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/batch_infer"))
                .body(Mono.just(body));
        StepVerifier.create(handler.handle(req, spec))
                .assertNext(resp -> assertEquals(HttpStatus.OK, resp.statusCode()))
                .verifyComplete();
        verify(fanout).dispatchChunks(eq("/batch_infer"), argThat(list -> list.size() == 2));
    }

    @Test
    void allFailedReturns500() {
        FanoutService fanout = mock(FanoutService.class);
        when(fanout.dispatchChunks(any(), anyList())).thenReturn(
                Mono.just(new MergedResponse(mapper.createObjectNode(), 0, 2, List.of(0, 1, 2, 3))));

        var handler = new GenericBatchHandler(fanout, mock(PassthroughClient.class), mapper, /*K=*/2);
        ObjectNode body = mapper.createObjectNode();
        body.putArray("prompt_batch").add("a").add("b").add("c").add("d");

        var req = MockServerRequest.builder()
                .method(org.springframework.http.HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/batch_infer"))
                .body(Mono.just(body));
        StepVerifier.create(handler.handle(req, spec))
                .assertNext(resp -> assertEquals(HttpStatus.INTERNAL_SERVER_ERROR, resp.statusCode()))
                .verifyComplete();
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `./mvnw test -pl flexlb-api -am -Dtest=GenericBatchHandlerTest -P-internal`
Expected: FAIL — handler/types not yet present.

- [ ] **Step 3: Implement**

`FeClient.java`:

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import reactor.core.publisher.Mono;

public interface FeClient {
    Mono<JsonNode> post(String feBaseUrl, String fePath, ObjectNode body);
}
```

`WebClientFeClient.java`: rename `postBatch(feUrl, body)` to `post(feUrl, fePath, body)`; change URI build to `feBaseUrl + fePath` (no hard-coded `/batch_infer`). Keep the 16 MiB cap.

`FanoutService.java` (rewritten):

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.util.ArrayList;
import java.util.List;

@Slf4j
@RequiredArgsConstructor
public class FanoutService {
    private final FeClient feClient;
    private final FePool fePool;
    private final ObjectMapper mapper;

    /** Each entry in {@code chunkBodies} is a fully-formed FE request (array field already sliced). */
    public Mono<List<SubBatchResult>> dispatchChunks(String fePath, List<ObjectNode> chunkBodies,
                                                     BatchEndpointSpec spec) {
        List<Mono<SubBatchResult>> calls = new ArrayList<>(chunkBodies.size());
        int start = 0;
        for (ObjectNode body : chunkBodies) {
            int chunkSize = body.get(spec.getRequestArrayField()).size();
            int startIndex = start;
            calls.add(Mono.fromCallable(fePool::next)
                    .flatMap(feUrl -> feClient.post(feUrl, fePath, body)
                            .map(resp -> SubBatchResult.ok(resp, chunkSize, startIndex))
                            .onErrorResume(e -> {
                                log.warn("FE chunk failed: url={}, path={}, size={}", feUrl, fePath, chunkSize, e);
                                return Mono.just(SubBatchResult.failed(chunkSize, startIndex, briefReason(e)));
                            }))
                    .onErrorResume(e -> {
                        log.warn("FE pick failed for chunk size={}", chunkSize, e);
                        return Mono.just(SubBatchResult.failed(chunkSize, startIndex, briefReason(e)));
                    }));
            start += chunkSize;
        }
        return Flux.mergeSequential(Flux.fromIterable(calls))
                .collectList()
                .publishOn(Schedulers.parallel());
    }

    private static String briefReason(Throwable e) {
        String m = e.getClass().getSimpleName();
        return e.getMessage() == null ? m : m + ": " + e.getMessage();
    }
}
```

`GenericBatchHandler.java`:

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import java.util.ArrayList;
import java.util.List;

@Slf4j
@RequiredArgsConstructor
public class GenericBatchHandler {
    private final FanoutService fanoutService;
    private final PassthroughClient passthroughClient;
    private final ObjectMapper mapper;
    private final int subBatchSize;

    public Mono<ServerResponse> handle(ServerRequest request, BatchEndpointSpec spec) {
        return request.bodyToMono(JsonNode.class).flatMap(body -> {
            if (!(body instanceof ObjectNode obj)) return passthroughClient.forward(request);
            JsonNode arr = obj.get(spec.getRequestArrayField());
            if (arr == null || !arr.isArray() || arr.size() <= subBatchSize) {
                return passthroughClient.forward(request);
            }
            List<ArrayNode> chunks = BatchSplitter.splitArray((ArrayNode) arr, subBatchSize, mapper);
            List<ObjectNode> chunkBodies = new ArrayList<>(chunks.size());
            for (ArrayNode chunk : chunks) {
                ObjectNode copy = obj.deepCopy();
                copy.set(spec.getRequestArrayField(), chunk);
                chunkBodies.add(copy);
            }
            return fanoutService.dispatchChunks(spec.getPath(), chunkBodies, spec)
                    .map(subs -> PartialFailureMerger.merge(subs, spec, mapper))
                    .flatMap(merged -> merged.allFailed()
                            ? errorResponse(merged)
                            : ServerResponse.ok().contentType(MediaType.APPLICATION_JSON).bodyValue(merged.body()));
        }).onErrorResume(e -> {
            log.warn("batch dispatch failed before fanout: spec={}", spec.getPath(), e);
            ObjectNode err = mapper.createObjectNode();
            err.put("error", "dispatch_failed");
            err.put("message", String.valueOf(e.getMessage()));
            return ServerResponse.status(500).contentType(MediaType.APPLICATION_JSON).bodyValue(err);
        });
    }

    private Mono<ServerResponse> errorResponse(MergedResponse merged) {
        ObjectNode body = mapper.createObjectNode();
        body.put("error", "all_sub_batches_failed");
        body.put("failed_count", merged.getFailedIndices().size());
        body.put("total_chunks", merged.getTotalChunks());
        return ServerResponse.status(500).contentType(MediaType.APPLICATION_JSON).bodyValue(body);
    }
}
```

Modify `DispatchHandler.java`: delete `handleBatch` and its dependency on `FanoutService` / `mapper`. Keep only the passthrough:

```java
@RequiredArgsConstructor
public class DispatchHandler {
    private final PassthroughClient passthroughClient;
    public Mono<ServerResponse> handlePassthrough(ServerRequest request) {
        return passthroughClient.forward(request);
    }
}
```

Delete `DispatchHandlerTest.java`'s batch cases (passthrough cases stay or move into a thin `DispatchHandlerTest.passthroughDelegates()`).

- [ ] **Step 4: Run to verify it passes**

Run: `./mvnw test -pl flexlb-api -am -Dtest='GenericBatchHandlerTest,PartialFailureMergerTest,BatchSplitterTest' -P-internal`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add flexlb-api/src/main/java/org/flexlb/dispatcher/GenericBatchHandler.java \
        flexlb-api/src/main/java/org/flexlb/dispatcher/FanoutService.java \
        flexlb-api/src/main/java/org/flexlb/dispatcher/FeClient.java \
        flexlb-api/src/main/java/org/flexlb/dispatcher/WebClientFeClient.java \
        flexlb-api/src/main/java/org/flexlb/dispatcher/DispatchHandler.java \
        flexlb-api/src/test/java/org/flexlb/dispatcher/GenericBatchHandlerTest.java
# If you deleted ResponseMerger as a shim in V4:
# git rm flexlb-api/src/main/java/org/flexlb/dispatcher/ResponseMerger.java \
#        flexlb-api/src/test/java/org/flexlb/dispatcher/ResponseMergerTest.java
git rm flexlb-api/src/test/java/org/flexlb/dispatcher/DispatchHandlerTest.java  # or trim to passthrough-only
git commit -m "feat(dispatcher): GenericBatchHandler dispatches any batch endpoint via spec"
```

---

### Task V8: Rewire `DispatchRouter` + `DispatcherConfiguration` from the registry

**Files:**
- Modify: `flexlb-api/src/main/java/org/flexlb/dispatcher/DispatchRouter.java`
- Modify: `flexlb-api/src/main/java/org/flexlb/dispatcher/DispatcherConfiguration.java`
- Modify: `flexlb-api/src/test/java/org/flexlb/dispatcher/DispatchRouterTest.java`
- Modify: `flexlb-api/src/test/java/org/flexlb/dispatcher/DispatcherConfigurationTest.java`

- [ ] **Step 1: Write the failing test**

In `DispatchRouterTest`, add (replacing the V1 placeholder that only knew `/dispatcher/batch_infer`):

```java
@Test
void registersOneRouteForEachSpec() {
    GenericBatchHandler batch = mock(GenericBatchHandler.class);
    when(batch.handle(any(), any()))
            .thenAnswer(inv -> ServerResponse.ok().bodyValue(((BatchEndpointSpec) inv.getArgument(1)).getPath()).build().block());
    DispatchHandler passthrough = mock(DispatchHandler.class);
    when(passthrough.handlePassthrough(any())).thenReturn(
            ServerResponse.ok().bodyValue("pass").build().block());

    List<BatchEndpointSpec> specs = List.of(
            new BatchEndpointSpec("/batch_infer", "prompt_batch", "response_batch", FailedItemFactory.NULL, null),
            new BatchEndpointSpec("/v1/embeddings", "input", "data", FailedItemFactory.EMBEDDING_NULL, new EmbeddingPostMerger()));

    var routes = new DispatchRouter(batch, passthrough, specs).routes();
    WebTestClient client = WebTestClient.bindToRouterFunction(routes).build();

    client.post().uri("/dispatcher/batch_infer").bodyValue("{}").exchange()
            .expectStatus().isOk().expectBody(String.class).isEqualTo("/batch_infer");
    client.post().uri("/dispatcher/v1/embeddings").bodyValue("{}").exchange()
            .expectStatus().isOk().expectBody(String.class).isEqualTo("/v1/embeddings");
    client.get().uri("/dispatcher/v1/models").exchange()
            .expectStatus().isOk().expectBody(String.class).isEqualTo("pass");
    client.post().uri("/batch_infer").bodyValue("{}").exchange().expectStatus().isNotFound();
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `./mvnw test -pl flexlb-api -am -Dtest=DispatchRouterTest -P-internal`
Expected: FAIL — new `DispatchRouter` constructor signature doesn't exist.

- [ ] **Step 3: Implement**

`DispatchRouter.java`:

```java
package org.flexlb.dispatcher;

import lombok.RequiredArgsConstructor;
import org.springframework.web.reactive.function.server.RequestPredicates;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;

import java.util.List;

import static org.springframework.web.reactive.function.server.RouterFunctions.route;

@RequiredArgsConstructor
public class DispatchRouter {

    private final GenericBatchHandler batchHandler;
    private final DispatchHandler passthroughHandler;
    private final List<BatchEndpointSpec> specs;

    public RouterFunction<ServerResponse> routes() {
        var b = route();
        for (BatchEndpointSpec spec : specs) {
            b.POST("/dispatcher" + spec.getPath(), req -> batchHandler.handle(req, spec));
        }
        return b.route(RequestPredicates.path("/dispatcher/**"), passthroughHandler::handlePassthrough)
                .build();
    }
}
```

`DispatcherConfiguration.java` — wire the new bean graph. Replace the `dispatcherRoutes` method body so that:
- It injects `List<BatchEndpointSpec>` from `BatchEndpointRegistry`.
- It constructs `GenericBatchHandler` and `DispatchHandler` (passthrough only).
- It passes all three into `new DispatchRouter(...)`.
- Removes the old `DispatchHandler(FanoutService, PassthroughClient, mapper)` ctor call.

The existing FE pool / `ServiceDiscovery` plumbing stays intact. Keep `@Order(Ordered.LOWEST_PRECEDENCE)` on the route bean for now — harmless and lets the cleanup of Master's `@Order(0)` happen in a separate change.

Update `DispatcherConfigurationTest.buildsRouterWhenEnabled` to also import/expect `BatchEndpointRegistry` in the slice.

- [ ] **Step 4: Run the full dispatcher test suite to confirm green**

Run: `./mvnw test -pl flexlb-api -am -Dtest='org.flexlb.dispatcher.*' -P-internal`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add flexlb-api/src/main/java/org/flexlb/dispatcher/DispatchRouter.java \
        flexlb-api/src/main/java/org/flexlb/dispatcher/DispatcherConfiguration.java \
        flexlb-api/src/test/java/org/flexlb/dispatcher/DispatchRouterTest.java \
        flexlb-api/src/test/java/org/flexlb/dispatcher/DispatcherConfigurationTest.java
git commit -m "feat(dispatcher): wire generic batch routing from BatchEndpointRegistry"
```

---

### Task V9: Stream-friendly passthrough timeouts

> v1's `WebClientPassthroughClient.forward()` wraps the whole exchange in `.timeout(3000ms)`. Streaming responses (SSE chat completion, typical 30–120 s) are killed at 3 s. Fix: drop the per-exchange wall-clock; rely on **connect timeout** (fast detection of dead FE) + **response timeout** (time to first response byte; short) + **total-duration cap** (very long, default 600 s).

**Files:**
- Modify: `flexlb-api/src/main/java/org/flexlb/dispatcher/WebClientPassthroughClient.java`
- Modify: `flexlb-api/src/main/java/org/flexlb/dispatcher/DispatchConfig.java`
- Modify: `flexlb-api/src/main/java/org/flexlb/dispatcher/DispatcherConfiguration.java` — pass the new timeouts through the existing `dispatcher-fe` `ConnectionProvider` / `HttpClient`.
- Create: `flexlb-api/src/test/java/org/flexlb/dispatcher/StreamingPassthroughTest.java`

- [ ] **Step 1: Write the failing test**

```java
package org.flexlb.dispatcher;

import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okio.Buffer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpMethod;
import org.springframework.mock.web.reactive.function.server.MockServerRequest;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.test.StepVerifier;

import java.net.URI;
import java.time.Duration;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class StreamingPassthroughTest {
    private MockWebServer fe;

    @BeforeEach void up() throws Exception { fe = new MockWebServer(); fe.start(); }
    @AfterEach  void down() throws Exception { fe.shutdown(); }

    @Test
    void streamingResponseLongerThanFiveSecondsIsNotCutOff() {
        Buffer body = new Buffer();
        for (int i = 0; i < 6; i++) body.writeUtf8("data: chunk" + i + "\n\n");
        fe.enqueue(new MockResponse()
                .setHeader("content-type", "text/event-stream")
                .setBody(body)
                .setBodyDelay(6, java.util.concurrent.TimeUnit.SECONDS));   // FE keeps the body open >5s

        var pool = new FePool(() -> List.of("http://" + fe.getHostName() + ":" + fe.getPort()));
        var client = new WebClientPassthroughClient(WebClient.create(), pool,
                /*connectMs=*/2000, /*responseMs=*/10000, /*maxDurationMs=*/600000);

        var req = MockServerRequest.builder()
                .method(HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/v1/chat/completions"))
                .body(reactor.core.publisher.Mono.empty());

        StepVerifier.create(client.forward(req).flatMapMany(resp ->
                        resp.bodyToFlux(org.springframework.core.io.buffer.DataBuffer.class)))
                .thenConsumeWhile(buf -> true)
                .expectComplete()
                .verify(Duration.ofSeconds(20));
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `./mvnw test -pl flexlb-api -am -Dtest=StreamingPassthroughTest -P-internal`
Expected: FAIL — the existing 3 s timeout cuts the body before all chunks arrive (test will see a TimeoutException).

- [ ] **Step 3: Implement**

Update `DispatchConfig.java`:

```java
private int feConnectTimeoutMs   = 2000;
private int feResponseTimeoutMs  = 5000;   // batch chunk wall-clock (FanoutService callers)
private int feMaxStreamDurationMs = 600_000;  // passthrough total-duration cap (10 min)
// feRequestTimeoutMs: drop or keep as deprecated alias of feResponseTimeoutMs for env-JSON compat
```

Update `WebClientPassthroughClient.java` constructor to take the three timeouts and drop the `.timeout(...)` on the exchange:

```java
@RequiredArgsConstructor
public class WebClientPassthroughClient implements PassthroughClient {
    private final WebClient webClient;
    private final FePool fePool;
    private final int connectTimeoutMs;
    private final int responseTimeoutMs;     // time to first byte
    private final int maxStreamDurationMs;   // hard cap on full body

    @Override
    public Mono<ServerResponse> forward(ServerRequest request) {
        return Mono.fromCallable(fePool::next).flatMap(feBaseUrl -> {
            URI src = request.uri();
            String fePath = src.getRawPath().startsWith("/dispatcher/")
                    ? src.getRawPath().substring("/dispatcher".length())
                    : src.getRawPath();
            String pathAndQuery = src.getRawQuery() == null ? fePath : fePath + "?" + src.getRawQuery();
            URI target = URI.create(feBaseUrl + pathAndQuery);
            Flux<DataBuffer> bodyStream = request.bodyToFlux(DataBuffer.class);
            return webClient.method(request.method())
                    .uri(target)
                    .headers(h -> h.addAll(request.headers().asHttpHeaders()))
                    .body(BodyInserters.fromDataBuffers(bodyStream))
                    .exchangeToMono(clientResponse ->
                            ServerResponse.status(clientResponse.statusCode())
                                    .headers(h -> h.addAll(clientResponse.headers().asHttpHeaders()))
                                    .body(clientResponse.bodyToFlux(DataBuffer.class)
                                            .timeout(Duration.ofMillis(maxStreamDurationMs)),
                                          DataBuffer.class));
        });
    }
}
```

In `DispatcherConfiguration`, configure the `HttpClient` underlying the `WebClient` with `connectTimeoutMs` (`ChannelOption.CONNECT_TIMEOUT_MILLIS`) and `responseTimeoutMs` (`HttpClient.responseTimeout(...)`) — these apply per request and short-circuit dead FEs without affecting in-flight stream bodies:

```java
HttpClient http = HttpClient.create(feConnections)
        .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, cfg.getFeConnectTimeoutMs())
        .responseTimeout(Duration.ofMillis(cfg.getFeResponseTimeoutMs()));
```

Pass `cfg.getFeMaxStreamDurationMs()` into `WebClientPassthroughClient`.

- [ ] **Step 4: Run to verify it passes**

Run: `./mvnw test -pl flexlb-api -am -Dtest='StreamingPassthroughTest,WebClientPassthroughClientTest' -P-internal`
Expected: PASS — streaming body arrives in full; existing prefix-strip test unaffected.

- [ ] **Step 5: Commit**

```bash
git add flexlb-api/src/main/java/org/flexlb/dispatcher/WebClientPassthroughClient.java \
        flexlb-api/src/main/java/org/flexlb/dispatcher/DispatchConfig.java \
        flexlb-api/src/main/java/org/flexlb/dispatcher/DispatcherConfiguration.java \
        flexlb-api/src/test/java/org/flexlb/dispatcher/StreamingPassthroughTest.java
git commit -m "fix(dispatcher): stream-friendly passthrough timeouts (connect/response/max-duration)"
```

---

### Task V10: End-to-end test — 4 endpoints with induced partial failure

> One integration test class binds the assembled `DispatchRouter` via `WebTestClient.bindToRouterFunction`, points the `FePool` at **3 MockWebServers**, and exercises all four registered specs. Each test induces one chunk failure (5xx from one FE) and asserts the response shape per spec.

**Files:**
- Create: `flexlb-api/src/test/java/org/flexlb/dispatcher/DispatcherE2ETest.java`

- [ ] **Step 1: Write the test**

Walks through 4 cases:

1. `POST /dispatcher/batch_infer` with `prompt_batch` of 9, `subBatchSize=3` → 3 chunks → 3 FEs. Make FE #2 return 500. Assert merged `response_batch.size()==9`, items `[3..5]==null`, top-level `_partial_failure` present with `failed_indices=[3,4,5]`, HTTP 200.
2. `POST /dispatcher/v1/batch/chat/completions` with `requests` of 4, K=2 → 2 chunks. Make one FE 500. Assert merged `responses.size()==4`, the failed positions hold `{index, error.code, error.message}`, HTTP 200.
3. `POST /dispatcher/v1/embeddings` with `input` of 6, K=2 → 3 chunks. Make one FE 500. Assert merged `data.size()==6`, indices are `0..5` absolute, `usage.prompt_tokens` is the sum of successful sub-bodies, failed item has `embedding: null + error`, HTTP 200.
4. `POST /dispatcher/v1/chat/completions` with no batch shape → goes through passthrough; assert one FE got the unchanged body at `/v1/chat/completions` (prefix stripped). HTTP 200.

(Full test class code is straightforward composition of MockWebServer + WebTestClient — engineer should reuse the patterns from `WebClientFeClientTest` and `StreamingPassthroughTest`. Do **not** stub `GenericBatchHandler`/`PartialFailureMerger` here; the point of E2E is to exercise the real wiring.)

- [ ] **Step 2: Run**

Run: `./mvnw test -pl flexlb-api -am -Dtest=DispatcherE2ETest -P-internal`
Expected: PASS — all 4 cases.

- [ ] **Step 3: Commit**

```bash
git add flexlb-api/src/test/java/org/flexlb/dispatcher/DispatcherE2ETest.java
git commit -m "test(dispatcher): E2E split/fanout/merge for all 4 batch endpoints incl. induced partial failure"
```

---

### Task V11: Docs — `DISPATCH_CONFIG` + endpoint table + partial-failed contract + migration

**Files:**
- Modify: `rtp_llm/flexlb/CLAUDE.md` (Configuration section)
- Modify: `rtp_llm/flexlb/README.md`

- [ ] **Step 1: Write the docs section** in `CLAUDE.md`, replacing/adding the v1 stub:

````markdown
### DISPATCH_CONFIG (optional, opt-in)

When set with `enabled=true`, FlexLB serves `/dispatcher/<original_fe_path>` on its 7001 listener. Requests are matched against a hard-coded **batch endpoint registry** (see `BatchEndpointRegistry`); batch-shaped requests (whose registered array field is a list larger than `subBatchSize`) are split across the FE pool and merged; everything else (including small-batch requests) is passthrough-forwarded to one FE.

```json
{
  "enabled": true,
  "fePoolServiceId": "rtp_llm.frontend.service",
  "subBatchSize": 5,
  "feConnectTimeoutMs": 2000,
  "feResponseTimeoutMs": 5000,
  "feMaxStreamDurationMs": 600000,
  "feMaxConnections": 200,
  "feMaxPendingAcquire": 1000,
  "feMaxResponseBytes": 16777216
}
```

**Batch endpoint registry (built-in):**

| Path under `/dispatcher/` | Request array field | Response array field | Failure shape | Cross-chunk aggregation |
|---|---|---|---|---|
| `/batch_infer` | `prompt_batch` | `response_batch` | `null` | — |
| `/` | `prompt` (when list) | `response_batch` *(verify per Task V6 Step 1)* | `null` | — |
| `/v1/batch/chat/completions` | `requests` | `responses` | `{index, error{code,message}}` | — |
| `/v1/embeddings` | `input` (when list) | `data` | `{index, embedding: null, error}` | `data[i].index` renumbered; `usage.{prompt,total}_tokens` summed |

**Partial-failure contract:**
- HTTP 200 on full success or any partial success.
- HTTP 500 only when **every** sub-batch failed.
- On partial success, the response body contains an extra top-level object: `_partial_failure: { failed_indices: [...], failed_count: N, total_count: M }`.
- Failed positions in the response array are filled in-place by the per-endpoint failure factory (see table). **Indices are preserved** so callers can correlate failures back to input positions.

**Migration:**
- Pre-dispatcher clients calling `<fe>/batch_infer` keep working — they hit FE directly. To opt in, change the URL to `<master>:7001/dispatcher/batch_infer` (everything else stays the same; the registered field names match FE's existing wire format).
- Streaming endpoints (e.g. `/v1/chat/completions` with `stream=true`) work through the passthrough as long as `feMaxStreamDurationMs` exceeds the longest expected response time.
- Direct-to-FE remains the bypass for any client that can't change URLs.

**Known limits (deferred):**
- `request_id` set by `frontend_server.py` overwrites any upstream id — dispatcher→FE trace linkage is broken. Tracked in `project_frontend_request_id_overwrite.md`.
- Failed pre-assigned FE targets do not auto-failover (BE pre-assignment is not enabled in Stage 2).
- Embedding variants (`/v1/embeddings/{dense,sparse,colbert,similarity}`, `/v1/reranker`, `/v1/classifier`) — not in the registry yet; add one row each after verifying wire shape.
````

Mirror a short version in `README.md` if its surface lists the env vars.

- [ ] **Step 2: Commit**

```bash
git add rtp_llm/flexlb/CLAUDE.md rtp_llm/flexlb/README.md
git commit -m "docs(dispatcher): /dispatcher/ prefix, endpoint registry, partial-failed contract"
```

---

### Task V12: Housekeeping — supersede v1 Tasks 14–17 in this plan; delete dead helpers

**Files:**
- Modify: `docs/superpowers/plans/2026-05-20-flexlb-dispatcher.md` — add status notes to Tasks 14, 15, 15b, 16, 16b, 17.
- Delete (if not already): `flexlb-api/src/main/java/org/flexlb/dispatcher/ResponseMerger.java` (and test), `DispatchProtocol.java` (constants moved to specs).

- [ ] **Step 1: Annotate superseded tasks**

In the existing plan body, prepend each of Tasks 14, 15, 15b, 16, 17 with:

```markdown
> **Status (2026-05-25):** SUPERSEDED by Stage 2 (Tasks V1–V11). Do not implement. Pre-assign deferred indefinitely (see Stage 2 §"What is NOT changing"). E2E coverage moved to Task V10. Docs moved to Task V11.
```

Task 16b stays (it's a perf concern, not a code task — still worth running once Stage 2 is in production).

- [ ] **Step 2: Delete dead helpers**

```bash
# If `DispatchProtocol`'s constants are now all dead (spec.fields replaced them):
git grep DispatchProtocol flexlb-api/src/main || git rm flexlb-api/src/main/java/org/flexlb/dispatcher/DispatchProtocol.java
# Repeat for ResponseMerger / ResponseMergerTest if they survived as a V4 shim:
git grep ResponseMerger flexlb-api/src/main || \
  git rm flexlb-api/src/main/java/org/flexlb/dispatcher/ResponseMerger.java \
         flexlb-api/src/test/java/org/flexlb/dispatcher/ResponseMergerTest.java
```

- [ ] **Step 3: Run the full module suite once**

Run: `./mvnw test -pl flexlb-api -am -P-internal`
Expected: PASS, no orphaned references.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/plans/2026-05-20-flexlb-dispatcher.md
git commit -m "chore(dispatcher): supersede v1 Tasks 14-17; remove dead helpers"
```

---

## Stage 2 Self-Review

- **Spec coverage:** `/dispatcher/` prefix (V1), table-driven specs (V2/V6), generic split (V3), partial-failure merge (V4), embedding aggregation (V5), one-handler-per-spec (V7), wiring (V8), streaming fix (V9), end-to-end coverage (V10), docs (V11), housekeeping (V12). The user's two concerns from the design discussion — (a) "other endpoints also need fanout, not just passthrough" and (b) "partial-failed should be unified" — are covered by V6 (4 endpoints) + V4 (single PartialFailureMerger applied to every spec). `/health` shadow bug is resolved by V1 (path disjointness).
- **Out of scope (recorded, not built):** BE pre-assignment (v1 Tasks 13b/14/15/15b — deferred); embedding variants (V6 covers `/v1/embeddings`; add one row each post-V10 per variant); `request_id` continuity; dynamic spec table (env-JSON–driven additions to the registry).
- **Type consistency (Stage 2):** `BatchEndpointSpec(path, requestArrayField, responseArrayField, FailedItemFactory, @Nullable PostMerger)`; `FailedItemFactory.build(int, String, ObjectMapper)→JsonNode`; `PostMerger.apply(ObjectNode, List<SubBatchResult>, List<Integer>, ObjectMapper)→void`; `SubBatchResult(success, chunkSize, startIndex, body, reason)`; `MergedResponse(body, succeededChunks, totalChunks, failedIndices)`; `BatchSplitter.splitArray(ArrayNode, int, ObjectMapper)→List<ArrayNode>`; `PartialFailureMerger.merge(List<SubBatchResult>, BatchEndpointSpec, ObjectMapper)→MergedResponse`; `FanoutService.dispatchChunks(String fePath, List<ObjectNode> chunkBodies, BatchEndpointSpec)→Mono<List<SubBatchResult>>`; `FeClient.post(String feUrl, String fePath, ObjectNode)→Mono<JsonNode>`; `GenericBatchHandler.handle(ServerRequest, BatchEndpointSpec)→Mono<ServerResponse>`; `DispatchRouter(GenericBatchHandler, DispatchHandler, List<BatchEndpointSpec>)`; `WebClientPassthroughClient(WebClient, FePool, int connectMs, int responseMs, int maxDurationMs)`.
- **Spec table coupling:** the registry hard-codes 4 paths. If FE adds a new batch endpoint, the new row is the only change required — no handler/merger/splitter touchup. Embedding variants share `EmbeddingPostMerger`; reranker/classifier may need their own `PostMerger` once their wire shape is confirmed.

---

## Scope & Phasing

| Phase | Delivers | Needs FE change? |
|-------|----------|------------------|
| 1 | Pure logic: split, merge, FE pool (no I/O) | No |
| 2 | Fanout over WebClient to FE `/batch_infer` | No |
| 3 | Transparent router on the shared 7001 listener, ordered after Master routes (split + passthrough) | No |
| 4 | BE pre-assign: master-aware coordinator + inject `generate_config.role_addrs` (optimization) | **No** — spike (Task 12) found the FE already honors `role_addrs`; FE work is test-only (Task 15) |
| 5 | End-to-end wiring, config, integration test | — |

Phases 1–3 + 5 deliver a **working transparent dispatcher against the unmodified FE** (FE already serves `/batch_infer`). Phase 4 is an optimization layer. **Spike result (Task 12, done):** the FE skips the Master whenever `generate_config.role_addrs` is pre-set, so pre-assign needs **no production FE change** — only the dispatcher populates `role_addrs` (Task 14) and a small Master-side `role` field (Task 13b). Pre-assign is gated behind `DISPATCH_CONFIG.preAssignBe` (default off → plain fanout).

## File Structure

All Java under `flexlb-api/src/main/java/org/flexlb/dispatcher/` (flexlb-api already depends on flexlb-sync, so in-process `RouteService` calls are available):

- `DispatchConfig.java` — parses `DISPATCH_CONFIG` env JSON: `dispatchPort`, `fePoolAddresses` (list), `subBatchSize` (K), `feRequestTimeoutMs`, `enabled`. One responsibility: config holder + validation.
- `FePool.java` — holds FE base URLs + atomic RR cursor; `next()` returns the next FE base URL.
- `BatchSplitter.java` — pure: split `List<String>` prompts into ordered chunks of size K.
- `SubBatchResult.java` — pure value type: one sub-batch outcome (FE response JSON, or a failure marker) carrying its chunk size so failures can be padded.
- `ResponseMerger.java` — pure: stitch sub-batch `response_batch` arrays back in order, padding any failed/short sub-batch with placeholders so the merged batch always has N entries; returns a `MergedResponse`. Modeled on ft_proxy's `mergeResults`.
- `MergedResponse.java` — pure value type: merged body + `succeededChunks`/`totalChunks`/`succeededPrompts` (drives the all-failed → 500 decision and metrics).
- `FanoutService.java` — WebClient: POST each sub-batch to a FE `/batch_infer` concurrently; a failed sub-batch becomes `SubBatchResult.failed` and never aborts its siblings; gather order-preserved and merge.
- `DispatchHandler.java` — WebFlux handler functions: `handleBatch` (split/fanout/merge), `handlePassthrough` (blind forward).
- `DispatchRouter.java` — builds the `RouterFunction` mirroring FE endpoints.
- (same-port default) dispatcher `RouterFunction` is registered as a Spring bean ordered last (Task 11) on the shared 7001 listener; the Master's route beans get `@Order(0)` (Task 10) so they match first. No separate `DispatchServer` unless the Task 10-alt separate-port variant is chosen.

Tests mirror under `flexlb-api/src/test/java/org/flexlb/dispatcher/`.

FE side (Phase 4, Python): `rtp_llm/frontend/frontend_worker.py` / `rtp_llm/pipeline/pipeline.py` — accept optional pre-assigned BE target (exact hook TBD by spike task 4.1).

## Build / Test commands

From `rtp_llm/flexlb`:

```bash
# Run one dispatcher test class (opensource profile; -P-internal avoids pulling kmonitor/vipserver)
./mvnw test -pl flexlb-api -am -Dtest=BatchSplitterTest -P-internal

# Full dispatcher package
./mvnw test -pl flexlb-api -am -Dtest='org.flexlb.dispatcher.*' -P-internal

# Format check before commit
./mvnw spotless:apply -Pspotless-check
```

> Note (from project memory): use `-P-internal` (single dash) to deactivate the internal profile; `-P'!internal'` gets eaten by nested shells and silently leaves it active.

---

## Phase 1 — Pure logic (no I/O)

### Task 1: BatchSplitter

**Files:**
- Create: `flexlb-api/src/main/java/org/flexlb/dispatcher/BatchSplitter.java`
- Test: `flexlb-api/src/test/java/org/flexlb/dispatcher/BatchSplitterTest.java`

> **Optional refinement (deferred):** `BatchSplitter` cuts naive contiguous K-sized chunks, which keeps the merge a plain in-order concat. ft_proxy (`splitPrompts`, `ft_proxy/rel_go/server.go:445`) instead balances total prompt *length* across chunks (sort desc, greedily fill the currently-shortest chunk) so one all-long chunk doesn't become the straggler that gates total latency. That makes chunks non-contiguous, so ft_proxy also tracks each prompt's original index and scatters results back by position in the merge. Adopt both together (length-balanced split + index-aware merge) only if a `SubLatencyGap`-style metric shows skew; v1 keeps contiguous chunks + concat.

- [ ] **Step 1: Write the failing test**

```java
package org.flexlb.dispatcher;

import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class BatchSplitterTest {

    @Test
    void splitsEvenlyPreservingOrder() {
        List<List<String>> chunks = BatchSplitter.split(List.of("a", "b", "c", "d"), 2);
        assertEquals(2, chunks.size());
        assertEquals(List.of("a", "b"), chunks.get(0));
        assertEquals(List.of("c", "d"), chunks.get(1));
    }

    @Test
    void lastChunkMayBeSmaller() {
        List<List<String>> chunks = BatchSplitter.split(List.of("a", "b", "c"), 2);
        assertEquals(2, chunks.size());
        assertEquals(List.of("c"), chunks.get(1));
    }

    @Test
    void singleChunkWhenBatchNotLargerThanK() {
        assertEquals(1, BatchSplitter.split(List.of("a", "b"), 5).size());
    }

    @Test
    void emptyInputYieldsNoChunks() {
        assertTrue(BatchSplitter.split(List.of(), 5).isEmpty());
    }

    @Test
    void rejectsNonPositiveK() {
        assertThrows(IllegalArgumentException.class, () -> BatchSplitter.split(List.of("a"), 0));
    }

    @Test
    void chunkCountIsTheSingleSourceForSplitSize() {
        // chunkCount(N, K) MUST equal split(...).size() — pre-assign sizes /batch_schedule by it
        assertEquals(BatchSplitter.split(List.of("a", "b", "c"), 2).size(), BatchSplitter.chunkCount(3, 2));
        assertEquals(0, BatchSplitter.chunkCount(0, 5));
        assertEquals(1, BatchSplitter.chunkCount(5, 5));
        assertEquals(100, BatchSplitter.chunkCount(500, 5));
        assertThrows(IllegalArgumentException.class, () -> BatchSplitter.chunkCount(3, 0));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./mvnw test -pl flexlb-api -am -Dtest=BatchSplitterTest -P-internal`
Expected: FAIL — `BatchSplitter` does not exist (compilation error).

- [ ] **Step 3: Write minimal implementation**

```java
package org.flexlb.dispatcher;

import java.util.ArrayList;
import java.util.List;

public final class BatchSplitter {

    private BatchSplitter() {
    }

    /**
     * Split prompts into ordered chunks of at most {@code subBatchSize}. Order is preserved;
     * the final chunk may be smaller.
     */
    public static List<List<String>> split(List<String> prompts, int subBatchSize) {
        if (subBatchSize < 1) {
            throw new IllegalArgumentException("subBatchSize must be >= 1, got " + subBatchSize);
        }
        List<List<String>> chunks = new ArrayList<>();
        for (int i = 0; i < prompts.size(); i += subBatchSize) {
            chunks.add(new ArrayList<>(prompts.subList(i, Math.min(i + subBatchSize, prompts.size()))));
        }
        return chunks;
    }

    /**
     * Number of chunks {@link #split} produces for {@code promptCount} prompts — the single source
     * of truth for the chunk count (e.g. sizing pre-assign's {@code /batch_schedule} call), so the
     * handler never recomputes {@code ceil(N/K)} independently of the split.
     */
    public static int chunkCount(int promptCount, int subBatchSize) {
        if (subBatchSize < 1) {
            throw new IllegalArgumentException("subBatchSize must be >= 1, got " + subBatchSize);
        }
        return (promptCount + subBatchSize - 1) / subBatchSize;
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./mvnw test -pl flexlb-api -am -Dtest=BatchSplitterTest -P-internal`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add flexlb-api/src/main/java/org/flexlb/dispatcher/BatchSplitter.java \
        flexlb-api/src/test/java/org/flexlb/dispatcher/BatchSplitterTest.java
git commit -m "feat(dispatcher): add BatchSplitter for K-sized prompt chunks"
```

### Task 2: ResponseMerger

**Files:**
- Create: `flexlb-api/src/main/java/org/flexlb/dispatcher/SubBatchResult.java`
- Create: `flexlb-api/src/main/java/org/flexlb/dispatcher/MergedResponse.java`
- Create: `flexlb-api/src/main/java/org/flexlb/dispatcher/ResponseMerger.java`
- Test: `flexlb-api/src/test/java/org/flexlb/dispatcher/ResponseMergerTest.java`

The FE batch response is `{"response_batch": [ {..}, {..} ]}` (verified: `frontend_worker.py` `BatchPipelineResponse`). Merging stitches each sub-batch's `response_batch` back together **in order** — but, modeled on ft_proxy's `mergeResults` (`ft_proxy/rel_go/server.go:487`), a sub-batch that **failed** or returned the **wrong number of entries** is padded with `{"response":"","finished":true}` placeholders so the merged batch always has exactly N entries in the original order. The merge takes `List<SubBatchResult>` (each carrying its chunk size) and returns a `MergedResponse` (the body plus how many chunks/prompts actually succeeded), so the handler can answer 200 on partial success and reserve 500 for the all-failed case. We operate on Jackson `JsonNode` so the dispatcher stays decoupled from the FE's pydantic schema — it only depends on the `response_batch` key.

> **Why not fail the whole batch on one bad chunk?** The naive `concat + throw-on-missing` design fails all N prompts when a single FE/BE is slow or dead — far too large a blast radius for a 500-prompt batch. ft_proxy (in production) instead soft-fails per chunk and pads, returning 500 only when *every* chunk failed. Tasks 2/5/7 adopt that model. See decision F5.

- [ ] **Step 1: Write the failing test**

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class ResponseMergerTest {

    private final ObjectMapper mapper = new ObjectMapper();

    private JsonNode batch(String... responses) throws Exception {
        StringBuilder sb = new StringBuilder("{\"response_batch\":[");
        for (int i = 0; i < responses.length; i++) {
            if (i > 0) sb.append(",");
            sb.append("{\"response\":\"").append(responses[i]).append("\",\"finished\":true}");
        }
        sb.append("]}");
        return mapper.readTree(sb.toString());
    }

    @Test
    void concatenatesSuccessfulSubBatchesInOrder() throws Exception {
        MergedResponse m = ResponseMerger.merge(
            List.of(SubBatchResult.ok(batch("a", "b"), 2), SubBatchResult.ok(batch("c"), 1)), mapper);
        JsonNode arr = m.body().get("response_batch");
        assertEquals(3, arr.size());
        assertEquals("a", arr.get(0).get("response").asText());
        assertEquals("c", arr.get(2).get("response").asText());
        assertEquals(2, m.succeededChunks());
        assertEquals(3, m.succeededPrompts());
    }

    @Test
    void failedSubBatchIsPaddedPreservingOrderAndSize() throws Exception {
        // chunk0 ok (2), chunk1 FAILED (size 1), chunk2 ok (1) -> 4 entries, slot 2 is a placeholder
        MergedResponse m = ResponseMerger.merge(
            List.of(SubBatchResult.ok(batch("a", "b"), 2),
                    SubBatchResult.failed(1),
                    SubBatchResult.ok(batch("d"), 1)), mapper);
        JsonNode arr = m.body().get("response_batch");
        assertEquals(4, arr.size());
        assertEquals("", arr.get(2).get("response").asText());
        assertTrue(arr.get(2).get("finished").asBoolean());
        assertEquals("d", arr.get(3).get("response").asText());
        assertEquals(2, m.succeededChunks());
        assertEquals(3, m.totalChunks());
        assertFalse(m.allFailed());
    }

    @Test
    void wrongLengthSubBatchIsTreatedAsFailedAndPadded() throws Exception {
        // claims chunk size 2 but FE returned only 1 entry -> padded to 2 placeholders, counted as failed
        MergedResponse m = ResponseMerger.merge(List.of(SubBatchResult.ok(batch("only"), 2)), mapper);
        JsonNode arr = m.body().get("response_batch");
        assertEquals(2, arr.size());
        assertEquals("", arr.get(0).get("response").asText());
        assertEquals(0, m.succeededChunks());
        assertTrue(m.allFailed());
    }

    @Test
    void emptyListYieldsEmptyBatch() {
        MergedResponse m = ResponseMerger.merge(List.of(), mapper);
        assertTrue(m.body().get("response_batch").isArray());
        assertEquals(0, m.body().get("response_batch").size());
        assertFalse(m.allFailed()); // nothing attempted is not a failure
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./mvnw test -pl flexlb-api -am -Dtest=ResponseMergerTest -P-internal`
Expected: FAIL — `ResponseMerger` / `SubBatchResult` / `MergedResponse` do not exist.

- [ ] **Step 3: Write minimal implementation**

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;

/**
 * One sub-batch outcome: the FE's response JSON ({@code ok}) or a failure marker ({@code failed}).
 * Always carries the chunk size so a failed sub-batch can be padded with the right number of
 * placeholders during merge.
 */
public final class SubBatchResult {

    private final JsonNode response;
    private final int chunkSize;

    private SubBatchResult(JsonNode response, int chunkSize) {
        this.response = response;
        this.chunkSize = chunkSize;
    }

    public static SubBatchResult ok(JsonNode response, int chunkSize) {
        return new SubBatchResult(response, chunkSize);
    }

    public static SubBatchResult failed(int chunkSize) {
        return new SubBatchResult(null, chunkSize);
    }

    public boolean isSuccess() {
        return response != null;
    }

    public JsonNode response() {
        return response;
    }

    public int chunkSize() {
        return chunkSize;
    }
}
```

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.node.ObjectNode;

/**
 * Result of merging sub-batches: the client-facing {@code body} plus how many sub-batches and
 * prompts actually succeeded. The handler returns 200 on any success and reserves 500 for the
 * all-failed case.
 */
public record MergedResponse(ObjectNode body, int succeededChunks, int totalChunks, int succeededPrompts) {

    public boolean allFailed() {
        return totalChunks > 0 && succeededChunks == 0;
    }
}
```

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.util.List;

public final class ResponseMerger {

    private ResponseMerger() {
    }

    /**
     * Stitch sub-batches back together in order. A failed sub-batch — or a successful one whose
     * {@code response_batch} length doesn't match its chunk size — is padded with placeholders so
     * the merged batch always has exactly sum(chunkSize) entries in the original order.
     */
    public static MergedResponse merge(List<SubBatchResult> subResults, ObjectMapper mapper) {
        ArrayNode merged = mapper.createArrayNode();
        int succeededChunks = 0;
        int succeededPrompts = 0;
        for (SubBatchResult sub : subResults) {
            JsonNode arr = sub.isSuccess() ? sub.response().get("response_batch") : null;
            if (arr != null && arr.isArray() && arr.size() == sub.chunkSize()) {
                merged.addAll((ArrayNode) arr);
                succeededChunks++;
                succeededPrompts += sub.chunkSize();
            } else {
                for (int i = 0; i < sub.chunkSize(); i++) {
                    merged.add(placeholder(mapper));
                }
            }
        }
        ObjectNode body = mapper.createObjectNode();
        body.set("response_batch", merged);
        return new MergedResponse(body, succeededChunks, subResults.size(), succeededPrompts);
    }

    private static ObjectNode placeholder(ObjectMapper mapper) {
        ObjectNode p = mapper.createObjectNode();
        p.put("response", "");
        p.put("finished", true);
        return p;
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./mvnw test -pl flexlb-api -am -Dtest=ResponseMergerTest -P-internal`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add flexlb-api/src/main/java/org/flexlb/dispatcher/SubBatchResult.java \
        flexlb-api/src/main/java/org/flexlb/dispatcher/MergedResponse.java \
        flexlb-api/src/main/java/org/flexlb/dispatcher/ResponseMerger.java \
        flexlb-api/src/test/java/org/flexlb/dispatcher/ResponseMergerTest.java
git commit -m "feat(dispatcher): merge sub-batches with placeholder padding for failed/short chunks"
```

### Task 3: FePool

**Files:**
- Create: `flexlb-api/src/main/java/org/flexlb/dispatcher/FePool.java`
- Test: `flexlb-api/src/test/java/org/flexlb/dispatcher/FePoolTest.java`

- [ ] **Step 1: Write the failing test**

```java
package org.flexlb.dispatcher;

import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class FePoolTest {

    @Test
    void roundRobinsAcrossAddresses() {
        FePool pool = new FePool(List.of("http://a:8088", "http://b:8088"));
        assertEquals("http://a:8088", pool.next());
        assertEquals("http://b:8088", pool.next());
        assertEquals("http://a:8088", pool.next());
    }

    @Test
    void reportsSize() {
        assertEquals(2, new FePool(List.of("http://a:8088", "http://b:8088")).size());
    }

    @Test
    void rejectsEmptyPool() {
        assertThrows(IllegalArgumentException.class, () -> new FePool(List.of()));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./mvnw test -pl flexlb-api -am -Dtest=FePoolTest -P-internal`
Expected: FAIL — `FePool` does not exist.

- [ ] **Step 3: Write minimal implementation**

```java
package org.flexlb.dispatcher;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class FePool {

    private final List<String> addresses;
    private final AtomicInteger cursor = new AtomicInteger(0);

    public FePool(List<String> addresses) {
        if (addresses == null || addresses.isEmpty()) {
            throw new IllegalArgumentException("FE pool must not be empty");
        }
        this.addresses = List.copyOf(addresses);
    }

    public String next() {
        return addresses.get(Math.floorMod(cursor.getAndIncrement(), addresses.size()));
    }

    public int size() {
        return addresses.size();
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./mvnw test -pl flexlb-api -am -Dtest=FePoolTest -P-internal`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add flexlb-api/src/main/java/org/flexlb/dispatcher/FePool.java \
        flexlb-api/src/test/java/org/flexlb/dispatcher/FePoolTest.java
git commit -m "feat(dispatcher): add FePool round-robin address provider"
```

---

## Phase 2 — Fanout over WebClient

### Task 4: DispatchConfig

**Files:**
- Create: `flexlb-api/src/main/java/org/flexlb/dispatcher/DispatchConfig.java`
- Test: `flexlb-api/src/test/java/org/flexlb/dispatcher/DispatchConfigTest.java`

Config is read from a dedicated `DISPATCH_CONFIG` env JSON (separate from `FLEXLB_CONFIG` to avoid touching shared `FlexlbConfig`). Shape:

```json
{ "enabled": true, "preAssignBe": false, "subBatchSize": 5,
  "feRequestTimeoutMs": 3000, "fePoolAddresses": ["http://10.0.0.1:8088", "http://10.0.0.2:8088"],
  "dispatchPort": 7005 }
```

> Field notes (the shape above is the **eventual full** config): Task 4 implements `enabled` / `subBatchSize` / `feRequestTimeoutMs` / `fePoolAddresses` / `dispatchPort`. `preAssignBe` (default `false`) is **added in Task 14** when Phase-4 pre-assignment lands — don't add it in Task 4. `dispatchPort` is **only** consumed by the optional separate-port variant (Task 10-alt); the same-port default (Task 10/11) ignores it — keep it parsed (harmless), don't wire it.

- [ ] **Step 1: Write the failing test**

```java
package org.flexlb.dispatcher;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class DispatchConfigTest {

    @Test
    void parsesFullJson() {
        DispatchConfig c = DispatchConfig.fromJson(
            "{\"enabled\":true,\"dispatchPort\":7005,\"subBatchSize\":5,"
          + "\"feRequestTimeoutMs\":3000,\"fePoolAddresses\":[\"http://a:8088\"]}");
        assertTrue(c.isEnabled());
        assertEquals(7005, c.getDispatchPort());
        assertEquals(5, c.getSubBatchSize());
        assertEquals(3000, c.getFeRequestTimeoutMs());
        assertEquals(1, c.getFePoolAddresses().size());
    }

    @Test
    void disabledWhenEnvNullOrBlank() {
        assertFalse(DispatchConfig.fromJson(null).isEnabled());
        assertFalse(DispatchConfig.fromJson("  ").isEnabled());
    }

    @Test
    void rejectsEnabledWithoutFePool() {
        assertThrows(IllegalArgumentException.class,
            () -> DispatchConfig.fromJson("{\"enabled\":true,\"dispatchPort\":7005}"));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./mvnw test -pl flexlb-api -am -Dtest=DispatchConfigTest -P-internal`
Expected: FAIL — `DispatchConfig` does not exist.

- [ ] **Step 3: Write minimal implementation**

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public class DispatchConfig {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @JsonProperty("enabled") private boolean enabled = false;
    @JsonProperty("dispatchPort") private int dispatchPort = 7005;
    @JsonProperty("subBatchSize") private int subBatchSize = 5;
    @JsonProperty("feRequestTimeoutMs") private int feRequestTimeoutMs = 3000;
    @JsonProperty("fePoolAddresses") private List<String> fePoolAddresses = List.of();

    public static DispatchConfig fromJson(String json) {
        if (json == null || json.isBlank()) {
            return new DispatchConfig();
        }
        try {
            DispatchConfig c = MAPPER.readValue(json, DispatchConfig.class);
            c.validate();
            return c;
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalArgumentException("invalid DISPATCH_CONFIG: " + e.getMessage(), e);
        }
    }

    private void validate() {
        if (enabled && fePoolAddresses.isEmpty()) {
            throw new IllegalArgumentException("DISPATCH_CONFIG.enabled=true requires fePoolAddresses");
        }
        if (subBatchSize < 1) {
            throw new IllegalArgumentException("subBatchSize must be >= 1");
        }
    }

    public boolean isEnabled() { return enabled; }
    public int getDispatchPort() { return dispatchPort; }
    public int getSubBatchSize() { return subBatchSize; }
    public int getFeRequestTimeoutMs() { return feRequestTimeoutMs; }
    public List<String> getFePoolAddresses() { return fePoolAddresses; }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./mvnw test -pl flexlb-api -am -Dtest=DispatchConfigTest -P-internal`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add flexlb-api/src/main/java/org/flexlb/dispatcher/DispatchConfig.java \
        flexlb-api/src/test/java/org/flexlb/dispatcher/DispatchConfigTest.java
git commit -m "feat(dispatcher): add DISPATCH_CONFIG parsing"
```

### Task 5: FanoutService

**Files:**
- Create: `flexlb-api/src/main/java/org/flexlb/dispatcher/FanoutService.java`
- Test: `flexlb-api/src/test/java/org/flexlb/dispatcher/FanoutServiceTest.java`

Uses a constructor-injected `WebClient` so the test can point at a stub server (OkHttp `MockWebServer` or a `WebClient` over a captured `ExchangeFunction`). We use Mockito to stub a small `FeClient` seam instead, keeping the test pure.

Design seam: `FanoutService` depends on a `FeClient` interface (`Mono<JsonNode> postBatch(String feBaseUrl, ObjectNode body)`), so fanout orchestration is tested without real HTTP. The WebClient-backed `FeClient` impl is Task 6.

- [ ] **Step 1: Write the failing test**

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;
import java.util.List;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class FanoutServiceTest {

    private final ObjectMapper mapper = new ObjectMapper();

    private JsonNode batchOf(String... responses) throws Exception {
        StringBuilder sb = new StringBuilder("{\"response_batch\":[");
        for (int i = 0; i < responses.length; i++) {
            if (i > 0) sb.append(",");
            sb.append("{\"response\":\"").append(responses[i]).append("\"}");
        }
        sb.append("]}");
        return mapper.readTree(sb.toString());
    }

    @Test
    void splitsFansOutAndMergesInOrder() throws Exception {
        FeClient feClient = mock(FeClient.class);
        FePool pool = new FePool(List.of("http://a", "http://b"));
        // chunk0 -> a -> ["r0","r1"], chunk1 -> b -> ["r2"]
        when(feClient.postBatch(eq("http://a"), any())).thenReturn(Mono.just(batchOf("r0", "r1")));
        when(feClient.postBatch(eq("http://b"), any())).thenReturn(Mono.just(batchOf("r2")));

        FanoutService svc = new FanoutService(feClient, pool, mapper, 2 /*K*/);

        StepVerifier.create(svc.dispatch(List.of("p0", "p1", "p2"), null))
            .assertNext(m -> {
                JsonNode arr = m.body().get("response_batch");
                assert arr.size() == 3;
                assert arr.get(0).get("response").asText().equals("r0");
                assert arr.get(2).get("response").asText().equals("r2");
                assert m.succeededChunks() == 2;
                assert !m.allFailed();
            })
            .verifyComplete();
    }

    @Test
    void failedChunkBecomesPlaceholdersNotAnError() throws Exception {
        FeClient feClient = mock(FeClient.class);
        FePool pool = new FePool(List.of("http://a", "http://b"));
        when(feClient.postBatch(eq("http://a"), any())).thenReturn(Mono.just(batchOf("r0", "r1")));
        when(feClient.postBatch(eq("http://b"), any())).thenReturn(Mono.error(new RuntimeException("FE down")));

        FanoutService svc = new FanoutService(feClient, pool, mapper, 2 /*K*/);

        StepVerifier.create(svc.dispatch(List.of("p0", "p1", "p2"), null))
            .assertNext(m -> {
                JsonNode arr = m.body().get("response_batch");
                assert arr.size() == 3;                            // still N, order preserved
                assert arr.get(0).get("response").asText().equals("r0");
                assert arr.get(2).get("response").asText().isEmpty(); // failed chunk -> placeholder
                assert m.succeededChunks() == 1;
                assert !m.allFailed();
            })
            .verifyComplete();
    }

    @Test
    void allChunksFailedReportedNotThrown() {
        FeClient feClient = mock(FeClient.class);
        FePool pool = new FePool(List.of("http://a"));
        when(feClient.postBatch(anyString(), any())).thenReturn(Mono.error(new RuntimeException("FE down")));

        FanoutService svc = new FanoutService(feClient, pool, mapper, 5 /*K*/);

        StepVerifier.create(svc.dispatch(List.of("p0", "p1"), null))
            .assertNext(m -> {
                assert m.allFailed();
                assert m.body().get("response_batch").size() == 2; // placeholders, no exception
            })
            .verifyComplete();
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./mvnw test -pl flexlb-api -am -Dtest=FanoutServiceTest -P-internal`
Expected: FAIL — `FanoutService` / `FeClient` do not exist.

- [ ] **Step 3: Write minimal implementation**

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import reactor.core.publisher.Mono;

public interface FeClient {
    Mono<JsonNode> postBatch(String feBaseUrl, ObjectNode body);
}
```

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;
import java.util.ArrayList;
import java.util.List;

public class FanoutService {

    private final FeClient feClient;
    private final FePool fePool;
    private final ObjectMapper mapper;
    private final int subBatchSize;

    public FanoutService(FeClient feClient, FePool fePool, ObjectMapper mapper, int subBatchSize) {
        this.feClient = feClient;
        this.fePool = fePool;
        this.mapper = mapper;
        this.subBatchSize = subBatchSize;
    }

    /**
     * Split prompts into K-sized chunks, POST each to one FE concurrently, and merge in order.
     * A chunk whose FE call errors becomes a {@link SubBatchResult#failed} — it never aborts its
     * siblings (ft_proxy semantics). The returned {@link MergedResponse} reports how many chunks
     * succeeded so the caller can 200 on partial success and 500 only when all chunks failed.
     */
    public Mono<MergedResponse> dispatch(List<String> prompts, JsonNode generateConfig) {
        List<List<String>> chunks = BatchSplitter.split(prompts, subBatchSize);
        List<Mono<SubBatchResult>> calls = new ArrayList<>(chunks.size());
        for (List<String> chunk : chunks) {
            ObjectNode body = mapper.createObjectNode();
            body.set("prompt_batch", mapper.valueToTree(chunk));
            if (generateConfig != null) {
                body.set("generate_config", generateConfig);
            }
            int chunkSize = chunk.size();
            calls.add(feClient.postBatch(fePool.next(), body)
                    .map(resp -> SubBatchResult.ok(resp, chunkSize))
                    .onErrorResume(e -> Mono.just(SubBatchResult.failed(chunkSize))));
        }
        // mergeSequential dispatches concurrently but collects results in chunk order.
        // publishOn(boundedElastic) moves the synchronous merge (JsonNode walk + ArrayNode addAll)
        // OFF the Reactor-Netty event loop, so a 5-MB N=500 merge can't add tail latency to the
        // co-located Master /schedule (1-5 ms SLA) sharing the same loop. See decision F4 / B4.
        return Flux.mergeSequential(Flux.fromIterable(calls))
                .collectList()
                .publishOn(Schedulers.boundedElastic())
                .map(subs -> ResponseMerger.merge(subs, mapper));
    }
}
```

> Note: `Flux.mergeSequential` dispatches all chunks concurrently but collects results in chunk order, so the merged batch stays ordered. The per-call `.onErrorResume(... -> SubBatchResult.failed(chunkSize))` is what gives ft_proxy's resilience: one dead/slow FE degrades only its K prompts to placeholders instead of failing the whole batch — never let a sub-call error escape into the merged stream. The `publishOn` before `merge` is required, not optional — same-JVM means the dispatcher's event loop is the Master's event loop, and `ResponseMerger.merge` on N=500 walks ~5 MB of JsonNode synchronously; without the schedule jump, a single big merge would gate every concurrent `/schedule` request scheduled on the same loop until it finishes.

- [ ] **Step 4: Run test to verify it passes**

Run: `./mvnw test -pl flexlb-api -am -Dtest=FanoutServiceTest -P-internal`
Expected: PASS (3 tests — happy path; one chunk failed → placeholders; all failed → reported, not thrown).

- [ ] **Step 5: Commit**

```bash
git add flexlb-api/src/main/java/org/flexlb/dispatcher/FeClient.java \
        flexlb-api/src/main/java/org/flexlb/dispatcher/FanoutService.java \
        flexlb-api/src/test/java/org/flexlb/dispatcher/FanoutServiceTest.java
git commit -m "feat(dispatcher): fan out with per-chunk soft-fail and ordered partial-success merge"
```

### Task 6: WebClient-backed FeClient

**Files:**
- Create: `flexlb-api/src/main/java/org/flexlb/dispatcher/WebClientFeClient.java`
- Test: `flexlb-api/src/test/java/org/flexlb/dispatcher/WebClientFeClientTest.java` (uses `okhttp3.mockwebserver.MockWebServer`)

- [ ] **Step 1: Add MockWebServer test dependency**

In `flexlb-api/pom.xml`, add under `<dependencies>` (scope test):

```xml
<dependency>
    <groupId>com.squareup.okhttp3</groupId>
    <artifactId>mockwebserver</artifactId>
    <version>4.12.0</version>
    <scope>test</scope>
</dependency>
```

- [ ] **Step 2: Write the failing test**

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import org.junit.jupiter.api.*;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.test.StepVerifier;

class WebClientFeClientTest {

    private MockWebServer server;
    private final ObjectMapper mapper = new ObjectMapper();

    @BeforeEach void start() throws Exception { server = new MockWebServer(); server.start(); }
    @AfterEach void stop() throws Exception { server.shutdown(); }

    @Test
    void postsToBatchInferAndParsesResponse() throws Exception {
        server.enqueue(new MockResponse()
            .setHeader("Content-Type", "application/json")
            .setBody("{\"response_batch\":[{\"response\":\"ok\"}]}"));
        WebClientFeClient client = new WebClientFeClient(WebClient.builder(), 3000);

        ObjectNode body = mapper.createObjectNode();
        body.putArray("prompt_batch").add("hi");

        String base = "http://" + server.getHostName() + ":" + server.getPort();
        StepVerifier.create(client.postBatch(base, body))
            .assertNext(n -> { assert n.get("response_batch").get(0).get("response").asText().equals("ok"); })
            .verifyComplete();

        RecordedRequest rec = server.takeRequest();
        Assertions.assertEquals("/batch_infer", rec.getPath());
        Assertions.assertEquals("POST", rec.getMethod());
    }

    @Test
    void rejectsResponseLargerThanMaxBytesCap() throws Exception {
        // Build a response_batch whose serialized body comfortably exceeds the 4 KiB cap.
        StringBuilder big = new StringBuilder("{\"response_batch\":[{\"response\":\"");
        for (int i = 0; i < 8 * 1024; i++) big.append('x');
        big.append("\"}]}");
        server.enqueue(new MockResponse().setHeader("Content-Type", "application/json").setBody(big.toString()));

        WebClientFeClient client = new WebClientFeClient(WebClient.builder(), 3000, /*maxResponseBytes*/ 4 * 1024);

        ObjectNode body = mapper.createObjectNode();
        body.putArray("prompt_batch").add("hi");
        String base = "http://" + server.getHostName() + ":" + server.getPort();

        StepVerifier.create(client.postBatch(base, body))
            .expectErrorMatches(e -> e instanceof org.springframework.core.io.buffer.DataBufferLimitException
                || (e.getCause() != null && e.getCause() instanceof org.springframework.core.io.buffer.DataBufferLimitException))
            .verify();
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `./mvnw test -pl flexlb-api -am -Dtest=WebClientFeClientTest -P-internal`
Expected: FAIL — `WebClientFeClient` does not exist.

- [ ] **Step 4: Write minimal implementation**

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.springframework.http.MediaType;
import org.springframework.http.codec.json.Jackson2JsonDecoder;
import org.springframework.web.reactive.function.client.ExchangeStrategies;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;
import java.time.Duration;

public class WebClientFeClient implements FeClient {

    /** Hard cap on a single FE response body. Protects same-JVM heap (Master's worker-status map
     *  shares the heap) from a runaway / misbehaving FE returning a multi-MB response_batch.
     *  16 MiB is generous for K=5 chunks of normal completions; tune via DispatchConfig if needed. */
    public static final int DEFAULT_MAX_RESPONSE_BYTES = 16 * 1024 * 1024;

    private final WebClient webClient;
    private final int timeoutMs;

    public WebClientFeClient(WebClient.Builder builder, int timeoutMs, int maxResponseBytes) {
        ExchangeStrategies strategies = ExchangeStrategies.builder()
                .codecs(c -> c.defaultCodecs().maxInMemorySize(maxResponseBytes))
                .build();
        this.webClient = builder.exchangeStrategies(strategies).build();
        this.timeoutMs = timeoutMs;
    }

    /** Convenience for tests/older call-sites; uses {@link #DEFAULT_MAX_RESPONSE_BYTES}. */
    public WebClientFeClient(WebClient.Builder builder, int timeoutMs) {
        this(builder, timeoutMs, DEFAULT_MAX_RESPONSE_BYTES);
    }

    @Override
    public Mono<JsonNode> postBatch(String feBaseUrl, ObjectNode body) {
        return webClient.post()
                .uri(feBaseUrl + "/batch_infer")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .retrieve()
                .bodyToMono(JsonNode.class)
                .timeout(Duration.ofMillis(timeoutMs));
    }
}
```

> The `maxResponseBytes` cap is the same-JVM heap guard: the dispatcher and the Master share heap (decision F4b), so one runaway FE return must not be allowed to grow `bodyToMono`'s in-memory buffer past a known ceiling. Without the cap, Spring WebFlux defaults to 256 KiB — too small for a normal merged FE response — but lifting it without an explicit ceiling would silently leave the heap exposed. Add a `WebClientFeClientTest` case sending a body bigger than the cap and asserting the `Mono` errors with `DataBufferLimitException` (don't ship without it).

- [ ] **Step 5: Run test to verify it passes**

Run: `./mvnw test -pl flexlb-api -am -Dtest=WebClientFeClientTest -P-internal`
Expected: PASS (2 tests — happy path; oversized response rejected with `DataBufferLimitException`).

- [ ] **Step 6: Commit**

```bash
git add flexlb-api/pom.xml \
        flexlb-api/src/main/java/org/flexlb/dispatcher/WebClientFeClient.java \
        flexlb-api/src/test/java/org/flexlb/dispatcher/WebClientFeClientTest.java
git commit -m "feat(dispatcher): add WebClient-backed FeClient with bounded response size"
```

---

## Phase 3 — Transparent router + second listener

### Task 7: DispatchHandler (batch + passthrough)

**Files:**
- Create: `flexlb-api/src/main/java/org/flexlb/dispatcher/DispatchHandler.java`
- Test: `flexlb-api/src/test/java/org/flexlb/dispatcher/DispatchHandlerTest.java`

`handleBatch` reads the request body, extracts `prompt_batch` + `generate_config`, calls `FanoutService.dispatch`, and returns the merged batch. Following ft_proxy (`ft_proxy/rel_go/server.go:210`), it answers **200 whenever at least one sub-batch succeeded** (failed chunks are already placeholder-padded by the merge) and reserves **500 for the all-failed case** (`MergedResponse.allFailed()`). `handlePassthrough` blind-forwards method/path/body to one FE via a `PassthroughClient` seam (`Mono<ServerResponse> forward(ServerRequest req)`), returning the FE's status + body verbatim.

For the unit test we exercise `handleBatch` with a stubbed `FanoutService` and an in-memory `MockServerRequest`.

- [ ] **Step 1: Write the failing test**

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpMethod;
import org.springframework.mock.web.reactive.function.server.MockServerRequest;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class DispatchHandlerTest {

    private final ObjectMapper mapper = new ObjectMapper();

    private MergedResponse mergedOf(int succeededChunks, int totalChunks, String... responses) {
        ObjectNode body = mapper.createObjectNode();
        ArrayNode arr = body.putArray("response_batch");
        for (String r : responses) {
            arr.addObject().put("response", r);
        }
        return new MergedResponse(body, succeededChunks, totalChunks, responses.length);
    }

    private MockServerRequest batchRequest() {
        ObjectNode reqBody = mapper.createObjectNode();
        reqBody.putArray("prompt_batch").add("p0").add("p1");
        return MockServerRequest.builder().method(HttpMethod.POST).body(Mono.just(reqBody));
    }

    @Test
    void handleBatchExtractsPromptsAndReturns200OnPartialSuccess() {
        FanoutService fanout = mock(FanoutService.class);
        when(fanout.dispatch(anyList(), any())).thenReturn(Mono.just(mergedOf(1, 2, "done", "")));

        DispatchHandler handler = new DispatchHandler(fanout, mock(PassthroughClient.class), mapper);

        StepVerifier.create(handler.handleBatch(batchRequest()))
            .assertNext(r -> { assert r.statusCode().value() == 200; })
            .verifyComplete();

        verify(fanout).dispatch(argThat(l -> l.size() == 2 && l.get(0).equals("p0")), any());
    }

    @Test
    void handleBatchReturns500WhenAllSubBatchesFailed() {
        FanoutService fanout = mock(FanoutService.class);
        when(fanout.dispatch(anyList(), any())).thenReturn(Mono.just(mergedOf(0, 2, "", "")));

        DispatchHandler handler = new DispatchHandler(fanout, mock(PassthroughClient.class), mapper);

        StepVerifier.create(handler.handleBatch(batchRequest()))
            .assertNext(r -> { assert r.statusCode().value() == 500; })
            .verifyComplete();
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./mvnw test -pl flexlb-api -am -Dtest=DispatchHandlerTest -P-internal`
Expected: FAIL — `DispatchHandler` / `PassthroughClient` do not exist.

- [ ] **Step 3: Write minimal implementation**

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import reactor.core.publisher.Mono;
import org.springframework.web.reactive.function.server.ServerRequest;

public interface PassthroughClient {
    /** Forward the request verbatim to one FE; complete the response onto the caller. */
    Mono<org.springframework.web.reactive.function.server.ServerResponse> forward(ServerRequest request);
}
```

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;
import java.util.ArrayList;
import java.util.List;

public class DispatchHandler {

    private final FanoutService fanoutService;
    private final PassthroughClient passthroughClient;
    private final ObjectMapper mapper;

    public DispatchHandler(FanoutService fanoutService, PassthroughClient passthroughClient, ObjectMapper mapper) {
        this.fanoutService = fanoutService;
        this.passthroughClient = passthroughClient;
        this.mapper = mapper;
    }

    public Mono<ServerResponse> handleBatch(ServerRequest request) {
        return request.bodyToMono(JsonNode.class).flatMap(body -> {
            List<String> prompts = new ArrayList<>();
            JsonNode arr = body.get("prompt_batch");
            if (arr != null && arr.isArray()) {
                arr.forEach(n -> prompts.add(n.asText()));
            }
            JsonNode generateConfig = body.get("generate_config");
            return fanoutService.dispatch(prompts, generateConfig).flatMap(merged -> {
                if (merged.allFailed()) {
                    ObjectNode err = mapper.createObjectNode();
                    err.put("error", "all_sub_batches_failed");
                    return ServerResponse.status(500).contentType(MediaType.APPLICATION_JSON).bodyValue(err);
                }
                return ServerResponse.ok().contentType(MediaType.APPLICATION_JSON).bodyValue(merged.body());
            });
        }).onErrorResume(e -> {
            // Fanout itself never errors (each chunk soft-fails); this guards genuine pre-flight
            // failures only — e.g. an unparseable request body.
            ObjectNode err = mapper.createObjectNode();
            err.put("error", "dispatch_failed");
            err.put("message", String.valueOf(e.getMessage()));
            return ServerResponse.status(500).contentType(MediaType.APPLICATION_JSON).bodyValue(err);
        });
    }

    public Mono<ServerResponse> handlePassthrough(ServerRequest request) {
        return passthroughClient.forward(request);
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./mvnw test -pl flexlb-api -am -Dtest=DispatchHandlerTest -P-internal`
Expected: PASS (2 tests — 200 on partial success, 500 on all-failed).

- [ ] **Step 5: Commit**

```bash
git add flexlb-api/src/main/java/org/flexlb/dispatcher/PassthroughClient.java \
        flexlb-api/src/main/java/org/flexlb/dispatcher/DispatchHandler.java \
        flexlb-api/src/test/java/org/flexlb/dispatcher/DispatchHandlerTest.java
git commit -m "feat(dispatcher): add DispatchHandler (batch split + passthrough delegation)"
```

### Task 8: WebClient-backed PassthroughClient

**Files:**
- Create: `flexlb-api/src/main/java/org/flexlb/dispatcher/WebClientPassthroughClient.java`
- Test: `flexlb-api/src/test/java/org/flexlb/dispatcher/WebClientPassthroughClientTest.java` (MockWebServer)

- [ ] **Step 1: Write the failing test**

```java
package org.flexlb.dispatcher;

import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import org.junit.jupiter.api.*;
import org.springframework.http.HttpMethod;
import org.springframework.mock.web.reactive.function.server.MockServerRequest;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;
import java.net.URI;
import java.util.List;

class WebClientPassthroughClientTest {

    private MockWebServer server;

    @BeforeEach void start() throws Exception { server = new MockWebServer(); server.start(); }
    @AfterEach void stop() throws Exception { server.shutdown(); }

    @Test
    void forwardsPathAndReturnsBodyVerbatim() throws Exception {
        server.enqueue(new MockResponse().setBody("{\"status\":\"ok\"}")
            .setHeader("Content-Type", "application/json"));
        String base = "http://" + server.getHostName() + ":" + server.getPort();
        FePool pool = new FePool(List.of(base));
        WebClientPassthroughClient client =
            new WebClientPassthroughClient(WebClient.builder().build(), pool, 3000);

        MockServerRequest request = MockServerRequest.builder()
            .method(HttpMethod.GET)
            .uri(URI.create("/worker_status"))
            .build();

        Mono<ServerResponse> resp = client.forward(request);
        StepVerifier.create(resp).assertNext(r -> { assert r.statusCode().value() == 200; }).verifyComplete();

        RecordedRequest rec = server.takeRequest();
        Assertions.assertEquals("/worker_status", rec.getPath());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./mvnw test -pl flexlb-api -am -Dtest=WebClientPassthroughClientTest -P-internal`
Expected: FAIL — `WebClientPassthroughClient` does not exist.

- [ ] **Step 3: Write minimal implementation**

```java
package org.flexlb.dispatcher;

import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import java.time.Duration;

public class WebClientPassthroughClient implements PassthroughClient {

    private final WebClient webClient;
    private final FePool fePool;
    private final int timeoutMs;

    public WebClientPassthroughClient(WebClient webClient, FePool fePool, int timeoutMs) {
        this.webClient = webClient;
        this.fePool = fePool;
        this.timeoutMs = timeoutMs;
    }

    @Override
    public Mono<ServerResponse> forward(ServerRequest request) {
        String target = fePool.next() + request.path();
        Flux<DataBuffer> bodyStream = request.bodyToFlux(DataBuffer.class);
        return webClient.method(request.method())
                .uri(target)
                .headers(h -> h.addAll(request.headers().asHttpHeaders()))
                .body(BodyInserters.fromDataBuffers(bodyStream))
                .exchangeToMono(clientResponse ->
                    ServerResponse.status(clientResponse.statusCode())
                        .headers(h -> h.addAll(clientResponse.headers().asHttpHeaders()))
                        .body(clientResponse.bodyToFlux(DataBuffer.class), DataBuffer.class))
                .timeout(Duration.ofMillis(timeoutMs));
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./mvnw test -pl flexlb-api -am -Dtest=WebClientPassthroughClientTest -P-internal`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add flexlb-api/src/main/java/org/flexlb/dispatcher/WebClientPassthroughClient.java \
        flexlb-api/src/test/java/org/flexlb/dispatcher/WebClientPassthroughClientTest.java
git commit -m "feat(dispatcher): add WebClient passthrough forwarding to FE pool"
```

### Task 9: DispatchRouter

**Files:**
- Create: `flexlb-api/src/main/java/org/flexlb/dispatcher/DispatchRouter.java`
- Test: `flexlb-api/src/test/java/org/flexlb/dispatcher/DispatchRouterTest.java` (WebTestClient over the RouterFunction)

Route table (mirrors the FE surface): `POST /batch_infer` → `handleBatch`; everything else → `handlePassthrough`. **Same-port (default):** this RouterFunction is registered as a Spring bean ordered AFTER the Master's routes (Task 11), so the catch-all is matched last — it cannot shadow `/rtp_llm/*` (Spring WebFlux is first-match-wins by `@Order`). The dispatcher shares the Master's 7001 listener; see Task 11 for the ordering, and Task 10 for the optional separate-port variant.

- [ ] **Step 1: Write the failing test**

```java
package org.flexlb.dispatcher;

import org.junit.jupiter.api.Test;
import org.springframework.test.web.reactive.server.WebTestClient;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class DispatchRouterTest {

    @Test
    void batchInferRoutesToHandleBatch() {
        DispatchHandler handler = mock(DispatchHandler.class);
        when(handler.handleBatch(any())).thenReturn(ServerResponse.ok().bodyValue("BATCH"));
        when(handler.handlePassthrough(any())).thenReturn(ServerResponse.ok().bodyValue("PASS"));

        RouterFunction<ServerResponse> rf = new DispatchRouter(handler).routes();
        WebTestClient client = WebTestClient.bindToRouterFunction(rf).build();

        client.post().uri("/batch_infer").bodyValue("{}")
            .exchange().expectStatus().isOk().expectBody(String.class).isEqualTo("BATCH");
        client.get().uri("/worker_status")
            .exchange().expectStatus().isOk().expectBody(String.class).isEqualTo("PASS");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./mvnw test -pl flexlb-api -am -Dtest=DispatchRouterTest -P-internal`
Expected: FAIL — `DispatchRouter` does not exist.

- [ ] **Step 3: Write minimal implementation**

```java
package org.flexlb.dispatcher;

import org.springframework.web.reactive.function.server.RequestPredicates;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;
import static org.springframework.web.reactive.function.server.RouterFunctions.route;

public class DispatchRouter {

    private final DispatchHandler handler;

    public DispatchRouter(DispatchHandler handler) {
        this.handler = handler;
    }

    public RouterFunction<ServerResponse> routes() {
        return route()
            .POST("/batch_infer", handler::handleBatch)
            // Everything else: blind passthrough to the FE pool.
            .route(RequestPredicates.all(), handler::handlePassthrough)
            .build();
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./mvnw test -pl flexlb-api -am -Dtest=DispatchRouterTest -P-internal`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add flexlb-api/src/main/java/org/flexlb/dispatcher/DispatchRouter.java \
        flexlb-api/src/test/java/org/flexlb/dispatcher/DispatchRouterTest.java
git commit -m "feat(dispatcher): add DispatchRouter (batch_infer split, catch-all passthrough)"
```

### Task 10: Same-port route precedence (default) — Master routes win over the catch-all

> **Decision (2026-05-20):** the dispatcher shares the Master's existing 7001 listener instead of opening a second port. Rationale: same-JVM was already chosen, so a second Netty listener would only isolate the event loop (not CPU/heap/GC) — a half-measure inconsistent with same-JVM, plus it costs a manual listener + a HIPPO/AONE port to provision. Same-port is correct as long as the dispatcher's catch-all is matched **after** the Master's routes. Spring WebFlux combines all `RouterFunction<ServerResponse>` beans and tries them in `@Order` order (first match wins). (Separate-port variant kept as Task 10-alt below for when data-plane fanout needs event-loop isolation from the Master SLA.)

> **Existing `RouterFunction<ServerResponse>` bean inventory (verified 2026-05-22, `grep RouterFunction flexlb-api/src/main/`):** exactly three, none currently annotated `@Order`:
> - `HttpLoadBalanceServer.loadBalancePrefill()` — 7 routes under `/rtp_llm/*` (`/schedule`, `/batch_schedule`, `/master/info`, `/schedule_snapshot`, `/notify_master`, `/update_log_level`, `/queue_snapshot`)
> - `HealthCheckServer.healthCheck()` — `/health` (note: the actuator `/health` lives on management port 7002 per `application.yml`; this 7001 `/health` is the dispatcher catch-all collision risk)
> - `AppStateHookServer.appStateHook()` — `/hook/*`
>
> All three are currently safe by **path mutual exclusion** (`/rtp_llm/*` vs `/health` vs `/hook/*`) — no `@Order` is needed for them to coexist today. Adding `@Order(0)` to all three is a no-op for their pairwise behavior; it only matters for the new dispatcher catch-all (`@Order(LOWEST_PRECEDENCE)`, Task 11) which would otherwise shadow `/health` and `/hook/*` on the same 7001 listener. If a fourth `RouterFunction` bean appears in the future, it inherits Spring's default precedence (LOWEST) — re-verify before merging.

**Files:**
- Modify: `flexlb-api/src/main/java/org/flexlb/httpserver/HttpLoadBalanceServer.java` (add `@Order(0)` to `loadBalancePrefill()` bean so Master routes take precedence)
- Modify: `flexlb-api/src/main/java/org/flexlb/httpserver/HealthCheckServer.java` + `AppStateHookServer.java` (same `@Order(0)`, so `/health` and `/hook/*` win over the catch-all)
- Test: `flexlb-api/src/test/java/org/flexlb/dispatcher/SamePortPrecedenceTest.java`

- [ ] **Step 1: Write the precedence contract test (Master route wins; unknown path falls to dispatcher)**

> This test exercises the actual mechanism the same-port design depends on: Spring combines all `RouterFunction` beans via `getBeanProvider(RouterFunction.class).orderedStream().reduce(RouterFunction::andOther)` (the exact call `RouterFunctionMapping` makes), honoring `@Order` on the `@Bean` methods. The dispatcher config is registered **first** on purpose — if `@Order` were ignored and registration order won, the catch-all would shadow `/rtp_llm/*` and this test would fail. (A hand-written `master.andOther(dispatcher)` would only prove `andOther` semantics, not that `@Order` controls the real combination — which is the thing we're unsure about.)

```java
package org.flexlb.dispatcher;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.test.web.reactive.server.WebTestClient;
import org.springframework.web.reactive.function.server.RequestPredicates;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;

import static org.springframework.web.reactive.function.server.RouterFunctions.route;

class SamePortPrecedenceTest {

    @Configuration
    static class DispatcherCatchAllConfig {
        @Bean
        @Order(Ordered.LOWEST_PRECEDENCE)
        RouterFunction<ServerResponse> dispatcherRoutes() {
            return route()
                .POST("/batch_infer", req -> ServerResponse.ok().bodyValue("BATCH"))
                .route(RequestPredicates.all(), req -> ServerResponse.ok().bodyValue("PASS"))
                .build();
        }
    }

    @Configuration
    static class MasterRoutesConfig {
        @Bean
        @Order(0)
        RouterFunction<ServerResponse> masterRoutes() {
            return route().POST("/rtp_llm/schedule", req -> ServerResponse.ok().bodyValue("MASTER")).build();
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    void orderedBeansMakeMasterWinAndCatchAllForwardsRest() {
        new ApplicationContextRunner()
            // dispatcher registered BEFORE master -> precedence must come from @Order, not bean order
            .withUserConfiguration(DispatcherCatchAllConfig.class, MasterRoutesConfig.class)
            .run(context -> {
                RouterFunction<ServerResponse> combined = context
                    .getBeanProvider(RouterFunction.class)
                    .orderedStream()
                    .map(rf -> (RouterFunction<ServerResponse>) rf)
                    .reduce(RouterFunction::andOther)
                    .orElseThrow();

                WebTestClient client = WebTestClient.bindToRouterFunction(combined).build();
                client.post().uri("/rtp_llm/schedule").bodyValue("{}")
                    .exchange().expectBody(String.class).isEqualTo("MASTER");
                client.post().uri("/batch_infer").bodyValue("{}")
                    .exchange().expectBody(String.class).isEqualTo("BATCH");
                client.get().uri("/worker_status")
                    .exchange().expectBody(String.class).isEqualTo("PASS");
            });
    }
}
```

- [ ] **Step 2: Run the contract test**

Run: `./mvnw test -pl flexlb-api -am -Dtest=SamePortPrecedenceTest -P-internal`
Expected: PASS. This characterizes Spring's mechanism (`orderedStream().reduce(andOther)` honors `@Order` on `RouterFunction` beans), so green-on-first-run is correct. If it ever FAILS, the same-port approach is invalid — stop and reconsider the Task 10-alt separate port.

- [ ] **Step 3: Set bean precedence on the existing route beans**

Add `@Order(0)` (highest precedence) to `HttpLoadBalanceServer.loadBalancePrefill()`, `HealthCheckServer.healthCheck()`, `AppStateHookServer.appStateHook()`. The dispatcher's RouterFunction bean (Task 11) gets `@Order(Ordered.LOWEST_PRECEDENCE)`. With Spring's `AnnotationAwareOrderComparator`, the combined `RouterFunction` tries Master/health/hook routes first and the dispatcher catch-all last.

- [ ] **Step 4: Regression — confirm the real Master beans still win after adding `@Order`**

Run the existing flexlb-api tests plus the contract test: `./mvnw test -pl flexlb-api -am -P-internal`
Expected: PASS — existing `HttpLoadBalanceServer` / `/health` / `/hook/*` route tests are unchanged (now `@Order(0)`); the dispatcher catch-all (Task 11) only handles unmatched paths. (Full-context coverage that the real beans carry the precedence is the Task 11 Step 5 regression.)

- [ ] **Step 5: Commit**

```bash
git add flexlb-api/src/main/java/org/flexlb/httpserver/HttpLoadBalanceServer.java \
        flexlb-api/src/main/java/org/flexlb/httpserver/HealthCheckServer.java \
        flexlb-api/src/main/java/org/flexlb/httpserver/AppStateHookServer.java \
        flexlb-api/src/test/java/org/flexlb/dispatcher/SamePortPrecedenceTest.java
git commit -m "feat(dispatcher): order Master routes ahead of dispatcher catch-all (same-port)"
```

### Task 10-alt (OPTIONAL, unlikely-needed): separate data-plane port

> **A separate port is a weak lever — prefer the cheaper fixes first.** Clarifying the actual contention:
> - The local dispatcher calls `batchSchedule` **in-process** (a method call) — it does NOT go through the 7001 listener, so it never competes there on the master node.
> - Reactive fanout does NOT block the event loop while awaiting FE responses (async I/O frees the loop). The only event-loop cost is **synchronous CPU**: parsing the big inbound batch JSON and merging/serializing the big response. Under high load this CPU can add tail latency to co-located control-plane requests that DO hit 7001 (remote slave-forwarded `batchSchedule`; in core mode, FE `/schedule`).
> - The **dominant** same-JVM cross-talk is **GC** (shared heap) — and a separate port does NOT fix that; only a separate process or ZGC does (ZGC already chosen).
>
> So: **first-line fix for CPU spikes = offload merge/serialize off the event loop** (`publishOn(Schedulers.boundedElastic())` around `ResponseMerger`), no port change. GC = ZGC. A separate port only helps the narrow event-loop-CPU case and only after offloading proves insufficient — provision a second Reactor-Netty `HttpServer` on `DISPATCH_CONFIG.dispatchPort` (`RouterFunctions.toHttpHandler` + `ReactorHttpHandlerAdapter`, `bindNow()`), keep the catch-all OFF the 7001 table. Costs: ~30 lines + a HIPPO/AONE port + VIP/healthcheck. Decide with the user; expected to stay unused.

### Task 11: Spring wiring (DispatcherConfiguration)

**Files:**
- Create: `flexlb-api/src/main/java/org/flexlb/dispatcher/DispatcherConfiguration.java`
- Test: `flexlb-api/src/test/java/org/flexlb/dispatcher/DispatcherConfigurationTest.java`

`@Configuration` that registers the dispatcher's `RouterFunction` as a Spring bean **ordered last** (`@Order(LOWEST_PRECEDENCE)`), so on the shared 7001 listener Spring tries Master routes first and the dispatcher catch-all last (Task 10). The bean is built **only when `DISPATCH_CONFIG.enabled`**, so disabled deployments register nothing. No second port (same-port default).

- [ ] **Step 1: Write the failing test**

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;
import static org.junit.jupiter.api.Assertions.*;

class DispatcherConfigurationTest {

    @Test
    void buildsRouterWhenEnabled() {
        DispatchConfig cfg = DispatchConfig.fromJson(
            "{\"enabled\":true,\"subBatchSize\":5,"
          + "\"feRequestTimeoutMs\":3000,\"fePoolAddresses\":[\"http://a:8088\"]}");
        DispatcherConfiguration conf = new DispatcherConfiguration();
        RouterFunction<ServerResponse> routes =
            conf.dispatcherRoutes(cfg, new ObjectMapper(), WebClient.builder());
        assertNotNull(routes);
    }

    @Test
    void noRouterWhenDisabled() {
        DispatchConfig cfg = DispatchConfig.fromJson(null); // disabled
        DispatcherConfiguration conf = new DispatcherConfiguration();
        assertNull(conf.dispatcherRoutes(cfg, new ObjectMapper(), WebClient.builder()));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./mvnw test -pl flexlb-api -am -Dtest=DispatcherConfigurationTest -P-internal`
Expected: FAIL — `DispatcherConfiguration` does not exist.

- [ ] **Step 3: Write minimal implementation**

```java
package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.netty.http.client.HttpClient;
import reactor.netty.resources.ConnectionProvider;

import java.time.Duration;

@Configuration
public class DispatcherConfiguration {

    @Bean
    public DispatchConfig dispatchConfig() {
        return DispatchConfig.fromJson(System.getenv("DISPATCH_CONFIG"));
    }

    /**
     * Dispatcher routes on the SHARED 7001 listener, ordered last so the catch-all never shadows
     * the Master's /rtp_llm/* (which are @Order(0) — Task 10). Returns null when disabled.
     *
     * <p>Same-JVM resource isolation (decision F4 / F4b):
     * <ul>
     *   <li>Reuses the auto-configured Spring {@link ObjectMapper} bean — no second copy with
     *       drifting Jackson config.</li>
     *   <li>Uses a dedicated, named {@link ConnectionProvider} ("dispatcher-fe") so dispatcher
     *       fanout cannot starve {@code GeneralHttpNettyService}'s connections to the master
     *       (which the Master's slave→master forward uses).</li>
     * </ul>
     */
    @Bean
    @Order(Ordered.LOWEST_PRECEDENCE)
    public RouterFunction<ServerResponse> dispatcherRoutes(DispatchConfig cfg,
                                                           ObjectMapper mapper,
                                                           WebClient.Builder webClientBuilder) {
        if (!cfg.isEnabled()) {
            return null;
        }
        ConnectionProvider feConnections = ConnectionProvider.builder("dispatcher-fe")
                .maxConnections(cfg.getFeMaxConnections())
                .pendingAcquireTimeout(Duration.ofMillis(cfg.getFeRequestTimeoutMs()))
                .pendingAcquireMaxCount(cfg.getFeMaxPendingAcquire())
                .build();
        WebClient.Builder feBuilder = webClientBuilder.clone()
                .clientConnector(new ReactorClientHttpConnector(HttpClient.create(feConnections)));
        FePool pool = new FePool(cfg.getFePoolAddresses());
        FeClient feClient = new WebClientFeClient(feBuilder, cfg.getFeRequestTimeoutMs(), cfg.getFeMaxResponseBytes());
        FanoutService fanout = new FanoutService(feClient, pool, mapper, cfg.getSubBatchSize());
        PassthroughClient passthrough = new WebClientPassthroughClient(feBuilder.build(), pool, cfg.getFeRequestTimeoutMs());
        DispatchHandler handler = new DispatchHandler(fanout, passthrough, mapper);
        return new DispatchRouter(handler).routes();
    }
}
```

> `DispatchConfig` adds three same-JVM-isolation knobs (sensible defaults — none required in env JSON):
> - `feMaxConnections` (default 200): hard cap on concurrent FE TCP connections held by the dispatcher.
> - `feMaxPendingAcquire` (default 1000): cap on connection-acquire queue (above this, fanout calls fail fast instead of piling up unbounded).
> - `feMaxResponseBytes` (default 16 MiB): per-response in-memory cap (heap guard, see Task 6).
>
> `DispatchConfig` no longer needs `dispatchPort` for the same-port default (drop it from Task 4's class, or keep it unused for the Task 10-alt separate-port variant). Update `DispatchConfigTest` to cover the three new fields' defaults and overrides.

- [ ] **Step 4: Run test to verify it passes**

Run: `./mvnw test -pl flexlb-api -am -Dtest=DispatcherConfigurationTest -P-internal`
Expected: PASS.

- [ ] **Step 5: Verify Master endpoints win (regression guard)**

Run the existing flexlb-api tests + the same-port precedence test (Task 10):
`./mvnw test -pl flexlb-api -am -P-internal`
Expected: PASS — `HttpLoadBalanceServer` routes (`/rtp_llm/*`), `/health`, `/hook/*` still match first; the dispatcher catch-all only handles unmatched paths; disabled deployments register no dispatcher bean.

- [ ] **Step 6: Commit**

```bash
git add flexlb-api/src/main/java/org/flexlb/dispatcher/DispatcherConfiguration.java \
        flexlb-api/src/test/java/org/flexlb/dispatcher/DispatcherConfigurationTest.java
git commit -m "feat(dispatcher): wire dispatcher beans behind DISPATCH_CONFIG.enabled"
```

---

## Phase 4 — BE pre-assign (optimization, requires FE change)

> This phase saves the per-request FE→Master round-trip by calling batch scheduling **once per batch** and tagging each sub-batch with its BE targets. **Master-aware:** the elected master runs the RR pick in-process (same JVM); a slave forwards to the real master — the dispatcher must NOT bypass this (Task 13 extracts the shared coordinator). Decoupled from Phases 1–3 — the dispatcher already works end-to-end against the unmodified FE without it.

### Task 12 (SPIKE — DONE 2026-05-20): FE→BE hook is `generate_config.role_addrs`

> Spike completed by reading the code. Findings below are the contract; no separate investigation needed. (Optionally mirror this into `docs/superpowers/specs/2026-05-20-fe-be-preassign-contract.md`.)

**How the FE picks its BE today** — `rtp_llm/server/backend_rpc_server_visitor.py::route_ips(input)`, called per-input from both `enqueue` and `batch_enqueue`:
1. `role_addrs_specified = bool(input.generate_config.role_addrs)` (`:217`).
2. If NOT specified, has a master addr, and token not 2D-batched → `get_master_route_addrs` → `MasterClient.get_backend_role_addrs` → on success writes `input.generate_config.role_addrs = route_result.role_addrs` (`:156`).
3. If roles still missing and fallback allowed → `get_domain_route_addrs` (host_service / domain).
4. If still empty → raises `ROUTE_ERROR`.

**The pre-assign hook ALREADY EXISTS — no FE happy-path change.** If `generate_config.role_addrs` is pre-populated, step 2 is skipped (`:226`). The Master's whole job is just to fill `role_addrs`; pre-assign fills it in advance.

**Wire format (already supported by pydantic):**
- `RoleAddr` (`rtp_llm/config/generate_config.py:19`) = `{role: RoleType, ip: str, http_port: int, grpc_port: int}`; `role` accepts a string like `"DECODE"`.
- `GenerateConfig.role_addrs: List[RoleAddr] = []` (`:126`); `create_generate_config(**dict)` parses it. So a request body `generate_config: {"role_addrs": [{"role":"DECODE","ip":"10.0.0.1","http_port":8088,"grpc_port":8089}]}` flows straight through.

**Two constraints that shape the dispatcher side:**
- **One BE per chunk, not per prompt.** `pipeline.batch_infer` builds ONE `generate_config` for the whole `prompt_batch` and shallow-copies it per input — the `role_addrs` list is shared, so all prompts in one `/batch_infer` call go to the SAME BE. ⇒ assign BE **per chunk**: `batchCount = number of chunks` (not prompt count). Per-prompt-different-BE would require an API extension; out of scope.
- **`RoleAddr` needs `role`, but `BatchScheduleTarget` has none.** `BatchScheduleTarget = {server_ip, http_port, grpc_port}` — no role. Since `/batch_schedule` is single-role, one role covers the batch ⇒ Task 13b adds a `role` to `BatchScheduleResponse`.

**Open item (NOT a blocker):** a dead pre-assigned target does NOT auto-fail-over (with `role_addrs` specified and complete, `need_domain_routing=False`, so no fallback) — the dead addr goes to gRPC and the request fails. v1 accepts "dead target → request fails → client retries" (targets are alive as-of selection, small window). A "dead → re-ask master" fallback is a later enhancement (would touch `route_ips`).

- [x] **Spike done** — findings above. Proceed to Tasks 13–15; the FE change (Task 15) is now just a verification test + optional fallback, not new routing code.

### Task 13: ~~Extract master-aware `BatchScheduleCoordinator` (refactor)~~ **ALREADY DONE — skip**

> **Status (2026-05-22):** the coordinator is already in the repo at `flexlb-sync/src/main/java/org/flexlb/service/BatchScheduleCoordinator.java`, and `HttpLoadBalanceServer` already injects and delegates to it (constructor field `batchScheduleCoordinator`). The extract step in this task is a no-op; **all 6 sub-steps below are obsolete history**. Task 14 picks up against the actual signature, which differs from the one this task originally proposed:
>
> | Aspect | This task's draft | Real implementation |
> |---|---|---|
> | Return | `Mono<BatchScheduleResponse>` | `Mono<Outcome>` where `Outcome = {BatchScheduleResponse response, Source source}` and `Source = LOCAL | FORWARDED` |
> | Transport failure | onErrorResume → return `BatchScheduleResponse.error(...)` (swallowed into success channel) | `Mono.error(BatchScheduleTransportException)` (raised into the error channel; caller handles) |
> | Master null | error response | `BatchScheduleTransportException(... "MASTER_NULL")` |
> | Spring stereotype | `@Service` | `@Component` (equivalent) |
>
> Task 14 below has been rewritten against the real signature. **Do NOT redefine the coordinator** — touching the signature would break the production `/rtp_llm/batch_schedule` path that already uses it.

- [x] **Step 1–6**: not applicable — coordinator already exists and is in production use.

<details>
<summary>Original task body (kept for history; do not execute)</summary>

> **Why:** the dispatcher must NOT call `routeService.batchSchedule()` directly — that bypasses the master-election / forward logic. Only the elected master may run the RR pick (single consistent cursor; a slave running locally would diverge and collide across nodes). `HttpLoadBalanceServer.processBatchScheduleRequest` already does the right thing (`if needConsistency && !isMaster → forwardBatchScheduleRequestToMaster; else routeService.batchSchedule`). Extract that into a shared coordinator so BOTH the HTTP endpoint and the dispatcher use one master-aware path — no drift.

**Files:**
- Create: `flexlb-sync/src/main/java/org/flexlb/service/BatchScheduleCoordinator.java`
- Modify: `flexlb-api/src/main/java/org/flexlb/httpserver/HttpLoadBalanceServer.java` (delegate the master-check + forward + local logic to the coordinator)
- Test: `flexlb-sync/src/test/java/org/flexlb/service/BatchScheduleCoordinatorTest.java`

> **Placement (codebase-style consistency):** put the coordinator in **flexlb-sync `org.flexlb.service`, next to `RouteService`** — NOT in flexlb-api. The codebase already follows "thin handler delegates to a `*Service`" (`HttpLoadBalanceServer` → `RouteService`, plus ~15 services under `flexlb-sync/org/flexlb/service/**`). All collaborators it needs already live in or below flexlb-sync: `RouteService` (same package), `LBStatusConsistencyService` (`org.flexlb.consistency`, flexlb-sync), `EngineHealthReporter` (`org.flexlb.service.monitor`, flexlb-sync), `GeneralHttpNettyService` (`org.flexlb.transport`, flexlb-common — a flexlb-sync dependency). Placing it in flexlb-api would split the `org.flexlb.service` package across two modules (a smell); flexlb-sync keeps it cohesive and co-locates it with the other master/consistency logic. `HttpLoadBalanceServer` (flexlb-api) injects it exactly as it injects `RouteService` today; the dispatcher injects it the same way.

- [ ] **Step 1: Write the failing test (master vs slave behavior)**

```java
package org.flexlb.service;

import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.transport.GeneralHttpNettyService;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;
import java.net.URI;
import java.util.List;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class BatchScheduleCoordinatorTest {

    private final RouteService routeService = mock(RouteService.class);
    private final LBStatusConsistencyService consistency = mock(LBStatusConsistencyService.class);
    private final GeneralHttpNettyService http = mock(GeneralHttpNettyService.class);
    private final EngineHealthReporter reporter = mock(EngineHealthReporter.class);
    private final BatchScheduleCoordinator coordinator =
        new BatchScheduleCoordinator(routeService, consistency, http, reporter);

    @Test
    void masterRunsInProcess() {
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(true);
        BatchScheduleResponse ok = BatchScheduleResponse.success(
            List.of(new BatchScheduleTarget("10.0.0.1", 8088, 8089)));
        when(routeService.batchSchedule(any())).thenReturn(Mono.just(ok));

        StepVerifier.create(coordinator.schedule(reqOf(1)))
            .assertNext(r -> { assert r.isSuccess(); })
            .verifyComplete();
        verify(http, never()).request(any(), any(), anyString(), any());
    }

    @Test
    void slaveForwardsToMaster() {
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(false);
        when(consistency.getMasterHostIpPort()).thenReturn("10.9.9.9:7001");
        BatchScheduleResponse forwarded = BatchScheduleResponse.success(
            List.of(new BatchScheduleTarget("10.0.0.2", 8088, 8089)));
        when(http.request(any(), any(URI.class), eq("/rtp_llm/batch_schedule"), eq(BatchScheduleResponse.class)))
            .thenReturn(Mono.just(forwarded));

        StepVerifier.create(coordinator.schedule(reqOf(1)))
            .assertNext(r -> { assert r.getServerStatus() != null; })
            .verifyComplete();
        verify(routeService, never()).batchSchedule(any());
    }

    private BatchScheduleRequest reqOf(int count) {
        BatchScheduleRequest req = new BatchScheduleRequest();
        req.setBatchCount(count);
        return req;
    }
}
```

> The targets accessor on `BatchScheduleResponse` is **`getServerStatus()`** (Lombok getter for the `serverStatus` field, JSON `server_status`) returning `List<BatchScheduleTarget>` — there is no `getTargets()`/`getServerStatusTargets()`. `isSuccess()`, `getErrorMessage()`, `getCode()` exist (verified against `BatchScheduleResponse.java`); `BatchScheduleResponse.success(targets)` is the factory `DefaultRouter` already uses.

- [ ] **Step 2: Run test to verify it fails**

Run: `./mvnw test -pl flexlb-sync -am -Dtest=BatchScheduleCoordinatorTest -P-internal`
Expected: FAIL — `BatchScheduleCoordinator` does not exist.

- [ ] **Step 3: Write the coordinator (move logic out of HttpLoadBalanceServer)**

```java
package org.flexlb.service;

import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.transport.GeneralHttpNettyService;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;
import java.net.URI;
import java.util.concurrent.TimeoutException;

/**
 * Single master-aware entry point for batch scheduling. The elected master runs the RR pick
 * in-process; a slave forwards to the real master's HTTP endpoint. Used by both the
 * {@code /rtp_llm/batch_schedule} endpoint and the dispatcher's BE pre-assign.
 */
@Service
public class BatchScheduleCoordinator {

    private final RouteService routeService;
    private final LBStatusConsistencyService consistency;
    private final GeneralHttpNettyService http;
    private final EngineHealthReporter reporter;

    public BatchScheduleCoordinator(RouteService routeService,
                                    LBStatusConsistencyService consistency,
                                    GeneralHttpNettyService http,
                                    EngineHealthReporter reporter) {
        this.routeService = routeService;
        this.consistency = consistency;
        this.http = http;
        this.reporter = reporter;
    }

    public Mono<BatchScheduleResponse> schedule(BatchScheduleRequest request) {
        if (consistency.isNeedConsistency() && !consistency.isMaster()) {
            return forwardToMaster(request);
        }
        return routeService.batchSchedule(request);
    }

    private Mono<BatchScheduleResponse> forwardToMaster(BatchScheduleRequest request) {
        String master = consistency.getMasterHostIpPort();
        if (master == null) {
            reporter.reportForwardToMasterResult("LOCAL", "MASTER_NULL");
            return Mono.just(BatchScheduleResponse.error(
                StrategyErrorType.NO_AVAILABLE_WORKER, "master unreachable"));
        }
        URI uri = URI.create("http://" + master);
        return http.request(request, uri, "/rtp_llm/batch_schedule", BatchScheduleResponse.class)
            .doOnNext(resp -> reporter.reportForwardToMasterResult(uri.getHost(), String.valueOf(resp.getCode())))
            .onErrorResume(e -> {
                String code = e instanceof TimeoutException ? "TIMEOUT" : "CONNECT_FAILED";
                Logger.error("[BatchSchedule] master unreachable, code={}", code, e);
                reporter.reportForwardToMasterResult("LOCAL", code);
                return Mono.just(BatchScheduleResponse.error(
                    StrategyErrorType.NO_AVAILABLE_WORKER, "master unreachable: " + code));
            });
    }
}
```

- [ ] **Step 4: Refactor `HttpLoadBalanceServer.processBatchScheduleRequest` to delegate**

Replace the inline master-check + `forwardBatchScheduleRequestToMaster` + `routeService.batchSchedule` body with a single `batchScheduleCoordinator.schedule(bctx.getBatchRequest())` call (inject the coordinator via the constructor; drop the now-dead `forwardBatchScheduleRequestToMaster` private method). The endpoint keeps its response-building / PV-logging shell unchanged.

- [ ] **Step 5: Run the coordinator test + both modules' tests (regression)**

Run: `./mvnw test -pl flexlb-sync,flexlb-api -am -P-internal`
Expected: PASS — coordinator test (flexlb-sync) passes AND existing `HttpLoadBalanceServer` batch-schedule behavior (flexlb-api) is unchanged (master/slave/forward identical, just relocated).

- [ ] **Step 6: Commit**

```bash
git add flexlb-sync/src/main/java/org/flexlb/service/BatchScheduleCoordinator.java \
        flexlb-api/src/main/java/org/flexlb/httpserver/HttpLoadBalanceServer.java \
        flexlb-sync/src/test/java/org/flexlb/service/BatchScheduleCoordinatorTest.java
git commit -m "refactor(flexlb): extract master-aware BatchScheduleCoordinator (reused by endpoint + dispatcher)"
```

</details>

### Task 13b: Add `role` to BatchScheduleResponse (Master side)

> The FE's `RoleAddr` needs a `role`, but `BatchScheduleTarget` carries none. Since `/batch_schedule` is single-role, return the one role on the response so the dispatcher can build `RoleAddr{role, ...}`.

**Files:**
- Modify: `flexlb-common/src/main/java/org/flexlb/dao/loadbalance/BatchScheduleResponse.java` (add `role` field + carry it through `success(...)`)
- Modify: `flexlb-sync/src/main/java/org/flexlb/balance/scheduler/DefaultRouter.java` (`batchSchedule` already knows `roleType` — set it on the success response)
- Test: `flexlb-sync/src/test/java/org/flexlb/balance/scheduler/DefaultRouterTest.java` (assert response carries the role)

- [ ] **Step 1: Write/extend the failing test** — in `DefaultRouterTest`, a single-role RR batch schedule returns a response whose `role` equals the deployed role (e.g. `RoleType.DECODE`).

- [ ] **Step 2: Run to verify it fails** — `./mvnw test -pl flexlb-sync -am -Dtest=DefaultRouterTest -P-internal`. Expected: FAIL (no `role` on response).

- [ ] **Step 3: Implement** — add `@JsonProperty("role") private String role;` to `BatchScheduleResponse`; in `DefaultRouter.batchSchedule`, after picking `roleType` and building the success response, set `response.setRole(roleType.name())`. Keep `success(targets)` and add `success(targets, role)` (or set role after construction) without breaking existing callers.

- [ ] **Step 4: Run to verify it passes.** Also run `./mvnw test -pl flexlb-sync,flexlb-api -am -P-internal` to confirm existing batch-schedule tests still pass.

- [ ] **Step 5: Commit**

```bash
git add flexlb-common/src/main/java/org/flexlb/dao/loadbalance/BatchScheduleResponse.java \
        flexlb-sync/src/main/java/org/flexlb/balance/scheduler/DefaultRouter.java \
        flexlb-sync/src/test/java/org/flexlb/balance/scheduler/DefaultRouterTest.java
git commit -m "feat(flexlb): carry resolved role on BatchScheduleResponse for dispatcher RoleAddr"
```

### Task 14: BeTargetAssigner (master-aware, via coordinator) + tag chunks with role_addrs

> **Status (2026-05-25):** SUPERSEDED by Stage 2 (Tasks V1–V11). Do not implement. Pre-assign deferred indefinitely (see Stage 2 §"What is NOT changing"). E2E coverage moved to Task V10. Docs moved to Task V11.

**Files:**
- Create: `flexlb-api/src/main/java/org/flexlb/dispatcher/BeTargetAssigner.java`
- Modify: `flexlb-api/src/main/java/org/flexlb/dispatcher/FanoutService.java` (target tagging overload)
- Test: `flexlb-api/src/test/java/org/flexlb/dispatcher/BeTargetAssignerTest.java`

`BeTargetAssigner` delegates to `BatchScheduleCoordinator` — so it inherits master→in-process / slave→forward automatically. It never calls `routeService` directly.

> **Chunk-count ceiling (verify before shipping pre-assign):** `assign(C)` calls `/batch_schedule` with `batchCount = C` (number of chunks). `DefaultRouter.batchSchedule` rejects `count > batchScheduleMaxCount` (`DefaultRouter.java:123`) with `INVALID_REQUEST`, which `BeTargetAssigner` surfaces as a failure → the whole batch fails. So **C must stay ≤ `batchScheduleMaxCount`**. For N=500, K=5 → C=100; confirm the Master-side cap covers that, or clamp K so `ceil(N/K) ≤ cap`. (ft_proxy similarly caps `numParts` at its pool size — `server.go:142,157`.) Note `selectBatch(count, …)` returns exactly `count` targets, so `targets.size() == C` holds — but `FanoutService` should still treat a missing target index as a failed chunk (placeholder-pad, per Task 2) rather than throw.

> **Single-role precondition (Phase 4 only):** pre-assign works only on a **single-role** fleet. `DefaultRouter.batchSchedule` rejects `roleTypes.size() > 1` (`DefaultRouter.java:148`), and on the FE side `route_ips` still domain-falls-back for any role not in the pre-set `role_addrs` (`backend_rpc_server_visitor.py:238-247`). So one `role` per response (Task 13b) is sufficient *because the regime is single-role* — for disaggregated PREFILL+DECODE, use core fanout mode (Phases 1–3), which has no such restriction.

- [ ] **Step 1: Write the failing test**

```java
package org.flexlb.dispatcher;

import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.exception.BatchScheduleTransportException;
import org.flexlb.service.BatchScheduleCoordinator;
import org.flexlb.service.BatchScheduleCoordinator.Outcome;
import org.flexlb.service.BatchScheduleCoordinator.Source;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Mono;
import java.util.List;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class BeTargetAssignerTest {

    @Test
    void delegatesToCoordinatorAndReturnsTargets() {
        BatchScheduleCoordinator coordinator = mock(BatchScheduleCoordinator.class);
        BatchScheduleTarget t0 = new BatchScheduleTarget("10.0.0.1", 8088, 8089);
        BatchScheduleTarget t1 = new BatchScheduleTarget("10.0.0.2", 8088, 8089);
        BatchScheduleResponse ok = BatchScheduleResponse.success(List.of(t0, t1));
        when(coordinator.schedule(any(BatchScheduleRequest.class)))
            .thenReturn(Mono.just(new Outcome(ok, Source.LOCAL)));

        BeTargetAssigner assigner = new BeTargetAssigner(coordinator);
        BatchScheduleResponse resp = assigner.assign(2).block();
        assertEquals(2, resp.getServerStatus().size());
        assertEquals("10.0.0.1", resp.getServerStatus().get(0).getServerIp());
    }

    @Test
    void surfacesBusinessFailureAsResponse() {
        // Master returned a non-success response (e.g. NO_AVAILABLE_WORKER) — this is NOT a
        // transport error, so the coordinator delivers it on the success channel.
        BatchScheduleCoordinator coordinator = mock(BatchScheduleCoordinator.class);
        BatchScheduleResponse err = BatchScheduleResponse.error(
            org.flexlb.dao.loadbalance.StrategyErrorType.NO_AVAILABLE_WORKER, "no be");
        when(coordinator.schedule(any())).thenReturn(Mono.just(new Outcome(err, Source.LOCAL)));
        BeTargetAssigner assigner = new BeTargetAssigner(coordinator);
        assertThrows(IllegalStateException.class, () -> assigner.assign(2).block());
    }

    @Test
    void surfacesTransportFailureAsException() {
        // Master is unreachable from a slave (or master null) — coordinator raises
        // BatchScheduleTransportException; the assigner does NOT mask it as a value.
        BatchScheduleCoordinator coordinator = mock(BatchScheduleCoordinator.class);
        when(coordinator.schedule(any()))
            .thenReturn(Mono.error(new BatchScheduleTransportException("master unreachable", "MASTER_NULL")));
        BeTargetAssigner assigner = new BeTargetAssigner(coordinator);
        assertThrows(BatchScheduleTransportException.class, () -> assigner.assign(2).block());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./mvnw test -pl flexlb-api -am -Dtest=BeTargetAssignerTest -P-internal`
Expected: FAIL — `BeTargetAssigner` does not exist.

- [ ] **Step 3: Write minimal implementation**

```java
package org.flexlb.dispatcher;

import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.service.BatchScheduleCoordinator;
import reactor.core.publisher.Mono;

public class BeTargetAssigner {

    private final BatchScheduleCoordinator coordinator;

    public BeTargetAssigner(BatchScheduleCoordinator coordinator) {
        this.coordinator = coordinator;
    }

    /**
     * One master-aware batch schedule per batch. {@code chunkCount} = number of K-sized chunks
     * (NOT prompt count): a /batch_infer call shares one generate_config, so one BE serves the
     * whole chunk. Returns the full response so the caller gets both targets and the resolved role.
     *
     * <p>Error channels match the coordinator contract:
     * <ul>
     *   <li>Business failure (e.g. {@code NO_AVAILABLE_WORKER}) — coordinator delivers a non-success
     *       response on the success channel; we raise {@link IllegalStateException} to keep the
     *       dispatch caller's flow uniform.</li>
     *   <li>Transport failure (master null / forward error) — coordinator raises
     *       {@link org.flexlb.exception.BatchScheduleTransportException}; we let it propagate,
     *       so the dispatch handler can fall back (e.g. degrade pre-assign to core-fanout mode)
     *       rather than mistake it for a per-prompt routing failure.</li>
     * </ul>
     */
    public Mono<BatchScheduleResponse> assign(int chunkCount) {
        BatchScheduleRequest req = new BatchScheduleRequest();
        req.setBatchCount(chunkCount);
        return coordinator.schedule(req).map(outcome -> {
            BatchScheduleResponse resp = outcome.getResponse();
            if (!resp.isSuccess()) {
                throw new IllegalStateException("batchSchedule failed: " + resp.getErrorMessage());
            }
            return resp; // carries targets (one per chunk) + role (Task 13b)
        });
    }
}
```

> The test above asserts on `assign(...).block().getServerStatus()` / `.getRole()` — the real getter names (`getServerStatus()` returns `List<BatchScheduleTarget>`; `getRole()` is added in Task 13b). Coordinator returns `Mono<Outcome>` not `Mono<BatchScheduleResponse>` directly — we unwrap with `outcome.getResponse()` and discard `outcome.getSource()` (the LOCAL/FORWARDED tag is for `HttpLoadBalanceServer`'s metrics, not relevant to the dispatcher path).

- [ ] **Step 4: Run test to verify it passes**

Run: `./mvnw test -pl flexlb-api -am -Dtest=BeTargetAssignerTest -P-internal`
Expected: PASS.

- [ ] **Step 5: Inject `role_addrs` into each chunk's generate_config (FanoutService)**

This is the actual pre-assign wiring, using the FE hook found in Task 12 (`generate_config.role_addrs`). Add an overload `dispatch(prompts, generateConfig, targets, role)` that, after `BatchSplitter.split` into C chunks:
- for chunk `i`, build one `RoleAddr`-shaped JSON object from `targets.get(i)` and `role`:
  `{"role": role, "ip": target.serverIp, "http_port": target.httpPort, "grpc_port": target.grpcPort}`
- set `generate_config.role_addrs = [thatOneObject]` on that chunk's body (one BE per chunk — see Task 12 constraint), merging into the caller-supplied `generate_config` rather than overwriting it.

`targets.size()` MUST equal the chunk count C. Keep C single-sourced: derive it from the same code the split uses — `C = BatchSplitter.chunkCount(prompts.size(), K)` (Task 1) — never a separate inline `ceil(N/K)`. Defensively, `FanoutService` should treat a missing `targets.get(i)` (size < chunk count) as a failed chunk (placeholder-pad, per Task 2) rather than throw. Add a `FanoutServiceTest` case asserting chunk 0's body has `generate_config.role_addrs[0].ip == targets[0].serverIp` and `.role == role`. Keep the no-targets `dispatch` for the disabled-preassign path. Wire `BeTargetAssigner` into `DispatchHandler.handleBatch`: compute `C = BatchSplitter.chunkCount(prompts.size(), K)`, `assign(C)`, then `dispatch(prompts, generateConfig, resp.getServerStatus(), resp.getRole())`. Update `DispatcherConfiguration` to build `BeTargetAssigner(batchScheduleCoordinator)` and pass it to `DispatchHandler` only when pre-assign is enabled (add `preAssignBe` flag to `DISPATCH_CONFIG`, default false → core fanout mode).

- [ ] **Step 6: Commit**

```bash
git add flexlb-api/src/main/java/org/flexlb/dispatcher/BeTargetAssigner.java \
        flexlb-api/src/main/java/org/flexlb/dispatcher/FanoutService.java \
        flexlb-api/src/main/java/org/flexlb/dispatcher/DispatchHandler.java \
        flexlb-api/src/main/java/org/flexlb/dispatcher/DispatcherConfiguration.java \
        flexlb-api/src/test/java/org/flexlb/dispatcher/BeTargetAssignerTest.java \
        flexlb-api/src/test/java/org/flexlb/dispatcher/FanoutServiceTest.java
git commit -m "feat(dispatcher): pre-assign BE targets via master-aware coordinator"
```

### Task 15: Verify FE honors pre-assigned `role_addrs` (no production FE change)

> **Status (2026-05-25):** SUPERSEDED by Stage 2 (Tasks V1–V11). Do not implement. Pre-assign deferred indefinitely (see Stage 2 §"What is NOT changing"). E2E coverage moved to Task V10. Docs moved to Task V11.

> Task 12 found the FE already skips the Master when `generate_config.role_addrs` is set (`route_ips:217,226`). So the happy path needs **no FE code change** — only a regression test pinning that behavior, so a future FE refactor can't silently break the dispatcher.

**Files:**
- Test only: `rtp_llm/server/test/backend_rpc_server_visitor_test.py` (or the existing test module for `BackendRPCServerVisitor` — confirm path with `find rtp_llm -name '*backend_rpc*test*'`)

- [ ] **Step 1: Write the test** — construct a `GenerateInput` whose `generate_config.role_addrs` is pre-populated with one `RoleAddr` (role matching `backend_role_list`); mock `MasterClient`; call `route_ips(input)`; assert `MasterClient.get_backend_role_addrs` is **never called** and `input.generate_config.role_addrs` is unchanged.

- [ ] **Step 2: Run to verify it passes immediately** (it should — behavior already exists). Run the FE test target (e.g. `bazelisk test //rtp_llm/server/test:backend_rpc_server_visitor_test` — confirm the exact target). This is a characterization test, so green-on-first-run is expected and correct here.

- [ ] **Step 3: Commit**

```bash
git add rtp_llm/server/test/backend_rpc_server_visitor_test.py
git commit -m "test(frontend): pin that pre-set generate_config.role_addrs bypasses master routing"
```

### Task 15b (OPTIONAL, deferred): dead pre-assigned target → re-ask master

> **Status (2026-05-25):** SUPERSEDED by Stage 2 (Tasks V1–V11). Do not implement. Pre-assign deferred indefinitely (see Stage 2 §"What is NOT changing"). E2E coverage moved to Task V10. Docs moved to Task V11.

> Not in the v1 happy path. Today a dead pre-assigned target is handed to gRPC and the request fails (client retries). If a deployment needs auto-failover, add it here.

- [ ] **Design + implement** a fallback in `route_ips`: when a request carries pre-assigned `role_addrs` AND the gRPC call to that target fails with a connection error, re-run `get_master_route_addrs` (clear the stale `role_addrs` first) and retry once. Gate behind a flag so default behavior is unchanged. Write a test simulating a dead target → master re-route. **Decide whether to do this in v1 with the user before implementing** — it touches the hot `route_ips`/enqueue path.

---

## Phase 5 — End-to-end & operability

### Task 16: End-to-end integration test (dispatcher → mock FE pool)

> **Status (2026-05-25):** SUPERSEDED by Stage 2 (Tasks V1–V11). Do not implement. Pre-assign deferred indefinitely (see Stage 2 §"What is NOT changing"). E2E coverage moved to Task V10. Docs moved to Task V11.

**Files:**
- Test: `flexlb-api/src/test/java/org/flexlb/dispatcher/DispatcherEndToEndTest.java`

Binds the dispatcher `RouterFunction` (from `DispatcherConfiguration.dispatcherRoutes`) via `WebTestClient.bindToRouterFunction`, pointing the `FePool` at **two** MockWebServer FEs, sends a batch of N>K to `/batch_infer`, asserts: (a) the merged `response_batch` has N entries in order, (b) the batch was split across both FEs (each MockWebServer received ≥1 request), (c) a non-batch path (`/worker_status`) is passed through. (No bound port needed — `WebTestClient` drives the RouterFunction directly.)

- [ ] **Step 1: Write the test** (two MockWebServers, each returns a `response_batch` echoing its chunk; assert merged size == N and both servers got a request).

- [ ] **Step 2: Run** `./mvnw test -pl flexlb-api -am -Dtest=DispatcherEndToEndTest -P-internal` — Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add flexlb-api/src/test/java/org/flexlb/dispatcher/DispatcherEndToEndTest.java
git commit -m "test(dispatcher): end-to-end split/fanout/merge over mock FE pool"
```

### Task 16b: Slave-node fanout + forward-to-master concurrency check

> **Why:** the same-JVM design has an asymmetric stack between master and slave nodes, and the existing perf data only covers the master case.
> - **Master node:** dispatcher calling `BatchScheduleCoordinator.schedule` is a method call (`Source.LOCAL`); the only HTTP traffic on 7001 is inbound `/batch_infer`.
> - **Slave node:** `BatchScheduleCoordinator.schedule` decides `!isMaster → forwardToMaster`, which is a real HTTP request out to the elected master's 7001. So a slave-node dispatcher's Reactor-Netty event loop simultaneously carries: (a) inbound `/batch_infer` from the client, (b) N parallel outbound `/batch_infer` to the FE pool, **and** (c) one outbound `/rtp_llm/batch_schedule` to the elected master.
>
> Decision B4 hand-waves this as "responsive fanout doesn't block the loop while awaiting" — true, but a slave node has +1 outbound HTTP-acquire path that the master node doesn't, so its event-loop and connection-pool contention is strictly higher. If `/schedule` (the Master's 1–5 ms SLA control plane) is co-located, the slave node is where it'll show up first. Verify before declaring same-JVM safe.
>
> Not blocking shipping the dispatcher — but blocking the "no separate port needed" claim (decision B4). If this test shows slave-node `/schedule` p99 regresses past SLA under concurrent fanout, the fix is the optional separate-port variant (Task 10-alt) on slaves only, not a code-level mitigation.

**Files:**
- Test: `flexlb-api/src/test/java/org/flexlb/dispatcher/SlaveNodeConcurrencyIntegrationTest.java`
- Or, if a JVM-internal test cannot model master forward realistically, document the manual procedure in `rtp_llm/flexlb/CLAUDE.md` and run it on staging before rollout.

- [ ] **Step 1:** Stand up a 2-node FlexLB locally (one master, one slave; the slave's `LBStatusConsistencyService.isMaster()=false` and `getMasterHostIpPort()` points at the master). Or mock by a dispatcher pointed at a `BatchScheduleCoordinator` configured as slave + a MockWebServer pretending to be the master endpoint.
- [ ] **Step 2:** From a load generator, hit the **slave** node's `/batch_infer` with N=20 concurrent batches of 100 prompts each (100 chunks per batch at K=5). Concurrently, hit the slave's `/rtp_llm/schedule` at 200 QPS. Both routes share the slave's event loop.
- [ ] **Step 3:** Record p50/p99 of `/rtp_llm/schedule` with and without the dispatcher load. Pass criterion: `/rtp_llm/schedule` p99 stays ≤ 5 ms under load (the published SLA). If it regresses past 10 ms, that's a deal with same-JVM on slave nodes — escalate to user and consider Task 10-alt for slaves.
- [ ] **Step 4:** Capture findings in `docs/dispatcher-slave-node-concurrency-2026-MM-DD.md` and link from decision B4. If the test passes, B4's "independent port not needed" claim is fully grounded; if it fails, B4 needs an asterisk for slave nodes.
- [ ] **Step 5: Commit** (test or docs only — no code change unless the result demands it).

### Task 17: Docs — DISPATCH_CONFIG + ports

> **Status (2026-05-25):** SUPERSEDED by Stage 2 (Tasks V1–V11). Do not implement. Pre-assign deferred indefinitely (see Stage 2 §"What is NOT changing"). E2E coverage moved to Task V10. Docs moved to Task V11.

**Files:**
- Modify: `rtp_llm/flexlb/CLAUDE.md` (Configuration section) + `rtp_llm/flexlb/README.md`

- [ ] **Step 1: Document** the new `DISPATCH_CONFIG` env (fields, example), the data-plane port vs Master control-plane port (7001) separation, the non-streaming v1 limitation, and the request_id trace-continuity caveat (known gap, deferred).

- [ ] **Step 2: Commit**

```bash
git add rtp_llm/flexlb/CLAUDE.md rtp_llm/flexlb/README.md
git commit -m "docs(dispatcher): document DISPATCH_CONFIG, port model, v1 limits"
```

---

## Phase 3 follow-up — FE Discovery via ServiceDiscovery (landed 2026-05-22)

> The Phase 3 plan above defaults to a static `fePoolAddresses: List<String>` in `DISPATCH_CONFIG`. That ships, but the FE fleet scales up/down/dies in production and a static list can't track that without a restart + env edit. Decision: reuse flexlb's existing `ServiceDiscovery` mechanism — the same one master uses for its BE pool today (`NoOpServiceDiscovery` reads `DOMAIN_ADDRESS:<service_id>=ip:port,...` in open-source / dev; the internal Maven profile swaps in a VipServer-backed impl with true push subscription, wired via `@ConditionalOnMissingBean`). Zero FE-side change (FE is already a discoverable service) and zero new infrastructure.

**Code path:**
- `FePool` now reads addresses through a `Supplier<List<String>>` on every `next()` (no internal cache). The static `List<String>` constructor is preserved as a convenience for tests and delegates to a supplier internally. Empty snapshot at call time throws `IllegalStateException("no FE endpoints available")` — `FanoutService` already wraps each chunk in `.onErrorResume(... → SubBatchResult.failed(...))` so a misconfig produces placeholder-padded chunks, not a JVM crash.
- `DispatchConfig.fePoolAddresses` is gone; replaced by `fePoolServiceId: String`. Validation rejects `enabled=true` with a blank serviceId.
- `DispatcherConfiguration` accepts an injected `ServiceDiscovery`, seeds an `AtomicReference<List<String>>` from `getHosts(serviceId)`, subscribes via `listen(serviceId, hosts -> ...)`, and gives `FePool` an `AtomicReference::get` supplier. URL shape: `"http://" + WorkerHost.getIpPort()` (httpPort only — HTTPS not in scope).

**Commits (on `feature/master-batch-schedule`):**
- `58b087264` `refactor(dispatcher): FePool reads addresses through a Supplier`
- `972ebc4b2` `refactor(dispatcher): DispatchConfig takes fePoolServiceId, not a static address list`
- `3456bcd7f` `feat(dispatcher): discover FE pool via ServiceDiscovery (DOMAIN_ADDRESS)`

**Tests added:** `FePoolTest.readsDynamicSupplierOnEveryNext`, `FePoolTest.emptySupplierSnapshotThrowsOnNext`, `DispatchConfigTest.rejectsEnabledWithBlankFePoolServiceId`, `DispatcherConfigurationTest.subscribesAndSeedsFromDiscovery`. `DispatchConfigTest.parsesFullJson` / `sameJvmIsolationKnobsOverridableViaJson` and `DispatcherConfigurationTest.buildsRouterWhenEnabled` updated to the new field. Full `flexlb-api` suite: 45/45 PASS.

**Out of scope (still deferred):**
- Rebalancing in-flight FE assignments when the host list shrinks — only fresh `FePool.next()` calls see the new list; in-flight chunks complete or time out against their original target.
- Multi-region / preferring local FEs — `WorkerHost.site` exists but ignored.
- Health-aware FE selection — service-discovery liveness is the only signal; no FE→master heartbeat.

---

## Open items deferred past v1 (record, don't build)

- **Streaming batch** — multi-stream SSE merge; out of v1 scope by decision.
- **Dead pre-assigned target → re-ask master** (Task 15b) — today a dead target fails the request (client retries); auto-failover touches the hot `route_ips` path, deferred pending need.
- **request_id trace continuity** — `frontend_server` overwrites upstream request_id; breaks dispatcher→FE trace linkage. Known, deferred.
- ~~**FE pool from service discovery**~~ — ✓ DONE in the Phase 3 follow-up above (`fePoolServiceId` + `ServiceDiscovery` + `DOMAIN_ADDRESS:<id>` env).
- **Per-request dynamic mode switch** — the agreed escape hatch (route load-sensitive deployments back to per-request `/schedule`); add a `DISPATCH_CONFIG` mode flag when a deployment needs it.

---

## Self-Review

- **Spec coverage:** transparent proxy (Tasks 9–11), batch split/fanout/merge (Tasks 1–8), non-streaming v1 (no streaming task — by decision), same-JVM **same-port** with route precedence (Task 10 + Task 11; separate-port is optional Task 10-alt), spike DONE (Task 12: FE hook = `generate_config.role_addrs`), **master-aware** BE pre-assign (Task 13 coordinator handles master/slave/forward; Task 13b adds response role; Task 14 injects role_addrs per chunk), FE verification test (Task 15) + optional dead-target fallback (Task 15b), end-to-end (Task 16), docs (Task 17), no Master endpoint shadowing (Task 10 precedence test + Task 11 Step 5 regression guard). Covered.
- **Master/slave correctness:** the dispatcher never calls `routeService.batchSchedule()` directly; it goes through `BatchScheduleCoordinator` (Task 13) which replicates `HttpLoadBalanceServer`'s `isNeedConsistency && !isMaster → forward` logic. Master → in-process (fast); slave → HTTP-forward to real master. One shared path, no cursor divergence across nodes.
- **Type consistency:** `FeClient.postBatch(String, ObjectNode)→Mono<JsonNode>`, `FanoutService.dispatch(List<String>, JsonNode)→Mono<MergedResponse>`, `ResponseMerger.merge(List<SubBatchResult>, ObjectMapper)→MergedResponse`, `DispatchHandler.handleBatch/handlePassthrough(ServerRequest)→Mono<ServerResponse>` (handleBatch: 200 on partial success, 500 only when `MergedResponse.allFailed()`), `DispatcherConfiguration.dispatcherRoutes(DispatchConfig)→RouterFunction<ServerResponse>` (same-port bean, `@Order(LOWEST_PRECEDENCE)`), `BatchScheduleCoordinator.schedule(BatchScheduleRequest)→Mono<BatchScheduleResponse>`, `BeTargetAssigner(BatchScheduleCoordinator)` — consistent across tasks.
- **FE hook — verified by spike (Task 12):** `route_ips` skips master when `generate_config.role_addrs` is set (`backend_rpc_server_visitor.py:217,226`); `RoleAddr={role,ip,http_port,grpc_port}` and `GenerateConfig.role_addrs` parse from the request `generate_config` dict (`generate_config.py:19,126`). Constraint: one BE per chunk (shared generate_config in `pipeline.batch_infer`). No production FE change for happy path.
- **External signatures (verified against the codebase 2026-05-21):** `RouteService.batchSchedule(BatchScheduleRequest)→Mono<BatchScheduleResponse>` (`RouteService.java:78`); the targets getter on `BatchScheduleResponse` is **`getServerStatus()`** (field `server_status`; **no** `getTargets`/`getServerStatusTargets`); `BatchScheduleTarget(String,int,int)` constructor (`@AllArgsConstructor`) with `getServerIp/getHttpPort/getGrpcPort`; `GeneralHttpNettyService.request(body, URI, path, Class)` (used at `HttpLoadBalanceServer.java:297`); `LBStatusConsistencyService.isNeedConsistency/isMaster/getMasterHostIpPort` — all confirmed present.
