package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.mock.web.reactive.function.server.MockServerRequest;
import org.springframework.web.reactive.function.server.EntityResponse;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.net.URI;
import java.util.List;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Unit tests for the combined inspection surface — {@code GET /dispatcher/_snapshot} and
 * {@code POST /dispatcher/_dryrun/<path>}. Mocks {@link FeHealthChecker},
 * {@link DispatcherFePoolRefresher}, and {@link BatchScheduleClient} directly so each case can
 * inject a known fixture and assert on the JSON response.
 *
 * <p>Chunk-shape correctness (force_batch, role_addrs wire format) lives in
 * {@link BatchChunkBuilderTest}; spec-aware splitting lives in {@link BatchSplitterTest}. These
 * tests assert <em>that</em> the handler invoked the utilities and surfaced their output, not
 * <em>how</em> the utilities shape it.
 */
class DispatcherInspectionHandlerTest {

    private static final String SERVICE_ID = "test.fe.publish";

    private final ObjectMapper mapper = new ObjectMapper();

    // ───────────────────────── snapshot ─────────────────────────

    @Nested
    class Snapshot {

        @Test
        void healthyPoolReportsAllAliveWithZeroFailures() {
            List<String> urls = List.of(
                    "http://10.0.0.1:23840", "http://10.0.0.2:23840", "http://10.0.0.3:23840");
            FeHealthChecker hc = mock(FeHealthChecker.class);
            when(hc.isAlive(anyString())).thenReturn(true);
            when(hc.consecFails(anyString())).thenReturn(0);

            ObjectNode body = invokeSnapshot(urls, hc);

            ObjectNode fePool = (ObjectNode) body.get("fePool");
            assertEquals(SERVICE_ID, fePool.get("serviceId").asText());
            assertEquals(3, fePool.get("size").asInt());
            ArrayNode hosts = (ArrayNode) fePool.get("hosts");
            for (int i = 0; i < 3; i++) {
                assertEquals(urls.get(i), hosts.get(i).get("url").asText(),
                        "hosts must preserve refresher's snapshot order — round-robin order");
                assertTrue(hosts.get(i).get("alive").asBoolean());
                assertEquals(0, hosts.get(i).get("consecFails").asInt());
            }
        }

        @Test
        void mixedAliveAndDeadReportsPerHostStateFaithfully() {
            // Three hosts, varying failure counts: warming-up (1 fail, still alive), dead (2 fails),
            // healthy (0 fails). Snapshot must distinguish all three so operators can spot a host
            // that's *about* to die vs one that just did.
            List<String> urls = List.of(
                    "http://10.0.0.1:23840", "http://10.0.0.2:23840", "http://10.0.0.3:23840");
            FeHealthChecker hc = mock(FeHealthChecker.class);
            when(hc.isAlive("http://10.0.0.1:23840")).thenReturn(true);
            when(hc.consecFails("http://10.0.0.1:23840")).thenReturn(1);
            when(hc.isAlive("http://10.0.0.2:23840")).thenReturn(false);
            when(hc.consecFails("http://10.0.0.2:23840")).thenReturn(2);
            when(hc.isAlive("http://10.0.0.3:23840")).thenReturn(true);
            when(hc.consecFails("http://10.0.0.3:23840")).thenReturn(0);

            ObjectNode body = invokeSnapshot(urls, hc);
            ArrayNode hosts = (ArrayNode) body.get("fePool").get("hosts");

            assertTrue(hosts.get(0).get("alive").asBoolean(),
                    "consecFails=1 still under threshold → alive=true (flap tolerance)");
            assertEquals(1, hosts.get(0).get("consecFails").asInt());

            assertFalse(hosts.get(1).get("alive").asBoolean(),
                    "consecFails=2 crosses FAIL_THRESHOLD → alive=false");
            assertEquals(2, hosts.get(1).get("consecFails").asInt());

            assertTrue(hosts.get(2).get("alive").asBoolean());
            assertEquals(0, hosts.get(2).get("consecFails").asInt());
        }

        @Test
        void emptyPoolReturns200WithSizeZeroAndEmptyHosts() {
            FeHealthChecker hc = mock(FeHealthChecker.class);
            ObjectNode body = invokeSnapshot(List.of(), hc);
            ObjectNode fePool = (ObjectNode) body.get("fePool");
            assertEquals(0, fePool.get("size").asInt());
            assertEquals(0, fePool.get("hosts").size());
            assertTrue(fePool.get("hosts").isArray(),
                    "even an empty hosts list must be a JSON array — never null or absent, so callers can iterate unconditionally");
        }

        @Test
        void serviceIdEchoesConfigVerbatim() {
            // Snapshot's serviceId field is the documented way to cross-check that dispatcher is
            // wired to the FE pool an operator expects, vs the boot WARN line which can scroll out
            // of log retention. Echo it verbatim so a typo'd config is immediately obvious.
            DispatchConfig cfg = new DispatchConfig();
            cfg.setFePoolServiceId("very.specific.weird.name.publish");
            cfg.setSubBatch("size:5");
            cfg.setSubBatchSpec(SubBatchSpec.parse("size:5"));
            DispatcherFePoolRefresher refresher = mock(DispatcherFePoolRefresher.class);
            when(refresher.source()).thenReturn(() -> List.<String>of());
            FeHealthChecker hc = mock(FeHealthChecker.class);
            BatchScheduleClient client = mock(BatchScheduleClient.class);
            DispatcherInspectionHandler handler =
                    new DispatcherInspectionHandler(cfg, refresher, hc, client, mapper);

            MockServerRequest req = MockServerRequest.builder()
                    .method(HttpMethod.GET)
                    .uri(URI.create("http://x/dispatcher/_snapshot"))
                    .build();

            StepVerifier.create(handler.snapshot(req))
                    .assertNext(resp -> {
                        assertEquals(HttpStatus.OK, resp.statusCode());
                        EntityResponse<?> entity = (EntityResponse<?>) resp;
                        ObjectNode body = (ObjectNode) entity.entity();
                        assertEquals("very.specific.weird.name.publish",
                                body.get("fePool").get("serviceId").asText());
                    })
                    .verifyComplete();
        }
    }

    // ───────────────────────── dryRun ─────────────────────────

    @Nested
    class DryRun {

        @Test
        void unknownPathReturns400WithRegistryContents() {
            // 400 message must list registered paths so a typo'd caller can self-correct without
            // having to grep the source. Counterfactual: if this fell through to passthrough, the
            // caller would get FE's 404 with no hint that the dispatcher even has a registry.
            BatchScheduleClient client = mock(BatchScheduleClient.class);
            DispatcherInspectionHandler handler = handlerWith(false, client);

            MockServerRequest req = MockServerRequest.builder()
                    .method(HttpMethod.POST)
                    .uri(URI.create("http://x/dispatcher/_dryrun/totally_made_up"))
                    .body(Mono.just(mapper.createObjectNode()));

            assertResponse(handler.dryRun(req), HttpStatus.BAD_REQUEST, body -> {
                assertEquals("invalid_inspection_request", body.get("error").asText());
                String msg = body.get("message").asText();
                assertTrue(msg.contains("/batch_infer"),
                        "error message must enumerate registered paths so caller can fix the URL");
                assertTrue(msg.contains("/v1/embeddings"));
            });
        }

        @Test
        void emptyArrayShortCircuitsWithoutCallingBatchScheduleClient() {
            // Critical: empty batch must NOT call batchScheduleClient.requestTargets(0) even when
            // pre_assign=true. Hitting the coordinator with batchCount=0 is at best wasted RTT and
            // at worst undefined behavior, and would advance master's batch RR cursor on a request
            // that the real handler would have served entirely locally.
            BatchScheduleClient client = mock(BatchScheduleClient.class);
            DispatcherInspectionHandler handler = handlerWith(true, client);

            ObjectNode body = mapper.createObjectNode();
            body.putArray("prompt_batch");

            MockServerRequest req = MockServerRequest.builder()
                    .method(HttpMethod.POST)
                    .uri(URI.create("http://x/dispatcher/_dryrun/batch_infer"))
                    .queryParam("pre_assign", "true")
                    .body(Mono.just(body));

            assertResponse(handler.dryRun(req), HttpStatus.OK, out -> {
                assertEquals(0, out.get("chunkCount").asInt());
                assertEquals(0, out.get("chunks").size());
                assertEquals(0, out.get("preAssignTargets").size());
                assertTrue(out.get("preAssignEffective").asBoolean(),
                        "preAssignEffective must still reflect the requested value even when no targets resolved");
            });
            verify(client, never()).requestTargets(anyInt());
        }

        @Test
        void queryParamTrueOverridesConfigDefaultFalse() {
            BatchScheduleClient client = mock(BatchScheduleClient.class);
            when(client.requestTargets(anyInt())).thenReturn(Mono.just(List.of(target("10.0.0.1"))));
            DispatcherInspectionHandler handler = handlerWith(false, client);

            ObjectNode out = invokeDryRun(handler, "true", List.of("a"));
            assertFalse(out.get("preAssignConfigDefault").asBoolean());
            assertTrue(out.get("preAssignEffective").asBoolean(),
                    "query param true must override config false");
            assertEquals(1, out.get("preAssignTargets").size(),
                    "target should have been resolved since effective=true");
        }

        @Test
        void queryParamFalseOverridesConfigDefaultTrue() {
            BatchScheduleClient client = mock(BatchScheduleClient.class);
            DispatcherInspectionHandler handler = handlerWith(true, client);

            ObjectNode out = invokeDryRun(handler, "false", List.of("a"));
            assertTrue(out.get("preAssignConfigDefault").asBoolean());
            assertFalse(out.get("preAssignEffective").asBoolean(),
                    "query param false must override config true");
            verify(client, never()).requestTargets(anyInt());
        }

        @Test
        void noQueryParamUsesConfigDefault() {
            // No query param → behavior matches what the real handler would do under this config.
            // This is the contract that makes dry-run usable as a "what would production do" check.
            BatchScheduleClient client = mock(BatchScheduleClient.class);
            when(client.requestTargets(anyInt())).thenReturn(Mono.just(List.of(target("10.0.0.1"))));
            DispatcherInspectionHandler handler = handlerWith(true, client);

            ObjectNode out = invokeDryRun(handler, null, List.of("a"));
            assertTrue(out.get("preAssignConfigDefault").asBoolean());
            assertTrue(out.get("preAssignEffective").asBoolean(),
                    "absent query param must fall back to config default");
        }

        @Test
        void preAssignTrueProducesStampedChunks() {
            BatchScheduleClient client = mock(BatchScheduleClient.class);
            when(client.requestTargets(anyInt())).thenReturn(Mono.just(List.of(
                    target("10.0.0.1"), target("10.0.0.2"))));
            DispatcherInspectionHandler handler = handlerWith(true, client);

            ObjectNode out = invokeDryRun(handler, "true", List.of("a", "b", "c"));
            ArrayNode chunks = (ArrayNode) out.get("chunks");
            assertTrue(chunks.size() > 0);
            for (JsonNode chunk : chunks) {
                JsonNode roleAddrs = chunk.get("generate_config").get("role_addrs");
                assertNotNull(roleAddrs,
                        "every chunk must show its stamped role_addrs when pre_assign effective");
            }
            // preAssignTargets is a flat dump of what BatchScheduleClient returned, in resolution order.
            assertEquals("10.0.0.1", out.get("preAssignTargets").get(0).get("ip").asText());
            assertEquals("10.0.0.2", out.get("preAssignTargets").get(1).get("ip").asText());
        }

        @Test
        void preAssignFalseLeavesChunksUnstamped() {
            BatchScheduleClient client = mock(BatchScheduleClient.class);
            DispatcherInspectionHandler handler = handlerWith(true, client);

            ObjectNode out = invokeDryRun(handler, "false", List.of("a", "b"));
            for (JsonNode chunk : (ArrayNode) out.get("chunks")) {
                JsonNode gc = chunk.get("generate_config");
                assertNull(gc == null ? null : gc.get("role_addrs"),
                        "preAssign=false must not stamp role_addrs even though config default is true");
                assertTrue(gc.get("force_batch").asBoolean(),
                        "force_batch still injected — that's independent of preAssign");
            }
            verify(client, never()).requestTargets(anyInt());
        }

        @Test
        void nonObjectBodyReturns400() {
            BatchScheduleClient client = mock(BatchScheduleClient.class);
            DispatcherInspectionHandler handler = handlerWith(false, client);

            MockServerRequest req = MockServerRequest.builder()
                    .method(HttpMethod.POST)
                    .uri(URI.create("http://x/dispatcher/_dryrun/batch_infer"))
                    .body(Mono.just(mapper.createArrayNode().add("a")));

            assertResponse(handler.dryRun(req), HttpStatus.BAD_REQUEST, body ->
                    assertEquals("invalid_inspection_request", body.get("error").asText()));
        }

        @Test
        void missingArrayFieldReturns400() {
            BatchScheduleClient client = mock(BatchScheduleClient.class);
            DispatcherInspectionHandler handler = handlerWith(false, client);

            ObjectNode body = mapper.createObjectNode();
            body.put("not_prompt_batch", "x");
            MockServerRequest req = MockServerRequest.builder()
                    .method(HttpMethod.POST)
                    .uri(URI.create("http://x/dispatcher/_dryrun/batch_infer"))
                    .body(Mono.just(body));

            assertResponse(handler.dryRun(req), HttpStatus.BAD_REQUEST, b -> {
                String msg = b.get("message").asText();
                assertTrue(msg.contains("prompt_batch"),
                        "400 message must name the field the caller forgot");
            });
        }

        @Test
        void internalErrorReturns500NotBadRequest() {
            // If BatchScheduleClient ever broke its no-throw contract (or any other code path
            // inside the handler raised an unexpected exception), the response must be 500 — not
            // 400. Mixing user-input failures with server-internal failures into one status defeats
            // the dry-run's diagnostic purpose.
            BatchScheduleClient client = mock(BatchScheduleClient.class);
            when(client.requestTargets(anyInt())).thenReturn(
                    Mono.error(new RuntimeException("simulated coordinator failure")));
            DispatcherInspectionHandler handler = handlerWith(true, client);

            ObjectNode body = mapper.createObjectNode();
            body.putArray("prompt_batch").add("a");
            MockServerRequest req = MockServerRequest.builder()
                    .method(HttpMethod.POST)
                    .uri(URI.create("http://x/dispatcher/_dryrun/batch_infer"))
                    .queryParam("pre_assign", "true")
                    .body(Mono.just(body));

            assertResponse(handler.dryRun(req), HttpStatus.INTERNAL_SERVER_ERROR, out -> {
                assertEquals("dryrun_internal_error", out.get("error").asText());
                assertTrue(out.get("message").asText().contains("simulated coordinator failure"));
            });
        }
    }

    // ───────────────────────── helpers ───────────────────────

    private ObjectNode invokeSnapshot(List<String> urls, FeHealthChecker hc) {
        DispatchConfig cfg = new DispatchConfig();
        cfg.setFePoolServiceId(SERVICE_ID);
        cfg.setSubBatch("size:5");
        cfg.setSubBatchSpec(SubBatchSpec.parse("size:5"));
        DispatcherFePoolRefresher refresher = mock(DispatcherFePoolRefresher.class);
        when(refresher.source()).thenReturn(() -> urls);
        BatchScheduleClient client = mock(BatchScheduleClient.class);
        DispatcherInspectionHandler handler =
                new DispatcherInspectionHandler(cfg, refresher, hc, client, mapper);

        MockServerRequest req = MockServerRequest.builder()
                .method(HttpMethod.GET)
                .uri(URI.create("http://x/dispatcher/_snapshot"))
                .build();

        ObjectNode[] captured = new ObjectNode[1];
        StepVerifier.create(handler.snapshot(req))
                .assertNext(resp -> {
                    assertEquals(HttpStatus.OK, resp.statusCode());
                    captured[0] = (ObjectNode) ((EntityResponse<?>) resp).entity();
                })
                .verifyComplete();
        return captured[0];
    }

    private DispatcherInspectionHandler handlerWith(boolean preAssignBeDefault, BatchScheduleClient client) {
        DispatchConfig cfg = new DispatchConfig();
        cfg.setFePoolServiceId("test.fe.publish");
        cfg.setSubBatch("size:2");
        cfg.setSubBatchSpec(SubBatchSpec.parse("size:2"));
        cfg.setPreAssignBe(preAssignBeDefault);
        DispatcherFePoolRefresher refresher = mock(DispatcherFePoolRefresher.class);
        when(refresher.source()).thenReturn(() -> List.<String>of());
        FeHealthChecker hc = mock(FeHealthChecker.class);
        return new DispatcherInspectionHandler(cfg, refresher, hc, client, mapper);
    }

    /**
     * @param preAssignParam pass {@code "true"} or {@code "false"} to set the query param
     *                       explicitly, or {@code null} to omit it (handler then falls back to
     *                       config default). {@link MockServerRequest} does not parse URI query
     *                       strings — query params must be set on the builder directly.
     */
    private ObjectNode invokeDryRun(DispatcherInspectionHandler handler, String preAssignParam,
                                    List<String> prompts) {
        ObjectNode body = mapper.createObjectNode();
        ArrayNode arr = body.putArray("prompt_batch");
        prompts.forEach(arr::add);

        MockServerRequest.Builder builder = MockServerRequest.builder()
                .method(HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/_dryrun/batch_infer"));
        if (preAssignParam != null) {
            builder.queryParam("pre_assign", preAssignParam);
        }
        MockServerRequest req = builder.body(Mono.just(body));

        ObjectNode[] captured = new ObjectNode[1];
        StepVerifier.create(handler.dryRun(req))
                .assertNext(resp -> {
                    assertEquals(HttpStatus.OK, resp.statusCode());
                    captured[0] = (ObjectNode) ((EntityResponse<?>) resp).entity();
                })
                .verifyComplete();
        return captured[0];
    }

    private void assertResponse(Mono<org.springframework.web.reactive.function.server.ServerResponse> mono,
                                HttpStatus expectedStatus, Consumer<ObjectNode> bodyAssertions) {
        StepVerifier.create(mono)
                .assertNext(resp -> {
                    assertEquals(expectedStatus, resp.statusCode());
                    EntityResponse<?> entity = (EntityResponse<?>) resp;
                    bodyAssertions.accept((ObjectNode) entity.entity());
                })
                .verifyComplete();
    }

    private static BatchScheduleTarget target(String ip) {
        return new BatchScheduleTarget(ip, 23840, 23841, RoleType.PDFUSION);
    }
}
