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
import java.nio.charset.StandardCharsets;
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
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * Unit tests for the combined inspection surface — {@code GET /dispatcher/_snapshot} and
 * {@code POST /dispatcher/_dryrun/<path>}. Mocks {@link FeHealthChecker},
 * {@link DispatcherFePoolRefresher}, and {@link BatchScheduleClient} directly so each case can
 * inject a known fixture and assert on the JSON response.
 *
 * <p>The handler emits its response as raw bytes (fastjson2 serialization). Tests parse those
 * bytes back into Jackson trees so the existing assertion idioms ({@code body.get("...").asText()})
 * stay readable — Jackson is no longer the production codec but it's still the most ergonomic
 * tree API for test assertions.
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

            assertTrue(hosts.get(0).get("alive").asBoolean());
            assertEquals(1, hosts.get(0).get("consecFails").asInt());
            assertFalse(hosts.get(1).get("alive").asBoolean());
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
                    "even an empty hosts list must be a JSON array — never null or absent");
        }

        @Test
        void serviceIdEchoesConfigVerbatim() {
            DispatchConfig cfg = new DispatchConfig();
            cfg.setFePoolServiceId("very.specific.weird.name.publish");
            cfg.setSubBatch("size:5");
            cfg.setSubBatchSpec(SubBatchSpec.parse("size:5"));
            DispatcherFePoolRefresher refresher = mock(DispatcherFePoolRefresher.class);
            when(refresher.source()).thenReturn(() -> List.<String>of());
            FeHealthChecker hc = mock(FeHealthChecker.class);
            BatchScheduleClient client = mock(BatchScheduleClient.class);
            DispatcherInspectionHandler handler =
                    new DispatcherInspectionHandler(cfg, refresher, hc, client);

            MockServerRequest req = MockServerRequest.builder()
                    .method(HttpMethod.GET)
                    .uri(URI.create("http://x/dispatcher/_snapshot"))
                    .build();

            StepVerifier.create(handler.snapshot(req))
                    .assertNext(resp -> {
                        assertEquals(HttpStatus.OK, resp.statusCode());
                        ObjectNode body = parseBody(resp);
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
            BatchScheduleClient client = mock(BatchScheduleClient.class);
            DispatcherInspectionHandler handler = handlerWith(false, client);

            MockServerRequest req = MockServerRequest.builder()
                    .method(HttpMethod.POST)
                    .uri(URI.create("http://x/dispatcher/_dryrun/totally_made_up"))
                    .body(Mono.just("{}".getBytes(StandardCharsets.UTF_8)));

            assertResponse(handler.dryRun(req), HttpStatus.BAD_REQUEST, body -> {
                assertEquals("invalid_inspection_request", body.get("error").asText());
                String msg = body.get("message").asText();
                assertTrue(msg.contains("/batch_infer"),
                        "error message must enumerate registered paths so caller can fix the URL");
                assertTrue(msg.contains("/v1/embeddings"));
            });
        }

        @Test
        void emptyBodyReturns400InsteadOfEmptyResponse() {
            BatchScheduleClient client = mock(BatchScheduleClient.class);
            DispatcherInspectionHandler handler = handlerWith(false, client);

            MockServerRequest req = MockServerRequest.builder()
                    .method(HttpMethod.POST)
                    .uri(URI.create("http://x/dispatcher/_dryrun/batch_infer"))
                    .body(Mono.<byte[]>empty());

            assertResponse(handler.dryRun(req), HttpStatus.BAD_REQUEST, body ->
                    assertEquals("invalid_inspection_request", body.get("error").asText()));
        }

        @Test
        void emptyArrayShortCircuitsWithoutCallingBatchScheduleClient() {
            BatchScheduleClient client = mock(BatchScheduleClient.class);
            DispatcherInspectionHandler handler = handlerWith(true, client);

            byte[] body = "{\"prompt_batch\":[]}".getBytes(StandardCharsets.UTF_8);

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
                        "requested && supported holds for /batch_infer, so preAssignEffective stays true even when the empty batch resolves no targets");
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
        void noQueryParamIsSideEffectFreeEvenWhenConfigDefaultIsOn() {
            // A diagnostic must not perturb production: resolving BE targets advances master's RR
            // cursor, so an unqualified dry-run never does it — not even when preAssignBe=true.
            BatchScheduleClient client = mock(BatchScheduleClient.class);
            when(client.requestTargets(anyInt())).thenReturn(Mono.just(List.of(target("10.0.0.1"))));
            DispatcherInspectionHandler handler = handlerWith(true, client);

            ObjectNode out = invokeDryRun(handler, null, List.of("a"));
            assertTrue(out.get("preAssignConfigDefault").asBoolean(),
                    "the config default is still reported for operators");
            assertFalse(out.get("preAssignEffective").asBoolean(),
                    "absent pre_assign must not resolve targets, whatever the config default says");
            verifyNoInteractions(client);
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
            assertEquals("10.0.0.1", out.get("preAssignTargets").get(0).get("ip").asText());
            assertEquals("10.0.0.2", out.get("preAssignTargets").get(1).get("ip").asText());
        }

        @Test
        void preAssignFalseLeavesChunksUnstamped() {
            BatchScheduleClient client = mock(BatchScheduleClient.class);
            DispatcherInspectionHandler handler = handlerWith(true, client);

            ObjectNode out = invokeDryRun(handler, "false", List.of("a", "b"));
            for (JsonNode chunk : (ArrayNode) out.get("chunks")) {
                // buildChunkBodies guarantees generate_config exists on every chunk.
                JsonNode gc = chunk.get("generate_config");
                assertNull(gc.get("role_addrs"),
                        "preAssign=false must not stamp role_addrs even though config default is true");
                assertTrue(gc.get("force_batch").asBoolean(),
                        "force_batch still injected — that's independent of preAssign");
            }
            verify(client, never()).requestTargets(anyInt());
        }

        @Test
        void preAssignRequestedOnNonPreAssignableEndpointStaysIneffective() {
            BatchScheduleClient client = mock(BatchScheduleClient.class);
            DispatcherInspectionHandler handler = handlerWith(true, client);

            // /v1/embeddings is not pre-assignable (FE's embeddings pydantic model drops unknown
            // top-level fields, so a stamped generate_config would be a dead write). Requesting
            // pre_assign=true must not make it effective and must not consume a /batch_schedule
            // round-trip — that would advance master's RR cursor for a write FE never reads.
            byte[] body = "{\"input\":[\"a\",\"b\",\"c\"]}".getBytes(StandardCharsets.UTF_8);
            MockServerRequest req = MockServerRequest.builder()
                    .method(HttpMethod.POST)
                    .uri(URI.create("http://x/dispatcher/_dryrun/v1/embeddings"))
                    .queryParam("pre_assign", "true")
                    .body(Mono.just(body));

            assertResponse(handler.dryRun(req), HttpStatus.OK, out -> {
                assertFalse(out.get("preAssignSupported").asBoolean(),
                        "/v1/embeddings must report itself as not pre-assignable");
                assertFalse(out.get("preAssignEffective").asBoolean(),
                        "preAssignEffective = requested AND supported; an unsupported endpoint stays false");
                assertEquals(0, out.get("preAssignTargets").size());
                assertTrue(out.get("chunkCount").asInt() > 0,
                        "the embedding batch itself still splits normally");
            });
            verify(client, never()).requestTargets(anyInt());
        }

        @Test
        void nonObjectBodyReturns400() {
            BatchScheduleClient client = mock(BatchScheduleClient.class);
            DispatcherInspectionHandler handler = handlerWith(false, client);

            MockServerRequest req = MockServerRequest.builder()
                    .method(HttpMethod.POST)
                    .uri(URI.create("http://x/dispatcher/_dryrun/batch_infer"))
                    .body(Mono.just("[\"a\"]".getBytes(StandardCharsets.UTF_8)));

            assertResponse(handler.dryRun(req), HttpStatus.BAD_REQUEST, body ->
                    assertEquals("invalid_inspection_request", body.get("error").asText()));
        }

        @Test
        void missingArrayFieldReportsPassthrough() {
            BatchScheduleClient client = mock(BatchScheduleClient.class);
            DispatcherInspectionHandler handler = handlerWith(false, client);

            // An absent array field is not an error: production passthrough-forwards the body
            // whole to one FE (BatchHandler treats arr==null as non-splittable), so dry-run must
            // report that same disposition rather than a 400.
            byte[] body = "{\"not_prompt_batch\":\"x\"}".getBytes(StandardCharsets.UTF_8);
            MockServerRequest req = MockServerRequest.builder()
                    .method(HttpMethod.POST)
                    .uri(URI.create("http://x/dispatcher/_dryrun/batch_infer"))
                    .body(Mono.just(body));

            assertResponse(handler.dryRun(req), HttpStatus.OK, out -> {
                assertEquals("passthrough", out.get("disposition").asText());
                assertEquals(0, out.get("chunkCount").asInt());
                assertEquals(0, out.get("totalItems").asInt());
            });
        }

        @Test
        void wholeBodyCompanionFieldReportsPassthrough() {
            BatchScheduleClient client = mock(BatchScheduleClient.class);
            DispatcherInspectionHandler handler = handlerWith(false, client);

            // prompt_batch carrying sample-aligned top-level images is passthrough-forwarded whole
            // in production (requiresWholeBody) — splitting would mis-align the companion field.
            // Dry-run must mirror that instead of fabricating chunks.
            byte[] body = ("{\"prompt_batch\":[\"a\",\"b\"],"
                    + "\"images\":[[\"http://x/0.png\"],[\"http://x/1.png\"]]}")
                    .getBytes(StandardCharsets.UTF_8);
            MockServerRequest req = MockServerRequest.builder()
                    .method(HttpMethod.POST)
                    .uri(URI.create("http://x/dispatcher/_dryrun/batch_infer"))
                    .body(Mono.just(body));

            assertResponse(handler.dryRun(req), HttpStatus.OK, out -> {
                assertEquals("passthrough", out.get("disposition").asText());
                assertEquals(0, out.get("chunkCount").asInt());
                assertEquals(2, out.get("totalItems").asInt());
            });
        }

        @Test
        void nonSplittableEmbeddingInputReportsPassthrough() {
            BatchScheduleClient client = mock(BatchScheduleClient.class);
            DispatcherInspectionHandler handler = handlerWith(true, client);

            // /v1/embeddings input given as a single multimodal input (array of objects) is ONE
            // input, not a batch — production passthrough-forwards it whole, so dry-run must
            // report that disposition instead of fabricating per-element chunks.
            byte[] body = "{\"input\":[{\"type\":\"text\",\"text\":\"hi\"}]}".getBytes(StandardCharsets.UTF_8);
            MockServerRequest req = MockServerRequest.builder()
                    .method(HttpMethod.POST)
                    .uri(URI.create("http://x/dispatcher/_dryrun/v1/embeddings"))
                    .queryParam("pre_assign", "true")
                    .body(Mono.just(body));

            assertResponse(handler.dryRun(req), HttpStatus.OK, out -> {
                assertEquals("passthrough", out.get("disposition").asText());
                assertEquals(0, out.get("chunkCount").asInt());
                assertEquals(1, out.get("totalItems").asInt());
            });
            verify(client, never()).requestTargets(anyInt());
        }

        @Test
        void internalErrorReturns500NotBadRequest() {
            BatchScheduleClient client = mock(BatchScheduleClient.class);
            when(client.requestTargets(anyInt())).thenReturn(
                    Mono.error(new RuntimeException("simulated coordinator failure")));
            DispatcherInspectionHandler handler = handlerWith(true, client);

            byte[] body = "{\"prompt_batch\":[\"a\"]}".getBytes(StandardCharsets.UTF_8);
            MockServerRequest req = MockServerRequest.builder()
                    .method(HttpMethod.POST)
                    .uri(URI.create("http://x/dispatcher/_dryrun/batch_infer"))
                    .queryParam("pre_assign", "true")
                    .body(Mono.just(body));

            assertResponse(handler.dryRun(req), HttpStatus.INTERNAL_SERVER_ERROR, out -> {
                assertEquals("dryrun_internal_error", out.get("error").asText());
                assertFalse(out.get("message").asText().contains("simulated coordinator failure"),
                        "the upstream exception text must stay in the log, not the client body");
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
                new DispatcherInspectionHandler(cfg, refresher, hc, client);

        MockServerRequest req = MockServerRequest.builder()
                .method(HttpMethod.GET)
                .uri(URI.create("http://x/dispatcher/_snapshot"))
                .build();

        ObjectNode[] captured = new ObjectNode[1];
        StepVerifier.create(handler.snapshot(req))
                .assertNext(resp -> {
                    assertEquals(HttpStatus.OK, resp.statusCode());
                    captured[0] = parseBody(resp);
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
        return new DispatcherInspectionHandler(cfg, refresher, hc, client);
    }

    private ObjectNode invokeDryRun(DispatcherInspectionHandler handler, String preAssignParam,
                                    List<String> prompts) {
        StringBuilder sb = new StringBuilder("{\"prompt_batch\":[");
        for (int i = 0; i < prompts.size(); i++) {
            if (i > 0) {
                sb.append(",");
            }
            sb.append("\"").append(prompts.get(i)).append("\"");
        }
        sb.append("]}");
        byte[] body = sb.toString().getBytes(StandardCharsets.UTF_8);

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
                    captured[0] = parseBody(resp);
                })
                .verifyComplete();
        return captured[0];
    }

    private void assertResponse(Mono<org.springframework.web.reactive.function.server.ServerResponse> mono,
                                HttpStatus expectedStatus, Consumer<ObjectNode> bodyAssertions) {
        StepVerifier.create(mono)
                .assertNext(resp -> {
                    assertEquals(expectedStatus, resp.statusCode());
                    bodyAssertions.accept(parseBody(resp));
                })
                .verifyComplete();
    }

    private ObjectNode parseBody(org.springframework.web.reactive.function.server.ServerResponse resp) {
        EntityResponse<?> entity = (EntityResponse<?>) resp;
        Object value = entity.entity();
        try {
            if (value instanceof byte[] bytes) {
                return (ObjectNode) mapper.readTree(bytes);
            }
            throw new IllegalStateException("unexpected entity type: " + value.getClass());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static BatchScheduleTarget target(String ip) {
        return new BatchScheduleTarget(ip, 23840, 23841, RoleType.PDFUSION);
    }
}
