package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.http.MediaType;
import org.springframework.http.server.reactive.ReactorHttpHandlerAdapter;
import org.springframework.test.web.reactive.server.WebTestClient;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.RouterFunctions;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.netty.DisposableServer;
import reactor.netty.http.server.HttpServer;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * End-to-end test for the assembled dispatcher: real {@link DispatchRouter} on top of real
 * {@link org.flexlb.dispatcher.BatchHandler}, real
 * {@link org.flexlb.dispatcher.FanoutService}, real {@link FeClient},
 * and a real {@link FePool} round-robin-ing across three {@link MockWebServer}s. Every batch
 * endpoint registered by
 * {@link org.flexlb.dispatcher.BatchEndpointSpec#SPECS} is exercised with
 * one chunk induced to fail (HTTP 500), and the non-batch path falls through to passthrough.
 */
@Timeout(30)
class DispatcherE2ETest {

    private MockWebServer fe1;
    private MockWebServer fe2;
    private MockWebServer fe3;
    private DisposableServer dispatcherServer;
    private reactor.netty.resources.ConnectionProvider feConnectionProvider;
    private final ObjectMapper mapper = new ObjectMapper();

    @BeforeEach
    void up() throws Exception {
        fe1 = new MockWebServer();
        fe1.start();
        fe2 = new MockWebServer();
        fe2.start();
        fe3 = new MockWebServer();
        fe3.start();
    }

    @AfterEach
    void down() throws Exception {
        if (dispatcherServer != null) {
            dispatcherServer.disposeNow();
            dispatcherServer = null;
        }
        if (feConnectionProvider != null) {
            feConnectionProvider.disposeLater().block(java.time.Duration.ofSeconds(5));
            feConnectionProvider = null;
        }
        fe1.shutdown();
        fe2.shutdown();
        fe3.shutdown();
    }

    @Test
    void openAiBatchChatFourSplitsTwoWithSecondChunkFailure() throws Exception {
        // chunk0 → fe1: ok; chunk1 → fe2: 500.
        fe1.enqueue(jsonResponse(200,
                "{\"responses\":[{\"id\":\"a\"},{\"id\":\"b\"}]}"));
        fe2.enqueue(new MockResponse().setResponseCode(500).setBody("boom"));

        WebTestClient client = buildClient(/*subBatchSize=*/2);

        ObjectNode body = mapper.createObjectNode();
        body.put("model", "qwen");
        var requests = body.putArray("requests");
        for (int i = 0; i < 4; i++) {
            ObjectNode req = requests.addObject();
            req.put("custom_id", "c" + i);
            req.putArray("messages").addObject().put("role", "user").put("content", "hi");
        }

        JsonNode resp = client.post().uri("/dispatcher/v1/batch/chat/completions")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .exchange()
                .expectStatus().isOk()
                .expectBody(JsonNode.class)
                .returnResult().getResponseBody();

        assertNotNull(resp);
        JsonNode arr = resp.get("responses");
        assertNotNull(arr);
        assertEquals(4, arr.size());
        // Successful chunk: positions 0..1 forwarded verbatim from fe1.
        assertEquals("a", arr.get(0).get("id").asText());
        assertEquals("b", arr.get(1).get("id").asText());
        // Failed chunk: positions 2..3 carry OPENAI_ERROR shape {index, error: {code, message}}.
        for (int i = 2; i <= 3; i++) {
            JsonNode item = arr.get(i);
            assertEquals(i, item.get("index").asInt());
            JsonNode err = item.get("error");
            assertNotNull(err, "expected error object at index " + i);
            assertEquals("dispatcher_sub_batch_failed", err.get("code").asText());
            assertFalse(err.get("message").asText().isBlank());
        }
        JsonNode pf = resp.get("_partial_failure");
        assertNotNull(pf);
        assertEquals(2, pf.get("failed_count").asInt());
        assertEquals(4, pf.get("total_count").asInt());
        List<Integer> failedIndices = new ArrayList<>();
        pf.get("failed_indices").forEach(n -> failedIndices.add(n.asInt()));
        assertEquals(List.of(2, 3), failedIndices);

        // Exactly two FE calls — fe3 never contacted.
        assertChunkRequest(fe1.takeRequest(), "/v1/batch/chat/completions", "requests", 2);
        assertChunkRequest(fe2.takeRequest(), "/v1/batch/chat/completions", "requests", 2);
        assertEquals(0, fe3.getRequestCount());
    }

    @Test
    void embeddingsSixSplitsThreeWithMiddleChunkFailureRenumbersAndSumsUsage() throws Exception {
        // chunk0 → fe1: ok; chunk1 → fe2: 500; chunk2 → fe3: ok.
        // Each successful sub-body uses local indices 0,1 and reports usage 4 / 4. The post-merger
        // must renumber to absolute indices and sum usage across only the successful chunks.
        fe1.enqueue(jsonResponse(200,
                "{\"data\":[{\"index\":0,\"embedding\":[1.0,2.0]},{\"index\":1,\"embedding\":[3.0,4.0]}],"
                        + "\"usage\":{\"prompt_tokens\":4,\"total_tokens\":4}}"));
        fe2.enqueue(new MockResponse().setResponseCode(500).setBody("boom"));
        fe3.enqueue(jsonResponse(200,
                "{\"data\":[{\"index\":0,\"embedding\":[5.0,6.0]},{\"index\":1,\"embedding\":[7.0,8.0]}],"
                        + "\"usage\":{\"prompt_tokens\":4,\"total_tokens\":4}}"));

        WebTestClient client = buildClient(/*subBatchSize=*/2);

        ObjectNode body = mapper.createObjectNode();
        body.put("model", "qwen-embed");
        var input = body.putArray("input");
        for (int i = 0; i < 6; i++) {
            input.add("text-" + i);
        }

        JsonNode resp = client.post().uri("/dispatcher/v1/embeddings")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .exchange()
                .expectStatus().isOk()
                .expectBody(JsonNode.class)
                .returnResult().getResponseBody();

        assertNotNull(resp);
        JsonNode data = resp.get("data");
        assertNotNull(data);
        assertEquals(6, data.size());
        // Indices must be renumbered to 0..5 absolute.
        for (int i = 0; i < 6; i++) {
            assertEquals(i, data.get(i).get("index").asInt(), "index at position " + i);
        }
        // Successful positions retain their embeddings.
        assertEquals(1.0, data.get(0).get("embedding").get(0).asDouble());
        assertEquals(4.0, data.get(1).get("embedding").get(1).asDouble());
        assertEquals(5.0, data.get(4).get("embedding").get(0).asDouble());
        assertEquals(8.0, data.get(5).get("embedding").get(1).asDouble());
        // Failed positions: embedding is null, plus an error field (FailedItemFactory.EMBEDDING_NULL).
        for (int i = 2; i <= 3; i++) {
            JsonNode item = data.get(i);
            assertTrue(item.get("embedding").isNull(), "expected null embedding at index " + i);
            assertNotNull(item.get("error"), "expected error reason at index " + i);
            assertFalse(item.get("error").asText().isBlank());
        }
        // Usage summed across successful sub-bodies only (4 + 4).
        JsonNode usage = resp.get("usage");
        assertNotNull(usage);
        assertEquals(8L, usage.get("prompt_tokens").asLong());
        assertEquals(8L, usage.get("total_tokens").asLong());
        // Partial failure metadata.
        JsonNode pf = resp.get("_partial_failure");
        assertNotNull(pf);
        assertEquals(2, pf.get("failed_count").asInt());
        assertEquals(6, pf.get("total_count").asInt());
        List<Integer> failedIndices = new ArrayList<>();
        pf.get("failed_indices").forEach(n -> failedIndices.add(n.asInt()));
        assertEquals(List.of(2, 3), failedIndices);

        assertChunkRequest(fe1.takeRequest(), "/v1/embeddings", "input", 2);
        assertChunkRequest(fe2.takeRequest(), "/v1/embeddings", "input", 2);
        assertChunkRequest(fe3.takeRequest(), "/v1/embeddings", "input", 2);
    }

    @Test
    void registeredPathWithNonBatchBodyFallsThroughToSingleFe() throws Exception {
        // /v1/embeddings is a registered batch endpoint, but OpenAI also allows `input` as a
        // plain string — that request must reach exactly one FE verbatim, not die with 400.
        String upstream = "{\"object\":\"list\",\"data\":[{\"index\":0,\"embedding\":[0.1,0.2]}]}";
        fe1.enqueue(jsonResponse(200, upstream));

        WebTestClient client = buildClient(/*subBatchSize=*/2);

        ObjectNode body = mapper.createObjectNode();
        body.put("model", "embed-model");
        body.put("input", "hello world");

        byte[] raw = client.post().uri("/dispatcher/v1/embeddings")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .exchange()
                .expectStatus().isOk()
                .expectBody(byte[].class).returnResult().getResponseBody();
        assertNotNull(raw);
        JsonNode resp = mapper.readTree(raw);
        assertEquals("list", resp.get("object").asText());
        assertNull(resp.get("_partial_failure"));

        assertEquals(1, fe1.getRequestCount() + fe2.getRequestCount() + fe3.getRequestCount(),
                "non-batch body must hit exactly one FE");
        RecordedRequest rec = fe1.takeRequest(5, java.util.concurrent.TimeUnit.SECONDS);
        assertNotNull(rec);
        JsonNode forwarded = mapper.readTree(rec.getBody().readUtf8());
        assertEquals("hello world", forwarded.get("input").asText(),
                "body must be forwarded verbatim, single-string input intact");
    }

    @Test
    void nonBatchPathFallsThroughPassthroughVerbatim() throws Exception {
        String upstream = "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion\","
                + "\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"hi\"}}]}";
        fe1.enqueue(jsonResponse(200, upstream));

        // Any subBatchSize works — passthrough doesn't use it.
        WebTestClient client = buildClient(/*subBatchSize=*/3);

        ObjectNode body = mapper.createObjectNode();
        body.put("model", "qwen");
        body.putArray("messages").addObject().put("role", "user").put("content", "hi");

        byte[] raw = client.post().uri("/dispatcher/v1/chat/completions")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .exchange()
                .expectStatus().isOk()
                .expectBody(byte[].class).returnResult().getResponseBody();
        assertNotNull(raw, "passthrough response body must not be null");
        JsonNode resp = mapper.readTree(raw);
        assertEquals("chatcmpl-1", resp.get("id").asText());
        assertEquals("chat.completion", resp.get("object").asText());
        assertEquals("hi", resp.get("choices").get(0).get("message").get("content").asText());
        // Dispatcher must not have stamped any _partial_failure on passthrough payloads.
        assertNull(resp.get("_partial_failure"));

        RecordedRequest rec = fe1.takeRequest();
        // Prefix stripped.
        assertEquals("/v1/chat/completions", rec.getPath());
        assertEquals("POST", rec.getMethod());
        // Body forwarded verbatim.
        JsonNode forwarded = mapper.readTree(rec.getBody().readUtf8());
        assertEquals("qwen", forwarded.get("model").asText());
        assertEquals("hi", forwarded.get("messages").get(0).get("content").asText());

        // Other FEs untouched.
        assertEquals(0, fe2.getRequestCount());
        assertEquals(0, fe3.getRequestCount());
    }

    @Test
    void preAssignBeStampsTargetsIntoGenerateConfigRoleAddrs() throws Exception {
        fe1.enqueue(jsonResponse(200,
                "{\"response_batch\":[{\"response\":\"r0\"},{\"response\":\"r1\"},{\"response\":\"r2\"}]}"));
        fe2.enqueue(jsonResponse(200,
                "{\"response_batch\":[{\"response\":\"r3\"},{\"response\":\"r4\"},{\"response\":\"r5\"}]}"));
        fe3.enqueue(jsonResponse(200,
                "{\"response_batch\":[{\"response\":\"r6\"},{\"response\":\"r7\"},{\"response\":\"r8\"}]}"));

        // Mock master /batch_schedule returning 3 BE targets, all PDFUSION (single-role cluster).
        // Dispatcher stamps them into each chunk's generate_config.role_addrs so FE's existing
        // role_addrs-aware path (rtp_llm.server.backend_rpc_server_visitor.route_ips) skips the
        // /schedule round-trip — no FE-side change required.
        List<org.flexlb.dao.loadbalance.BatchScheduleTarget> targets = List.of(
                new org.flexlb.dao.loadbalance.BatchScheduleTarget("10.0.0.1", 23840, 23841,
                        org.flexlb.dao.route.RoleType.PDFUSION),
                new org.flexlb.dao.loadbalance.BatchScheduleTarget("10.0.0.2", 23840, 23841,
                        org.flexlb.dao.route.RoleType.PDFUSION),
                new org.flexlb.dao.loadbalance.BatchScheduleTarget("10.0.0.3", 23840, 23841,
                        org.flexlb.dao.route.RoleType.PDFUSION));
        WebTestClient client = buildClient(/*subBatchSize=*/3, /*preAssignBe=*/true, targets);

        ObjectNode body = mapper.createObjectNode();
        body.put("model", "qwen");
        var pb = body.putArray("prompt_batch");
        for (int i = 0; i < 9; i++) {
            pb.add("p" + i);
        }

        client.post().uri("/dispatcher/batch_infer")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .exchange()
                .expectStatus().isOk();

        // Each FE saw one chunk; each chunk's generate_config.role_addrs carries the i-th target.
        verifyChunkHasRoleAddr(fe1.takeRequest(), "10.0.0.1");
        verifyChunkHasRoleAddr(fe2.takeRequest(), "10.0.0.2");
        verifyChunkHasRoleAddr(fe3.takeRequest(), "10.0.0.3");
    }

    @Test
    void emptyBodyOnBatchEndpointIsRejectedWith400() {
        WebTestClient client = buildClient(/*subBatchSize=*/2);

        JsonNode resp = client.post().uri("/dispatcher/batch_infer")
                .contentType(MediaType.APPLICATION_JSON)
                .exchange()
                .expectStatus().isBadRequest()
                .expectBody(JsonNode.class)
                .returnResult().getResponseBody();

        assertNotNull(resp);
        assertEquals("invalid_batch_request", resp.get("error").asText());
        assertEquals(0, fe1.getRequestCount());
        assertEquals(0, fe2.getRequestCount());
        assertEquals(0, fe3.getRequestCount());
    }

    @Test
    void emptyBatchArrayShortCircuitsTo200EmptyEnvelopeWithoutContactingFe() throws Exception {
        // A present-but-empty batch array is answered locally: the dispatcher must not fan out a
        // zero-item request (no FE contacted) and must return 200 with an empty response array.
        WebTestClient client = buildClient(/*subBatchSize=*/2);

        ObjectNode body = mapper.createObjectNode();
        body.put("model", "qwen");
        body.putArray("prompt_batch");

        JsonNode resp = client.post().uri("/dispatcher/batch_infer")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .exchange()
                .expectStatus().isOk()
                .expectBody(JsonNode.class)
                .returnResult().getResponseBody();

        assertNotNull(resp);
        JsonNode arr = resp.get("response_batch");
        assertNotNull(arr, "empty batch must still carry the response array field");
        assertTrue(arr.isArray());
        assertEquals(0, arr.size());
        assertNull(resp.get("_partial_failure"));
        // Empty batch is answered locally — no FE round-trip.
        assertEquals(0, fe1.getRequestCount());
        assertEquals(0, fe2.getRequestCount());
        assertEquals(0, fe3.getRequestCount());
    }

    @Test
    void allChunksFailWithSharedClientErrorSurfaces4xxNot500() throws Exception {
        // 4 items at size:2 → 2 chunks → fe1 + fe2, both returning 400. The dispatcher must surface
        // the shared client-error status (400), not collapse it to 500 — exercising the real
        // WebClientResponseException → httpStatusOf → feStatus → commonErrorStatus → errorStatus chain
        // end to end (the unit test pins the merge vote with a hand-built SubBatchResult; this pins
        // the transport glue that feeds it).
        fe1.enqueue(new MockResponse().setResponseCode(400).setBody("bad request"));
        fe2.enqueue(new MockResponse().setResponseCode(400).setBody("bad request"));

        WebTestClient client = buildClient(/*subBatchSize=*/2);

        ObjectNode body = mapper.createObjectNode();
        body.put("model", "qwen");
        var requests = body.putArray("requests");
        for (int i = 0; i < 4; i++) {
            ObjectNode req = requests.addObject();
            req.put("custom_id", "c" + i);
            req.putArray("messages").addObject().put("role", "user").put("content", "hi");
        }

        JsonNode resp = client.post().uri("/dispatcher/v1/batch/chat/completions")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .exchange()
                .expectStatus().isBadRequest()
                .expectBody(JsonNode.class)
                .returnResult().getResponseBody();

        assertNotNull(resp);
        assertEquals("all_sub_batches_failed", resp.get("error").asText());
        assertEquals(4, resp.get("failed_count").asInt());
        assertEquals(2, resp.get("total_chunks").asInt());
        assertTrue(resp.get("failed_reasons").isArray() && resp.get("failed_reasons").size() > 0,
                "all-failed body must carry at least one failure reason");
        // Both failing chunks landed on fe1/fe2; fe3 never contacted.
        assertEquals(0, fe3.getRequestCount());
    }

    @Test
    void snapshotEndpointReturnsCurrentFePoolThroughRealRouter() {
        // Wires the real DispatcherSnapshotHandler through the real DispatchRouter and HTTP
        // transport to verify the route is registered (and not shadowed by the /dispatcher/**
        // passthrough catch-all that comes after it in the registration order).
        WebTestClient client = buildClient(/*subBatchSize=*/2);

        JsonNode resp = client.get().uri("/dispatcher/_snapshot")
                .exchange()
                .expectStatus().isOk()
                .expectHeader().contentTypeCompatibleWith(MediaType.APPLICATION_JSON)
                .expectBody(JsonNode.class)
                .returnResult().getResponseBody();

        assertNotNull(resp);
        JsonNode fePool = resp.get("fePool");
        assertNotNull(fePool);
        assertEquals("e2e.fe.publish", fePool.get("serviceId").asText());
        assertEquals(3, fePool.get("size").asInt());
        JsonNode hosts = fePool.get("hosts");
        assertEquals(3, hosts.size());
        // Order matches the supplier's snapshot order — round-robin order.
        assertEquals(baseUrl(fe1), hosts.get(0).get("url").asText());
        assertEquals(baseUrl(fe2), hosts.get(1).get("url").asText());
        assertEquals(baseUrl(fe3), hosts.get(2).get("url").asText());
        for (int i = 0; i < 3; i++) {
            assertTrue(hosts.get(i).get("alive").asBoolean());
            assertEquals(0, hosts.get(i).get("consecFails").asInt());
        }
        // No FE traffic — snapshot reads dispatcher-local state only.
        assertEquals(0, fe1.getRequestCount());
        assertEquals(0, fe2.getRequestCount());
        assertEquals(0, fe3.getRequestCount());
    }

    @Test
    void dryRunEndpointReturnsStampedChunksThroughRealRouter() throws Exception {
        // End-to-end proof that POST /dispatcher/_dryrun/<path> is routed through HTTP transport,
        // delegates to the real BatchScheduleClient for pre-assign, and returns chunk bodies
        // byte-equivalent to what the real fanout would have written to FE — without actually
        // calling any FE.
        List<org.flexlb.dao.loadbalance.BatchScheduleTarget> targets = List.of(
                new org.flexlb.dao.loadbalance.BatchScheduleTarget("10.0.0.1", 23840, 23841,
                        org.flexlb.dao.route.RoleType.PDFUSION),
                new org.flexlb.dao.loadbalance.BatchScheduleTarget("10.0.0.2", 23840, 23841,
                        org.flexlb.dao.route.RoleType.PDFUSION));
        WebTestClient client = buildClient(/*subBatchSize=*/2, /*preAssignBe=*/true, targets);

        ObjectNode body = mapper.createObjectNode();
        body.put("model", "qwen");
        var pb = body.putArray("prompt_batch");
        for (int i = 0; i < 3; i++) {
            pb.add("p" + i);
        }

        JsonNode resp = client.post().uri("/dispatcher/_dryrun/batch_infer")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .exchange()
                .expectStatus().isOk()
                .expectBody(JsonNode.class)
                .returnResult().getResponseBody();

        assertNotNull(resp);
        assertEquals("/batch_infer", resp.get("path").asText());
        assertEquals(3, resp.get("totalItems").asInt());
        assertEquals(2, resp.get("chunkCount").asInt(), "3 items at size:2 → 2 chunks");
        assertTrue(resp.get("preAssignEffective").asBoolean());
        assertEquals(2, resp.get("preAssignTargets").size());

        JsonNode chunks = resp.get("chunks");
        assertEquals(2, chunks.size());
        for (int i = 0; i < 2; i++) {
            JsonNode chunk = chunks.get(i);
            assertEquals("qwen", chunk.get("model").asText(),
                    "envelope top-level fields preserved per chunk");
            JsonNode gc = chunk.get("generate_config");
            assertTrue(gc.get("force_batch").asBoolean());
            JsonNode roleAddrs = gc.get("role_addrs");
            assertEquals(1, roleAddrs.size());
            assertEquals("10.0.0." + (i + 1), roleAddrs.get(0).get("ip").asText());
        }
        // Dry-run must not touch any FE.
        assertEquals(0, fe1.getRequestCount());
        assertEquals(0, fe2.getRequestCount());
        assertEquals(0, fe3.getRequestCount());
    }

    private void verifyChunkHasRoleAddr(RecordedRequest rec, String expectedIp) throws Exception {
        assertNotNull(rec);
        JsonNode bodyJson = mapper.readTree(rec.getBody().readUtf8());
        // Dispatcher must NOT use a new top-level field (pydantic extra=ignore would drop it);
        // it must write into generate_config.role_addrs which FE already honors.
        assertNull(bodyJson.get("pre_assigned_be"),
                "stamping must use generate_config.role_addrs, not a top-level pre_assigned_be field");
        JsonNode roleAddrs = bodyJson.get("generate_config").get("role_addrs");
        assertNotNull(roleAddrs, "generate_config.role_addrs must be present when preAssignBe stamps");
        assertTrue(roleAddrs.isArray() && roleAddrs.size() == 1,
                "exactly one role_addr per chunk for single-role batch_schedule");
        JsonNode addr = roleAddrs.get(0);
        // Match Python rtp_llm.config.generate_config.RoleAddr field names exactly.
        assertEquals("PDFUSION", addr.get("role").asText(),
                "role enum must serialize to its name() so Python's RoleType enum parses it");
        assertEquals(expectedIp, addr.get("ip").asText());
        assertEquals(23840, addr.get("http_port").asInt());
        assertEquals(23841, addr.get("grpc_port").asInt());
    }

    private WebTestClient buildClient(int subBatchSize) {
        return buildClient(subBatchSize, /*preAssignBe=*/false, /*targets=*/List.of());
    }

    private WebTestClient buildClient(int subBatchSize,
                                      boolean preAssignBe,
                                      List<org.flexlb.dao.loadbalance.BatchScheduleTarget> targets) {
        List<String> urls = List.of(baseUrl(fe1), baseUrl(fe2), baseUrl(fe3));
        FePool pool = DispatcherTestSupport.fePool(urls);

        DispatchConfig cfg = new DispatchConfig();
        cfg.setBatchTimeoutMs(5000);
        cfg.setFePoolServiceId("e2e.fe.publish");
        cfg.setSubBatch("size:" + subBatchSize);
        cfg.setSubBatchSpec(SubBatchSpec.parse("size:" + subBatchSize));
        cfg.setPreAssignBe(preAssignBe);

        feConnectionProvider = reactor.netty.resources.ConnectionProvider.builder("e2e").build();
        FeClient feClient = new FeClient(WebClient.builder(), feConnectionProvider, cfg);
        org.flexlb.dispatcher.FanoutService fanout =
                new org.flexlb.dispatcher.FanoutService(feClient, pool, DispatcherTestSupport.noopMetrics());
        BatchScheduleClient batchScheduleClient = new BatchScheduleClient(null) {
            @Override
            public reactor.core.publisher.Mono<List<org.flexlb.dao.loadbalance.BatchScheduleTarget>> requestTargets(int count) {
                return reactor.core.publisher.Mono.just(targets);
            }
        };
        PassthroughClient passthrough =
                new PassthroughClient(WebClient.create(), pool, DispatcherTestSupport.noopMetrics(), cfg);
        org.flexlb.dispatcher.BatchHandler batchHandler =
                new org.flexlb.dispatcher.BatchHandler(fanout, cfg, batchScheduleClient, passthrough,
                        DispatcherTestSupport.noopMetrics());

        // Real inspection handler — refresher source returns the same pool URLs the router fans
        // out to, so /dispatcher/_snapshot reflects what dispatcher actually sees. FeHealthChecker
        // is mocked rather than instantiated to avoid starting the background probe loop in tests;
        // snapshot reads of isAlive/consecFails route through these stubs. The same handler also
        // serves /dispatcher/_dryrun, sharing cfg + ObjectMapper + BatchScheduleClient so its
        // ?pre_assign behavior and stamping logic exercise the same code paths as production.
        DispatcherFePoolRefresher inspectionRefresher = mock(DispatcherFePoolRefresher.class);
        when(inspectionRefresher.source()).thenReturn(() -> urls);
        FeHealthChecker hc = mock(FeHealthChecker.class);
        when(hc.isAlive(anyString())).thenReturn(true);
        when(hc.consecFails(anyString())).thenReturn(0);
        DispatcherInspectionHandler inspectionHandler =
                new DispatcherInspectionHandler(cfg, inspectionRefresher, hc, batchScheduleClient);

        List<org.flexlb.dispatcher.BatchEndpointSpec> specs =
                org.flexlb.dispatcher.BatchEndpointSpec.SPECS;
        DispatchRouter router = new DispatchRouter(
                batchHandler, passthrough, inspectionHandler,
                new org.flexlb.service.grace.ActiveRequestCounter(), specs);

        // Bind to a real Reactor Netty server (rather than WebTestClient.bindToRouterFunction's
        // in-memory connector) so the passthrough's raw DataBuffer body actually traverses an HTTP
        // transport — closer to how this runs in production, and the in-memory connector does not
        // surface the body bytes back to the test when the response is built from a lazy
        // Publisher<DataBuffer>.
        RouterFunction<ServerResponse> routes = router.routes();
        ReactorHttpHandlerAdapter adapter =
                new ReactorHttpHandlerAdapter(RouterFunctions.toHttpHandler(routes));
        dispatcherServer = HttpServer.create().port(0).handle(adapter).bindNow();

        return WebTestClient.bindToServer()
                .baseUrl("http://localhost:" + dispatcherServer.port())
                .responseTimeout(Duration.ofSeconds(10))
                .build();
    }

    private JsonNode assertChunkBatchInferRequest(RecordedRequest rec, int expectedSliceSize) throws Exception {
        JsonNode bodyJson = assertChunkRequest(rec, "/batch_infer", "prompt_batch", expectedSliceSize);
        // Verify the envelope was deep-copied: top-level "model" stays put per chunk body.
        assertEquals("qwen", bodyJson.get("model").asText());
        return bodyJson;
    }

    private JsonNode assertChunkRequest(RecordedRequest rec, String expectedPath, String arrayField,
                                        int expectedSliceSize) throws Exception {
        assertNotNull(rec);
        assertEquals("POST", rec.getMethod());
        assertEquals(expectedPath, rec.getPath());
        JsonNode bodyJson = mapper.readTree(rec.getBody().readUtf8());
        JsonNode slice = bodyJson.get(arrayField);
        assertNotNull(slice, "request body missing array field " + arrayField);
        assertTrue(slice.isArray(), "expected " + arrayField + " to be an array");
        assertEquals(expectedSliceSize, slice.size(), "chunk slice size for " + arrayField);
        return bodyJson;
    }

    private static String baseUrl(MockWebServer server) {
        return "http://" + server.getHostName() + ":" + server.getPort();
    }

    private static MockResponse jsonResponse(int code, String body) {
        return new MockResponse()
                .setResponseCode(code)
                .setHeader("Content-Type", "application/json")
                .setBody(body);
    }
}
