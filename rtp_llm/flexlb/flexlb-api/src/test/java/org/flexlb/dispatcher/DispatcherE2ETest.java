package org.flexlb.dispatcher;

import static org.flexlb.dispatcher.BatchEndpointSpec.FailedItemFactory;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
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

/**
 * End-to-end test for the assembled dispatcher: real {@link DispatchRouter} on top of real
 * {@link GenericBatchHandler}, real {@link FanoutService}, real {@link FeClient}, and a
 * real {@link FePool} round-robin-ing across three {@link MockWebServer}s. Every batch endpoint
 * registered by {@link BatchEndpointSpec#SPECS} is exercised with one chunk induced to fail
 * (HTTP 500), and the non-batch path falls through to passthrough.
 */
class DispatcherE2ETest {

    private MockWebServer fe1;
    private MockWebServer fe2;
    private MockWebServer fe3;
    private DisposableServer dispatcherServer;
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
        fe1.shutdown();
        fe2.shutdown();
        fe3.shutdown();
    }

    @Test
    void batchInferNineSplitsThreeWithMiddleChunkFailure() throws Exception {
        // chunk0 → fe1: ok; chunk1 → fe2: 500; chunk2 → fe3: ok.
        fe1.enqueue(jsonResponse(200,
                "{\"response_batch\":[{\"response\":\"r0\"},{\"response\":\"r1\"},{\"response\":\"r2\"}]}"));
        fe2.enqueue(new MockResponse().setResponseCode(500).setBody("internal boom"));
        fe3.enqueue(jsonResponse(200,
                "{\"response_batch\":[{\"response\":\"r0\"},{\"response\":\"r1\"},{\"response\":\"r2\"}]}"));

        WebTestClient client = buildClient(/*subBatchSize=*/3);

        ObjectNode body = mapper.createObjectNode();
        body.put("model", "qwen");
        var pb = body.putArray("prompt_batch");
        for (int i = 0; i < 9; i++) {
            pb.add("p" + i);
        }

        JsonNode resp = client.post().uri("/dispatcher/batch_infer")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .exchange()
                .expectStatus().isOk()
                .expectBody(JsonNode.class)
                .returnResult().getResponseBody();

        assertNotNull(resp);
        JsonNode arr = resp.get("response_batch");
        assertNotNull(arr);
        assertEquals(9, arr.size());
        // Successful items 0..2 carry the FE payload; 6..8 from fe3 also do.
        for (int i = 0; i < 3; i++) {
            assertEquals("r" + i, arr.get(i).get("response").asText());
            assertEquals("r" + i, arr.get(6 + i).get("response").asText());
        }
        // Failed positions 3..5 are null per FailedItemFactory.NULL.
        for (int i = 3; i <= 5; i++) {
            assertTrue(arr.get(i).isNull(), "expected null at index " + i + " but was " + arr.get(i));
        }
        // Top-level _partial_failure envelope.
        JsonNode pf = resp.get("_partial_failure");
        assertNotNull(pf);
        assertEquals(3, pf.get("failed_count").asInt());
        assertEquals(9, pf.get("total_count").asInt());
        List<Integer> failedIndices = new ArrayList<>();
        pf.get("failed_indices").forEach(n -> failedIndices.add(n.asInt()));
        assertEquals(List.of(3, 4, 5), failedIndices);

        // Each FE received one /batch_infer call with the deep-copied envelope and a 3-element slice.
        assertChunkBatchInferRequest(fe1.takeRequest(), 3);
        assertChunkBatchInferRequest(fe2.takeRequest(), 3);
        assertChunkBatchInferRequest(fe3.takeRequest(), 3);
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

        DispatchConfig feClientCfg = new DispatchConfig();
        feClientCfg.setBatchTimeoutMs(5000);
        FeClient feClient = new FeClient(
                WebClient.builder(),
                reactor.netty.resources.ConnectionProvider.builder("e2e").build(),
                feClientCfg);
        FanoutService fanout = new FanoutService(feClient, pool);
        BatchScheduleClient batchScheduleClient = new BatchScheduleClient(null) {
            @Override
            public reactor.core.publisher.Mono<List<org.flexlb.dao.loadbalance.BatchScheduleTarget>> requestTargets(int count) {
                return reactor.core.publisher.Mono.just(targets);
            }
        };
        GenericBatchHandler batchHandler = DispatcherTestSupport.genericBatchHandler(
                fanout, mapper, "size:" + subBatchSize, batchScheduleClient, preAssignBe);

        PassthroughClient passthrough =
                new PassthroughClient(WebClient.create(), pool);

        List<BatchEndpointSpec> specs = BatchEndpointSpec.SPECS;
        DispatchRouter router = new DispatchRouter(batchHandler, passthrough, specs);

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
