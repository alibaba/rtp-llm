package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.BatchScheduleCoordinator;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.springframework.http.MediaType;
import org.springframework.http.server.reactive.ReactorHttpHandlerAdapter;
import org.springframework.test.web.reactive.server.WebTestClient;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.server.RouterFunctions;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;
import reactor.netty.DisposableServer;
import reactor.netty.http.server.HttpServer;
import reactor.netty.resources.ConnectionProvider;

import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Real HTTP coverage for routing, fanout, wire schemas and streaming passthrough. */
@Timeout(30)
class DispatcherE2ETest {
    private final List<MockWebServer> frontends = List.of(new MockWebServer(), new MockWebServer(), new MockWebServer());
    private final DispatchConfig cfg = new DispatchConfig();
    private List<BatchScheduleTarget> backendTargets = List.of();
    private boolean allocationFails;
    private WebTestClient client;
    private DisposableServer server;
    private ConnectionProvider connections;

    @BeforeEach
    void startFrontends() throws Exception {
        for (MockWebServer frontend : frontends) {
            frontend.start();
        }
    }

    @AfterEach
    void closeConnections() throws Exception {
        if (server != null) {
            server.disposeNow();
        }
        if (connections != null) {
            connections.disposeLater().block(Duration.ofSeconds(5));
        }
        for (MockWebServer frontend : frontends) {
            frontend.shutdown();
        }
    }

    @Test
    void chatPartialFailureKeepsPositionsAndErrorSchema() throws Exception {
        reply(0, 200, "{\"responses\":[{\"id\":\"a\"},{\"id\":\"b\"}]}");
        reply(1, 500, "boom");
        startDispatcher(2);
        JSONObject response = post("/v1/batch/chat/completions", chatRequest(), 200);
        JSONArray items = response.getJSONArray("responses");
        assertEquals(4, items.size());
        assertEquals("a", items.getJSONObject(0).getString("id"));
        assertEquals("b", items.getJSONObject(1).getString("id"));
        for (int i = 2; i < 4; i++) {
            assertEquals(i, items.getJSONObject(i).getInteger("index"));
            JSONObject error = items.getJSONObject(i).getJSONObject("error");
            assertEquals("dispatcher_sub_batch_failed", error.getString("code"));
            assertFalse(error.getString("message").isBlank());
        }
        assertPartialFailure(response, 4);
        takeChunk(0, "/v1/batch/chat/completions", "requests", 2);
        takeChunk(1, "/v1/batch/chat/completions", "requests", 2);
        assertEquals(0, frontends.get(2).getRequestCount());
    }

    @Test
    void embeddingPartialFailureRebasesIndicesAndSumsUsage() throws Exception {
        reply(0, 200, """
                {"data":[{"index":0,"embedding":[1.0,2.0]},{"index":1,"embedding":[3.0,4.0]}],"usage":{"prompt_tokens":4,"total_tokens":4}}
                """);
        reply(1, 500, "boom");
        reply(2, 200, """
                {"data":[{"index":0,"embedding":[5.0,6.0]},{"index":1,"embedding":[7.0,8.0]}],"usage":{"prompt_tokens":4,"total_tokens":4}}
                """);
        startDispatcher(2);
        JSONObject response = post("/v1/embeddings", "{\"model\":\"qwen-embed\",\"input\":[\"a\",\"b\",\"c\",\"d\",\"e\",\"f\"]}", 200);
        JSONArray items = response.getJSONArray("data");
        assertEquals(6, items.size());
        for (int i = 0; i < 6; i++) {
            assertEquals(i, items.getJSONObject(i).getInteger("index"));
        }
        assertEquals(JSON.parseArray("[1.0,2.0]"), items.getJSONObject(0).getJSONArray("embedding"));
        assertEquals(JSON.parseArray("[3.0,4.0]"), items.getJSONObject(1).getJSONArray("embedding"));
        assertEquals(JSON.parseArray("[5.0,6.0]"), items.getJSONObject(4).getJSONArray("embedding"));
        assertEquals(JSON.parseArray("[7.0,8.0]"), items.getJSONObject(5).getJSONArray("embedding"));
        for (int i = 2; i < 4; i++) {
            assertTrue(items.getJSONObject(i).containsKey("embedding"));
            assertNull(items.getJSONObject(i).get("embedding"));
            assertFalse(items.getJSONObject(i).getString("error").isBlank());
        }
        assertEquals(JSON.parseObject("{\"prompt_tokens\":8,\"total_tokens\":8}"), response.getJSONObject("usage"));
        assertPartialFailure(response, 6);
        for (int i = 0; i < 3; i++) {
            takeChunk(i, "/v1/embeddings", "input", 2);
        }
    }

    @ParameterizedTest
    @CsvSource({"false,200", "true,500"})
    void rerankerSortsGloballyAndFailsClosed(boolean failSecond, int status) throws Exception {
        reply(0, 200, """
                {"results":[{"index":0,"document":"d0","relevance_score":0.2},{"index":1,"document":"d1","relevance_score":0.9}],"total_tokens":11}
                """);
        reply(1, failSecond ? 500 : 200, """
                {"results":[{"index":0,"document":"d2","relevance_score":0.9},{"index":1,"document":"d3","relevance_score":0.4}],"total_tokens":17}
                """);
        startDispatcher(2);
        JSONObject out = post("/v1/reranker", "{\"query\":\"cape pants\",\"documents\":[\"d0\",\"d1\",\"d2\",\"d3\"],\"top_k\":2}", status);
        if (failSecond) {
            assertEquals("sub_batch_failed", out.getString("error"));
            assertEquals(2, out.getInteger("failed_count"));
            assertEquals(4, out.getInteger("total_count"));
            assertEquals(2, out.getInteger("total_chunks"));
        } else {
            JSONArray results = out.getJSONArray("results");
            assertEquals(2, results.size());
            for (int i = 0; i < 2; i++) {
                assertEquals(i + 1, results.getJSONObject(i).getInteger("index"));
                assertEquals("d" + (i + 1), results.getJSONObject(i).getString("document"));
            }
            assertEquals(28L, out.getLong("total_tokens"));
        }
        for (int i = 0; i < 2; i++) {
            JSONObject chunk = takeChunk(i, "/v1/reranker", "documents", 2);
            assertFalse(chunk.getBoolean("sorted"));
            assertFalse(chunk.containsKey("top_k"));
        }
        assertEquals(0, frontends.get(2).getRequestCount());
    }

    @Test
    void unicodeDocumentsSurviveSplittingAndGlobalIndexing() throws Exception {
        JSONArray documents = new JSONArray();
        for (int i = 0; i < 100; i++) {
            documents.add("标题：古风男长裤与斗篷搭配🧥\n内容：中文、emoji、换行、\"引号\"和\\反斜线 " + i);
        }
        int[] counts = {40, 40, 20};
        int[] tokens = {6000, 7000, 6343};
        for (int chunk = 0; chunk < 3; chunk++) {
            JSONArray results = new JSONArray();
            for (int i = 0; i < counts[chunk]; i++) {
                int index = chunk * 40 + i;
                results.add(JSONObject.of("index", i, "document", documents.get(index), "relevance_score", index / 100.0));
            }
            reply(chunk, 200, JSONObject.of("results", results, "total_tokens", tokens[chunk]).toJSONString());
        }
        startDispatcher(40);
        JSONObject request = JSONObject.of("query", "古风男长裤怎么搭配斗篷", "model", "bge-reranker",
                "__request_id__", 146280, "documents", documents);
        JSONObject out = post("/v1/reranker", request.toJSONString(), 200);
        JSONArray results = out.getJSONArray("results");
        assertEquals(100, results.size());
        assertEquals(19343L, out.getLong("total_tokens"));
        for (int i = 0; i < 100; i++) {
            assertEquals(99 - i, results.getJSONObject(i).getInteger("index"));
            assertEquals(documents.get(99 - i), results.getJSONObject(i).getString("document"));
            assertEquals((99 - i) / 100.0, results.getJSONObject(i).getDouble("relevance_score"));
        }
        for (int i = 0; i < 3; i++) {
            JSONObject child = takeChunk(i, "/v1/reranker", "documents", counts[i]);
            for (String key : List.of("query", "model", "__request_id__")) {
                assertEquals(request.get(key), child.get(key));
            }
            assertFalse(child.getBoolean("sorted"));
            assertFalse(child.containsKey("top_k"));
            assertEquals(documents.subList(i * 40, i * 40 + counts[i]), child.getJSONArray("documents"));
        }
    }

    @ParameterizedTest
    @CsvSource(delimiter = '|', quoteCharacter = '~', textBlock = """
            /v1/embeddings | {"model":"embed-model","input":"hello world"} | {"object":"list","data":[{"index":0,"embedding":[0.1,0.2]}]}
            /v1/chat/completions | {"model":"qwen","messages":[{"role":"user","content":"hi"}]} | {"id":"chatcmpl-1","object":"chat.completion","choices":[{"message":{"role":"assistant","content":"hi"}}]}
            """)
    void passthroughPreservesRequestAndResponseBytes(String path, String json, String upstream) throws Exception {
        reply(0, 200, upstream);
        startDispatcher(2);
        byte[] response = send(path, json, 200);
        assertArrayEquals(upstream.getBytes(StandardCharsets.UTF_8), response);
        RecordedRequest received = frontends.getFirst().takeRequest(5, TimeUnit.SECONDS);
        assertNotNull(received);
        assertEquals(path, received.getPath());
        assertEquals("POST", received.getMethod());
        assertEquals(json, received.getBody().readUtf8());
        assertEquals(0, frontends.get(1).getRequestCount() + frontends.get(2).getRequestCount());
    }

    @Test
    void preassignedBackendsAppearOnTheFeWire() throws Exception {
        cfg.setPreAssignBe(true);
        List<BatchScheduleTarget> targets = new ArrayList<>();
        JSONArray prompts = new JSONArray();
        for (int i = 0; i < 3; i++) {
            targets.add(new BatchScheduleTarget("10.0.0." + (i + 1), 23840, 23841, RoleType.PDFUSION));
            JSONArray responses = new JSONArray();
            for (int j = 0; j < 3; j++) {
                prompts.add("p" + (i * 3 + j));
                responses.add(JSONObject.of("response", "r" + (i * 3 + j)));
            }
            reply(i, 200, JSONObject.of("response_batch", responses).toJSONString());
        }
        backendTargets = targets;
        startDispatcher(3);
        post("/batch_infer", JSONObject.of("model", "qwen", "prompt_batch", prompts).toJSONString(), 200);
        for (int i = 0; i < 3; i++) {
            JSONObject chunk = takeChunk(i, "/batch_infer", "prompt_batch", 3);
            assertFalse(chunk.containsKey("pre_assigned_be"));
            assertEquals(JSONArray.of(JSONObject.of("role", "PDFUSION", "ip", "10.0.0." + (i + 1),
                    "http_port", 23840, "grpc_port", 23841)), chunk.getJSONObject("generate_config").getJSONArray("role_addrs"));
        }
    }

    @Test
    void emptyRequestsAndBatchesContactNoFe() {
        startDispatcher(2);
        assertEquals("invalid_batch_request", post("/batch_infer", "", 400).getString("error"));
        assertEquals(JSON.parseObject("{\"response_batch\":[]}"), post("/batch_infer", "{\"prompt_batch\":[]}", 200));
        assertNoFeTraffic();
    }

    @Test
    void uniformFeClientErrorsReachTheCaller() {
        reply(0, 400, "bad request");
        reply(1, 400, "bad request");
        startDispatcher(2);
        JSONObject out = post("/v1/batch/chat/completions", chatRequest(), 400);
        assertEquals("all_sub_batches_failed", out.getString("error"));
        assertEquals(4, out.getInteger("failed_count"));
        assertEquals(2, out.getInteger("total_chunks"));
        assertFalse(out.getJSONArray("failed_reasons").isEmpty());
        assertEquals(0, frontends.get(2).getRequestCount());
    }

    @Test
    void allocationFailureContactsNoFe() {
        allocationFails = true;
        startDispatcher(2);
        assertEquals("batch_schedule_failed", post("/v1/batch/chat/completions", chatRequest(), 503).getString("error"));
        assertNoFeTraffic();
    }

    @Test
    void diagnosticRoutesReturnLocalStateAndStampedChunksWithoutFeTraffic() {
        cfg.setPreAssignBe(true);
        backendTargets = List.of(new BatchScheduleTarget("10.0.0.1", 23840, 23841, RoleType.PDFUSION),
                new BatchScheduleTarget("10.0.0.2", 23840, 23841, RoleType.PDFUSION));
        startDispatcher(2);
        byte[] snapshot = client.get().uri("/dispatcher/_snapshot").exchange().expectStatus().isOk()
                .expectHeader().contentTypeCompatibleWith(MediaType.APPLICATION_JSON).expectBody().returnResult().getResponseBody();
        JSONObject pool = JSON.parseObject(snapshot).getJSONObject("fePool");
        assertEquals("e2e.fe.publish", pool.getString("serviceId"));
        assertEquals(3, pool.getInteger("size"));
        JSONArray hosts = pool.getJSONArray("hosts");
        assertEquals(3, hosts.size());
        for (int i = 0; i < 3; i++) {
            assertEquals(url(frontends.get(i)), hosts.getJSONObject(i).getString("url"));
            assertTrue(hosts.getJSONObject(i).getBoolean("alive"));
            assertEquals(0, hosts.getJSONObject(i).getInteger("consecFails"));
        }
        JSONObject out = post("/_dryrun/batch_infer?pre_assign=true", "{\"model\":\"qwen\",\"prompt_batch\":[\"p0\",\"p1\",\"p2\"]}", 200);
        assertEquals("/batch_infer", out.getString("path"));
        assertEquals(3, out.getInteger("totalItems"));
        assertEquals(2, out.getInteger("chunkCount"));
        assertTrue(out.getBoolean("preAssignEffective"));
        assertEquals(2, out.getJSONArray("preAssignTargets").size());
        JSONArray chunks = out.getJSONArray("chunks");
        assertEquals(2, chunks.size());
        for (int i = 0; i < 2; i++) {
            JSONObject chunk = chunks.getJSONObject(i);
            assertEquals("qwen", chunk.getString("model"));
            JSONObject config = chunk.getJSONObject("generate_config");
            assertTrue(config.getBoolean("force_batch"));
            assertEquals(1, config.getJSONArray("role_addrs").size());
            assertEquals("10.0.0." + (i + 1), config.getJSONArray("role_addrs").getJSONObject(0).getString("ip"));
        }
        assertNoFeTraffic();
    }

    private void startDispatcher(int chunkSize) {
        List<String> urls = frontends.stream().map(DispatcherE2ETest::url).toList();
        FePool pool = DispatcherTestSupport.fePool(urls);
        cfg.setBatchTimeoutMs(5000);
        cfg.setFePoolServiceId("e2e.fe.publish");
        cfg.setSubBatch("size:" + chunkSize);
        cfg.setSubBatchSpec(SubBatchSpec.parse(cfg.getSubBatch()));
        connections = ConnectionProvider.builder("e2e").build();
        DispatcherMetricsReporter metrics = DispatcherTestSupport.noopMetrics();
        FanoutService fanout = new FanoutService(new FeClient(WebClient.builder(), connections, cfg), metrics, pool, cfg);
        BatchScheduleCoordinator coordinator = mock(BatchScheduleCoordinator.class);
        when(coordinator.schedule(any())).thenAnswer(call -> {
            if (allocationFails) {
                return Mono.just(BatchScheduleResponse.error(StrategyErrorType.NO_AVAILABLE_WORKER, "no FE endpoints available"));
            }
            int count = ((BatchScheduleRequest) call.getArgument(0)).getBatchCount();
            List<BatchScheduleTarget> targets = new ArrayList<>();
            for (int i = 0; i < count; i++) {
                BatchScheduleTarget target = i < backendTargets.size() ? backendTargets.get(i) : new BatchScheduleTarget();
                target.setFeUrl(urls.get(i % urls.size()));
                targets.add(target);
            }
            return Mono.just(BatchScheduleResponse.success(targets));
        });
        PassthroughClient passthrough = new PassthroughClient(WebClient.create(), pool, metrics, cfg);
        var configService = DispatcherTestSupport.configService(new FlexlbConfig());
        BatchHandler handler = new BatchHandler(fanout, cfg, coordinator, passthrough, metrics, configService, Schedulers.immediate());
        DispatcherFePoolRefresher refresher = mock(DispatcherFePoolRefresher.class);
        when(refresher.source()).thenReturn(() -> urls);
        FeHealthChecker health = mock(FeHealthChecker.class);
        when(health.isAlive(anyString())).thenReturn(true);
        DispatcherInspectionHandler inspection = new DispatcherInspectionHandler(cfg, refresher, health, coordinator, configService, Schedulers.immediate());
        DispatchRouter router = new DispatchRouter(handler, passthrough, inspection);
        // A real transport is required to exercise lazy DataBuffer bodies and their ownership.
        server = HttpServer.create().port(0).handle(new ReactorHttpHandlerAdapter(RouterFunctions.toHttpHandler(router.routes()))).bindNow();
        client = WebTestClient.bindToServer().baseUrl("http://localhost:" + server.port()).responseTimeout(Duration.ofSeconds(10)).build();
    }

    private byte[] send(String path, String json, int status) {
        return client.post().uri("/dispatcher" + path).contentType(MediaType.APPLICATION_JSON)
                .bodyValue(json.getBytes(StandardCharsets.UTF_8)).exchange().expectStatus().isEqualTo(status)
                .expectBody().returnResult().getResponseBody();
    }

    private JSONObject post(String path, String json, int status) {
        return JSON.parseObject(send(path, json, status));
    }

    private void reply(int frontend, int status, String body) {
        frontends.get(frontend).enqueue(new MockResponse().setResponseCode(status)
                .setHeader("Content-Type", "application/json").setBody(body));
    }

    private JSONObject takeChunk(int frontend, String path, String field, int size) throws Exception {
        RecordedRequest received = frontends.get(frontend).takeRequest(5, TimeUnit.SECONDS);
        assertNotNull(received);
        assertEquals("POST", received.getMethod());
        assertEquals(path, received.getPath());
        JSONObject body = JSON.parseObject(received.getBody().readUtf8());
        assertEquals(size, body.getJSONArray(field).size());
        if (field.equals("prompt_batch")) {
            assertTrue(body.getJSONObject("generate_config").getBoolean("force_batch"));
        }
        return body;
    }

    private void assertNoFeTraffic() {
        frontends.forEach(fe -> assertEquals(0, fe.getRequestCount()));
    }

    private static void assertPartialFailure(JSONObject response, int total) {
        assertEquals(JSON.parseObject("{\"failed_count\":2,\"total_count\":" + total + ",\"failed_indices\":[2,3]}"), response.getJSONObject("_partial_failure"));
    }

    private static String chatRequest() {
        JSONArray requests = new JSONArray();
        for (int i = 0; i < 4; i++) {
            requests.add(JSONObject.of("custom_id", "c" + i, "messages", JSONArray.of(JSONObject.of("role", "user", "content", "hi"))));
        }
        return JSONObject.of("model", "qwen", "requests", requests).toJSONString();
    }

    private static String url(MockWebServer server) {
        return server.url("/").toString().replaceAll("/$", "");
    }
}
