package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.TrafficPolicyConfig;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dispatcher.DispatchConfig.FeAllocation;
import org.flexlb.enums.EngineType;
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
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/** Real HTTP coverage for routing, fanout, wire schemas and streaming passthrough. */
@Timeout(30)
class DispatcherE2ETest {
    private final List<MockWebServer> frontends = List.of(new MockWebServer(), new MockWebServer(), new MockWebServer());
    private final DispatchConfig cfg = new DispatchConfig();
    private final FlexlbConfig lb = new FlexlbConfig();
    private final BatchScheduleCoordinator coordinator = mock(BatchScheduleCoordinator.class);
    private boolean policy;
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

    @ParameterizedTest
    @CsvSource(delimiter = '|', quoteCharacter = '~', textBlock = """
            / | {"prompt_batch":["a","b"]} | {"response_batch":[1]} | {"response_batch":[2]} | 200 | 200 | 200 | {"response_batch":[1,2]}
            /batch_infer | {"prompt_batch":["a","b"]} | {"response_batch":[]} | {"response_batch":[2]} | 200 | 200 | 200 | {"response_batch":[null,2],"_partial_failure":{"failed_count":1,"total_count":2,"failed_indices":[0]}}
            /v1/batch/chat/completions | {"requests":[{},{}]} | {"responses":[{"id":"a"}]} | boom | 200 | 500 | 200 | {"responses":[{"id":"a"},{"index":1,"error":{"code":"dispatcher_sub_batch_failed","message":"fe_server_error"}}],"_partial_failure":{"failed_count":1,"total_count":2,"failed_indices":[1]}}
            /v1/embeddings | {"input":["a","b"]} | {"data":[{"index":0,"embedding":[1.0]}],"usage":{"prompt_tokens":4,"total_tokens":4}} | boom | 200 | 500 | 200 | {"data":[{"index":0,"embedding":[1.0]},{"index":1,"embedding":null,"error":"fe_server_error"}],"object":"list","model":"","usage":{"prompt_tokens":4,"total_tokens":4},"_partial_failure":{"failed_count":1,"total_count":2,"failed_indices":[1]}}
            /batch_infer | {"prompt_batch":["a","b"]} | bad | bad | 400 | 400 | 400 | {"error":"all_sub_batches_failed","failed_count":2,"total_count":2,"total_chunks":2,"failed_reasons":["fe_client_error"]}
            /batch_infer | {"prompt_batch":["a","b"]} | bad | bad | 400 | 500 | 500 | {"error":"all_sub_batches_failed","failed_count":2,"total_count":2,"total_chunks":2,"failed_reasons":["fe_client_error","fe_server_error"]}
            /batch_infer | {"prompt_batch":["a","b"]} | bad | bad | 400 | 404 | 500 | {"error":"all_sub_batches_failed","failed_count":2,"total_count":2,"total_chunks":2,"failed_reasons":["fe_client_error"]}
            /batch_infer | {"prompt_batch":["a","b"]} | [] | bad | 200 | 500 | 500 | {"error":"all_sub_batches_failed","failed_count":2,"total_count":2,"total_chunks":2,"failed_reasons":["malformed_sub_batch","fe_server_error"]}
            /batch_infer | {"prompt_batch":["a","b"]} | {} | bad | 200 | 500 | 500 | {"error":"all_sub_batches_failed","failed_count":2,"total_count":2,"total_chunks":2,"failed_reasons":["malformed_sub_batch","fe_server_error"]}
            """)
    void realHttpFanoutPreservesWireSchemas(String path, String input, String first, String second,
                                           int firstStatus, int secondStatus, int status, String expected) throws Exception {
        reply(0, firstStatus, first);
        reply(1, secondStatus, second);
        startDispatcher(1);
        assertEquals(JSON.parseObject(expected), post(path, input, status));
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get(path);
        JSONArray items = JSON.parseObject(input).getJSONArray(spec.getRequestArrayField());
        for (int i = 0; i < 2; i++) {
            JSONObject chunk = takeChunk(i, path, spec.getRequestArrayField(), 1);
            assertEquals(items.subList(i, i + 1), chunk.getJSONArray(spec.getRequestArrayField()));
        }
        assertEquals(0, frontends.get(2).getRequestCount());
    }

    @ParameterizedTest
    @CsvSource({"false,200", "true,500"})
    void rerankerSortsGloballyAndFailsClosed(boolean failSecond, int status) throws Exception {
        reply(0, 200, """
                {"results":[{"index":0,"document":"文档🧥0","relevance_score":0.2},{"index":1,"document":"文档🧥1","relevance_score":0.9}],"total_tokens":11}
                """);
        reply(1, failSecond ? 500 : 200, """
                {"results":[{"index":0,"document":"文档🧥2","relevance_score":0.9},{"index":1,"document":"文档🧥3","relevance_score":0.4}],"total_tokens":17}
                """);
        startDispatcher(2);
        JSONObject out = post("/v1/reranker", "{\"query\":\"cape pants\",\"documents\":[\"文档🧥0\",\"文档🧥1\",\"文档🧥2\",\"文档🧥3\"],\"top_k\":2}", status);
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
                assertEquals("文档🧥" + (i + 1), results.getJSONObject(i).getString("document"));
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

    @ParameterizedTest
    @CsvSource(delimiter = '|', quoteCharacter = '~', textBlock = """
            /v1/embeddings | {"model":"embed-model","input":"hello world"} | {"object":"list","data":[{"index":0,"embedding":[0.1,0.2]}]}
            /v1/chat/completions | {"model":"qwen","messages":[{"role":"user","content":"hi"}]} | {"id":"chatcmpl-1","object":"chat.completion","choices":[{"message":{"role":"assistant","content":"hi"}}]}
            / | {"prompt_batch":["a","b"],"images":[["u0"],["u1"]]} | {"response_batch":[]}
            / | {"prompt_batch":["a","b"],"generation_config":{"adapter_name":["a","b"]}} | {"response_batch":[]}
            / | {"prompt_batch":["a","b"],"generate_config":{"is_streaming":true}} | raw-stream
            / | {"prompt_batch":["a","b"],"yield_generator":true} | raw-stream
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

    @ParameterizedTest
    @CsvSource({"MASTER,true,false", "MASTER,false,false", "LOCAL,true,false", "LOCAL,false,false", "MASTER,true,true"})
    void allocationDimensionsAppearOnTheFeWire(FeAllocation mode, boolean preassign, boolean groupPolicy) throws Exception {
        cfg.setPreAssignBe(preassign);
        cfg.setFeAllocation(mode);
        policy = groupPolicy;
        if (policy) {
            TrafficPolicyConfig.Target target = new TrafficPolicyConfig.Target();
            target.setGroup("tenant");
            target.setWeight(1);
            TrafficPolicyConfig group = new TrafficPolicyConfig();
            group.setDefaultTargets(List.of(target));
            lb.getRouter().setGroupSelector(group);
        }
        for (int i = 0; i < 3; i++) {
            reply(i, 200, "{\"response_batch\":[\"ok\"]}");
        }
        startDispatcher(1);
        post("/batch_infer", "{\"prompt_batch\":[\"a\",\"b\",\"c\"]}", 200);
        for (int i = 0; i < 3; i++) {
            JSONObject chunk = takeChunk(i, "/batch_infer", "prompt_batch", 1);
            assertFalse(chunk.containsKey("pre_assigned_be"));
            Object expected = preassign && !policy ? JSONArray.of(JSONObject.of("role", "PDFUSION", "ip", "10.0.0." + (i + 1),
                    "http_port", 23840, "grpc_port", 23841)) : null;
            assertEquals(expected, chunk.getJSONObject("generate_config").get("role_addrs"));
        }
        if (mode == FeAllocation.MASTER || preassign && !policy) {
            verify(coordinator).schedule(argThat(r -> r.isAssignBe() == (preassign && !policy)
                    && r.isAssignFe() == (mode == FeAllocation.MASTER)));
        } else {
            verifyNoInteractions(coordinator);
        }
    }

    @ParameterizedTest
    @CsvSource(delimiter = '|', quoteCharacter = '~', textBlock = """
            []
            {"generate_config":null}
            {"prompt_batch":["a"],"role_addrs":[]}
            {"prompt_batch":["a"],"generate_config":{"role_addrs":[]},"stream":true}
            {"prompt_batch":["a"],"generation_config":{"role_addrs":[]},"images":[]}
            """)
    void invalidRequestsFailBeforeAnyFrontendIsContacted(String body) {
        startDispatcher(1);
        post("/batch_infer", body, 400);
        assertNoFeTraffic();
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(strings = {"request", "response", "count"})
    void requestAndResponseBudgetsReturn413(String limit) {
        boolean responseLimit = limit.equals("response");
        if (limit.equals("count")) {
            lb.getRouter().setBatchScheduleMaxCount(1);
        } else if (responseLimit) {
            cfg.setMaxAggregateResponseBytes(1);
            reply(0, 200, "{\"response_batch\":[1]}");
        } else {
            cfg.setMaxAggregateRequestBytes(1);
        }
        startDispatcher(1);
        post("/batch_infer", "{\"prompt_batch\":[\"a\",\"b\"]}", 413);
        assertEquals(responseLimit ? 1 : 0, frontends.get(0).getRequestCount());
        assertEquals(0, frontends.get(1).getRequestCount());
    }

    @Test
    void emptyRequestsAndBatchesContactNoFe() {
        startDispatcher(2);
        assertEquals("invalid_batch_request", post("/batch_infer", "", 400).getString("error"));
        assertEquals(JSON.parseObject("{\"response_batch\":[]}"), post("/batch_infer", "{\"prompt_batch\":[]}", 200));
        assertEquals(JSON.parseObject("{\"object\":\"list\",\"model\":\"\",\"data\":[],\"usage\":{\"prompt_tokens\":0,\"total_tokens\":0}}"),
                post("/v1/embeddings", "{\"input\":[]}", 200));
        assertEquals(JSON.parseObject("{\"results\":[],\"total_tokens\":0}"),
                post("/v1/reranker", "{\"query\":\"q\",\"documents\":[]}", 200));
        assertNoFeTraffic();
    }

    @Test
    void allocationFailureContactsNoFe() {
        allocationFails = true;
        startDispatcher(2);
        assertEquals("batch_schedule_failed", post("/v1/batch/chat/completions", "{\"requests\":[{},{}]}", 503).getString("error"));
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
        FanoutService fanout = new FanoutService(new FeClient(WebClient.builder(), connections, cfg), metrics, cfg);
        when(coordinator.schedule(any())).thenAnswer(call -> {
            if (allocationFails) {
                return Mono.just(BatchScheduleResponse.error(StrategyErrorType.NO_AVAILABLE_WORKER, "no FE endpoints available"));
            }
            int count = ((BatchScheduleRequest) call.getArgument(0)).getBatchCount();
            List<BatchScheduleTarget> targets = new ArrayList<>();
            for (int i = 0; i < count; i++) {
                BatchScheduleTarget target = BatchScheduleTarget.of(new WorkerHost("10.0.0." + (i + 1), 23840), RoleType.PDFUSION, EngineType.LLM);
                target.setFeUrl(((BatchScheduleRequest) call.getArgument(0)).isAssignFe() ? urls.get(i % urls.size()) : "http://must-not-use");
                targets.add(target);
            }
            return Mono.just(BatchScheduleResponse.success(targets));
        });
        PassthroughClient passthrough = new PassthroughClient(WebClient.create(), pool, metrics, cfg);
        var configService = DispatcherTestSupport.configService(lb);
        BatchHandler handler = new BatchHandler(fanout, cfg, coordinator, passthrough, metrics, configService, pool, Schedulers.immediate());
        DispatchRouter router = new DispatchRouter(handler, passthrough);
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
            assertEquals(!policy, body.getJSONObject("generate_config").getBoolean("force_batch"));
        }
        return body;
    }

    private void assertNoFeTraffic() {
        frontends.forEach(fe -> assertEquals(0, fe.getRequestCount()));
    }

    private static String url(MockWebServer server) {
        return server.url("/").toString().replaceAll("/$", "");
    }
}
