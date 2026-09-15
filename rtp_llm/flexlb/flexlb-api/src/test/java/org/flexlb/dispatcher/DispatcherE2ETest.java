package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import okhttp3.mockwebserver.SocketPolicy;
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
            /batch_infer | {"prompt_batch":["a","b"]} | disconnected | bad | 0 | 400 | 500 | {"error":"all_sub_batches_failed","failed_count":2,"total_count":2,"total_chunks":2,"failed_reasons":["fe_unavailable","fe_client_error"]}
            /batch_infer | {"prompt_batch":["a","b"]} | bad | disconnected | 404 | 0 | 500 | {"error":"all_sub_batches_failed","failed_count":2,"total_count":2,"total_chunks":2,"failed_reasons":["fe_client_error","fe_unavailable"]}
            /batch_infer | {"prompt_batch":["a","b"]} | [] | bad | 200 | 500 | 500 | {"error":"all_sub_batches_failed","failed_count":2,"total_count":2,"total_chunks":2,"failed_reasons":["malformed_sub_batch","fe_server_error"]}
            /batch_infer | {"prompt_batch":["a","b"]} | {} | bad | 200 | 500 | 500 | {"error":"all_sub_batches_failed","failed_count":2,"total_count":2,"total_chunks":2,"failed_reasons":["malformed_sub_batch","fe_server_error"]}
            /v1/reranker | {"query":"cape pants","documents":["文档🧥0","文档🧥1","文档🧥2","文档🧥3"],"top_k":2} | {"results":[{"index":0,"document":"文档🧥0","relevance_score":0.2},{"index":1,"document":"文档🧥1","relevance_score":0.9}],"total_tokens":11} | {"results":[{"index":0,"document":"文档🧥2","relevance_score":0.9},{"index":1,"document":"文档🧥3","relevance_score":0.4}],"total_tokens":17} | 200 | 200 | 200 | {"results":[{"index":1,"document":"文档🧥1","relevance_score":0.9},{"index":2,"document":"文档🧥2","relevance_score":0.9}],"total_tokens":28}
            /v1/reranker | {"query":"cape pants","documents":["文档🧥0","文档🧥1","文档🧥2","文档🧥3"],"top_k":2} | {"results":[{"index":0,"document":"文档🧥0","relevance_score":0.2},{"index":1,"document":"文档🧥1","relevance_score":0.9}],"total_tokens":11} | {"results":[{"index":0,"document":"文档🧥2","relevance_score":0.9},{"index":1,"document":"文档🧥3","relevance_score":0.4}],"total_tokens":17} | 200 | 500 | 500 | {"error":"sub_batch_failed","failed_count":2,"total_count":4,"total_chunks":2,"failed_reasons":["fe_server_error"]}
            """)
    void realHttpFanoutPreservesWireSchemas(String path, String input, String first, String second,
                                           int firstStatus, int secondStatus, int status, String expected) throws Exception {
        reply(0, firstStatus, first);
        reply(1, secondStatus, second);
        BatchEndpointSpec spec = BatchEndpointSpec.BY_PATH.get(path);
        JSONArray items = JSON.parseObject(input).getJSONArray(spec.getRequestArrayField());
        int chunkSize = items.size() / 2;
        startDispatcher(chunkSize);
        JSONArray preview = preview(path, input, "split", 2);
        assertEquals(JSON.parseObject(expected), post(path, input, status));
        for (int i = 0; i < 2; i++) {
            JSONObject chunk = takeChunk(i, path, spec.getRequestArrayField(), chunkSize);
            assertEquals(items.subList(i * chunkSize, (i + 1) * chunkSize), chunk.getJSONArray(spec.getRequestArrayField()));
            if (spec == BatchEndpointSpec.RERANKER) {
                assertFalse(chunk.getBoolean("sorted"));
                assertFalse(chunk.containsKey("top_k"));
            }
            assertEquals(preview.getJSONObject(i), chunk);
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
        if (BatchEndpointSpec.BY_PATH.containsKey(path)) {
            assertEquals(JSONArray.of(JSON.parseObject(json)), preview(path, json, "passthrough", 1));
        } else {
            post("/_dryrun" + path, json, 400);
            assertNoFeTraffic();
        }
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
        JSONArray preview = preview("/batch_infer", "{\"prompt_batch\":[\"a\",\"b\"]}", "split", 2);
        assertFalse(preview.toJSONString().contains("role_addrs"));
        post("/batch_infer", "{\"prompt_batch\":[\"a\",\"b\",\"c\"]}", 200);
        for (int i = 0; i < 3; i++) {
            JSONObject chunk = takeChunk(i, "/batch_infer", "prompt_batch", 1);
            assertEquals(String.valueOf((char) ('a' + i)), chunk.getJSONArray("prompt_batch").getString(0));
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
        post("/_dryrun/batch_infer", body, 400);
        post("/batch_infer", body, 400);
        verifyNoInteractions(coordinator);
        assertNoFeTraffic();
    }

    @ParameterizedTest
    @CsvSource({"request,1,413", "response,1,413", "count,1,413", "request,122,200", "request,121,413"})
    void requestAndResponseBudgetsEnforceTheWireBoundary(String limit, long bytes, int status) {
        boolean responseLimit = limit.equals("response");
        if (limit.equals("count")) {
            lb.getRouter().setBatchScheduleMaxCount(1);
        } else if (responseLimit) {
            cfg.setMaxAggregateResponseBytes(1);
        } else {
            cfg.setMaxAggregateRequestBytes(bytes);
        }
        reply(0, 200, "{\"response_batch\":[1]}");
        reply(1, 200, "{\"response_batch\":[2]}");
        startDispatcher(1);
        post("/_dryrun/batch_infer", "{\"prompt_batch\":[\"a\",\"b\"]}", responseLimit ? 200 : 413);
        if (limit.equals("request") && bytes == 1) {
            post("/_dryrun", "{}", 413);
            post("/_dryrun/", "{\"prompt_batch\":[]}", 413);
        }
        verifyNoInteractions(coordinator);
        assertNoFeTraffic();
        post("/batch_infer", "{\"prompt_batch\":[\"a\",\"b\"]}", status);
        assertEquals(responseLimit || status == 200 ? 1 : 0, frontends.get(0).getRequestCount());
        assertEquals(status == 200 ? 1 : 0, frontends.get(1).getRequestCount());
    }

    @Test
    void emptyRequestsAndBatchesContactNoFe() {
        startDispatcher(2);
        post("/_dryrun/batch_infer", "", 400);
        preview("/batch_infer", "{\"prompt_batch\":[]}", "split", 0);
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
        cfg.setTrustedRoutingToken("dryrun-secret");
        startDispatcher(2);
        assertFalse(preview("/v1/batch/chat/completions", "{\"requests\":[{},{}]}", "split", 1).toJSONString().contains("dryrun-secret"));
        client.get().uri("/dispatcher/_dryrun").exchange().expectStatus().isBadRequest();
        post("/_dryrun/unknown", "{}", 400);
        verifyNoInteractions(coordinator);
        assertEquals("batch_schedule_failed", post("/v1/batch/chat/completions", "{\"requests\":[{},{}]}", 503).getString("error"));
        verify(coordinator).schedule(any());
        assertNoFeTraffic();
    }

    private void startDispatcher(int chunkSize) {
        List<String> urls = frontends.stream().map(fe -> fe.url("/").toString().replaceAll("/$", "")).toList();
        FePool pool = DispatcherTestSupport.fePool(urls, cfg);
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
        BatchHandler handler = new BatchHandler(fanout, cfg, coordinator, passthrough, metrics, DispatcherTestSupport.configService(lb), pool, Schedulers.immediate());
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
        MockResponse response = new MockResponse().setResponseCode(status == 0 ? 200 : status)
                .setHeader("Content-Type", "application/json").setBody(body);
        if (status == 0) {
            response.setSocketPolicy(SocketPolicy.DISCONNECT_DURING_RESPONSE_BODY);
        }
        frontends.get(frontend).enqueue(response);
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

    private JSONArray preview(String path, String body, String mode, int count) {
        JSONObject result = post("/_dryrun" + path, body, 200);
        assertEquals(mode, result.getString("mode"));
        assertEquals(count, result.getInteger("chunk_count"));
        assertEquals(count, result.getJSONArray("chunks").size());
        verifyNoInteractions(coordinator);
        assertNoFeTraffic();
        return result.getJSONArray("chunks");
    }

    private void assertNoFeTraffic() {
        frontends.forEach(fe -> assertEquals(0, fe.getRequestCount()));
    }
}
