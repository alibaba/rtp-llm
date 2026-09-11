package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONObject;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.TrafficPolicyConfig;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.exception.BatchScheduleTransportException;
import org.flexlb.service.BatchScheduleCoordinator;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.springframework.core.io.buffer.DataBufferLimitException;
import org.springframework.http.HttpHeaders;
import org.springframework.web.reactive.function.server.EntityResponse;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Scheduler;
import reactor.core.scheduler.Schedulers;

import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

@Timeout(30)
class BatchHandlerContractTest {
    private final FanoutService fanout = mock(FanoutService.class);
    private final BatchScheduleCoordinator coordinator = mock(BatchScheduleCoordinator.class);
    private final PassthroughClient passthrough = mock(PassthroughClient.class);
    private final ServerRequest request = mock(ServerRequest.class);
    private final DispatchConfig cfg = new DispatchConfig();
    private final FlexlbConfig lbConfig = new FlexlbConfig();

    @BeforeEach
    void setUp() {
        cfg.setSubBatchSpec(SubBatchSpec.parse("count:2"));
        ServerRequest.Headers headers = mock(ServerRequest.Headers.class);
        when(headers.asHttpHeaders()).thenReturn(new HttpHeaders());
        when(request.headers()).thenReturn(headers);
        when(request.uri()).thenReturn(URI.create("http://master/dispatcher/batch_infer"));
        when(coordinator.schedule(any())).thenReturn(Mono.just(BatchScheduleResponse.success(List.of())));
    }

    @ParameterizedTest
    @CsvSource(delimiter = '|', quoteCharacter = '~', textBlock = """
            BATCH_INFER | {"prompt":["a","b"]}
            EMBEDDING | {"model":"m","input":"hello world"}
            EMBEDDING | {"input":[{"type":"image_url","image_url":{"url":"http://x/y.png"}},{"type":"text","text":"describe"}]}
            BATCH_INFER | {"prompt_batch":["a","b"],"images":[["http://x/0.png"],["http://x/1.png"]]}
            BATCH_INFER | {"prompt_batch":["a","b"],"generate_config":{"adapter_name":["lora0","lora1"]}}
            BATCH_INFER | {"prompt_batch":["a","b"],"adapter_name":["lora0","lora1"]}
            ROOT | {"prompt_batch":["a","b"],"yield_generator":true}
            """)
    void wholeBodyRequestsAreForwardedVerbatim(BatchEndpointSpec spec, String json) {
        ServerResponse expected = ServerResponse.ok().bodyValue("fe-says-hi").block();
        when(passthrough.forward(eq(request), any(byte[].class))).thenReturn(Mono.just(expected));
        assertSame(expected, handle(spec, json));
        verify(passthrough).forward(request, json.getBytes(StandardCharsets.UTF_8));
        verifyNoInteractions(fanout, coordinator);
    }

    @ParameterizedTest
    @CsvSource(delimiter = '|', quoteCharacter = '~', textBlock = """
            BATCH_INFER | [1,2,3] | JSON object
            BATCH_INFER | {"prompt_batch":["a","b"],"generate_config":"oops"} | generate_config
            BATCH_INFER | {"prompt_batch":["a"],"generate_config":{"role_addrs":[{"role":"PDFUSION","ip":"1.2.3.4"}]}} | role_addrs
            ROOT | {"prompt_batch":["a"],"role_addrs":[{"role":"PDFUSION","ip":"1.2.3.4"}]} | role_addrs
            ROOT | {"prompt_batch":["a"],"stream":true,"generation_config":{"role_addrs":[{"role":"PDFUSION","ip":"1.2.3.4"}]}} | role_addrs
            BATCH_INFER | {"prompt_batch":["a"],"images":[["https://example/image.png"]],"generate_config":{"role_addrs":[{"role":"PDFUSION","ip":"1.2.3.4"}]}} | role_addrs
            RERANKER | {"query":"cape pants","documents":["a","b"],"top_k":1.5} | top_k
            """)
    void invalidRequestsStopBeforeAllocation(BatchEndpointSpec spec, String json, String reason) {
        ServerResponse response = handle(spec, json);
        assertError(response, 400, "invalid_batch_request");
        assertTrue(body(response).getString("message").contains(reason));
        verifyNoInteractions(fanout, coordinator, passthrough);
    }

    @ParameterizedTest
    @CsvSource(delimiter = '|', quoteCharacter = '~', textBlock = """
            RERANKER | {"query":"cape pants","documents":[]} | {"results":[],"total_tokens":0}
            EMBEDDING | {"model":"embed-v1","input":[]} | {"data":[],"object":"list","model":"embed-v1","usage":{"prompt_tokens":0,"total_tokens":0}}
            """)
    void emptyBatchesReturnCompleteSchemas(BatchEndpointSpec spec, String json, String expected) {
        ServerResponse response = handle(spec, json);
        assertEquals(200, response.rawStatusCode());
        assertEquals(JSON.parseObject(expected), body(response));
        verifyNoInteractions(fanout, coordinator, passthrough);
    }

    @ParameterizedTest
    @CsvSource({"master,INVALID_REQUEST,400", "local,INVALID_REQUEST,400",
            "master,NO_AVAILABLE_WORKER,503", "local,NO_AVAILABLE_WORKER,503"})
    void allocationFailureStopsBeforeFanout(String mode, StrategyErrorType error, int status) {
        cfg.setFeAllocation(mode);
        cfg.setPreAssignBe(true);
        when(coordinator.schedule(any())).thenReturn(Mono.just(BatchScheduleResponse.error(error, "unavailable")));
        assertError(handle(BatchEndpointSpec.BATCH_INFER, "{\"prompt_batch\":[\"a\",\"b\"]}"),
                status, "batch_schedule_failed");
        verifyNoInteractions(fanout, passthrough);
    }

    @Test
    void unavailableMasterDoesNotLeakItsAddress() {
        when(coordinator.schedule(any())).thenReturn(Mono.error(
                new BatchScheduleTransportException("internal-master:7001", "CONNECT_FAILED")));
        ServerResponse response = handle(BatchEndpointSpec.BATCH_INFER, "{\"prompt_batch\":[\"a\",\"b\"]}");
        assertError(response, 503, "batch_schedule_failed");
        assertEquals("batch target allocation failed", body(response).getString("message"));
        verifyNoInteractions(fanout, passthrough);
    }

    @ParameterizedTest
    @CsvSource({"CHAT,master,true,false", "EMBEDDING,local,false,false",
            "BATCH_INFER,master,true,false", "BATCH_INFER,master,true,true",
            "BATCH_INFER,local,true,false"})
    void allocationRequestsOnlyConsumedDimensions(BatchEndpointSpec spec, String mode,
                                                  boolean preassign, boolean trafficPolicy) {
        cfg.setFeAllocation(mode);
        cfg.setPreAssignBe(preassign);
        if (trafficPolicy) {
            TrafficPolicyConfig policy = new TrafficPolicyConfig();
            TrafficPolicyConfig.Target target = new TrafficPolicyConfig.Target();
            target.setGroup("tenant-a");
            target.setWeight(1);
            policy.setDefaultTargets(List.of(target));
            lbConfig.getRouter().setGroupSelector(policy);
        }
        boolean assignBe = preassign && spec.isPreAssignable() && !trafficPolicy;
        boolean assignFe = mode.equals("master");
        BatchScheduleTarget target = new BatchScheduleTarget("10.0.0.1", 8088, 50051, RoleType.PDFUSION);
        target.setFeUrl("http://fe-1");
        when(coordinator.schedule(any())).thenReturn(Mono.just(BatchScheduleResponse.success(List.of(target, target))));
        when(fanout.dispatchChunks(anyString(), anyList(), anyList(), any(), any(), any()))
                .thenAnswer(call -> {
                    List<JSONObject> chunks = call.getArgument(1);
                    List<String> urls = call.getArgument(2);
                    assertEquals(assignFe ? List.of("http://fe-1", "http://fe-1") : List.of(), urls);
                    for (JSONObject chunk : chunks) {
                        JSONObject gc = chunk.getJSONObject("generate_config");
                        assertEquals(assignBe, gc != null && gc.containsKey("role_addrs"));
                        if (trafficPolicy) {
                            assertFalse(gc.getBooleanValue("force_batch"));
                        }
                    }
                    return Mono.just(List.of(SubBatchResult.failed(2, 0, "fe_http_500")));
                });
        handle(spec, "{\"" + spec.getRequestArrayField() + "\":[\"a\",\"b\"]}");
        if (assignBe || assignFe) {
            verify(coordinator).schedule(argThat(r -> r.isAssignBe() == assignBe && r.isAssignFe() == assignFe));
        } else {
            verifyNoInteractions(coordinator);
        }
        verify(fanout).dispatchChunks(eq(spec.getPath()), anyList(), anyList(), eq(spec), any(), any());
        verifyNoInteractions(passthrough);
    }

    @Test
    void stringListEmbeddingsStillSplit() {
        when(fanout.dispatchChunks(anyString(), anyList(), anyList(), any(), any(), any()))
                .thenReturn(Mono.just(List.of(SubBatchResult.failed(3, 0, "fe_http_500"))));
        handle(BatchEndpointSpec.EMBEDDING, "{\"model\":\"m\",\"input\":[\"a\",\"b\",\"c\"]}");
        verify(fanout).dispatchChunks(eq("/v1/embeddings"), anyList(), anyList(), eq(BatchEndpointSpec.EMBEDDING), any(), any());
        verifyNoInteractions(passthrough);
    }

    @Test
    void tooManyChunksAreRejectedBeforeScheduling() {
        cfg.setSubBatchSpec(SubBatchSpec.parse("size:2"));
        lbConfig.getRouter().setBatchScheduleMaxCount(2);
        ServerResponse response = handle(BatchEndpointSpec.EMBEDDING, "{\"input\":[\"a\",\"b\",\"c\",\"d\",\"e\"]}");
        assertError(response, 413, "too_many_sub_batches");
        assertTrue(body(response).getString("message").contains("maximum is 2"));
        verifyNoInteractions(fanout, coordinator, passthrough);
    }

    @Test
    void repeatedEnvelopeIsBudgetedBeforeScheduling() {
        cfg.setMaxAggregateRequestBytes(2048);
        assertError(handle(BatchEndpointSpec.EMBEDDING, "{\"model\":\"" + "x".repeat(1500)
                + "\",\"input\":[\"a\",\"b\"]}"), 413, "batch_request_too_large");
        verifyNoInteractions(fanout, coordinator, passthrough);
    }

    @Test
    void aggregateFanoutLimitMapsTo413() {
        when(fanout.dispatchChunks(anyString(), anyList(), anyList(), any(), any(), any()))
                .thenReturn(Mono.error(new AggregateResponseTooLargeException(8)));
        assertError(handle(BatchEndpointSpec.BATCH_INFER, "{\"prompt_batch\":[\"a\"]}"),
                413, "batch_response_too_large");
    }

    @Test
    void inboundBodyLimitMapsTo413() {
        when(request.bodyToMono(byte[].class)).thenReturn(Mono.error(new DataBufferLimitException("too large")));
        assertError(handler(Schedulers.immediate()).handle(request, BatchEndpointSpec.BATCH_INFER).block(),
                413, "request_body_too_large");
        verifyNoInteractions(fanout, coordinator, passthrough);
    }

    @ParameterizedTest
    @CsvSource({"500,fe_server_error", "400,fe_client_error"})
    void allFailedChunksPreserveStatusAndDeduplicateReasons(int status, String publicReason) {
        when(fanout.dispatchChunks(anyString(), anyList(), anyList(), any(), any(), any()))
                .thenReturn(Mono.just(List.of(SubBatchResult.failed(2, 0, "boom", status),
                        SubBatchResult.failed(2, 2, "boom", status))));
        ServerResponse response = handle(BatchEndpointSpec.BATCH_INFER, "{\"prompt_batch\":[\"a\",\"b\",\"c\",\"d\"]}");
        assertError(response, status, "all_sub_batches_failed");
        JSONObject result = body(response);
        assertEquals(4, result.getInteger("failed_count"));
        assertEquals(4, result.getInteger("total_count"));
        assertEquals(2, result.getInteger("total_chunks"));
        assertEquals(List.of(publicReason), result.getJSONArray("failed_reasons"));
        verifyNoInteractions(passthrough);
    }

    @Test
    void rerankerFailsClosedOnPartialFailure() {
        JSONObject ok = JSON.parseObject("""
                {"results":[{"index":0,"relevance_score":0.1},{"index":1,"relevance_score":0.2}],"total_tokens":8}
                """);
        when(fanout.dispatchChunks(anyString(), anyList(), anyList(), any(), any(), any()))
                .thenReturn(Mono.just(List.of(SubBatchResult.ok(ok, 2, 0), SubBatchResult.failed(2, 2, "timeout"))));
        ServerResponse response = handle(BatchEndpointSpec.RERANKER,
                "{\"query\":\"cape pants\",\"documents\":[\"d0\",\"d1\",\"d2\",\"d3\"]}");
        assertError(response, 500, "sub_batch_failed");
        assertEquals(2, body(response).getInteger("failed_count"));
        assertEquals(4, body(response).getInteger("total_count"));
        assertEquals(2, body(response).getInteger("total_chunks"));
    }

    @Test
    void jsonWorkUsesTheCpuScheduler() {
        Scheduler scheduler = Schedulers.newSingle("batch-json-test");
        try {
            when(request.bodyToMono(byte[].class)).thenReturn(Mono.just("not-json".getBytes(StandardCharsets.UTF_8)));
            AtomicReference<String> thread = new AtomicReference<>();
            ServerResponse response = handler(scheduler).handle(request, BatchEndpointSpec.BATCH_INFER)
                    .doOnNext(ignored -> thread.set(Thread.currentThread().getName())).block();
            assertError(response, 400, "invalid_batch_request");
            assertTrue(thread.get().startsWith("batch-json-test"));
        } finally {
            scheduler.dispose();
        }
    }

    private BatchHandler handler(Scheduler scheduler) {
        return new BatchHandler(fanout, cfg, coordinator, passthrough, DispatcherTestSupport.noopMetrics(),
                DispatcherTestSupport.configService(lbConfig), scheduler);
    }

    private ServerResponse handle(BatchEndpointSpec spec, String json) {
        when(request.bodyToMono(byte[].class)).thenReturn(Mono.just(json.getBytes(StandardCharsets.UTF_8)));
        return handler(Schedulers.immediate()).handle(request, spec).block();
    }

    private static JSONObject body(ServerResponse response) {
        return JSON.parseObject((byte[]) ((EntityResponse<?>) response).entity());
    }

    private static void assertError(ServerResponse response, int status, String code) {
        assertEquals(status, response.rawStatusCode());
        assertEquals(code, body(response).getString("error"));
    }
}
