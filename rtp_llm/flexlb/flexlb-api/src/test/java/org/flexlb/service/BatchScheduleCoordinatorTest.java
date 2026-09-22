package org.flexlb.service;

import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import org.flexlb.balance.strategy.RoundRobinLoadBalancer;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleRequest.AllocationType;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dispatcher.FePool;
import org.flexlb.enums.EngineType;
import org.flexlb.exception.BatchScheduleTransportException;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.http.HttpStatus;
import org.springframework.web.reactive.function.client.ClientResponse;
import org.springframework.web.reactive.function.client.ExchangeFunction;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;
import reactor.test.StepVerifier;

import java.time.Duration;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class BatchScheduleCoordinatorTest {
    @Mock private RoundRobinLoadBalancer scheduler;
    @Mock private LBStatusConsistencyService consistency;
    @Mock private ExchangeFunction transport;
    @Mock private EngineHealthReporter reporter;
    @Mock private ObjectProvider<FePool> pools;
    @Mock private FePool pool;
    @Mock private ConfigService configService;
    private final FlexlbConfig config = new FlexlbConfig();
    private BatchScheduleCoordinator coordinator;

    @BeforeEach
    void setUp() {
        config.getHttpDispatcher().setEnabled(true);
        when(configService.loadBalanceConfig()).thenReturn(config);
        coordinator = new BatchScheduleCoordinator(scheduler, consistency, WebClient.builder().exchangeFunction(transport), reporter, pools, configService);
    }

    private static BatchScheduleRequest request(AllocationType type) {
        BatchScheduleRequest request = new BatchScheduleRequest();
        request.setBatchCount(1);
        request.setAllocationType(type);
        return request;
    }

    private static BatchScheduleTarget target() {
        return BatchScheduleTarget.of(new WorkerHost("10.0.0.9", 8000), RoleType.PDFUSION, EngineType.LLM);
    }

    private void follower() {
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.getMasterHostIpPort()).thenReturn("10.0.0.2:7001");
    }

    private void forward(Mono<BatchScheduleResponse> result) {
        when(transport.exchange(any())).thenReturn(result.map(response -> ClientResponse.create(HttpStatus.OK)
                .header("Content-Type", "application/json").body(JsonUtils.toString(response)).build()));
    }

    @ParameterizedTest
    @EnumSource(AllocationType.class)
    void localAllocationSelectsOnlyTheRequestedType(AllocationType type) {
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(true);
        when(consistency.getMasterHostIpPort()).thenReturn("10.0.0.1:7001");
        BatchScheduleTarget target = target();
        if (type == AllocationType.BE) {
            when(scheduler.schedule(1)).thenReturn(Mono.just(BatchScheduleResponse.success(List.of(target))));
        } else {
            when(pools.getIfAvailable()).thenReturn(pool);
            when(pool.nextBatch(1)).thenReturn(List.of("http://fe:8000"));
        }
        BatchScheduleResponse response = coordinator.schedule(request(type)).block();
        assertTrue(response.isSuccess());
        assertEquals("10.0.0.1:7001", response.getRealMasterHost());
        assertEquals(1, response.getServerStatus().size());
        if (type == AllocationType.BE) {
            assertSame(target, response.getServerStatus().getFirst());
            verifyNoInteractions(pools, pool);
        } else {
            assertEquals("http://fe:8000", response.getServerStatus().getFirst().httpUrl());
            assertEquals(RoleType.FRONTEND, response.getServerStatus().getFirst().getRole());
            verify(pool).nextBatch(1);
            verifyNoInteractions(scheduler);
        }
        verifyNoInteractions(transport);
    }

    @Test
    void legacyRtpRequestNeedsNoFrontendPool() {
        config.getHttpDispatcher().setEnabled(false);
        BatchScheduleRequest request = JsonUtils.toObject("{\"batch_count\":1}", BatchScheduleRequest.class);
        when(scheduler.schedule(1)).thenReturn(Mono.just(BatchScheduleResponse.success(List.of(target()))));
        BatchScheduleResponse response = coordinator.schedule(request).block();
        assertTrue(response.isSuccess());
        assertEquals("10.0.0.9", response.getServerStatus().getFirst().getServerIp());
        verifyNoInteractions(pools, pool);
    }

    @ParameterizedTest
    @EnumSource(AllocationType.class)
    void invalidCountsFailBeforeEitherAllocationOrForwarding(AllocationType type) {
        config.getRouter().setBatchScheduleMaxCount(2);
        for (int count : new int[]{-1, 0, 3, Integer.MAX_VALUE}) {
            BatchScheduleRequest request = request(type);
            request.setBatchCount(count);
            assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), coordinator.schedule(request).block().getCode());
        }
        verifyNoInteractions(consistency, scheduler, pools, pool, transport);
    }

    @Test
    void nullAllocationTypeIsRejected() {
        BatchScheduleRequest request = JsonUtils.toObject("{\"batch_count\":1,\"allocation_type\":null}", BatchScheduleRequest.class);
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), coordinator.schedule(request).block().getCode());
        verifyNoInteractions(consistency, scheduler, pools, pool, transport);
    }

    @Test
    void followerNeverReassignsForwardedTargetsAfterPromotion() {
        follower();
        AtomicBoolean master = new AtomicBoolean(false);
        when(consistency.isMaster()).thenAnswer(invocation -> master.get());
        Sinks.One<BatchScheduleResponse> pending = Sinks.one();
        forward(pending.asMono());
        BatchScheduleRequest request = request(AllocationType.BE);
        var future = coordinator.schedule(request).toFuture();
        master.set(true);
        BatchScheduleResponse response = BatchScheduleResponse.success(List.of(target()));
        pending.tryEmitValue(response);
        assertEquals("10.0.0.9", future.join().getServerStatus().getFirst().getServerIp());
        assertEquals(0, request.getForwardHop());
        verifyNoInteractions(scheduler, pools, pool);
    }

    @Test
    void localCompletionDoesNotChangeOwnerAfterDemotion() {
        when(consistency.isNeedConsistency()).thenReturn(true);
        AtomicBoolean master = new AtomicBoolean(true);
        when(consistency.isMaster()).thenAnswer(invocation -> master.get());
        Sinks.One<BatchScheduleResponse> pending = Sinks.one();
        when(scheduler.schedule(anyInt())).thenReturn(pending.asMono());
        var future = coordinator.schedule(request(AllocationType.BE)).toFuture();
        master.set(false);
        pending.tryEmitValue(BatchScheduleResponse.success(List.of(target())));
        assertEquals("10.0.0.9", future.join().getServerStatus().getFirst().getServerIp());
        verifyNoInteractions(transport, pools, pool);
    }

    @ParameterizedTest
    @CsvSource({"1,false,false,FORWARD_HOP_LIMIT", "-1,false,false,FORWARD_HOP_LIMIT",
            "2147483647,false,false,FORWARD_HOP_LIMIT", "0,true,false,SELF_FORWARD_BLOCKED",
            "0,false,true,MASTER_NULL"})
    void rejectsLoopsAndUnknownLeadersWithoutLocalFallback(int hop, boolean self, boolean missing, String code) {
        follower();
        if (self) {
            when(consistency.getLocalHostIp()).thenReturn("10.0.0.2");
        }
        if (missing) {
            when(consistency.getMasterHostIpPort()).thenReturn(null);
        }
        BatchScheduleRequest request = request(AllocationType.BE);
        request.setForwardHop(hop);
        assertEquals(code, assertThrows(BatchScheduleTransportException.class,
                () -> coordinator.schedule(request).block()).getErrorCode());
        verifyNoInteractions(transport, scheduler, pools);
    }

    @ParameterizedTest
    @CsvSource({"200,success", "302,success", "307,success", "404,{}", "502,garbage",
            "500,null", "400,business", "200,missing", "200,empty", "200,fe-only", "200,legacy", "200,wrong-role"})
    void realHttpForwardingPreservesTheWireContract(int status, String body) throws Exception {
        boolean be = !List.of("fe-only", "legacy", "wrong-role").contains(body);
        boolean success = body.equals("success") || body.equals("fe-only");
        try (MockWebServer master = new MockWebServer()) {
            master.start();
            when(consistency.isNeedConsistency()).thenReturn(true);
            when(consistency.getMasterHostIpPort()).thenReturn("localhost:" + master.getPort());
            coordinator = new BatchScheduleCoordinator(scheduler, consistency, WebClient.builder(), reporter, pools, configService);
            BatchScheduleResponse completed = BatchScheduleResponse.success(List.of(
                    be ? target() : BatchScheduleTarget.frontend("http://selected-fe:8000")));
            String payload = switch (body) {
                case "success", "fe-only" -> JsonUtils.toString(completed);
                case "legacy" -> "{\"success\":true,\"server_status\":[],\"frontend_urls\":[\"http://old-fe:8000\"]}";
                case "wrong-role" -> JsonUtils.toString(BatchScheduleResponse.success(List.of(target())));
                case "business" -> JsonUtils.toString(BatchScheduleResponse.error(StrategyErrorType.INVALID_REQUEST, "bad count"));
                case "missing" -> "{\"success\":true}";
                case "empty" -> JsonUtils.toString(BatchScheduleResponse.success(List.of()));
                default -> body;
            };
            master.enqueue(new MockResponse().setResponseCode(status)
                    .setHeader("Content-Type", "application/json").setBody(payload));
            if (status == 200 || body.equals("business")) {
                BatchScheduleResponse response = coordinator.schedule(request(be ? AllocationType.BE : AllocationType.FE)).block(Duration.ofSeconds(5));
                assertEquals(success, response.isSuccess());
                if (response.isSuccess()) {
                    assertEquals(be ? "10.0.0.9" : "selected-fe", response.getServerStatus().getFirst().getServerIp());
                } else if (body.equals("business")) {
                    assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), response.getCode());
                    assertEquals("bad count", response.getErrorMessage());
                }
            } else {
                BatchScheduleTransportException error = assertThrows(BatchScheduleTransportException.class,
                        () -> coordinator.schedule(request(AllocationType.BE)).block(Duration.ofSeconds(5)));
                assertEquals("HTTP_ERROR", error.getErrorCode());
            }
            var sent = master.takeRequest(5, java.util.concurrent.TimeUnit.SECONDS);
            assertEquals("/rtp_llm/batch_schedule", sent.getPath());
            var json = new com.fasterxml.jackson.databind.ObjectMapper().readTree(sent.getBody().readUtf8());
            assertEquals(1, json.get("batch_count").asInt());
            assertEquals(1, json.get("forward_hop").asInt());
            assertEquals(be ? "BE" : "FE", json.get("allocation_type").asText());
            assertFalse(json.has("assign_be"));
            assertFalse(json.has("assign_fe"));
            assertEquals(1, master.getRequestCount());
            verifyNoInteractions(scheduler, pools);
        }
    }

    @Test
    void localBusinessFailureDoesNotConsumeAnFeCursor() {
        when(scheduler.schedule(anyInt())).thenReturn(Mono.just(
                BatchScheduleResponse.error(StrategyErrorType.INVALID_REQUEST, "unsupported topology")));
        assertFalse(coordinator.schedule(request(AllocationType.BE)).block().isSuccess());
        verifyNoInteractions(pools, pool);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void requestedFrontendCannotSucceedWithoutAnAvailablePool(boolean configured) {
        if (configured) {
            when(pools.getIfAvailable()).thenReturn(pool);
            when(pool.nextBatch(1)).thenThrow(new IllegalStateException("empty"));
        }
        BatchScheduleResponse response = coordinator.schedule(request(AllocationType.FE)).block();
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), response.getCode());
        verifyNoInteractions(scheduler);
    }

    @ParameterizedTest
    @CsvSource({"connect,CONNECT_FAILED", "empty,EMPTY_RESPONSE", "timeout,TIMEOUT"})
    void transportFailureNeverFallsBack(String mode, String code) {
        if (mode.equals("connect")) {
            follower();
            when(transport.exchange(any())).thenReturn(Mono.error(new java.net.ConnectException("refused")));
        } else {
            when(scheduler.schedule(anyInt())).thenReturn(mode.equals("empty") ? Mono.empty() : Mono.never());
        }
        StepVerifier.withVirtualTime(() -> coordinator.schedule(request(AllocationType.BE)))
                .thenAwait(Duration.ofSeconds(3))
                .expectErrorSatisfies(error -> assertEquals(code,
                        ((BatchScheduleTransportException) error).getErrorCode()))
                .verify(Duration.ofSeconds(5));
        if (mode.equals("connect")) {
            verifyNoInteractions(scheduler);
        }
        verifyNoInteractions(pools);
    }
}
