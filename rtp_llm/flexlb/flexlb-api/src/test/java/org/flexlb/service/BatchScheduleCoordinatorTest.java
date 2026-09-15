package org.flexlb.service;

import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import org.flexlb.balance.strategy.RoundRobinLoadBalancer;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
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
    private BatchScheduleCoordinator coordinator;

    @BeforeEach
    void setUp() {
        coordinator = new BatchScheduleCoordinator(scheduler, consistency, WebClient.builder().exchangeFunction(transport), reporter, pools);
    }

    private static BatchScheduleRequest request(boolean be, boolean fe) {
        BatchScheduleRequest request = new BatchScheduleRequest();
        request.setBatchCount(1);
        request.setAssignBe(be);
        request.setAssignFe(fe);
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
    @CsvSource({"true,true", "false,true", "true,false"})
    void localAllocationCompletesOnlyTheRequestedDimensions(boolean be, boolean fe) {
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(true);
        when(consistency.getMasterHostIpPort()).thenReturn("10.0.0.1:7001");
        BatchScheduleRequest request = request(be, fe);
        BatchScheduleTarget target = be ? target() : new BatchScheduleTarget();
        when(scheduler.schedule(request)).thenReturn(Mono.just(BatchScheduleResponse.success(List.of(target))));
        if (fe) {
            when(pools.getIfAvailable()).thenReturn(pool);
            when(pool.nextBatch(1)).thenReturn(List.of("http://fe:8000"));
        }
        BatchScheduleResponse response = coordinator.schedule(request).block();
        assertTrue(response.isSuccess());
        assertEquals("10.0.0.1:7001", response.getRealMasterHost());
        assertSame(target, response.getServerStatus().getFirst());
        if (fe) {
            assertEquals("http://fe:8000", target.getFeUrl());
            verify(pool).nextBatch(1);
        } else {
            verifyNoInteractions(pools, pool);
        }
        verifyNoInteractions(transport);
    }

    @Test
    void followerPreservesFlagsAndNeverReassignsForwardedTargetsAfterPromotion() {
        follower();
        AtomicBoolean master = new AtomicBoolean(false);
        when(consistency.isMaster()).thenAnswer(invocation -> master.get());
        Sinks.One<BatchScheduleResponse> pending = Sinks.one();
        forward(pending.asMono());
        BatchScheduleRequest request = request(true, true);
        var future = coordinator.schedule(request).toFuture();
        master.set(true);
        BatchScheduleTarget target = target();
        target.setFeUrl("http://master-selected-fe");
        pending.tryEmitValue(BatchScheduleResponse.success(List.of(target)));
        assertEquals("http://master-selected-fe", future.join().getServerStatus().getFirst().getFeUrl());
        assertEquals(0, request.getForwardHop());
        verifyNoInteractions(scheduler, pools, pool);
    }

    @Test
    void localCompletionDoesNotChangeOwnerAfterDemotion() {
        when(consistency.isNeedConsistency()).thenReturn(true);
        AtomicBoolean master = new AtomicBoolean(true);
        when(consistency.isMaster()).thenAnswer(invocation -> master.get());
        Sinks.One<BatchScheduleResponse> pending = Sinks.one();
        when(scheduler.schedule(any())).thenReturn(pending.asMono());
        when(pools.getIfAvailable()).thenReturn(pool);
        when(pool.nextBatch(1)).thenReturn(List.of("http://local-fe"));
        var future = coordinator.schedule(request(false, true)).toFuture();
        master.set(false);
        pending.tryEmitValue(BatchScheduleResponse.success(List.of(new BatchScheduleTarget())));
        assertEquals("http://local-fe", future.join().getServerStatus().getFirst().getFeUrl());
        verify(pool).nextBatch(1);
        verifyNoInteractions(transport);
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
        BatchScheduleRequest request = request(true, false);
        request.setForwardHop(hop);
        assertEquals(code, assertThrows(BatchScheduleTransportException.class,
                () -> coordinator.schedule(request).block()).getErrorCode());
        verifyNoInteractions(transport, scheduler, pools);
    }

    @ParameterizedTest
    @CsvSource({"200,success", "302,success", "307,success", "404,{}", "502,garbage",
            "500,null", "400,business", "200,missing", "200,empty"})
    void realHttpForwardingPreservesTheWireContract(int status, String body) throws Exception {
        try (MockWebServer master = new MockWebServer()) {
            master.start();
            when(consistency.isNeedConsistency()).thenReturn(true);
            when(consistency.getMasterHostIpPort()).thenReturn("localhost:" + master.getPort());
            coordinator = new BatchScheduleCoordinator(scheduler, consistency, WebClient.builder(), reporter, pools);
            BatchScheduleTarget target = target();
            target.setFeUrl("http://selected-fe");
            String payload = switch (body) {
                case "success" -> JsonUtils.toString(BatchScheduleResponse.success(List.of(target)));
                case "business" -> JsonUtils.toString(BatchScheduleResponse.error(StrategyErrorType.INVALID_REQUEST, "bad count"));
                case "missing" -> JsonUtils.toString(BatchScheduleResponse.success(List.of(target())));
                case "empty" -> JsonUtils.toString(BatchScheduleResponse.success(List.of()));
                default -> body;
            };
            master.enqueue(new MockResponse().setResponseCode(status)
                    .setHeader("Content-Type", "application/json").setBody(payload));
            if (status == 200 || body.equals("business")) {
                BatchScheduleResponse response = coordinator.schedule(request(true, true)).block(Duration.ofSeconds(5));
                assertEquals(body.equals("success"), response.isSuccess());
                if (response.isSuccess()) {
                    assertEquals("http://selected-fe", response.getServerStatus().getFirst().getFeUrl());
                } else if (body.equals("business")) {
                    assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), response.getCode());
                    assertEquals("bad count", response.getErrorMessage());
                }
            } else {
                BatchScheduleTransportException error = assertThrows(BatchScheduleTransportException.class,
                        () -> coordinator.schedule(request(true, true)).block(Duration.ofSeconds(5)));
                assertEquals("HTTP_ERROR", error.getErrorCode());
            }
            var sent = master.takeRequest(5, java.util.concurrent.TimeUnit.SECONDS);
            assertEquals("/rtp_llm/batch_schedule", sent.getPath());
            var json = new com.fasterxml.jackson.databind.ObjectMapper().readTree(sent.getBody().readUtf8());
            assertEquals(1, json.get("batch_count").asInt());
            assertEquals(1, json.get("forward_hop").asInt());
            assertTrue(json.get("assign_be").asBoolean());
            assertTrue(json.get("assign_fe").asBoolean());
            assertEquals(1, master.getRequestCount());
            verifyNoInteractions(scheduler, pools);
        }
    }

    @Test
    void localBusinessFailureDoesNotConsumeAnFeCursor() {
        when(scheduler.schedule(any())).thenReturn(Mono.just(
                BatchScheduleResponse.error(StrategyErrorType.INVALID_REQUEST, "unsupported topology")));
        assertFalse(coordinator.schedule(request(true, true)).block().isSuccess());
        verifyNoInteractions(pools, pool);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void requestedFrontendCannotSucceedWithoutAnAvailablePool(boolean configured) {
        when(scheduler.schedule(any())).thenReturn(Mono.just(
                BatchScheduleResponse.success(List.of(new BatchScheduleTarget()))));
        if (configured) {
            when(pools.getIfAvailable()).thenReturn(pool);
            when(pool.nextBatch(1)).thenThrow(new IllegalStateException("empty"));
        }
        BatchScheduleResponse response = coordinator.schedule(request(false, true)).block();
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), response.getCode());
    }

    @ParameterizedTest
    @CsvSource({"connect,CONNECT_FAILED", "empty,EMPTY_RESPONSE", "timeout,TIMEOUT"})
    void transportFailureNeverFallsBack(String mode, String code) {
        if (mode.equals("connect")) {
            follower();
            when(transport.exchange(any())).thenReturn(Mono.error(new java.net.ConnectException("refused")));
        } else {
            when(scheduler.schedule(any())).thenReturn(mode.equals("empty") ? Mono.empty() : Mono.never());
        }
        StepVerifier.withVirtualTime(() -> coordinator.schedule(request(true, false)))
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
