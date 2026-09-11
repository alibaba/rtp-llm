package org.flexlb.service;

import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import org.flexlb.balance.strategy.RoundRobinLoadBalancer;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dispatcher.FePool;
import org.flexlb.exception.BatchScheduleTransportException;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.http.HttpStatus;
import org.springframework.web.reactive.function.client.ClientRequest;
import org.springframework.web.reactive.function.client.ClientResponse;
import org.springframework.web.reactive.function.client.ExchangeFunction;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;
import reactor.test.StepVerifier;

import java.net.URI;
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
        return new BatchScheduleTarget("10.0.0.9", 8000, 8001, RoleType.PDFUSION);
    }

    private void follower() {
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(false);
        when(consistency.getMasterHostIpPort()).thenReturn("10.0.0.2:7001");
    }

    private void forward(Mono<BatchScheduleResponse> result) {
        when(transport.exchange(any())).thenReturn(result.map(response -> ClientResponse.create(HttpStatus.OK)
                .header("Content-Type", "application/json").body(JsonUtils.toString(response)).build()));
    }

    @ParameterizedTest
    @CsvSource({"true,true", "false,true", "true,false"})
    void localAllocationCompletesOnlyTheRequestedDimensions(boolean be, boolean fe) {
        BatchScheduleRequest request = request(be, fe);
        BatchScheduleTarget target = be ? target() : new BatchScheduleTarget();
        when(scheduler.schedule(request)).thenReturn(Mono.just(BatchScheduleResponse.success(List.of(target))));
        if (fe) {
            when(pools.getIfAvailable()).thenReturn(pool);
            when(pool.nextBatch(1)).thenReturn(List.of("http://fe:8000"));
        }
        BatchScheduleResponse response = coordinator.schedule(request).block();
        assertTrue(response.isSuccess());
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
    void masterStampsItsAddressAndReturnsTheCompletedAllocation() {
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(true);
        when(consistency.getMasterHostIpPort()).thenReturn("10.0.0.1:7001");
        when(scheduler.schedule(any())).thenReturn(Mono.just(BatchScheduleResponse.success(List.of(target()))));
        assertEquals("10.0.0.1:7001", coordinator.schedule(request(true, false)).block().getRealMasterHost());
        verifyNoInteractions(transport, pools);
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
        ArgumentCaptor<ClientRequest> sent = ArgumentCaptor.forClass(ClientRequest.class);
        verify(transport).exchange(sent.capture());
        assertEquals(URI.create("http://10.0.0.2:7001/rtp_llm/batch_schedule"), sent.getValue().url());
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
    @ValueSource(ints = {1, -1, Integer.MAX_VALUE})
    void rejectsForwardingLoops(int hop) {
        follower();
        BatchScheduleRequest request = request(true, false);
        request.setForwardHop(hop);
        BatchScheduleTransportException error = assertThrows(BatchScheduleTransportException.class,
                () -> coordinator.schedule(request).block());
        assertEquals("FORWARD_HOP_LIMIT", error.getErrorCode());
        verifyNoInteractions(transport, scheduler, pools);
    }

    @Test
    void rejectsSelfForwarding() {
        follower();
        when(consistency.getLocalHostIp()).thenReturn("10.0.0.2");
        BatchScheduleTransportException error = assertThrows(BatchScheduleTransportException.class,
                () -> coordinator.schedule(request(true, false)).block());
        assertEquals("SELF_FORWARD_BLOCKED", error.getErrorCode());
        verifyNoInteractions(transport, scheduler);
    }

    @Test
    void missingLeaderDoesNotAllocateLocally() {
        follower();
        when(consistency.getMasterHostIpPort()).thenReturn(null);
        BatchScheduleTransportException error = assertThrows(BatchScheduleTransportException.class,
                () -> coordinator.schedule(request(true, false)).block());
        assertEquals("MASTER_NULL", error.getErrorCode());
        verifyNoInteractions(transport, scheduler, pools);
    }

    @ParameterizedTest
    @CsvSource({"404,{}", "502,Bad Gateway", "500,null"})
    void nonBusinessHttpErrorsRemainTransportFailures(int status, String body) {
        follower();
        when(transport.exchange(any())).thenReturn(Mono.just(ClientResponse.create(HttpStatus.valueOf(status))
                .header("Content-Type", "application/json").body(body).build()));
        BatchScheduleTransportException error = assertThrows(BatchScheduleTransportException.class,
                () -> coordinator.schedule(request(true, false)).block());
        assertEquals("HTTP_ERROR", error.getErrorCode());
        verifyNoInteractions(scheduler, pools);
    }

    @Test
    void forwardedBusinessFailureKeepsItsCodeAndMessage() {
        follower();
        BatchScheduleResponse rejection = BatchScheduleResponse.error(StrategyErrorType.INVALID_REQUEST, "bad count");
        when(transport.exchange(any())).thenReturn(Mono.just(ClientResponse.create(HttpStatus.BAD_REQUEST)
                .header("Content-Type", "application/json").body(JsonUtils.toString(rejection)).build()));
        BatchScheduleResponse response = coordinator.schedule(request(true, false)).block();
        assertFalse(response.isSuccess());
        assertEquals(rejection.getCode(), response.getCode());
        assertEquals("bad count", response.getErrorMessage());
    }

    @ParameterizedTest
    @ValueSource(ints = {200, 302, 307})
    void realHttpForwardingPreservesTheWireContractAndRejectsRedirects(int status) throws Exception {
        try (MockWebServer master = new MockWebServer()) {
            master.start();
            when(consistency.isNeedConsistency()).thenReturn(true);
            when(consistency.getMasterHostIpPort()).thenReturn("localhost:" + master.getPort());
            coordinator = new BatchScheduleCoordinator(scheduler, consistency, WebClient.builder(), reporter, pools);
            BatchScheduleTarget target = target();
            target.setFeUrl("http://selected-fe");
            master.enqueue(new MockResponse().setResponseCode(status)
                    .setHeader("Content-Type", "application/json")
                    .setBody(JsonUtils.toString(BatchScheduleResponse.success(List.of(target)))));
            if (status == 200) {
                BatchScheduleResponse response = coordinator.schedule(request(true, true)).block(Duration.ofSeconds(5));
                assertEquals("http://selected-fe", response.getServerStatus().getFirst().getFeUrl());
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
    void connectionFailureHasNoLocalFallback() {
        follower();
        when(transport.exchange(any())).thenReturn(Mono.error(new java.net.ConnectException("refused")));
        BatchScheduleTransportException error = assertThrows(BatchScheduleTransportException.class,
                () -> coordinator.schedule(request(true, false)).block());
        assertEquals("CONNECT_FAILED", error.getErrorCode());
        verifyNoInteractions(scheduler, pools);
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

    @Test
    void forwardingRejectsMissingRequestedAssignments() {
        follower();
        forward(Mono.just(BatchScheduleResponse.success(List.of(target()))));
        assertFalse(coordinator.schedule(request(true, true)).block().isSuccess());
        forward(Mono.just(BatchScheduleResponse.success(List.of())));
        assertFalse(coordinator.schedule(request(true, false)).block().isSuccess());
        verifyNoInteractions(scheduler, pools);
    }

    @Test
    void emptyResponseIsAnExplicitTransportFailure() {
        when(scheduler.schedule(any())).thenReturn(Mono.empty());
        BatchScheduleTransportException error = assertThrows(BatchScheduleTransportException.class,
                () -> coordinator.schedule(request(true, false)).block());
        assertEquals("EMPTY_RESPONSE", error.getErrorCode());
    }

    @Test
    void allocationHasABoundedDeadline() {
        when(scheduler.schedule(any())).thenReturn(Mono.never());
        StepVerifier.withVirtualTime(() -> coordinator.schedule(request(true, false)))
                .thenAwait(Duration.ofSeconds(3))
                .expectErrorSatisfies(error -> assertEquals("TIMEOUT",
                        ((BatchScheduleTransportException) error).getErrorCode()))
                .verify(Duration.ofSeconds(5));
    }
}
