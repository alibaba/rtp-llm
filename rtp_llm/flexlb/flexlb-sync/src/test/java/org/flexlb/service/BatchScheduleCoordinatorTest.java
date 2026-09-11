package org.flexlb.service;

import org.flexlb.balance.strategy.RoundRobinLoadBalancer;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.enums.StatusEnum;
import org.flexlb.exception.BatchScheduleTransportException;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.transport.GeneralHttpNettyService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import reactor.core.publisher.Mono;

import java.net.URI;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class BatchScheduleCoordinatorTest {

    @Mock
    private RoundRobinLoadBalancer batchScheduler;
    @Mock
    private LBStatusConsistencyService consistency;
    @Mock
    private GeneralHttpNettyService httpNettyService;
    @Mock
    private EngineHealthReporter engineHealthReporter;

    private BatchScheduleCoordinator coordinator;

    @BeforeEach
    void setUp() {
        coordinator = new BatchScheduleCoordinator(
                batchScheduler, consistency, httpNettyService, engineHealthReporter);
    }

    @Test
    void schedule_master_resolvesLocallyAndStampsMasterHost() {
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(true);
        when(consistency.getMasterHostIpPort()).thenReturn("10.0.0.1:7001");

        BatchScheduleResponse response = BatchScheduleResponse.success(null);
        when(batchScheduler.schedule(any())).thenReturn(Mono.just(response));

        BatchScheduleRequest forwarded = new BatchScheduleRequest();
        forwarded.setForwardHop(1);
        BatchScheduleResponse returned = coordinator.schedule(forwarded).block();

        assertSame(response, returned);
        assertEquals("10.0.0.1:7001", returned.getRealMasterHost());
        assertTrue(returned.isResolvedLocally());
        verifyNoInteractions(httpNettyService, engineHealthReporter);
    }

    @Test
    void schedule_consistencyDisabled_takesLocalPath() {
        when(consistency.isNeedConsistency()).thenReturn(false);

        BatchScheduleResponse response = BatchScheduleResponse.success(null);
        when(batchScheduler.schedule(any())).thenReturn(Mono.just(response));

        BatchScheduleResponse returned =
                coordinator.schedule(new BatchScheduleRequest()).block();

        assertSame(response, returned);
        assertTrue(returned.isResolvedLocally());
        verifyNoInteractions(httpNettyService);
    }

    @Test
    void schedule_slave_forwardsToMaster() {
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(false);
        when(consistency.getMasterHostIpPort()).thenReturn("10.0.0.2:7001");

        BatchScheduleResponse response = BatchScheduleResponse.success(null);
        response.setCode(200);
        when(httpNettyService.request(
                any(BatchScheduleRequest.class),
                any(URI.class),
                eq("/rtp_llm/batch_schedule"),
                eq(BatchScheduleResponse.class)))
                .thenReturn(Mono.just(response));

        BatchScheduleRequest request = new BatchScheduleRequest();
        request.setBatchCount(3);
        request.setAssignBe(false);
        request.setAssignFe(true);
        BatchScheduleResponse returned = coordinator.schedule(request).block();

        assertSame(response, returned);
        assertFalse(returned.isResolvedLocally());
        verify(engineHealthReporter).reportForwardToMasterResult("10.0.0.2", "200");
        verifyNoInteractions(batchScheduler);
        ArgumentCaptor<BatchScheduleRequest> forwarded = ArgumentCaptor.forClass(BatchScheduleRequest.class);
        verify(httpNettyService).request(forwarded.capture(), eq(URI.create("http://10.0.0.2:7001")),
                eq("/rtp_llm/batch_schedule"), eq(BatchScheduleResponse.class));
        assertEquals(1, forwarded.getValue().getForwardHop());
        assertEquals(3, forwarded.getValue().getBatchCount());
        assertFalse(forwarded.getValue().isAssignBe());
        assertTrue(forwarded.getValue().isAssignFe());
        assertEquals(0, request.getForwardHop(), "forwarding must not mutate the caller's request");
    }

    @Test
    void demotedMasterCannotForwardToItsCachedSelfAddress() {
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(false);
        when(consistency.getMasterHostIpPort()).thenReturn("10.0.0.1:7001");
        when(consistency.getLocalHostIp()).thenReturn("10.0.0.1");

        BatchScheduleTransportException ex = assertThrows(
                BatchScheduleTransportException.class,
                () -> coordinator.schedule(new BatchScheduleRequest()).block());

        assertEquals("SELF_FORWARD_BLOCKED", ex.getErrorCode());
        verify(engineHealthReporter).reportForwardToMasterResult("10.0.0.1", "SELF_FORWARD_BLOCKED");
        verifyNoInteractions(batchScheduler, httpNettyService);
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2, -1})
    void followerCannotRelayAnAlreadyForwardedRequest(int hop) {
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(false);
        when(consistency.getMasterHostIpPort()).thenReturn("10.0.0.2:7001");
        BatchScheduleRequest request = new BatchScheduleRequest();
        request.setForwardHop(hop);

        BatchScheduleTransportException ex = assertThrows(
                BatchScheduleTransportException.class,
                () -> coordinator.schedule(request).block());

        assertEquals("FORWARD_HOP_LIMIT", ex.getErrorCode());
        verify(engineHealthReporter).reportForwardToMasterResult("10.0.0.2", "FORWARD_HOP_LIMIT");
        verifyNoInteractions(batchScheduler, httpNettyService);
    }

    @Test
    void localOriginIsCapturedBeforeAsyncCompletionEvenAfterRoleDowngrade() {
        java.util.concurrent.atomic.AtomicBoolean master =
                new java.util.concurrent.atomic.AtomicBoolean(true);
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenAnswer(ignored -> master.get());
        when(consistency.getMasterHostIpPort()).thenReturn("10.0.0.1:7001");
        BatchScheduleResponse response = BatchScheduleResponse.success(null);
        when(batchScheduler.schedule(any())).thenReturn(Mono.just(response));

        Mono<BatchScheduleResponse> scheduled =
                coordinator.schedule(new BatchScheduleRequest());
        master.set(false);

        BatchScheduleResponse returned = scheduled.block();
        assertTrue(returned.isResolvedLocally(),
                "a role change after branch selection must not reclassify local work");
        assertFalse(org.flexlb.util.JsonUtils.toString(returned).contains("resolvedLocally"),
                "in-process provenance must never cross the wire");
    }

    @Test
    void forwardedOriginStaysRemoteEvenAfterRoleUpgrade() {
        java.util.concurrent.atomic.AtomicBoolean master =
                new java.util.concurrent.atomic.AtomicBoolean(false);
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenAnswer(ignored -> master.get());
        when(consistency.getMasterHostIpPort()).thenReturn("10.0.0.2:7001");
        BatchScheduleResponse response = BatchScheduleResponse.success(null);
        when(httpNettyService.request(any(BatchScheduleRequest.class), any(URI.class),
                eq("/rtp_llm/batch_schedule"), eq(BatchScheduleResponse.class)))
                .thenReturn(Mono.just(response));

        Mono<BatchScheduleResponse> scheduled =
                coordinator.schedule(new BatchScheduleRequest());
        master.set(true);

        assertFalse(scheduled.block().isResolvedLocally(),
                "a role change after forwarding must not authorize a second FE stamp");
    }

    @Test
    void schedule_slave_masterNull_throwsMasterNull() {
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(false);
        when(consistency.getMasterHostIpPort()).thenReturn(null);

        BatchScheduleTransportException ex = assertThrows(
                BatchScheduleTransportException.class,
                () -> coordinator.schedule(new BatchScheduleRequest()).block());

        assertEquals("MASTER_NULL", ex.getErrorCode());
        verify(engineHealthReporter).reportForwardToMasterResult("LOCAL", "MASTER_NULL");
        verifyNoInteractions(batchScheduler, httpNettyService);
    }

    @Test
    void schedule_slave_forwardTimeout_throwsTimeout() {
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(false);
        when(consistency.getMasterHostIpPort()).thenReturn("10.0.0.2:7001");
        when(httpNettyService.request(any(BatchScheduleRequest.class), any(URI.class),
                eq("/rtp_llm/batch_schedule"), eq(BatchScheduleResponse.class)))
                .thenReturn(Mono.error(StatusEnum.READ_TIME_OUT.toException()));

        BatchScheduleTransportException ex = assertThrows(
                BatchScheduleTransportException.class,
                () -> coordinator.schedule(new BatchScheduleRequest()).block());

        assertEquals("TIMEOUT", ex.getErrorCode());
        verify(engineHealthReporter).reportForwardToMasterResult("LOCAL", "TIMEOUT");
    }

    @Test
    void schedule_slave_forwardOtherError_throwsConnectFailed() {
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(false);
        when(consistency.getMasterHostIpPort()).thenReturn("10.0.0.2:7001");
        when(httpNettyService.request(any(BatchScheduleRequest.class), any(URI.class),
                eq("/rtp_llm/batch_schedule"), eq(BatchScheduleResponse.class)))
                .thenReturn(Mono.error(new RuntimeException("connection refused")));

        BatchScheduleTransportException ex = assertThrows(
                BatchScheduleTransportException.class,
                () -> coordinator.schedule(new BatchScheduleRequest()).block());

        assertEquals("CONNECT_FAILED", ex.getErrorCode());
        verify(engineHealthReporter).reportForwardToMasterResult("LOCAL", "CONNECT_FAILED");
    }

    @Test
    void schedule_slave_masterBusinessFailureOverHttp500_returnsTypedResponse() {
        // The master answers business failures as HTTP 500 with a full BatchScheduleResponse
        // body. The transport layer surfaces that as HttpErrorResponseException; the
        // coordinator must reconstruct the typed response instead of reporting CONNECT_FAILED.
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(false);
        when(consistency.getMasterHostIpPort()).thenReturn("10.0.0.2:7001");

        BatchScheduleResponse masterError = BatchScheduleResponse.error(
                StrategyErrorType.INVALID_REQUEST, "batch_count must be in [1, 1000]");
        String body = org.flexlb.util.JsonUtils.toString(masterError);
        when(httpNettyService.request(any(BatchScheduleRequest.class), any(URI.class),
                eq("/rtp_llm/batch_schedule"), eq(BatchScheduleResponse.class)))
                .thenReturn(Mono.error(new org.flexlb.exception.HttpErrorResponseException(500, body)));

        BatchScheduleResponse returned =
                coordinator.schedule(new BatchScheduleRequest()).block();

        assertFalse(returned.isSuccess());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), returned.getCode());
        assertEquals("batch_count must be in [1, 1000]", returned.getErrorMessage());
        verify(engineHealthReporter).reportForwardToMasterResult(
                "10.0.0.2", String.valueOf(StrategyErrorType.INVALID_REQUEST.getErrorCode()));
    }

    @Test
    void schedule_slave_missingMasterEndpoint_doesNotScheduleOnFollower() {
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(false);
        when(consistency.getMasterHostIpPort()).thenReturn("10.0.0.2:7001");
        when(httpNettyService.request(any(BatchScheduleRequest.class), any(URI.class),
                eq("/rtp_llm/batch_schedule"), eq(BatchScheduleResponse.class)))
                .thenReturn(Mono.error(new org.flexlb.exception.HttpErrorResponseException(
                        404, "{\"timestamp\":1,\"status\":404,\"path\":\"/rtp_llm/batch_schedule\"}")));
        BatchScheduleTransportException error = assertThrows(BatchScheduleTransportException.class,
                () -> coordinator.schedule(new BatchScheduleRequest()).block());
        assertEquals("HTTP_ERROR", error.getErrorCode());
        verifyNoInteractions(batchScheduler);
    }

    @ParameterizedTest
    @ValueSource(strings = {"Bad Gateway", "null", "{}"})
    void schedule_slave_http500WithoutBusinessResponse_throwsHttpError(String body) {
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(false);
        when(consistency.getMasterHostIpPort()).thenReturn("10.0.0.2:7001");
        when(httpNettyService.request(any(BatchScheduleRequest.class), any(URI.class),
                eq("/rtp_llm/batch_schedule"), eq(BatchScheduleResponse.class)))
                .thenReturn(Mono.error(new org.flexlb.exception.HttpErrorResponseException(502, body)));

        BatchScheduleTransportException ex = assertThrows(
                BatchScheduleTransportException.class,
                () -> coordinator.schedule(new BatchScheduleRequest()).block());

        // The connection worked — master answered 502 with a non-business body. HTTP_ERROR
        // keeps the metric from pointing operators at the network.
        assertEquals("HTTP_ERROR", ex.getErrorCode());
        verify(engineHealthReporter).reportForwardToMasterResult("LOCAL", "HTTP_ERROR");
    }

    @Test
    void schedule_slave_masterReturnsBusinessError_stillReturnsResponse() {
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(false);
        when(consistency.getMasterHostIpPort()).thenReturn("10.0.0.2:7001");

        BatchScheduleResponse response = BatchScheduleResponse.error(
                StrategyErrorType.NO_AVAILABLE_WORKER, "no worker");
        when(httpNettyService.request(any(BatchScheduleRequest.class), any(URI.class),
                eq("/rtp_llm/batch_schedule"), eq(BatchScheduleResponse.class)))
                .thenReturn(Mono.just(response));

        BatchScheduleResponse returned =
                coordinator.schedule(new BatchScheduleRequest()).block();

        assertSame(response, returned);
        assertFalse(returned.isSuccess());
        verify(engineHealthReporter).reportForwardToMasterResult(
                "10.0.0.2", String.valueOf(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode()));
    }
}
