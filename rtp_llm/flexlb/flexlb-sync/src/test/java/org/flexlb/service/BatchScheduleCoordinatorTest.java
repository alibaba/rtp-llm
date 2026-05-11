package org.flexlb.service;

import java.net.URI;
import java.util.concurrent.TimeoutException;

import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.exception.BatchScheduleTransportException;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.transport.GeneralHttpNettyService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import reactor.core.publisher.Mono;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class BatchScheduleCoordinatorTest {

    @Mock
    private RouteService routeService;
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
                routeService, consistency, httpNettyService, engineHealthReporter);
    }

    @Test
    void schedule_master_returnsLocalOutcomeAndStampsMasterHost() {
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(true);
        when(consistency.getMasterHostIpPort()).thenReturn("10.0.0.1:7001");

        BatchScheduleResponse response = BatchScheduleResponse.success(null);
        when(routeService.batchSchedule(any())).thenReturn(Mono.just(response));

        BatchScheduleCoordinator.Outcome outcome =
                coordinator.schedule(new BatchScheduleRequest()).block();

        assertEquals(BatchScheduleCoordinator.Source.LOCAL, outcome.getSource());
        assertSame(response, outcome.getResponse());
        assertEquals("10.0.0.1:7001", outcome.getResponse().getRealMasterHost());
        verifyNoInteractions(httpNettyService, engineHealthReporter);
    }

    @Test
    void schedule_consistencyDisabled_takesLocalPath() {
        when(consistency.isNeedConsistency()).thenReturn(false);

        BatchScheduleResponse response = BatchScheduleResponse.success(null);
        when(routeService.batchSchedule(any())).thenReturn(Mono.just(response));

        BatchScheduleCoordinator.Outcome outcome =
                coordinator.schedule(new BatchScheduleRequest()).block();

        assertEquals(BatchScheduleCoordinator.Source.LOCAL, outcome.getSource());
        verifyNoInteractions(httpNettyService);
    }

    @Test
    void schedule_slave_forwardsAndReturnsForwardedOutcome() {
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

        BatchScheduleCoordinator.Outcome outcome =
                coordinator.schedule(new BatchScheduleRequest()).block();

        assertEquals(BatchScheduleCoordinator.Source.FORWARDED, outcome.getSource());
        assertSame(response, outcome.getResponse());
        verify(engineHealthReporter).reportForwardToMasterResult("10.0.0.2", "200");
        verifyNoInteractions(routeService);
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
        verifyNoInteractions(routeService, httpNettyService);
    }

    @Test
    void schedule_slave_forwardTimeout_throwsTimeout() {
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(false);
        when(consistency.getMasterHostIpPort()).thenReturn("10.0.0.2:7001");
        when(httpNettyService.request(any(BatchScheduleRequest.class), any(URI.class),
                eq("/rtp_llm/batch_schedule"), eq(BatchScheduleResponse.class)))
                .thenReturn(Mono.error(new TimeoutException("read timeout")));

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
    void schedule_slave_masterReturnsBusinessError_stillReturnsForwardedOutcome() {
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(false);
        when(consistency.getMasterHostIpPort()).thenReturn("10.0.0.2:7001");

        BatchScheduleResponse response = BatchScheduleResponse.error(
                StrategyErrorType.NO_AVAILABLE_WORKER, "no worker");
        when(httpNettyService.request(any(BatchScheduleRequest.class), any(URI.class),
                eq("/rtp_llm/batch_schedule"), eq(BatchScheduleResponse.class)))
                .thenReturn(Mono.just(response));

        BatchScheduleCoordinator.Outcome outcome =
                coordinator.schedule(new BatchScheduleRequest()).block();

        assertEquals(BatchScheduleCoordinator.Source.FORWARDED, outcome.getSource());
        assertFalse(outcome.getResponse().isSuccess());
        verify(engineHealthReporter).reportForwardToMasterResult(
                "10.0.0.2", String.valueOf(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode()));
    }
}
