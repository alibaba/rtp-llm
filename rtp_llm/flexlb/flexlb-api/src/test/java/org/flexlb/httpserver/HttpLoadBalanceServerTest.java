package org.flexlb.httpserver;

import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.BatchScheduleContext;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.BatchScheduleCoordinator;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.transport.GeneralHttpNettyService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.web.reactive.function.server.ServerRequest;
import reactor.core.publisher.Mono;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class HttpLoadBalanceServerTest {

    @Mock
    private GeneralHttpNettyService generalHttpNettyService;
    @Mock
    private RouteService routeService;
    @Mock
    private LBStatusConsistencyService lbStatusConsistencyService;
    @Mock
    private EngineHealthReporter engineHealthReporter;
    @Mock
    private QueueManager queueManager;
    @Mock
    private ActiveRequestCounter activeRequestCounter;
    @Mock
    private BatchScheduleCoordinator batchScheduleCoordinator;
    @Mock
    private ServerRequest serverRequest;

    private HttpLoadBalanceServer server;

    @BeforeEach
    void setUp() {
        server = new HttpLoadBalanceServer(generalHttpNettyService, routeService,
                lbStatusConsistencyService, engineHealthReporter, queueManager,
                activeRequestCounter, batchScheduleCoordinator);
    }

    private BatchScheduleContext capturedBatchContext() {
        ArgumentCaptor<BatchScheduleContext> captor = ArgumentCaptor.forClass(BatchScheduleContext.class);
        verify(engineHealthReporter).reportBatchSchedule(captor.capture());
        return captor.getValue();
    }

    @Test
    void batch_schedule_success_returns_200_and_records_response_in_context() {
        BatchScheduleRequest batchRequest = new BatchScheduleRequest();
        batchRequest.setBatchCount(2);
        when(serverRequest.bodyToMono(BatchScheduleRequest.class)).thenReturn(Mono.just(batchRequest));
        when(activeRequestCounter.acquire()).thenReturn(org.mockito.Mockito.mock(
                org.flexlb.service.grace.ActiveRequestCounter.RequestToken.class));

        org.flexlb.dao.loadbalance.BatchScheduleResponse success =
                org.flexlb.dao.loadbalance.BatchScheduleResponse.success(java.util.List.of(
                        new org.flexlb.dao.loadbalance.BatchScheduleTarget("10.0.0.1", 8088, 50051),
                        new org.flexlb.dao.loadbalance.BatchScheduleTarget("10.0.0.2", 8088, 50051)));
        when(batchScheduleCoordinator.schedule(batchRequest)).thenReturn(Mono.just(success));

        org.springframework.web.reactive.function.server.ServerResponse out =
                server.batchScheduleRequest(serverRequest).block();

        assertNotNull(out);
        assertEquals(200, out.statusCode().value());
        BatchScheduleContext bctx = capturedBatchContext();
        assertEquals(success, bctx.getBatchResponse());
        org.junit.jupiter.api.Assertions.assertTrue(bctx.isSuccess());
    }

    @Test
    void batch_schedule_outer_error_backfills_response_for_pv_log() {
        when(serverRequest.bodyToMono(BatchScheduleRequest.class))
                .thenReturn(Mono.error(new IllegalArgumentException("malformed json")));

        org.springframework.web.reactive.function.server.ServerResponse out =
                server.batchScheduleRequest(serverRequest).block();

        assertNotNull(out);
        assertEquals(400, out.statusCode().value(),
                "a malformed body is a deterministic client error and must map to 400, not 500");
        BatchScheduleContext bctx = capturedBatchContext();
        assertFalse(bctx.isSuccess());
        assertNotNull(bctx.getBatchResponse(),
                "outer errors must backfill the response so the PV record carries the real code, not 0");
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), bctx.getBatchResponse().getCode());
    }

    @Test
    void batch_schedule_empty_body_is_reported_as_invalid_request() {
        when(serverRequest.bodyToMono(BatchScheduleRequest.class)).thenReturn(Mono.empty());

        org.springframework.web.reactive.function.server.ServerResponse out =
                server.batchScheduleRequest(serverRequest).block();

        assertNotNull(out);
        assertEquals(400, out.statusCode().value(),
                "an empty body is a deterministic client error and must map to 400, not 500");
        BatchScheduleContext bctx = capturedBatchContext();
        assertFalse(bctx.isSuccess());
        assertNotNull(bctx.getBatchResponse(),
                "an empty body must produce an explicit INVALID_REQUEST response, not an empty Mono");
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), bctx.getBatchResponse().getCode());
    }

    @Test
    void batch_schedule_invalid_request_business_rejection_returns_400() {
        // batch_count out of range / multi-role rejection come back from the coordinator as a
        // success=false response with the INVALID_REQUEST code — deterministic client errors
        // that retrying cannot fix, so the HTTP status must be 400.
        BatchScheduleRequest batchRequest = new BatchScheduleRequest();
        batchRequest.setBatchCount(5000);
        when(serverRequest.bodyToMono(BatchScheduleRequest.class)).thenReturn(Mono.just(batchRequest));
        when(activeRequestCounter.acquire()).thenReturn(org.mockito.Mockito.mock(
                org.flexlb.service.grace.ActiveRequestCounter.RequestToken.class));
        when(batchScheduleCoordinator.schedule(batchRequest)).thenReturn(Mono.just(
                org.flexlb.dao.loadbalance.BatchScheduleResponse.error(
                        StrategyErrorType.INVALID_REQUEST, "batch_count must be in [1, 1000]")));

        org.springframework.web.reactive.function.server.ServerResponse out =
                server.batchScheduleRequest(serverRequest).block();

        assertNotNull(out);
        assertEquals(400, out.statusCode().value());
        BatchScheduleContext bctx = capturedBatchContext();
        assertFalse(bctx.isSuccess());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), bctx.getBatchResponse().getCode());
    }

    @Test
    void batch_schedule_no_available_worker_stays_500() {
        // NO_AVAILABLE_WORKER is a server-side condition (fleet not ready) — retryable, must
        // keep the 5xx class so clients and monitoring treat it as a server failure.
        BatchScheduleRequest batchRequest = new BatchScheduleRequest();
        batchRequest.setBatchCount(2);
        when(serverRequest.bodyToMono(BatchScheduleRequest.class)).thenReturn(Mono.just(batchRequest));
        when(activeRequestCounter.acquire()).thenReturn(org.mockito.Mockito.mock(
                org.flexlb.service.grace.ActiveRequestCounter.RequestToken.class));
        when(batchScheduleCoordinator.schedule(batchRequest)).thenReturn(Mono.just(
                org.flexlb.dao.loadbalance.BatchScheduleResponse.error(
                        StrategyErrorType.NO_AVAILABLE_WORKER, "no BE")));

        org.springframework.web.reactive.function.server.ServerResponse out =
                server.batchScheduleRequest(serverRequest).block();

        assertNotNull(out);
        assertEquals(500, out.statusCode().value());
    }

    @Test
    void batch_schedule_transport_error_stays_500() {
        BatchScheduleRequest batchRequest = new BatchScheduleRequest();
        batchRequest.setBatchCount(2);
        when(serverRequest.bodyToMono(BatchScheduleRequest.class)).thenReturn(Mono.just(batchRequest));
        when(activeRequestCounter.acquire()).thenReturn(org.mockito.Mockito.mock(
                org.flexlb.service.grace.ActiveRequestCounter.RequestToken.class));
        when(batchScheduleCoordinator.schedule(batchRequest)).thenReturn(Mono.error(
                new org.flexlb.exception.BatchScheduleTransportException("master unreachable", "MASTER_NULL")));

        org.springframework.web.reactive.function.server.ServerResponse out =
                server.batchScheduleRequest(serverRequest).block();

        assertNotNull(out);
        assertEquals(500, out.statusCode().value(),
                "transport failures are server-side and must not be downgraded to 4xx");
        BatchScheduleContext bctx = capturedBatchContext();
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), bctx.getBatchResponse().getCode());
    }
}
