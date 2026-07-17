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
import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.core.codec.DecodingException;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.server.ServerWebInputException;
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
        // The real codec surfaces a malformed body as ServerWebInputException wrapping a
        // DecodingException — mocking a hand-rolled type here would let the classification drift
        // away from what the decode stage actually throws.
        when(serverRequest.bodyToMono(BatchScheduleRequest.class))
                .thenReturn(Mono.error(new ServerWebInputException("Failed to read HTTP message", null,
                        new DecodingException("JSON decode error: unexpected character"))));

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
    void batch_schedule_decoding_exception_from_the_codec_returns_400() {
        when(serverRequest.bodyToMono(BatchScheduleRequest.class))
                .thenReturn(Mono.error(new DecodingException("JSON decode error: unexpected character")));

        org.springframework.web.reactive.function.server.ServerResponse out =
                server.batchScheduleRequest(serverRequest).block();

        assertNotNull(out);
        assertEquals(400, out.statusCode().value(),
                "a body the codec could not decode is the caller's error whatever type it arrives as");
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                capturedBatchContext().getBatchResponse().getCode());
    }

    @Test
    void batch_schedule_illegal_argument_raised_after_decoding_stays_500() {
        // The decoded body is fine; the fault is ours — e.g. forwardToMaster building a URI from a
        // malformed elected-master address. Classifying by exception type reported this as 400,
        // telling the caller to fix a request that was never the problem.
        BatchScheduleRequest batchRequest = new BatchScheduleRequest();
        batchRequest.setBatchCount(2);
        when(serverRequest.bodyToMono(BatchScheduleRequest.class)).thenReturn(Mono.just(batchRequest));
        when(activeRequestCounter.acquire()).thenReturn(org.mockito.Mockito.mock(
                org.flexlb.service.grace.ActiveRequestCounter.RequestToken.class));
        when(batchScheduleCoordinator.schedule(batchRequest)).thenReturn(Mono.error(
                new IllegalArgumentException("Illegal character in authority at index 7: http://:not-a-host")));

        org.springframework.web.reactive.function.server.ServerResponse out =
                server.batchScheduleRequest(serverRequest).block();

        assertNotNull(out);
        assertEquals(500, out.statusCode().value(),
                "a server-side fault raised after the body decoded must alert as 5xx, not be blamed on the caller");
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(),
                capturedBatchContext().getBatchResponse().getCode());
    }

    @Test
    void batch_schedule_unexpected_runtime_error_stays_500() {
        BatchScheduleRequest batchRequest = new BatchScheduleRequest();
        batchRequest.setBatchCount(2);
        when(serverRequest.bodyToMono(BatchScheduleRequest.class)).thenReturn(Mono.just(batchRequest));
        when(activeRequestCounter.acquire()).thenReturn(org.mockito.Mockito.mock(
                org.flexlb.service.grace.ActiveRequestCounter.RequestToken.class));
        when(batchScheduleCoordinator.schedule(batchRequest))
                .thenReturn(Mono.error(new RuntimeException("boom")));

        org.springframework.web.reactive.function.server.ServerResponse out =
                server.batchScheduleRequest(serverRequest).block();

        assertNotNull(out);
        assertEquals(500, out.statusCode().value());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(),
                capturedBatchContext().getBatchResponse().getCode());
    }

    @Test
    void batch_schedule_server_failure_keeps_500_after_a_slave_forwards_it() {
        // End-to-end shape of a forwarded failure: a slave rebuilds the master's response from the
        // body alone (BatchScheduleCoordinator#forwardToMaster parses it out of the HTTP error and
        // returns it as a business failure, discarding the status) and re-derives the status from
        // that body. So the master's body code must agree with the master's status, or the slave
        // answers 400 for the master's 500 — a retryable server fault read as "your request is bad".
        BatchScheduleRequest batchRequest = new BatchScheduleRequest();
        batchRequest.setBatchCount(2);
        when(serverRequest.bodyToMono(BatchScheduleRequest.class)).thenReturn(Mono.just(batchRequest));
        when(activeRequestCounter.acquire()).thenReturn(org.mockito.Mockito.mock(
                org.flexlb.service.grace.ActiveRequestCounter.RequestToken.class));
        when(batchScheduleCoordinator.schedule(batchRequest))
                .thenReturn(Mono.error(new RuntimeException("boom")));

        org.springframework.web.reactive.function.server.ServerResponse master =
                server.batchScheduleRequest(serverRequest).block();
        assertNotNull(master);
        assertEquals(500, master.statusCode().value());

        // What the slave gets back from forwardToMaster: the master's body, off the wire.
        org.flexlb.dao.loadbalance.BatchScheduleResponse asParsedBySlave = JsonUtils.toObjectOrNull(
                JsonUtils.toStringOrEmpty(capturedBatchContext().getBatchResponse()),
                org.flexlb.dao.loadbalance.BatchScheduleResponse.class);
        assertNotNull(asParsedBySlave, "the master's error body must round-trip as a BatchScheduleResponse");

        ServerRequest slaveRequest = org.mockito.Mockito.mock(ServerRequest.class);
        BatchScheduleCoordinator slaveCoordinator = org.mockito.Mockito.mock(BatchScheduleCoordinator.class);
        ActiveRequestCounter slaveCounter = org.mockito.Mockito.mock(ActiveRequestCounter.class);
        HttpLoadBalanceServer slave = new HttpLoadBalanceServer(generalHttpNettyService, routeService,
                lbStatusConsistencyService, org.mockito.Mockito.mock(EngineHealthReporter.class), queueManager,
                slaveCounter, slaveCoordinator);
        BatchScheduleRequest forwarded = new BatchScheduleRequest();
        forwarded.setBatchCount(2);
        when(slaveRequest.bodyToMono(BatchScheduleRequest.class)).thenReturn(Mono.just(forwarded));
        when(slaveCounter.acquire()).thenReturn(org.mockito.Mockito.mock(
                org.flexlb.service.grace.ActiveRequestCounter.RequestToken.class));
        when(slaveCoordinator.schedule(forwarded)).thenReturn(Mono.just(asParsedBySlave));

        org.springframework.web.reactive.function.server.ServerResponse out =
                slave.batchScheduleRequest(slaveRequest).block();

        assertNotNull(out);
        assertEquals(500, out.statusCode().value(),
                "the master's 500 must still read as a server failure once a slave has forwarded it");
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
