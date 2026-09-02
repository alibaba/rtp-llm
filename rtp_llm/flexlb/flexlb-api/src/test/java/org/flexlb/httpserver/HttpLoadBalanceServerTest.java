package org.flexlb.httpserver;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.config.ConfigService;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.BatchScheduleContext;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dispatcher.FePool;
import org.flexlb.dispatcher.MasterFeAssigner;
import org.flexlb.service.BatchScheduleCoordinator;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.synchronizer.MasterEngineSynchronizer;
import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.core.codec.DecodingException;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.server.ServerWebInputException;
import reactor.core.publisher.Mono;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class HttpLoadBalanceServerTest {

    @Mock
    private LBStatusConsistencyService lbStatusConsistencyService;
    @Mock
    private EngineHealthReporter engineHealthReporter;
    @Mock
    private QueueManager queueManager;
    @Mock
    private ConfigService configService;
    @Mock
    private FlexlbBatchScheduler batchScheduler;
    @Mock
    private EndpointRegistry endpointRegistry;
    @Mock
    private MasterEngineSynchronizer masterEngineSynchronizer;
    @Mock
    private ServerScheduleLatencyRecorder serverLatencyRecorder;
    @Mock
    private ActiveRequestCounter activeRequestCounter;
    @Mock
    private BatchScheduleCoordinator batchScheduleCoordinator;
    @Mock
    private ObjectProvider<FePool> fePoolProvider;
    @Mock
    private ServerRequest serverRequest;

    private HttpLoadBalanceServer server;

    private MasterFeAssigner masterFeAssigner;

    @BeforeEach
    void setUp() {
        // Real assigner over the mocked pool provider + consistency view: the stamping tests drive
        // fePoolProvider.getIfAvailable() / isMaster() / isNeedConsistency() exactly as before, now
        // through the shared MasterFeAssigner bean instead of a private method.
        masterFeAssigner = new MasterFeAssigner(fePoolProvider);
        server = new HttpLoadBalanceServer(
                lbStatusConsistencyService,
                queueManager,
                configService,
                batchScheduler,
                endpointRegistry,
                masterEngineSynchronizer,
                serverLatencyRecorder,
                activeRequestCounter,
                batchScheduleCoordinator,
                masterFeAssigner,
                engineHealthReporter);
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
        success.setResolvedLocally(true);
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
    void master_stamps_fe_url_on_each_target_from_local_pool() {
        // On the elected master (resolves locally) with a dispatcher FE pool present, every target
        // gets an fe_url from the single local cursor — this is what lets one master coordinate FE
        // selection across the whole dispatcher fleet. Mutation guard: drop the stamping and the
        // targets come back with null fe_url.
        BatchScheduleRequest batchRequest = new BatchScheduleRequest();
        batchRequest.setBatchCount(2);
        when(serverRequest.bodyToMono(BatchScheduleRequest.class)).thenReturn(Mono.just(batchRequest));
        when(activeRequestCounter.acquire()).thenReturn(org.mockito.Mockito.mock(
                org.flexlb.service.grace.ActiveRequestCounter.RequestToken.class));
        org.flexlb.dao.loadbalance.BatchScheduleResponse success =
                org.flexlb.dao.loadbalance.BatchScheduleResponse.success(java.util.List.of(
                        new org.flexlb.dao.loadbalance.BatchScheduleTarget("10.0.0.1", 8088, 50051),
                        new org.flexlb.dao.loadbalance.BatchScheduleTarget("10.0.0.2", 8088, 50051)));
        success.setResolvedLocally(true);
        when(batchScheduleCoordinator.schedule(batchRequest)).thenReturn(Mono.just(success));

        FePool pool = org.mockito.Mockito.mock(FePool.class);
        when(pool.nextBatch(2)).thenReturn(java.util.List.of("http://fe-1", "http://fe-2"));
        when(fePoolProvider.getIfAvailable()).thenReturn(pool);
        server.batchScheduleRequest(serverRequest).block();

        assertEquals("http://fe-1", success.getServerStatus().get(0).getFeUrl());
        assertEquals("http://fe-2", success.getServerStatus().get(1).getFeUrl());
    }

    @Test
    void slave_forwarding_does_not_restamp_the_master_fe_url() {
        // A slave that forwarded to the master already holds the master's fe_url in the response;
        // it must NOT overwrite it with its own cursor (that would reintroduce the collision the
        // feature removes). The forwarded response keeps resolvedLocally=false, so the pool is
        // never consulted even if this node's role changes before the async response arrives.
        BatchScheduleRequest batchRequest = new BatchScheduleRequest();
        batchRequest.setBatchCount(1);
        when(serverRequest.bodyToMono(BatchScheduleRequest.class)).thenReturn(Mono.just(batchRequest));
        when(activeRequestCounter.acquire()).thenReturn(org.mockito.Mockito.mock(
                org.flexlb.service.grace.ActiveRequestCounter.RequestToken.class));
        org.flexlb.dao.loadbalance.BatchScheduleTarget fromMaster =
                new org.flexlb.dao.loadbalance.BatchScheduleTarget("10.0.0.1", 8088, 50051);
        fromMaster.setFeUrl("http://master-picked-fe");
        org.flexlb.dao.loadbalance.BatchScheduleResponse forwarded =
                org.flexlb.dao.loadbalance.BatchScheduleResponse.success(java.util.List.of(fromMaster));
        when(batchScheduleCoordinator.schedule(batchRequest)).thenReturn(Mono.just(forwarded));
        server.batchScheduleRequest(serverRequest).block();

        assertEquals("http://master-picked-fe", forwarded.getServerStatus().get(0).getFeUrl(),
                "a forwarding slave must preserve the master's fe_url, not restamp it");
        org.mockito.Mockito.verifyNoInteractions(fePoolProvider);
    }

    @Test
    void master_empty_fe_pool_leaves_fe_url_null_without_failing_the_schedule() {
        // An empty FE snapshot makes fePool.nextBatch(...) throw. MasterFeAssigner.assign must
        // swallow it: BE assignment already succeeded, so the schedule response still returns 200
        // with its targets; the affected chunks fail later in the dispatcher (no fallback), not here.
        BatchScheduleRequest batchRequest = new BatchScheduleRequest();
        batchRequest.setBatchCount(1);
        when(serverRequest.bodyToMono(BatchScheduleRequest.class)).thenReturn(Mono.just(batchRequest));
        when(activeRequestCounter.acquire()).thenReturn(org.mockito.Mockito.mock(
                org.flexlb.service.grace.ActiveRequestCounter.RequestToken.class));
        org.flexlb.dao.loadbalance.BatchScheduleResponse success =
                org.flexlb.dao.loadbalance.BatchScheduleResponse.success(java.util.List.of(
                        new org.flexlb.dao.loadbalance.BatchScheduleTarget("10.0.0.1", 8088, 50051)));
        success.setResolvedLocally(true);
        when(batchScheduleCoordinator.schedule(batchRequest)).thenReturn(Mono.just(success));

        FePool pool = org.mockito.Mockito.mock(FePool.class);
        when(pool.nextBatch(1)).thenThrow(new IllegalStateException("no FE endpoints available"));
        when(fePoolProvider.getIfAvailable()).thenReturn(pool);
        org.springframework.web.reactive.function.server.ServerResponse out =
                server.batchScheduleRequest(serverRequest).block();

        assertNotNull(out);
        assertEquals(200, out.statusCode().value(), "an empty FE pool must not fail the schedule");
        assertNull(success.getServerStatus().get(0).getFeUrl(),
                "an empty FE pool leaves fe_url null; the chunk fails later in the dispatcher");
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
        HttpLoadBalanceServer slave = new HttpLoadBalanceServer(
                lbStatusConsistencyService,
                queueManager,
                configService,
                batchScheduler,
                endpointRegistry,
                masterEngineSynchronizer,
                serverLatencyRecorder,
                slaveCounter,
                slaveCoordinator,
                masterFeAssigner,
                org.mockito.Mockito.mock(EngineHealthReporter.class));
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
