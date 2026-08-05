package org.flexlb.balance.scheduler;

import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.constant.MetricConstant;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.FlexlbMetricHelper;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.slf4j.LoggerFactory;

import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

/**
 * Full evidence-chain tests for two of the three inflight defence paths
 * (red-team audit: these paths never fire in integration tests, so their
 * behaviour is locked down at the UT layer here):
 *
 * <ol>
 *   <li><b>TTL safety net</b> — {@link InflightStore#evict()} driving
 *       {@link InflightItem#timeoutWithError()} for over-age RUNNING items:
 *       terminal state, unified error code, future completion, exactly-once
 *       activeCount decrement, decode-EP KV reservation release, eviction
 *       log, and terminal metric. Plus the negative guards: terminal
 *       tombstones and fresh RUNNING items are never killed.</li>
 *   <li><b>Cancel</b> — {@link InflightItem#cancel()} resource cleanup:
 *       immediate KV release (no TTL wait), terminal state, exceptional
 *       future completion, activeCount decrement, cancel metric, and
 *       idempotent second cancel.</li>
 * </ol>
 *
 * <p>The third path (STALE round-based engine-task eviction) is EP-layer
 * only and covered in {@code EndpointStaleEvictionChainTest}.
 *
 * <p>No sleeps: TTL expiry is triggered through the configurable
 * {@code flexlbInflightTtlMs} (set to {@code -1} so any age exceeds it),
 * matching the approach of {@link InflightItemTtlExpiryTest}.
 */
class InflightDefenseChainTest {

    private ch.qos.logback.classic.Logger flexlbLogger;
    private ListAppender<ILoggingEvent> logAppender;

    @BeforeEach
    void attachLogCapture() {
        flexlbLogger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("flexlbLogger");
        logAppender = new ListAppender<>();
        logAppender.start();
        flexlbLogger.addAppender(logAppender);
    }

    @AfterEach
    void detachLogCapture() {
        flexlbLogger.detachAppender(logAppender);
        logAppender.stop();
    }

    private boolean hasWarnLogContaining(String fragment) {
        return logAppender.list.stream()
                .anyMatch(event -> event.getFormattedMessage().contains(fragment));
    }

    // ---- fixtures ----

    private static InflightStore newStore(long inflightTtlMs) {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbInflightTtlMs(inflightTtlMs);
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.lenient().when(configService.loadBalanceConfig()).thenReturn(config);
        return new InflightStore(mock(BatchSchedulerReporter.class), configService);
    }

    private static InflightItem newItem(long requestId, CompletableFuture<Response> future) {
        Request request = new Request();
        request.setRequestId(requestId);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        return new InflightItem(ctx, future, null);
    }

    private static DecodeEndpoint newDecodeEndpoint() {
        WorkerStatus status = new WorkerStatus();
        status.setIp("10.0.0.1");
        status.setPort(8080);
        status.setGrpcPort(8081);
        return new DecodeEndpoint(status, null);
    }

    // ==================== 1. TTL safety-net full chain ====================

    @Test
    void ttlEvictOfRunningItemRunsFullCleanupChain() throws Exception {
        InflightStore store = newStore(-1); // any age exceeds the TTL — no sleep needed
        DecodeEndpoint decodeEp = newDecodeEndpoint();
        PrefillEndpoint prefillEp = mock(PrefillEndpoint.class);
        FlexMonitor monitor = mock(FlexMonitor.class);
        try {
            CompletableFuture<Response> future = new CompletableFuture<>();
            InflightItem item = newItem(100L, future);

            // decode-EP binding with a live KV reservation
            decodeEp.reserve(100L, 500, 800);
            assertEquals(500, decodeEp.decodeInflightHardKvReserved());
            assertEquals(800, decodeEp.decodeInflightExpectedKvReserved());
            item.setDecodeEp(decodeEp);
            item.setPrefillEp(prefillEp);
            item.setMetricHelper(new FlexlbMetricHelper(monitor, MetricConstant.PATH_BATCH));

            store.putIfAbsent(item.requestId(), item);
            assertEquals(1, store.activeCount());

            store.evict();

            // terminal state + unified error code + future completion
            assertEquals(InflightState.TIMED_OUT, item.state());
            assertTrue(future.isDone());
            Response response = future.join();
            assertFalse(response.isSuccess());
            assertEquals(StrategyErrorType.INFLIGHT_TTL_EXPIRED.getErrorCode(), response.getCode());
            assertEquals("inflight TTL expired", response.getErrorMessage());

            // activeCount decremented; KV reservation fully released (both counters)
            assertEquals(0, store.activeCount());
            assertEquals(0, decodeEp.decodeInflightCount());
            assertEquals(0, decodeEp.decodeInflightHardKvReserved());
            assertEquals(0, decodeEp.decodeInflightExpectedKvReserved());
            verify(prefillEp).release(100L);

            // terminal metric (TIMED_OUT → timeout QPS) + eviction log
            verify(monitor).report(eq(MetricConstant.REQUEST_TIMEOUT_QPS),
                    any(FlexMetricTags.class), eq(1.0));
            assertTrue(hasWarnLogContaining("inflight TTL expired: request_id=100"),
                    "TTL eviction must emit the safety-net warn log");
        } finally {
            store.shutdown();
        }
    }

    @Test
    void ttlEvictDecrementsActiveCountExactlyOnce() {
        InflightStore store = newStore(-1);
        DecodeEndpoint decodeEp = newDecodeEndpoint();
        try {
            InflightItem item = newItem(101L, new CompletableFuture<>());
            decodeEp.reserve(101L, 300, 300);
            item.setDecodeEp(decodeEp);
            store.putIfAbsent(item.requestId(), item);

            store.evict();
            assertEquals(0, store.activeCount());

            // a second sweep and a late terminal attempt must not decrement again
            store.evict();
            assertFalse(item.cancel());
            assertEquals(0, store.activeCount(), "activeCount must never go negative");
            // KV release is also exactly-once: counters stay at zero, not negative
            assertEquals(0, decodeEp.decodeInflightHardKvReserved());
            assertEquals(0, decodeEp.decodeInflightExpectedKvReserved());
        } finally {
            store.shutdown();
        }
    }

    @Test
    void completedItemIsNeverKilledByTtlSweep() {
        InflightStore store = newStore(-1); // even with an instantly-expiring TTL
        FlexMonitor monitor = mock(FlexMonitor.class);
        try {
            CompletableFuture<Response> future = new CompletableFuture<>();
            InflightItem item = newItem(102L, future);
            item.setMetricHelper(new FlexlbMetricHelper(monitor, MetricConstant.PATH_BATCH));
            store.putIfAbsent(item.requestId(), item);

            Response success = new Response();
            success.setSuccess(true);
            item.complete(success);
            assertEquals(0, store.activeCount());

            store.evict();

            // tombstone untouched: state and original response preserved
            assertEquals(InflightState.COMPLETED, item.state());
            assertTrue(future.join().isSuccess());
            verify(monitor, never()).report(eq(MetricConstant.REQUEST_TIMEOUT_QPS),
                    any(FlexMetricTags.class), eq(1.0));
            assertFalse(hasWarnLogContaining("inflight TTL expired: request_id=102"));
        } finally {
            store.shutdown();
        }
    }

    @Test
    void freshRunningItemSurvivesTtlSweep() {
        InflightStore store = newStore(300_000); // generous TTL: item is not over-age
        try {
            CompletableFuture<Response> future = new CompletableFuture<>();
            InflightItem item = newItem(103L, future);
            store.putIfAbsent(item.requestId(), item);

            store.evict();

            assertEquals(InflightState.RUNNING, item.state());
            assertFalse(future.isDone());
            assertEquals(1, store.activeCount());
        } finally {
            store.shutdown();
        }
    }

    // ==================== 3. Cancel full chain ====================

    @Test
    void cancelReleasesDecodeKvImmediatelyAndRunsFullChain() {
        InflightStore store = newStore(300_000); // TTL far away — cancel must not wait for it
        DecodeEndpoint decodeEp = newDecodeEndpoint();
        FlexMonitor monitor = mock(FlexMonitor.class);
        try {
            CompletableFuture<Response> future = new CompletableFuture<>();
            InflightItem item = newItem(200L, future);
            decodeEp.reserve(200L, 500, 800);
            item.setDecodeEp(decodeEp);
            item.setMetricHelper(new FlexlbMetricHelper(monitor, MetricConstant.PATH_QUEUE));
            store.putIfAbsent(item.requestId(), item);
            assertEquals(1, store.activeCount());

            assertTrue(item.cancel());

            // KV reservation released immediately, not by the TTL sweep
            assertEquals(0, decodeEp.decodeInflightCount());
            assertEquals(0, decodeEp.decodeInflightHardKvReserved());
            assertEquals(0, decodeEp.decodeInflightExpectedKvReserved());

            // terminal state: CANCELLED now maps to InflightState.CANCELLED
            assertEquals(InflightState.CANCELLED, item.state());
            assertTrue(item.isTerminated());

            // future completed exceptionally with CancellationException
            // (CompletableFuture surfaces it directly from get(), unwrapped)
            assertTrue(future.isCompletedExceptionally());
            assertThrows(CancellationException.class, future::get);

            // activeCount decremented; cancel metric reported
            assertEquals(0, store.activeCount());
            verify(monitor).report(eq(MetricConstant.REQUEST_CANCEL_QPS),
                    any(FlexMetricTags.class), eq(1.0));
        } finally {
            store.shutdown();
        }
    }

    @Test
    void repeatedCancelIsIdempotentAndReturnsFalse() {
        InflightStore store = newStore(300_000);
        DecodeEndpoint decodeEp = newDecodeEndpoint();
        FlexMonitor monitor = mock(FlexMonitor.class);
        try {
            InflightItem item = newItem(201L, new CompletableFuture<>());
            decodeEp.reserve(201L, 400, 400);
            item.setDecodeEp(decodeEp);
            item.setMetricHelper(new FlexlbMetricHelper(monitor, MetricConstant.PATH_QUEUE));
            store.putIfAbsent(item.requestId(), item);

            assertTrue(item.cancel());
            assertFalse(item.cancel(), "second cancel must lose the CAS and return false");

            // no double decrement / double KV release
            assertEquals(0, store.activeCount());
            assertEquals(0, decodeEp.decodeInflightHardKvReserved());
            assertEquals(0, decodeEp.decodeInflightExpectedKvReserved());
            // cancel metric reported exactly once
            verify(monitor, Mockito.times(1)).report(eq(MetricConstant.REQUEST_CANCEL_QPS),
                    any(FlexMetricTags.class), eq(1.0));
        } finally {
            store.shutdown();
        }
    }
}
