package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.constant.MetricConstant;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.FlexlbMetricHelper;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/**
 * TTL safety-net error semantics for {@link InflightItem#complete(Response, InflightState)}.
 *
 * <p>After the TTL error-code unification, all scheduling paths
 * (BATCH/QUEUE/DIRECT, and items registered without a scheduler) uniformly
 * expire with {@link StrategyErrorType#INFLIGHT_TTL_EXPIRED}. The batch
 * dispatch-timeout paths ({@code BatchItem.failTimeout}/{@code failExpired})
 * keep {@code BATCH_SLO_EXPIRED} — covered by {@code GrpcTimeoutTest}.
 */
class InflightItemTtlExpiryTest {

    private static InflightItem newItem(long requestId, AbstractScheduler scheduler,
                                        CompletableFuture<Response> future) {
        Request request = new Request();
        request.setRequestId(requestId);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        return new InflightItem(ctx, future, scheduler);
    }

    private static ConfigService configService(FlexlbConfig config) {
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(config);
        return configService;
    }

    private static BatchScheduler newBatchScheduler(InflightStore store) {
        return new BatchScheduler(configService(new FlexlbConfig()), mock(Router.class),
                mock(EndpointRegistry.class), mock(BatchSchedulerReporter.class),
                store, new FlexlbMetricHelper(null, MetricConstant.PATH_BATCH));
    }

    // ==================== unified error code across paths ====================

    @Test
    void batchSchedulerItemExpiresWithInflightTtlExpired() {
        InflightStore store = newStore();
        try {
            CompletableFuture<Response> future = new CompletableFuture<>();
            InflightItem item = newItem(1L, newBatchScheduler(store), future);

            assertTrue(item.complete(Response.error(StrategyErrorType.INFLIGHT_TTL_EXPIRED,
                    "inflight TTL expired"), InflightState.TIMED_OUT));

            Response response = future.join();
            assertFalse(response.isSuccess());
            assertEquals(StrategyErrorType.INFLIGHT_TTL_EXPIRED.getErrorCode(), response.getCode());
            assertEquals("inflight TTL expired", response.getErrorMessage());
            assertEquals(InflightState.TIMED_OUT, item.state());
        } finally {
            store.shutdown();
        }
    }

    @Test
    void directSchedulerItemExpiresWithInflightTtlExpired() {
        InflightStore store = newStore();
        try {
            DirectScheduler direct = new DirectScheduler(mock(Router.class), store,
                    new FlexlbMetricHelper(null, MetricConstant.PATH_DIRECT));
            CompletableFuture<Response> future = new CompletableFuture<>();
            InflightItem item = newItem(2L, direct, future);

            assertTrue(item.complete(Response.error(StrategyErrorType.INFLIGHT_TTL_EXPIRED,
                    "inflight TTL expired"), InflightState.TIMED_OUT));
            assertEquals(StrategyErrorType.INFLIGHT_TTL_EXPIRED.getErrorCode(),
                    future.join().getCode());
        } finally {
            store.shutdown();
        }
    }

    @Test
    void schedulerlessItemExpiresWithInflightTtlExpired() {
        CompletableFuture<Response> future = new CompletableFuture<>();
        InflightItem item = newItem(3L, null, future);

        assertTrue(item.complete(Response.error(StrategyErrorType.INFLIGHT_TTL_EXPIRED,
                "inflight TTL expired"), InflightState.TIMED_OUT));
        assertEquals(StrategyErrorType.INFLIGHT_TTL_EXPIRED.getErrorCode(),
                future.join().getCode());
        assertEquals(InflightState.TIMED_OUT, item.state());
    }

    // ==================== CAS correctness ====================

    @Test
    void completeTimedOutLosesCasWhenAlreadyTerminal() {
        CompletableFuture<Response> future = new CompletableFuture<>();
        InflightItem item = newItem(4L, null, future);

        Response success = new Response();
        success.setSuccess(true);
        item.complete(success);

        assertFalse(item.complete(Response.error(StrategyErrorType.INFLIGHT_TTL_EXPIRED,
                "inflight TTL expired"), InflightState.TIMED_OUT));
        assertEquals(InflightState.COMPLETED, item.state());
        assertTrue(future.join().isSuccess()); // original response preserved
    }

    @Test
    void secondCompleteTimedOutIsNoOp() {
        CompletableFuture<Response> future = new CompletableFuture<>();
        InflightItem item = newItem(5L, null, future);

        assertTrue(item.complete(Response.error(StrategyErrorType.INFLIGHT_TTL_EXPIRED,
                "inflight TTL expired"), InflightState.TIMED_OUT));
        assertFalse(item.complete(Response.error(StrategyErrorType.INFLIGHT_TTL_EXPIRED,
                "inflight TTL expired"), InflightState.TIMED_OUT));
        assertFalse(item.complete(Response.error(StrategyErrorType.CANCELLED, "cancelled"),
                InflightState.CANCELLED)); // any later terminal attempt loses the CAS
        assertEquals(StrategyErrorType.INFLIGHT_TTL_EXPIRED.getErrorCode(),
                future.join().getCode());
    }

    // ==================== evictor-driven TTL sweep ====================

    @Test
    void evictSweepDeliversUnifiedTtlErrorForBatchRegisteredItem() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbInflightTtlMs(-1); // any age exceeds the TTL immediately
        InflightStore store = new InflightStore(mock(BatchSchedulerReporter.class),
                configService(config));
        try {
            CompletableFuture<Response> future = new CompletableFuture<>();
            InflightItem item = newItem(6L, newBatchScheduler(store), future);
            store.putIfAbsent(item.requestId(), item);

            store.evict();

            assertTrue(item.isTerminated());
            assertEquals(InflightState.TIMED_OUT, item.state());
            assertEquals(StrategyErrorType.INFLIGHT_TTL_EXPIRED.getErrorCode(),
                    future.join().getCode());
            assertEquals(0, store.activeCount());
        } finally {
            store.shutdown();
        }
    }

    private static InflightStore newStore() {
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.lenient().when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        return new InflightStore(mock(BatchSchedulerReporter.class), configService);
    }
}
