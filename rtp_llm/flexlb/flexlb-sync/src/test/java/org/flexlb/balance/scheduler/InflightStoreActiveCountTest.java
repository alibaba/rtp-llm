package org.flexlb.balance.scheduler;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Concurrency tests for the {@link InflightStore#putIfAbsent} active counter.
 *
 * <p>Covers the race between registration (map insert → increment → callback
 * wiring → compensation check) and a concurrent terminal transition: the
 * decrement must run exactly once regardless of interleaving, so
 * {@code activeCount} always returns to zero — never leaks (+1 forever,
 * jamming the maxInflight soft breaker) and never double-decrements
 * (going negative).
 */
class InflightStoreActiveCountTest {

    private static InflightStore newStore() {
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        return new InflightStore(Mockito.mock(BatchSchedulerReporter.class), configService);
    }

    /** Helper: cancel an item via the unified complete() API. */
    private static boolean cancel(InflightItem item) {
        return item.complete(Response.error(StrategyErrorType.CANCELLED, "cancelled"),
                InflightState.CANCELLED);
    }

    private static InflightItem newItem(long requestId) {
        return newItem(requestId, null);
    }

    private static InflightItem newItem(long requestId, AbstractScheduler scheduler) {
        Request request = new Request();
        request.setRequestId(requestId);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        return new InflightItem(ctx, new CompletableFuture<Response>(), scheduler);
    }

    /** Minimal scheduler stub for the per-scheduler bucket tests. */
    private static AbstractScheduler newScheduler() {
        return new AbstractScheduler(null, null) {
            @Override
            public CompletableFuture<Response> submit(BalanceContext ctx) {
                return new CompletableFuture<>();
            }
        };
    }

    @Test
    void putIfAbsentThenTerminateDecrementsOnce() {
        InflightStore store = newStore();
        try {
            InflightItem item = newItem(1L);
            assertNull(store.putIfAbsent("1", item));
            assertEquals(1, store.activeCount());

            assertTrue(cancel(item));
            assertEquals(0, store.activeCount());

            // second terminal attempt loses the CAS and must not decrement again
            item.timeout();
            assertEquals(0, store.activeCount());
        } finally {
            store.shutdown();
        }
    }

    @Test
    void perSchedulerBucketsAreIsolated() {
        InflightStore store = newStore();
        try {
            AbstractScheduler batch = newScheduler();
            AbstractScheduler direct = newScheduler();

            InflightItem batchItem = newItem(10L, batch);
            InflightItem directItem = newItem(11L, direct);
            InflightItem unowned = newItem(12L);
            assertNull(store.putIfAbsent("10", batchItem));
            assertNull(store.putIfAbsent("11", directItem));
            assertNull(store.putIfAbsent("12", unowned));

            // global counter sees all three; buckets only see their own items
            assertEquals(3, store.activeCount());
            assertEquals(1, store.activeCount(batch));
            assertEquals(1, store.activeCount(direct));

            // terminating the DIRECT item must not touch the BATCH bucket
            assertTrue(cancel(directItem));
            assertEquals(2, store.activeCount());
            assertEquals(1, store.activeCount(batch));
            assertEquals(0, store.activeCount(direct));

            // second terminal attempt loses the CAS: bucket not double-decremented
            assertTrue(cancel(batchItem));
            batchItem.timeout();
            assertEquals(0, store.activeCount(batch));

            assertTrue(cancel(unowned));
            assertEquals(0, store.activeCount());
        } finally {
            store.shutdown();
        }
    }

    @Test
    void perSchedulerBucketCompensatedWhenTerminalBeforeWiring() {
        InflightStore store = newStore();
        try {
            AbstractScheduler batch = newScheduler();
            InflightItem item = newItem(20L, batch);
            assertTrue(cancel(item));

            // terminal before registration: compensation path must roll the
            // bucket back together with the global counter
            assertNull(store.putIfAbsent("20", item));
            assertEquals(0, store.activeCount());
            assertEquals(0, store.activeCount(batch));
        } finally {
            store.shutdown();
        }
    }

    @Test
    void terminalBeforeCallbackWiringIsCompensated() {
        InflightStore store = newStore();
        try {
            // Terminate before registration: transitionTo sees a null callback,
            // so putIfAbsent's compensation path must claim and run the decrement.
            InflightItem item = newItem(2L);
            assertTrue(cancel(item));

            assertNull(store.putIfAbsent("2", item));
            assertEquals(0, store.activeCount());
        } finally {
            store.shutdown();
        }
    }

    @Test
    void concurrentPutIfAbsentAndTerminateNeverLeaksActiveCount() throws Exception {
        InflightStore store = newStore();
        ExecutorService pool = Executors.newFixedThreadPool(2);
        try {
            int rounds = 5_000;
            for (int i = 0; i < rounds; i++) {
                InflightItem item = newItem(i);
                String requestId = String.valueOf(i);
                CyclicBarrier barrier = new CyclicBarrier(2);

                Future<?> register = pool.submit(() -> {
                    barrier.await();
                    store.putIfAbsent(requestId, item);
                    return null;
                });
                Future<?> terminate = pool.submit(() -> {
                    barrier.await();
                    cancel(item);
                    return null;
                });
                register.get(5, TimeUnit.SECONDS);
                terminate.get(5, TimeUnit.SECONDS);

                assertTrue(item.isTerminated());
                assertEquals(0, store.activeCount(),
                        "activeCount leaked or double-decremented at round " + i);
            }
            assertEquals(0, store.activeCount());
        } finally {
            pool.shutdownNow();
            store.shutdown();
        }
    }
}
