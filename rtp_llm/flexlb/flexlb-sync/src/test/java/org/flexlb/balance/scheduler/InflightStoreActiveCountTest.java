package org.flexlb.balance.scheduler;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
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

    private static InflightItem newItem(long requestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        return new InflightItem(ctx, new CompletableFuture<Response>(), null);
    }

    @Test
    void putIfAbsentThenTerminateDecrementsOnce() {
        InflightStore store = new InflightStore(Mockito.mock(BatchSchedulerReporter.class));
        try {
            InflightItem item = newItem(1L);
            assertNull(store.putIfAbsent("1", item));
            assertEquals(1, store.activeCount());

            assertTrue(item.cancel());
            assertEquals(0, store.activeCount());

            // second terminal attempt loses the CAS and must not decrement again
            item.timeout();
            assertEquals(0, store.activeCount());
        } finally {
            store.shutdown();
        }
    }

    @Test
    void terminalBeforeCallbackWiringIsCompensated() {
        InflightStore store = new InflightStore(Mockito.mock(BatchSchedulerReporter.class));
        try {
            // Terminate before registration: transitionTo sees a null callback,
            // so putIfAbsent's compensation path must claim and run the decrement.
            InflightItem item = newItem(2L);
            assertTrue(item.cancel());

            assertNull(store.putIfAbsent("2", item));
            assertEquals(0, store.activeCount());
        } finally {
            store.shutdown();
        }
    }

    @Test
    void concurrentPutIfAbsentAndTerminateNeverLeaksActiveCount() throws Exception {
        InflightStore store = new InflightStore(Mockito.mock(BatchSchedulerReporter.class));
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
                    item.cancel();
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
