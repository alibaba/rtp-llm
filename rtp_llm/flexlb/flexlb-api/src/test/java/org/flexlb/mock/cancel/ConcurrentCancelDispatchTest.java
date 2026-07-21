package org.flexlb.mock.cancel;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.mock.InflightAssertions;
import org.flexlb.mock.MockWorkerBehavior;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Concurrent cancel + dispatch race: multiple requests submitted concurrently
 * while some are being canceled — verify no deadlock, no resource leak.
 *
 * <p>Flow:
 * 1. Configure mock prefill with 500ms delay (gives cancel window before ACK)
 * 2. Concurrently submit 5 requests
 * 3. Concurrently cancel 2 of them (racing with dispatch)
 * 4. Wait for all to complete
 * 5. Verify: no deadlock (completes within timeout), no inflight leak,
 *    non-cancelled requests succeed, cancelled requests are CANCELLED or SUCCESS
 *
 * <p>Uses {@link CompletableFuture} + {@link ExecutorService} + {@link CountDownLatch}
 * to create true concurrency between submit and cancel operations.
 */
class ConcurrentCancelDispatchTest extends FlexLBMockTestBase {

    @Override
    protected MockWorkerBehavior createPrefillBehavior() {
        return MockWorkerBehavior.builder()
                .enqueueDelayMs(500)  // moderate delay: gives cancel window before ACK
                .build();
    }

    @Override
    protected FlexlbConfig createConfig() {
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.setFlexlbBatchEnabled(true);
        cfg.setFlexlbBatchSizeMax(1);        // each request dispatches independently
        cfg.setFlexlbBatchWindowMs(300);
        cfg.setCostSloMs(50_000L);
        cfg.setCostSloRiskMarginMs(50L);
        cfg.setFlexlbBatchEnqueueDeadlineMs(5_000L);
        cfg.setFlexlbInflightTtlMs(300_000L);
        return cfg;
    }

    @Test
    @Timeout(30)
    void concurrentCancelAndDispatch_noDeadlockNoLeak() throws Exception {
        int totalRequests = 5;
        List<Long> requestIds = List.of(9001L, 9002L, 9003L, 9004L, 9005L);
        List<Long> cancelIds = List.of(9001L, 9002L);

        ExecutorService executor = Executors.newFixedThreadPool(totalRequests + cancelIds.size());
        try {
            // Latch ensures all threads start at the same time
            CountDownLatch startLatch = new CountDownLatch(1);

            // --- Submit tasks: each submits a request and stores the future ---
            Map<Long, CompletableFuture<Response>> futureMap = new ConcurrentHashMap<>();
            List<CompletableFuture<Void>> submitTasks = new ArrayList<>();
            for (Long id : requestIds) {
                submitTasks.add(CompletableFuture.runAsync(() -> {
                    try {
                        startLatch.await();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        return;
                    }
                    futureMap.put(id, submitRequest(id));
                }, executor));
            }

            // --- Cancel tasks: cancel 2 requests with a small delay so submit runs first ---
            List<CompletableFuture<Void>> cancelTasks = new ArrayList<>();
            for (Long id : cancelIds) {
                cancelTasks.add(CompletableFuture.runAsync(() -> {
                    try {
                        startLatch.await();
                        Thread.sleep(50);  // small delay: let dispatch start before cancel
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        return;
                    }
                    cancelRequest(id);
                }, executor));
            }

            // --- Start all concurrent operations ---
            startLatch.countDown();

            // Wait for all submissions and cancels to execute
            CompletableFuture.allOf(submitTasks.toArray(new CompletableFuture[0]))
                    .get(5, TimeUnit.SECONDS);
            CompletableFuture.allOf(cancelTasks.toArray(new CompletableFuture[0]))
                    .get(5, TimeUnit.SECONDS);

            // Wait for all request futures to complete (ACK or CANCELLED)
            List<CompletableFuture<Response>> futures = new ArrayList<>(futureMap.values());
            CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
                    .get(10, TimeUnit.SECONDS);

            // 1. Verify: all futures completed (no deadlock)
            for (Long id : requestIds) {
                assertTrue(futureMap.get(id).isDone(),
                        "Request " + id + " should be done (no deadlock)");
            }

            // 2. Verify: non-cancelled requests succeeded
            for (Long id : requestIds) {
                if (!cancelIds.contains(id)) {
                    Response r = futureMap.get(id).getNow(null);
                    assertTrue(r != null && r.isSuccess(),
                            "Non-cancelled request " + id + " should succeed");
                }
            }

            // 3. Verify: cancelled requests are CANCELLED or SUCCESS (race-dependent)
            for (Long id : cancelIds) {
                Response r = futureMap.get(id).getNow(null);
                assertTrue(r != null,
                        "Cancelled request " + id + " should have a response");
                boolean isCancelled = StrategyErrorType.REQUEST_CANCELLED.getErrorCode() == r.getCode();
                boolean isSuccess = r.isSuccess();
                assertTrue(isCancelled || isSuccess,
                        "Cancelled request " + id + " should be CANCELLED or SUCCESS, got code=" + r.getCode());
            }

            // 4. Cleanup: cancel all requests to release inflight resources.
            //    Acknowledged requests are cancelled and removed from their prefill batches;
            //    terminal requests are idempotent no-ops.
            //    In production, this cleanup is done by onWorkerStatusUpdate() when
            //    the engine reports finished tasks — but mock workers don't send updates.
            for (Long id : requestIds) {
                cancelRequest(id);
            }

            // 5. Verify: three-layer inflight cleanup (no resource leak)
            InflightAssertions.assertResourcesReleasedWithin(
                    getPrefillEndpoint(), getDecodeEndpoint(), 5000);
        } finally {
            executor.shutdownNow();
        }
    }
}
