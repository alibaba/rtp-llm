package org.flexlb.mock.cancel;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.mock.InflightAssertions;
import org.flexlb.mock.MockWorkerBehavior;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Concurrent cancel + onSuccess race: cancel and ACK happen simultaneously,
 * verify rollback only executes once (rolledBack CAS protection).
 *
 * <p>Flow:
 * 1. Configure mock prefill with very short delay (10ms) so ACK returns quickly
 * 2. Submit request
 * 3. Cancel concurrently with ACK using CountDownLatch for timing control
 * 4. Verify: final state is CANCELLED or SUCCESS, inflight clean, no double-release
 *
 * <p>Key invariant: regardless of whether cancel() or onSuccess() wins the race,
 * the {@code rolledBack} AtomicBoolean CAS in {@code rollbackOnce()} ensures
 * that resource rollback (decode KV release) executes exactly once.
 *
 * <p>Race outcomes:
 * <ul>
 *   <li>cancel wins: entry removed before ACK → future=CANCELLED, cancelPrefill sent,
 *       onSuccess finds null entry → returns. Cancel count = 1.</li>
 *   <li>onSuccess wins: ACK processed first → future=SUCCESS, entry stays in inflight,
 *       cancel removes entry → cancelPrefill sent. Cancel count = 1.</li>
 *   <li>True race: both hold entry reference → synchronized(entry) serializes access.
 *       rolledBack CAS ensures rollback once. Cancel count = 1.</li>
 * </ul>
 */
class ConcurrentCancelOnSuccessTest extends FlexLBMockTestBase {

    @Override
    protected MockWorkerBehavior createPrefillBehavior() {
        return MockWorkerBehavior.builder()
                .enqueueDelayMs(10)  // very short delay: ACK returns almost immediately
                .build();
    }

    @Override
    protected FlexlbConfig createConfig() {
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.setFlexlbBatchEnabled(true);
        cfg.setFlexlbBatchSizeMax(1);        // immediate dispatch
        cfg.setFlexlbBatchWindowMs(300);
        cfg.setCostSloMs(50_000L);
        cfg.setCostSloRiskMarginMs(50L);
        cfg.setFlexlbBatchFillThreshold(1.0);
        cfg.setFlexlbBatchEnqueueDeadlineMs(5_000L);
        cfg.setFlexlbInflightTtlMs(300_000L);
        return cfg;
    }

    @Test
    @Timeout(30)
    void concurrentCancelAndOnSuccess_noDoubleRelease() throws Exception {
        long requestId = 9101;

        // CountDownLatch: cancel thread starts the moment submit returns,
        // maximizing the chance of racing with the 10ms-delayed ACK
        CountDownLatch submitDone = new CountDownLatch(1);

        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            // --- Cancel thread: waits for submit to complete, then cancels ---
            CompletableFuture<Void> cancelTask = CompletableFuture.runAsync(() -> {
                try {
                    submitDone.await();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return;
                }
                cancelRequest(requestId);
            }, executor);

            // --- Main thread: submit request, then signal cancel thread ---
            CompletableFuture<Response> future = submitRequest(requestId);
            submitDone.countDown();

            // --- Wait for both the request future and the cancel task ---
            Response response = future.get(5, TimeUnit.SECONDS);
            cancelTask.get(5, TimeUnit.SECONDS);

            // 1. Verify: final state is CANCELLED or SUCCESS (race-dependent, both valid)
            boolean isCancelled = StrategyErrorType.REQUEST_CANCELLED.getErrorCode() == response.getCode();
            boolean isSuccess = response.isSuccess();
            assertTrue(isCancelled || isSuccess,
                    "Response should be SUCCESS or CANCELLED, got success=" + response.isSuccess()
                            + " code=" + response.getCode());

            // 2. Verify: three-layer inflight cleanup (no resource leak)
            //    If double-release occurred (rolledBack CAS failed), DecodeEndpoint
            //    inflight count would be negative — assertResourcesReleasedWithin catches this
            InflightAssertions.assertResourcesReleasedWithin(
                    getPrefillEndpoint(), getDecodeEndpoint(), 5000);

            // 3. Verify: no double-release — cancel called at most once
            //    rolledBack CAS in rollbackOnce() ensures rollback executes exactly once
            assertTrue(mockPrefillWorker.getCancelCount() <= 1,
                    "Cancel should be called at most once (no double-release), got: "
                            + mockPrefillWorker.getCancelCount());
        } finally {
            executor.shutdownNow();
        }
    }
}
