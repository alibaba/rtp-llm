package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTimeout;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Locks the sweep-based outstanding-result collection semantics of
 * {@link JavaLoadClient#collectOutstandingResults}.
 *
 * <p>Regression guard for the serial-blocking defect observed on run
 * 20260829_094522: the legacy finalization loop blocked on
 * {@code future.get(remaining)} per future in submission order, so one slow
 * RPC parked the collection cursor until the global deadline and every later
 * future — including ones that had completed — was synthesized as an empty
 * timeout row without send_start, truncating client_events.jsonl real rows to
 * the first ~58% of the send window.
 */
class JavaLoadClientResultCollectionTest {

    private static JavaLoadClient.RequestResult realRow(String rid, double sendStartEpochMs) {
        JavaLoadClient.RequestResult result = new JavaLoadClient.RequestResult();
        result.rid = rid;
        result.status = "ok";
        result.sendStartEpochMs = sendStartEpochMs;
        result.totalMs = 42.0;
        return result;
    }

    private static JavaLoadClient.RequestResult delayedRow(String rid, long sleepMs)
            throws InterruptedException {
        Thread.sleep(sleepMs);
        return realRow(rid, 1_000.0);
    }

    private static JavaLoadClient.RequestResult failingRow(long sleepMs) throws InterruptedException {
        Thread.sleep(sleepMs);
        throw new RuntimeException("boom");
    }

    @Test
    void slowHeadFutureDoesNotBlockCompletedTailFutures() {
        // Defect shape: future[0] is a stuck RPC; futures behind it complete
        // normally. The legacy serial cursor parked on future[0] until the
        // deadline and then synthesized every later row without send_start.
        assertTimeout(Duration.ofSeconds(10), () -> {
            ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
            try {
                List<Future<JavaLoadClient.RequestResult>> futures = new ArrayList<>();
                futures.add(new CompletableFuture<>()); // never completes
                futures.add(executor.submit(() -> delayedRow("fast-1", 50)));
                futures.add(executor.submit(() -> delayedRow("fast-2", 250)));
                futures.add(executor.submit(() -> delayedRow("fast-3", 400)));

                long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
                List<JavaLoadClient.RequestResult> rows =
                        JavaLoadClient.collectOutstandingResults(futures, deadline);

                assertEquals(4, rows.size());
                // stuck head: synthesized timeout row, no send_start
                assertEquals("timeout", rows.get(0).status);
                assertEquals("response deadline exceeded", rows.get(0).error);
                assertEquals("", rows.get(0).rid);
                assertEquals(0.0, rows.get(0).sendStartEpochMs);
                // completed tail: real rows with send_start, in submission order
                for (int i = 1; i <= 3; i++) {
                    assertEquals("fast-" + i, rows.get(i).rid);
                    assertEquals("ok", rows.get(i).status);
                    assertEquals(1_000.0, rows.get(i).sendStartEpochMs);
                }
            } finally {
                executor.shutdownNow();
            }
        });
    }

    @Test
    void rowCountConservedAcrossMixedTerminalStates() {
        assertTimeout(Duration.ofSeconds(10), () -> {
            ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
            try {
                List<Future<JavaLoadClient.RequestResult>> futures = new ArrayList<>();
                futures.add(CompletableFuture.completedFuture(realRow("done-0", 100.0)));
                futures.add(executor.submit(() -> delayedRow("done-1", 80)));
                futures.add(executor.submit(() -> failingRow(40)));
                futures.add(new CompletableFuture<>()); // never completes

                long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(600);
                List<JavaLoadClient.RequestResult> rows =
                        JavaLoadClient.collectOutstandingResults(futures, deadline);

                // one row per future, in submission order
                assertEquals(4, rows.size());
                assertEquals("done-0", rows.get(0).rid);
                assertEquals("ok", rows.get(0).status);
                assertEquals("done-1", rows.get(1).rid);
                assertEquals("ok", rows.get(1).status);
                assertEquals("exception", rows.get(2).status);
                assertTrue(rows.get(2).error.contains("boom"), rows.get(2).error);
                assertEquals("timeout", rows.get(3).status);
            } finally {
                executor.shutdownNow();
            }
        });
    }

    @Test
    void collectionStopsAtGlobalDeadline() {
        // A single stuck future with a short deadline: collection must return
        // shortly after the deadline with a synthesized timeout row — never
        // block on the individual future's own (nonexistent) completion.
        assertTimeout(Duration.ofSeconds(5), () -> {
            List<Future<JavaLoadClient.RequestResult>> futures =
                    List.of(new CompletableFuture<>());
            long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(300);
            long startNanos = System.nanoTime();
            List<JavaLoadClient.RequestResult> rows =
                    JavaLoadClient.collectOutstandingResults(futures, deadline);
            long elapsedMs = (System.nanoTime() - startNanos) / 1_000_000;
            assertEquals(1, rows.size());
            assertEquals("timeout", rows.get(0).status);
            // bounded by the deadline plus at most one sweep interval
            // (generous margin for slow CI machines)
            assertTrue(elapsedMs < 2_000, "collection took " + elapsedMs + "ms");
        });
    }

    @Test
    void alreadyCompletedFuturesReturnImmediatelyWithRealRows() {
        assertTimeout(Duration.ofSeconds(5), () -> {
            List<Future<JavaLoadClient.RequestResult>> futures = new ArrayList<>();
            for (int i = 0; i < 100; i++) {
                futures.add(CompletableFuture.completedFuture(realRow("rid-" + i, i * 10.0)));
            }
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(60);
            long startNanos = System.nanoTime();
            List<JavaLoadClient.RequestResult> rows =
                    JavaLoadClient.collectOutstandingResults(futures, deadline);
            long elapsedMs = (System.nanoTime() - startNanos) / 1_000_000;
            assertEquals(100, rows.size());
            for (int i = 0; i < 100; i++) {
                assertEquals("rid-" + i, rows.get(i).rid);
                assertEquals(i * 10.0, rows.get(i).sendStartEpochMs);
            }
            // nothing pending: the sweep loop must not sleep at all
            assertTrue(elapsedMs < 500, "collection took " + elapsedMs + "ms");
        });
    }

    @Test
    void futureCompletingAfterDeadlineIsSynthesizedNotAwaited() {
        assertTimeout(Duration.ofSeconds(10), () -> {
            ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
            try {
                List<Future<JavaLoadClient.RequestResult>> futures = new ArrayList<>();
                futures.add(executor.submit(() -> delayedRow("late", 800)));
                long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(300);
                List<JavaLoadClient.RequestResult> rows =
                        JavaLoadClient.collectOutstandingResults(futures, deadline);
                assertEquals(1, rows.size());
                assertEquals("timeout", rows.get(0).status);
                assertEquals("", rows.get(0).rid);
            } finally {
                executor.shutdownNow();
            }
        });
    }
}
