package org.flexlb.sync.synchronizer;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The sync scheduler ticks every 20ms but only hands work off, while a discovery lookup is bounded
 * at 500ms — so without this gate ~25 rounds for the same model/role run concurrently, and an older
 * round finishing last reverts the newer round's membership decisions.
 */
class SingleFlightGateTest {

    private final ExecutorService executor = Executors.newFixedThreadPool(8);
    private final SingleFlightGate gate = new SingleFlightGate();

    @AfterEach
    void tearDown() {
        executor.shutdownNow();
    }

    @Test
    void tickArrivingWhileTheKeyIsBusyIsSkippedNotOverlapped() throws Exception {
        CountDownLatch release = new CountDownLatch(1);
        CountDownLatch started = new CountDownLatch(1);
        AtomicInteger concurrent = new AtomicInteger();
        AtomicInteger maxConcurrent = new AtomicInteger();
        AtomicInteger ran = new AtomicInteger();

        Runnable slowRound = () -> {
            maxConcurrent.accumulateAndGet(concurrent.incrementAndGet(), Math::max);
            ran.incrementAndGet();
            started.countDown();
            try {
                release.await(5, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } finally {
                concurrent.decrementAndGet();
            }
        };

        assertTrue(gate.submit("m/PDFUSION", executor, slowRound), "first tick must run");
        assertTrue(started.await(5, TimeUnit.SECONDS), "first round should have started");

        // 24 further ticks land while the first round is still resolving discovery.
        for (int i = 0; i < 24; i++) {
            assertFalse(gate.submit("m/PDFUSION", executor, slowRound),
                    "a tick arriving while the key is busy must be skipped, not queued");
        }

        release.countDown();
        executor.shutdown();
        assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));

        assertEquals(1, ran.get(), "exactly one round may run per key while it is in flight");
        assertEquals(1, maxConcurrent.get(), "rounds for the same key must never overlap");
    }

    @Test
    void keyIsReleasedSoTheNextTickRuns() throws Exception {
        AtomicInteger ran = new AtomicInteger();
        for (int i = 0; i < 5; i++) {
            CountDownLatch done = new CountDownLatch(1);
            assertTrue(gate.submit("m/PDFUSION", executor, () -> {
                ran.incrementAndGet();
                done.countDown();
            }), "a tick after the previous round finished must run");
            assertTrue(done.await(5, TimeUnit.SECONDS));
            // The gate is cleared in a finally *inside* the task, so give the wrapper a moment to
            // return before the next submit.
            for (int spin = 0; spin < 100 && ran.get() != i + 1; spin++) {
                Thread.sleep(1);
            }
        }
        assertEquals(5, ran.get(), "the gate must not wedge a key permanently");
    }

    @Test
    void differentKeysDoNotBlockEachOther() {
        assertTrue(gate.submit("m/PREFILL", executor, () -> { }));
        assertTrue(gate.submit("m/DECODE", executor, () -> { }),
                "a busy PREFILL round must not gate DECODE");
    }

    @Test
    void throwingTaskReleasesTheKey() throws Exception {
        CountDownLatch threw = new CountDownLatch(1);
        assertTrue(gate.submit("m/PDFUSION", executor, () -> {
            threw.countDown();
            throw new RuntimeException("sync round blew up");
        }));
        assertTrue(threw.await(5, TimeUnit.SECONDS));

        // The release lives in a finally inside the wrapper; give it a moment to run, then the
        // key must accept work again — otherwise one bad round would stop a model/role forever.
        boolean resubmitted = false;
        for (int spin = 0; spin < 5000 && !resubmitted; spin++) {
            resubmitted = gate.submit("m/PDFUSION", executor, () -> { });
            if (!resubmitted) {
                Thread.sleep(1);
            }
        }
        assertTrue(resubmitted, "a task that throws must still release its key");
    }

    @Test
    void rejectedSubmissionReleasesTheKey() {
        ExecutorService rejecting = Executors.newFixedThreadPool(1);
        rejecting.shutdown();

        assertThrows(RejectedExecutionException.class,
                () -> gate.submit("m/PDFUSION", rejecting, () -> { }));
        // If the gate stayed latched on a rejected submit, this key would never sync again.
        assertTrue(gate.submit("m/PDFUSION", executor, () -> { }),
                "a rejected submission must not wedge the key");
    }
}
