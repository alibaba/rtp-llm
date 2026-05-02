package org.flexlb.balance.dp;

import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class PrefillQueueTest {

    private ScheduledExecutorService timer;
    private ServerStatus prefill;
    private List<PrefillBatch> flushed;

    @BeforeEach
    void setUp() {
        timer = Executors.newSingleThreadScheduledExecutor();
        prefill = new ServerStatus();
        prefill.setServerIp("10.0.0.1");
        prefill.setHttpPort(8080);
        prefill.setGrpcPort(9080);
        flushed = java.util.Collections.synchronizedList(new ArrayList<>());
    }

    @AfterEach
    void tearDown() {
        timer.shutdownNow();
    }

    private PrefillQueue newQueue(int dpSize, int batchSize, long windowMs, long timeoutMs) {
        return new PrefillQueue(prefill, dpSize, batchSize, windowMs, timeoutMs, timer, flushed::add);
    }

    private PendingRequest newReq() {
        return PendingRequest.of(null, prefill, null, new CompletableFuture<Response>());
    }

    // ============== Trigger conditions ==============

    @Test
    void flushes_immediately_when_batch_size_reached() {
        PrefillQueue q = newQueue(4, 4, 5_000, 10_000);  // window/timeout large enough to not interfere
        for (int i = 0; i < 4; i++) {
            q.offer(newReq());
        }
        assertEquals(1, flushed.size(), "reaching batchSize should flush immediately");
        assertEquals(4, flushed.get(0).size());
        assertEquals(0, q.size(), "queue should be empty after drain");
    }

    @Test
    void flushes_on_window_timeout_when_under_batch_size() throws Exception {
        PrefillQueue q = newQueue(4, 4, 30, 10_000);
        q.offer(newReq());
        q.offer(newReq());
        // Below batchSize; wait for the window to expire.
        Thread.sleep(80);
        assertEquals(1, flushed.size(), "window timeout should flush a partial batch");
        assertEquals(2, flushed.get(0).size());
    }

    @Test
    void flushes_on_single_request_timeout_even_within_window() throws Exception {
        // Per-request timeout is very small (10ms): the first request enqueues, and
        // even though the batch window has not yet expired, the second offer detects
        // the head has been waiting too long and force-flushes.
        PrefillQueue q = newQueue(4, 4, 5_000, 10);
        q.offer(newReq());
        Thread.sleep(20);  // ensure head has exceeded the per-request timeout
        q.offer(newReq()); // triggers the head-too-long check
        assertEquals(1, flushed.size(), "head timeout must trigger a flush so low-QPS isn't stalled forever");
        assertEquals(2, flushed.get(0).size());
    }

    @Test
    void no_flush_when_queue_empty_at_window_timeout() throws Exception {
        PrefillQueue q = newQueue(4, 4, 30, 10_000);
        q.offer(newReq());
        // Drain it via the size trigger by adding 3 more — emulates the queue being
        // emptied before the window timer fires.
        for (int i = 0; i < 3; i++) q.offer(newReq());
        assertEquals(1, flushed.size());
        Thread.sleep(80); // The window timer would otherwise fire here, but the queue is empty.
        assertEquals(1, flushed.size(), "window timeout on an empty queue must be a no-op");
    }

    // ============== Residual handling ==============

    @Test
    void residual_after_size_flush_is_dispatched_on_next_window() throws Exception {
        // Push 6 items; with batchSize=4 the first flush takes 4, leaving 2 residual.
        PrefillQueue q = newQueue(4, 4, 30, 10_000);
        // Same-thread serial offer to keep the test deterministic vs size-trigger races.
        for (int i = 0; i < 6; i++) q.offer(newReq());
        // First size flush carried 4.
        assertEquals(1, flushed.size());
        assertEquals(4, flushed.get(0).size());
        // Remaining 2 wait for the next window.
        Thread.sleep(80);
        assertEquals(2, flushed.size(), "residual must be flushed by the next window");
        assertEquals(2, flushed.get(1).size());
    }

    // ============== Concurrency ==============

    @Test
    void concurrent_offers_no_lost_requests() throws Exception {
        int total = 200;
        int batchSize = 4;
        PrefillQueue q = newQueue(batchSize, batchSize, 30, 10_000);
        ExecutorService pool = Executors.newFixedThreadPool(8);
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch done = new CountDownLatch(total);
        for (int i = 0; i < total; i++) {
            pool.submit(() -> {
                try {
                    start.await();
                    q.offer(newReq());
                } catch (InterruptedException ignored) {
                    Thread.currentThread().interrupt();
                } finally {
                    done.countDown();
                }
            });
        }
        start.countDown();
        assertTrue(done.await(5, TimeUnit.SECONDS));
        // Wait for all size-triggered flushes plus the final window flush.
        Thread.sleep(80);

        int totalFlushed = flushed.stream().mapToInt(PrefillBatch::size).sum();
        assertEquals(total, totalFlushed, "concurrent offer must not lose requests");
        for (PrefillBatch b : flushed) {
            assertTrue(b.size() <= batchSize, "no batch may exceed batchSize");
        }
        pool.shutdownNow();
    }

    // ============== Flush callback exceptions ==============

    @Test
    void flush_callback_throwing_completes_futures_exceptionally() {
        PrefillQueue q = new PrefillQueue(prefill, 4, 4, 5_000, 10_000, timer,
                batch -> { throw new RuntimeException("simulated"); });
        List<CompletableFuture<Response>> futures = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            CompletableFuture<Response> f = new CompletableFuture<>();
            futures.add(f);
            q.offer(PendingRequest.of(null, prefill, null, f));
        }
        // Size flush fires, callback throws → every pending future must complete exceptionally.
        for (CompletableFuture<Response> f : futures) {
            assertTrue(f.isCompletedExceptionally(),
                    "when the callback throws, all waiting futures MUST completeExceptionally, otherwise the HTTP request hangs");
        }
    }

    // ============== Constructor validation ==============

    @Test
    void invalid_dpSize_throws() {
        AtomicInteger counter = new AtomicInteger();
        assertThrows(IllegalArgumentException.class,
                () -> new PrefillQueue(prefill, 0, 4, 30, 100, timer, b -> counter.incrementAndGet()));
    }

    @Test
    void invalid_batchSize_throws() {
        AtomicInteger counter = new AtomicInteger();
        assertThrows(IllegalArgumentException.class,
                () -> new PrefillQueue(prefill, 4, 0, 30, 100, timer, b -> counter.incrementAndGet()));
    }
}
