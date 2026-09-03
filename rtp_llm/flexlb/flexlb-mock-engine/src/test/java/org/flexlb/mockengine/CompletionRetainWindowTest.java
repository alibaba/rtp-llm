package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.input;
import static org.flexlb.mockengine.MockEngineTestSupport.performanceModel;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.flexlb.mockengine.MockEngineTestSupport.workerStatus;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for the completion backlog delivery protocol rework
 * (dual-flexlb HA prerequisite): {@code getWorkerStatus} used to head-trim
 * the shared {@code completions} queue by the CALLER's cursor, so with two
 * masters polling the same engine each keeping an independent cursor, the
 * first poller permanently destroyed the second poller's unconsumed records
 * (active-master inflight leak). The rework keeps the read path a pure
 * cursor filter (no destruction) and bounds the backlog with a retain
 * window trimmed by {@code periodicCleanup()}.
 *
 * <p>Coverage: dual-consumer non-starvation, single-consumer increment
 * protocol byte-compatibility (same cursor → same increment list), retain
 * window trimming (oldest first, latestVersion never regresses), and a
 * lagging consumer inside the window still receiving its backlog.
 *
 * <p>Completion readiness is awaited via {@code latestFinishedVersion}
 * (monotonic, unaffected by trimming) — after a cleanup the trimmed records
 * are no longer visible to a cursor-0 poll by design.
 */
class CompletionRetainWindowTest {

    private static final AtomicInteger PORT_ALLOCATOR = new AtomicInteger(63500);

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private JavaMockEngineCluster.FastRpcService prefill;

    @BeforeEach
    void setUp() throws IOException {
        scheduler = Executors.newScheduledThreadPool(4, r -> {
            Thread t = new Thread(r, "completion-window-test-scheduler");
            t.setDaemon(true);
            return t;
        });
        services = new ConcurrentHashMap<>();
        int port = PORT_ALLOCATOR.getAndAdd(10);
        prefill = new JavaMockEngineCluster.FastRpcService(
                "prefill", EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                port, services, scheduler, performanceModel(tempDir, "10"), 100,
                new JavaMockEngineCluster.ClusterStats());
        services.put(port, prefill);
    }

    @AfterEach
    void tearDown() throws InterruptedException {
        if (services != null) {
            for (JavaMockEngineCluster.FastRpcService service : services.values()) {
                service.shutdown();
            }
            services = null;
        }
        if (scheduler != null) {
            scheduler.shutdownNow();
            scheduler.awaitTermination(3, TimeUnit.SECONDS);
            scheduler = null;
        }
        prefill = null;
    }

    // ════════════════════════════════════════════════════════════════
    //  Dual-consumer: independent cursors must not starve each other
    // ════════════════════════════════════════════════════════════════

    @Test
    void secondMasterWithIndependentCursorStillReceivesFullBacklog() throws Exception {
        publishCompletions(1, 2);

        // Master A polls first with its own cursor at 0 and receives the
        // full increment.
        EngineRpcService.WorkerStatusPB first = workerStatus(prefill, 0);
        assertEquals(2, first.getFinishedTaskListCount());
        assertEquals(2, first.getLatestFinishedVersion());

        // Master B polls AFTER A with ITS OWN cursor still at 0: the shared
        // backlog must survive A's read (the old head-trim protocol consumed
        // it here, leaking B's inflight entries forever).
        EngineRpcService.WorkerStatusPB second = workerStatus(prefill, 0);
        assertEquals(2, second.getFinishedTaskListCount(),
                "A's poll must not consume B's unconsumed backlog");
        assertEquals(2, second.getLatestFinishedVersion());
        assertEquals(finishedRids(first), finishedRids(second),
                "both consumers with the same cursor see the same increment list");

        // B advances its own cursor to 2 → B's next poll is empty (its own
        // consumption state, independent of A's).
        EngineRpcService.WorkerStatusPB bCaughtUp = workerStatus(prefill, 2);
        assertEquals(0, bCaughtUp.getFinishedTaskListCount());
        assertEquals(2, bCaughtUp.getLatestFinishedVersion());

        // A NEW completion shows up: both consumers independently receive it
        // from their own (now equal) cursors.
        publishCompletions(3, 1);
        EngineRpcService.WorkerStatusPB forA = workerStatus(prefill, 2);
        assertEquals(1, forA.getFinishedTaskListCount());
        assertEquals(3, forA.getLatestFinishedVersion());
        EngineRpcService.WorkerStatusPB forB = workerStatus(prefill, 2);
        assertEquals(1, forB.getFinishedTaskListCount());
        assertEquals(finishedRids(forA), finishedRids(forB));
    }

    @Test
    void repeatedPollAtSameCursorIsIdempotent() throws Exception {
        publishCompletions(1, 2);

        // Single-master compatibility: polling twice at the SAME cursor
        // returns the SAME increment list both times (the master may retry
        // or re-poll before advancing its cursor).
        EngineRpcService.WorkerStatusPB first = workerStatus(prefill, 0);
        EngineRpcService.WorkerStatusPB again = workerStatus(prefill, 0);
        assertEquals(finishedRids(first), finishedRids(again));
        assertEquals(2, again.getLatestFinishedVersion());
    }

    // ════════════════════════════════════════════════════════════════
    //  Single-consumer increment protocol stays unchanged
    // ════════════════════════════════════════════════════════════════

    @Test
    void singleConsumerCursorAdvanceProtocolUnchanged() throws Exception {
        publishCompletions(1, 2);

        // Cursor 0 → the full increment.
        EngineRpcService.WorkerStatusPB all = workerStatus(prefill, 0);
        assertEquals(List.of(1L, 2L), finishedRids(all));
        assertEquals(2, all.getLatestFinishedVersion());

        // Cursor advanced to latest → empty increment, cursor unchanged.
        EngineRpcService.WorkerStatusPB caughtUp = workerStatus(prefill, 2);
        assertEquals(0, caughtUp.getFinishedTaskListCount());
        assertEquals(2, caughtUp.getLatestFinishedVersion());

        // New completion after the cursor → exactly the delta.
        publishCompletions(3, 1);
        EngineRpcService.WorkerStatusPB delta = workerStatus(prefill, 2);
        assertEquals(List.of(3L), finishedRids(delta));
        assertEquals(3, delta.getLatestFinishedVersion());

        // Mid-window cursor → only the tail past that cursor.
        EngineRpcService.WorkerStatusPB tail = workerStatus(prefill, 1);
        assertEquals(List.of(2L, 3L), finishedRids(tail));
    }

    // ════════════════════════════════════════════════════════════════
    //  Retain window: periodicCleanup trims oldest-first
    // ════════════════════════════════════════════════════════════════

    @Test
    void periodicCleanupTrimsBacklogToRetainWindow() throws Exception {
        prefill.setCompletionRetainWindow(2);
        publishCompletions(1, 4);

        // Before cleanup the full backlog is still readable (reads never trim).
        assertEquals(4, workerStatus(prefill, 0).getFinishedTaskListCount());

        // One 60s cleanup pass: only the 2 most recent records survive.
        prefill.periodicCleanup();

        EngineRpcService.WorkerStatusPB afterTrim = workerStatus(prefill, 0);
        assertEquals(2, afterTrim.getFinishedTaskListCount(),
                "backlog is capped at the retain window");
        assertEquals(List.of(3L, 4L), finishedRids(afterTrim),
                "the most recent records are retained, oldest trimmed first");
        assertEquals(4, afterTrim.getLatestFinishedVersion(),
                "latestFinishedVersion never regresses on trim");

        // A slow consumer whose cursor is still inside the retained window
        // keeps receiving its slice.
        EngineRpcService.WorkerStatusPB lagging = workerStatus(prefill, 3);
        assertEquals(List.of(4L), finishedRids(lagging));
    }

    @Test
    void periodicCleanupKeepsBacklogWhenInsideWindow() throws Exception {
        prefill.setCompletionRetainWindow(8);
        publishCompletions(1, 3);

        prefill.periodicCleanup();

        EngineRpcService.WorkerStatusPB afterCleanup = workerStatus(prefill, 0);
        assertEquals(3, afterCleanup.getFinishedTaskListCount(),
                "cleanup must not trim inside the retain window");
        assertEquals(List.of(1L, 2L, 3L), finishedRids(afterCleanup));
    }

    @Test
    void backlogStaysBoundedAcrossPublishAndCleanupRounds() throws Exception {
        prefill.setCompletionRetainWindow(3);
        // Three rounds of (publish 2 completions + cleanup): the queue must
        // converge to the window, never grow past it. Readiness is awaited
        // via latestFinishedVersion — earlier rounds' trimmed records are
        // legitimately invisible to a cursor-0 poll.
        for (int round = 0; round < 3; round++) {
            publishCompletions(1 + round * 2, 2);
            prefill.periodicCleanup();
        }

        EngineRpcService.WorkerStatusPB bounded = workerStatus(prefill, 0);
        assertTrue(bounded.getFinishedTaskListCount() <= 3,
                "backlog never exceeds the retain window across rounds");
        assertEquals(6, bounded.getLatestFinishedVersion());
        // The last round's records (rid 5, 6) plus one older (rid 4) — the
        // 3 most recent by version.
        assertEquals(List.of(4L, 5L, 6L), finishedRids(bounded));
    }

    // ════════════════════════════════════════════════════════════════
    //  Scaffolding
    // ════════════════════════════════════════════════════════════════

    /**
     * Enqueues one single-request batch per requestId (each finishing in
     * ~10ms per the "10" prefill formula) and waits until their
     * completions are published (latestFinishedVersion reaches the last
     * rid — rids are consecutive starting at 1 and the completion version
     * counter is per-engine, so version == rid here).
     */
    private void publishCompletions(long firstRequestId, int count) throws Exception {
        for (int i = 0; i < count; i++) {
            long rid = firstRequestId + i;
            enqueue(prefill, batch(9000 + rid, slot(0, input(rid, 10))));
        }
        awaitVersion(firstRequestId + count - 1);
    }

    /**
     * Polls (at cursor 0 — reads are non-destructive under the rework) until
     * {@code latestFinishedVersion} reaches {@code expectedVersion}. The
     * version counter is monotonic and unaffected by retain-window trimming,
     * so this is the reliable readiness signal even after cleanups.
     */
    private void awaitVersion(long expectedVersion) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(5_000);
        EngineRpcService.WorkerStatusPB last = workerStatus(prefill, 0);
        while (last.getLatestFinishedVersion() < expectedVersion
                && System.nanoTime() < deadline) {
            Thread.sleep(10);
            last = workerStatus(prefill, 0);
        }
        assertEquals(expectedVersion, last.getLatestFinishedVersion(),
                "completions did not publish within the timeout");
    }

    private static List<Long> finishedRids(EngineRpcService.WorkerStatusPB status) {
        return status.getFinishedTaskListList().stream()
                .map(EngineRpcService.TaskInfoPB::getRequestId)
                .collect(Collectors.toList());
    }
}
