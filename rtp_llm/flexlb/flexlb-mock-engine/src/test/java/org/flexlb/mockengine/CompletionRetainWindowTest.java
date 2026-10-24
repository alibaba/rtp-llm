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

import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.input;
import static org.flexlb.mockengine.MockEngineTestSupport.performanceModel;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.flexlb.mockengine.MockEngineTestSupport.workerStatus;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Real runtime-meta protocol: independent opaque cursors, fixed FIFO history and wall-clock status. */
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
                port, services, scheduler, performanceModel(tempDir, "10"), 4096,
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

    @Test
    void consumersHaveIndependentOpaqueCursors() throws Exception {
        assertEquals(0, workerStatus(prefill, 0).getLatestFinishedVersion());
        publishCompletions(1, 2);
        var first = workerStatus(prefill, 0);
        long cursor = first.getLatestFinishedVersion();
        assertTrue(cursor > System.currentTimeMillis() * 1000 - 10_000_000L);
        assertEquals(List.of(1L, 2L), finishedRids(first));
        assertEquals(first.getFinishedTaskListList(), workerStatus(prefill, 0).getFinishedTaskListList());
        var caughtUp = workerStatus(prefill, cursor);
        assertEquals(0, caughtUp.getFinishedTaskListCount());
        assertEquals(cursor, caughtUp.getLatestFinishedVersion());
        publishCompletions(3, 1);
        var delta = workerStatus(prefill, cursor);
        assertEquals(List.of(3L), finishedRids(delta));
        assertEquals(cursor + 1, delta.getLatestFinishedVersion());
        assertEquals(delta.getFinishedTaskListList(), workerStatus(prefill, cursor).getFinishedTaskListList());
        prefill.periodicCleanup();
        assertEquals(3, workerStatus(prefill, 0).getFinishedTaskListCount());
    }

    @Test
    void emptyDeltaEchoesFutureCursorAndStatusVersionIsWallTime() throws Exception {
        long before = System.currentTimeMillis() * 1000;
        var empty = workerStatus(prefill, Long.MAX_VALUE);
        assertEquals(Long.MAX_VALUE, empty.getLatestFinishedVersion());
        assertTrue(empty.getStatusVersion() >= before);
        assertTrue(empty.getStatusVersion() <= System.currentTimeMillis() * 1000 + 999);
        publishCompletions(1, 1);
        var future = workerStatus(prefill, Long.MAX_VALUE);
        assertEquals(0, future.getFinishedTaskListCount());
        assertEquals(Long.MAX_VALUE, future.getLatestFinishedVersion());
    }

    @Test
    void defaultCapacityCutsThe1001stRecordWithoutConsumerOrCleanup() throws Exception {
        assertEquals(1000, JavaMockEngineCluster.FastRpcService.DEFAULT_COMPLETION_RETAIN_WINDOW);
        prefill.setAutoFetch(true);
        EngineRpcService.GenerateInputPB[] wave = new EngineRpcService.GenerateInputPB[1000];
        for (int i = 0; i < wave.length; i++) {
            wave[i] = input(i + 1, 10);
        }
        // Retention is independent of prepare-pool overload: dispatch bounded waves without status reads.
        for (int start = 0; start < wave.length; start += 64) {
            var group = java.util.Arrays.copyOfRange(wave, start, Math.min(wave.length, start + 64));
            var ack = enqueue(prefill, batch(10001 + start, slot(0, group)));
            assertEquals(group.length, ack.getSuccessesCount(), ack.toString());
            assertEquals(0, ack.getErrorsCount());
        }
        // Observe local completion only; the slow master has made NO status read.
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        while (prefill.getCompletedCount() < 1000 && System.nanoTime() < deadline) {
            Thread.sleep(2);
        }
        assertEquals(1000, prefill.getCompletedCount());
        var atCapacity = workerStatus(prefill, 0);
        assertEquals(1000, atCapacity.getFinishedTaskListCount());
        assertEquals(1L, finishedRids(atCapacity).get(0));
        assertEquals(1000L, finishedRids(atCapacity).get(999));

        assertEquals(1, enqueue(prefill, batch(10002, slot(0, input(1001, 10)))).getSuccessesCount());
        deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (prefill.getCompletedCount() < 1001 && System.nanoTime() < deadline) {
            Thread.sleep(2);
        }
        assertEquals(1001, prefill.getCompletedCount());
        var delayed = workerStatus(prefill, 0);
        assertEquals(1000, delayed.getFinishedTaskListCount());
        assertEquals(2L, finishedRids(delayed).get(0));
        assertEquals(1001L, finishedRids(delayed).get(999));
        assertEquals(atCapacity.getLatestFinishedVersion() + 1, delayed.getLatestFinishedVersion());
        assertEquals(List.of(1001L), finishedRids(workerStatus(prefill, atCapacity.getLatestFinishedVersion())));
        assertEquals(finishedRids(delayed), finishedRids(workerStatus(prefill, 0)),
                "another consumer cannot recover the clipped rid or consume this one's backlog");
    }

    private void publishCompletions(long firstRequestId, int count) throws Exception {
        long initial = firstRequestId - 1;
        for (int i = 0; i < count; i++) {
            long rid = firstRequestId + i;
            enqueue(prefill, batch(9000 + rid, slot(0, input(rid, 10))));
        }
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (workerStatus(prefill, 0).getFinishedTaskListCount() < Math.min(1000, initial + count)
                && System.nanoTime() < deadline) Thread.sleep(2);
        assertEquals(Math.min(1000, initial + count), workerStatus(prefill, 0).getFinishedTaskListCount());
    }

    private static List<Long> finishedRids(EngineRpcService.WorkerStatusPB status) {
        return status.getFinishedTaskListList().stream().map(EngineRpcService.TaskInfoPB::getRequestId).toList();
    }
}
