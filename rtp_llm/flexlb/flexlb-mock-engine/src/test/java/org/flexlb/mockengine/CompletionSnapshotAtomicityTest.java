package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.HashSet;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.flexlb.mockengine.MockEngineTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;

/** Actual P/D execution while independent status consumers race completion. */
class CompletionSnapshotAtomicityTest {
    @TempDir Path tempDir;

    @Test
    void acknowledgedTasksNeverDisappearBeforeTheirTerminalOnEitherRole() throws Exception {
        try (var cluster = MockEngineTestCluster.create(performanceModel(tempDir, "3"),
                62840, 1, 1, 8); var senders = Executors.newSingleThreadExecutor()) {
            cluster.prefill(0).setAutoFetch(true);
            AtomicInteger acknowledged = new AtomicInteger();
            var producer = senders.submit(() -> {
                for (int wave = 0; wave < 20; wave++) {
                    EngineRpcService.GenerateInputPB[] inputs = new EngineRpcService.GenerateInputPB[16];
                    for (int j = 0; j < inputs.length; j++) {
                        inputs[j] = inputWithDecode(wave * 16L + j + 1, 128,
                                cluster.decode(0).getGrpcPort(), 2);
                    }
                    assertEquals(16, enqueue(cluster.prefill(0), batch(wave + 1, slot(0, inputs)))
                            .getSuccessesCount());
                    acknowledged.set((wave + 1) * 16);
                }
            });
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(15);
            int snapshots = 0;
            while ((!producer.isDone() || cluster.decode(0).getCompletedCount() < 320)
                    && System.nanoTime() < deadline) {
                int known = acknowledged.get();
                for (var service : cluster.services().values()) {
                    var status = workerStatus(service, 0);
                    var visible = new HashSet<Long>();
                    status.getRunningTaskInfoList().forEach(t -> visible.add(t.getRequestId()));
                    status.getFinishedTaskListList().forEach(t -> visible.add(t.getRequestId()));
                    for (long id = 1; id <= known; id++) {
                        assertTrue(visible.contains(id), "missing acknowledged rid=" + id
                                + " port=" + service.getGrpcPort() + " status=" + status);
                    }
                    snapshots++;
                }
                Thread.sleep(1);
            }
            producer.get(1, TimeUnit.SECONDS);
            assertEquals(320, cluster.decode(0).getCompletedCount());
            assertTrue(snapshots > 2);
            for (var service : cluster.services().values()) {
                var finalStatus = workerStatus(service, 0);
                assertEquals(320, finalStatus.getFinishedTaskListCount());
                assertEquals(0, finalStatus.getRunningTaskInfoCount());
                assertEquals(320, finalStatus.getFinishedTaskListList().stream()
                        .map(EngineRpcService.TaskInfoPB::getRequestId).distinct().count());
            }
        }
    }
}
