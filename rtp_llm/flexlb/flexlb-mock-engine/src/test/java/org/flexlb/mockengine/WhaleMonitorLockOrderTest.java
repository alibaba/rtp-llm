package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.metric.NoOpFlexMonitor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertTrue;

class WhaleMonitorLockOrderTest {
    @TempDir Path directory;

    @Test void engineSnapshotDoesNotHoldMonitorLockAgainstEngineCallbacks() throws Exception {
        var scheduler = Executors.newScheduledThreadPool(1);
        var threads = Executors.newFixedThreadPool(2);
        var samplingThread = new AtomicReference<Thread>();
        var started = new CountDownLatch(1);
        var model = MockEngineTestSupport.performanceModel(directory, "10", 1);
        var service = new JavaMockEngineCluster.FastRpcService("prefill",
                EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL, 63190,
                new ConcurrentHashMap<>(), scheduler, model, 16,
                new JavaMockEngineCluster.ClusterStats());
        var field = JavaMockEngineCluster.FastRpcService.class.getDeclaredField("cache");
        field.setAccessible(true);
        var cache = field.get(service);
        var monitor = new WhaleMockMonitor(NoOpFlexMonitor.getInstance());
        try {
            java.util.concurrent.Future<?> sampling;
            synchronized (cache) {
                sampling = threads.submit(() -> {
                    samplingThread.set(Thread.currentThread());
                    started.countDown();
                    monitor.sample(service);
                });
                assertTrue(started.await(2, TimeUnit.SECONDS));
                AutoTpmE2EHarness.await(() -> samplingThread.get().getState() == Thread.State.BLOCKED,
                        2_000, "sampling must wait for the held engine cache");
                // Sampling is blocked on this cache. The eviction callback
                // must still acquire the monitor or the opposite lock order
                // would deadlock an actual eviction on the cache owner.
                threads.submit(() -> monitor.reportEvent(
                        Map.of("mock_cache_direct_evicted_blocks", 1), service.whaleMetricTags()))
                        .get(1, TimeUnit.SECONDS);
            }
            sampling.get(2, TimeUnit.SECONDS);
        } finally {
            threads.shutdownNow();
            service.shutdown();
            scheduler.shutdownNow();
        }
    }
}
