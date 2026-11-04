package org.flexlb.cache.match.localsync;

import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.EndpointTrace;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

@Timeout(20)
class CacheDiscoveryConcurrencyTest {
    @Test
    void discoveryCleanupWaitsForSnapshotApplicationAndForcesSameVersionRebuild() throws Exception {
        CountDownLatch adding = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        AtomicBoolean pause = new AtomicBoolean();
        GlobalCacheIndex global =
                new GlobalCacheIndex() {
                    @Override
                    public void addCacheBlock(Long key, String owner, EndpointTrace.Poll trace) {
                        if (pause.get()) {
                            adding.countDown();
                            await(release);
                        }
                        super.addCacheBlock(key, owner, trace);
                    }
                };
        WorkerStatus owner = worker(0);
        try (Fixture fixture = new Fixture(global);
                var threads = Executors.newFixedThreadPool(2)) {
            fixture.update(owner, Set.of(1L), 1L);
            pause.set(true);
            var updating = threads.submit(() -> fixture.update(owner, Set.of(2L), 2L));
            CountDownLatch removing = new CountDownLatch(1);
            try {
                assertTrue(adding.await(5, TimeUnit.SECONDS));
                var cleanup =
                        threads.submit(
                                () -> {
                                    removing.countDown();
                                    fixture.manager.removeStaleEngineCaches(List.of());
                                });
                assertTrue(removing.await(5, TimeUnit.SECONDS));
                long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
                while (!owner.cacheMutationLock.hasQueuedThreads()
                        && System.nanoTime() < deadline) {
                    Thread.onSpinWait();
                }
                assertTrue(owner.cacheMutationLock.hasQueuedThreads());
                assertFalse(cleanup.isDone());
                release.countDown();
                updating.get(5, TimeUnit.SECONDS);
                cleanup.get(5, TimeUnit.SECONDS);
            } finally {
                release.countDown();
            }
            assertEquals(0L, global.totalMappings());
            assertTrue(fixture.local.getEngineCacheBlocks(owner.getLogicalIpPort()).isEmpty());
            assertFalse(owner.cacheIndexSnapshot().indexInitialized());
            assertEquals(-1L, owner.cacheIndexSnapshot().indexedVersion());
            pause.set(false);
            fixture.update(owner, Set.of(2L), 2L);
            assertEquals(
                    Map.of(owner.getLogicalIpPort(), 1),
                    fixture.manager.findMatchingEngines(
                            List.of(2L), List.of(owner.getLogicalIpPort())));
        }
    }

    @Test
    void staleGenerationCleanupCannotRemoveReplacementOrSiblingEngine() {
        WorkerStatus previous = worker(0);
        WorkerStatus replacement = worker(0);
        WorkerStatus sibling = worker(1);
        try (Fixture fixture = new Fixture(new GlobalCacheIndex())) {
            fixture.update(previous, Set.of(1L), 1L);
            fixture.update(sibling, Set.of(1L, 2L), 1L);
            fixture.manager.removeEngineCache(previous, previous.trace());
            fixture.update(replacement, Set.of(1L, 2L), 2L);
            fixture.manager.removeEngineCache(previous, previous.trace());
            fixture.manager.removeStaleEngineCaches(List.of("10.0.0.1:8080"));
            assertEquals(
                    Map.of(replacement.getLogicalIpPort(), 2, sibling.getLogicalIpPort(), 2),
                    fixture.manager.findMatchingEngines(
                            List.of(1L, 2L),
                            List.of(replacement.getLogicalIpPort(), sibling.getLogicalIpPort())));
            fixture.manager.removeEngineCache(replacement, replacement.trace());
            assertEquals(
                    Map.of(sibling.getLogicalIpPort(), 2),
                    fixture.manager.findMatchingEngines(
                            List.of(1L, 2L),
                            List.of(replacement.getLogicalIpPort(), sibling.getLogicalIpPort())));
            fixture.manager.removeStaleEngineCaches(List.of());
            assertEquals(0L, fixture.global.totalMappings());
        }
    }

    private static WorkerStatus worker(int engineIndex) {
        return WorkerStatus.createDiscovered(
                RoleType.PREFILL,
                "group",
                "10.0.0.1",
                8080,
                8081,
                "site",
                "deployment",
                engineIndex,
                2);
    }

    private static void await(CountDownLatch latch) {
        try {
            assertTrue(latch.await(5, TimeUnit.SECONDS));
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new AssertionError(interrupted);
        }
    }

    private static final class Fixture implements AutoCloseable {
        final EngineLocalView local = new EngineLocalView();
        final GlobalCacheIndex global;
        final KvCacheManager manager = new KvCacheManager();

        Fixture(GlobalCacheIndex global) {
            this.global = global;
            ReflectionTestUtils.setField(
                    local, "dynamicIntervalManager", mock(DynamicCacheIntervalService.class));
            ReflectionTestUtils.setField(manager, "engineLocalView", local);
            ReflectionTestUtils.setField(manager, "globalCacheIndex", global);
            ReflectionTestUtils.setField(
                    manager, "cacheMetricsReporter", mock(CacheMetricsReporter.class));
        }

        void update(WorkerStatus owner, Set<Long> keys, long version) {
            owner.cacheMutationLock.lock();
            try {
                owner.publishCacheStatus(
                        CacheStatus.builder().cachedKeys(keys).version(version).build());
                manager.updateEngineCache(owner, "PREFILL", keys);
                owner.publishCacheIndexedVersion(version);
            } finally {
                owner.cacheMutationLock.unlock();
            }
        }

        @Override
        public void close() {
            local.shutdown();
        }
    }
}
