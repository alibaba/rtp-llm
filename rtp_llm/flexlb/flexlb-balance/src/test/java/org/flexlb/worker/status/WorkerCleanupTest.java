package org.flexlb.worker.status;

import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.junit.jupiter.api.Test;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.scheduling.TaskScheduler;
import org.springframework.scheduling.annotation.ScheduledAnnotationBeanPostProcessor;
import org.springframework.scheduling.config.FixedRateTask;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks scheduled cleanup detaches unavailable workers before cache cleanup and finishes earlier cleanups if a later one fails.
 */
class WorkerCleanupTest {

    @Test
    void springSchedulesWorkerCleanupAtTheConfiguredInterval() {
        FlexlbConfig config =
                ConfigService.parse(
                        """
                {"requestLifecycle":{"request":{"timeoutMs":60000}},
                 "workerRegistry":{"health":{"cleanupIntervalMs":1250}}}
                """);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        try (AnnotationConfigApplicationContext context =
                new AnnotationConfigApplicationContext()) {
            for (String name :
                    java.util.List.of(
                            "environmentConfigSource",
                            "nacosConfigSource",
                            "uniConfigConfigSource")) {
                context.registerBean(
                        name,
                        org.flexlb.config.loading.source.ConfigSource.class,
                        () -> mock(org.flexlb.config.loading.source.ConfigSource.class));
            }
            context.registerBean("configService", ConfigService.class, () -> configService);
            context.registerBean(CacheAwareService.class, () -> mock(CacheAwareService.class));
            context.registerBean(EndpointRegistry.class, () -> mock(EndpointRegistry.class));
            context.registerBean(TaskScheduler.class, () -> mock(TaskScheduler.class));
            context.registerBean(ScheduledAnnotationBeanPostProcessor.class);
            context.registerBean(
                    org.flexlb.worker.status.WorkerStatusSynchronizer.class,
                    () ->
                            org.flexlb.worker.status.WorkerSyncTestSupport.cleanups(
                                    context.getBean(CacheAwareService.class),
                                    context.getBean(EndpointRegistry.class)));
            context.refresh();
            var tasks =
                    context
                            .getBean(ScheduledAnnotationBeanPostProcessor.class)
                            .getScheduledTasks()
                            .stream()
                            .filter(
                                    task ->
                                            ((org.springframework.scheduling.support
                                                                            .ScheduledMethodRunnable)
                                                                    task.getTask().getRunnable())
                                                            .getTarget()
                                                    instanceof
                                                    org.flexlb.worker.status
                                                            .WorkerStatusSynchronizer)
                            .toList();
            assertEquals(1, tasks.size());
            FixedRateTask task = assertInstanceOf(FixedRateTask.class, tasks.getFirst().getTask());
            assertEquals(1250L, task.getInterval());
        }
    }

    @Test
    void detachesEveryUnavailableWorkerBeforeCacheCleanup() throws Exception {
        EndpointRegistry registry =
                RunnerTestSupport.endpointRegistry(
                        org.flexlb.balance.schedule.SchedulingTestConfig.configService(
                                org.flexlb.balance.schedule.SchedulingTestConfig.newConfig()));
        CacheAwareService cache = mock(CacheAwareService.class);
        WorkerStatus first = status("127.0.0.1", 8080);
        WorkerStatus second = status("127.0.0.2", 8080);
        EndpointRegistry directory = registry;
        directory.currentOrDiscover(RoleType.PREFILL, first.getLogicalIpPort(), () -> first);
        directory.currentOrDiscover(RoleType.PREFILL, second.getLogicalIpPort(), () -> second);
        first.markRemovedFromDiscovery();
        second.markRemovedFromDiscovery();

        CountDownLatch cleanupEntered = new CountDownLatch(1);
        CountDownLatch releaseCleanup = new CountDownLatch(1);
        org.mockito.Mockito.doAnswer(
                        invocation -> {
                            cleanupEntered.countDown();
                            assertTrue(releaseCleanup.await(5, TimeUnit.SECONDS));
                            return null;
                        })
                .when(cache)
                .removeEngineBlockCache(first.getLogicalIpPort());
        org.flexlb.worker.status.WorkerStatusSynchronizer cleaner =
                org.flexlb.worker.status.WorkerSyncTestSupport.cleanups(cache, directory);
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<?> cleaning = executor.submit(cleaner::closeUnavailableWorkers);

            assertTrue(cleanupEntered.await(2, TimeUnit.SECONDS));
            assertFalse(first.isActiveGeneration());
            assertFalse(
                    second.isActiveGeneration(),
                    "both routing gates must close before cache cleanup begins");
            cleaning.get(2, TimeUnit.SECONDS); // The scan must not wait for external cleanup.
            releaseCleanup.countDown();
            org.flexlb.worker.status.RunnerTestSupport.awaitCondition(
                    () -> directory.statusSnapshot(RoleType.PREFILL).isEmpty());
            verify(cache).removeEngineBlockCache(first.getLogicalIpPort());
            verify(cache).removeEngineBlockCache(second.getLogicalIpPort());
        } finally {
            releaseCleanup.countDown();
            executor.shutdownNow();
            registry.close();
        }
    }

    @Test
    void failurePreparingOneRetirementStillStartsOtherAcceptedCleanups() {
        var registry =
                org.mockito.Mockito.spy(
                        RunnerTestSupport.endpointRegistry(
                                org.flexlb.balance.schedule.SchedulingTestConfig.configService(
                                        org.flexlb.balance.schedule.SchedulingTestConfig
                                                .newConfig())));
        var cache = mock(CacheAwareService.class);
        var first = status("127.0.0.1", 8080);
        var second = status("127.0.0.2", 8080);
        registry.currentOrDiscover(RoleType.PREFILL, first.getLogicalIpPort(), () -> first);
        registry.currentOrDiscover(RoleType.PREFILL, second.getLogicalIpPort(), () -> second);
        first.markRemovedFromDiscovery();
        second.markRemovedFromDiscovery();
        var failure = new IllegalStateException("endpoint lookup failed");
        org.mockito.Mockito.doThrow(failure)
                .when(registry)
                .getEndpoint(RoleType.PREFILL, second.getLogicalIpPort());
        var cleaner = org.flexlb.worker.status.WorkerSyncTestSupport.cleanups(cache, registry);
        org.junit.jupiter.api.Assertions.assertSame(
                failure,
                org.junit.jupiter.api.Assertions.assertThrows(
                        IllegalStateException.class, cleaner::closeUnavailableWorkers));
        RunnerTestSupport.awaitCondition(
                () -> !registry.statusSnapshot(RoleType.PREFILL).containsValue(first));
        assertTrue(second.isActiveGeneration());
        verify(cache).removeEngineBlockCache(first.getLogicalIpPort());
        registry.close();
    }

    @Test
    void busyStatusTransactionDoesNotPreventAnotherWorkerRetirement() {
        var registry =
                RunnerTestSupport.endpointRegistry(
                        org.flexlb.balance.schedule.SchedulingTestConfig.configService(
                                org.flexlb.balance.schedule.SchedulingTestConfig.newConfig()));
        var cache = mock(CacheAwareService.class);
        var first = status("127.0.0.1", 8080);
        var second = status("127.0.0.2", 8080);
        registry.currentOrDiscover(RoleType.PREFILL, first.getLogicalIpPort(), () -> first);
        registry.currentOrDiscover(RoleType.PREFILL, second.getLogicalIpPort(), () -> second);
        first.markRemovedFromDiscovery();
        second.markRemovedFromDiscovery();
        var cleaner = org.flexlb.worker.status.WorkerSyncTestSupport.cleanups(cache, registry);
        // A busy status transaction must not prevent retirement of the other worker.
        second.lock.lock();
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            executor.submit(cleaner::closeUnavailableWorkers).get(2, TimeUnit.SECONDS);
            RunnerTestSupport.awaitCondition(
                    () -> !registry.statusSnapshot(RoleType.PREFILL).containsValue(first));
            org.junit.jupiter.api.Assertions.assertSame(
                    second,
                    registry.statusSnapshot(RoleType.PREFILL).get(second.getLogicalIpPort()));
            assertTrue(second.isActiveGeneration());
            verify(cache).removeEngineBlockCache(first.getLogicalIpPort());
            org.mockito.Mockito.verify(cache, org.mockito.Mockito.never())
                    .removeEngineBlockCache(second.getLogicalIpPort());
        } catch (Exception failure) {
            throw new AssertionError(failure);
        } finally {
            second.lock.unlock();
        }
        cleaner.closeUnavailableWorkers();
        RunnerTestSupport.awaitCondition(() -> registry.statusSnapshot(RoleType.PREFILL).isEmpty());
        registry.close();
    }

    private static WorkerStatus status(String ip, int port) {
        return WorkerStatus.createDiscovered(
                RoleType.PREFILL, null, ip, port, port + 1, "test-site");
    }
}
