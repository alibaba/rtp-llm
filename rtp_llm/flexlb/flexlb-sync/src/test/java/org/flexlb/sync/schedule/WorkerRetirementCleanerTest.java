package org.flexlb.sync.schedule;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.runner.RunnerTestSupport;
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

class WorkerRetirementCleanerTest {

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
                context.registerBean(name, Object.class, Object::new);
            }
            context.registerBean("configService", ConfigService.class, () -> configService);
            context.registerBean(CacheAwareService.class, () -> mock(CacheAwareService.class));
            context.registerBean(EndpointRegistry.class, () -> mock(EndpointRegistry.class));
            context.registerBean(TaskScheduler.class, () -> mock(TaskScheduler.class));
            context.registerBean(ScheduledAnnotationBeanPostProcessor.class);
            context.registerBean(WorkerRetirementCleaner.class);
            context.refresh();
            var tasks =
                    context.getBean(ScheduledAnnotationBeanPostProcessor.class).getScheduledTasks();
            assertEquals(1, tasks.size());
            FixedRateTask task =
                    assertInstanceOf(FixedRateTask.class, tasks.iterator().next().getTask());
            assertEquals(1250L, task.getInterval());
        }
    }

    @Test
    void detachesEveryUnavailableWorkerBeforeCacheCleanup() throws Exception {
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(mock(ConfigService.class));
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
        WorkerRetirementCleaner cleaner = new WorkerRetirementCleaner(cache, directory);
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<?> cleaning = executor.submit(cleaner::retireUnavailableWorkers);

            assertTrue(cleanupEntered.await(2, TimeUnit.SECONDS));
            assertFalse(first.isActiveGeneration());
            assertFalse(
                    second.isActiveGeneration(),
                    "both routing gates must close before cache cleanup begins");
            cleaning.get(2, TimeUnit.SECONDS); // The scan must not wait for external cleanup.
            releaseCleanup.countDown();
            org.flexlb.sync.runner.RunnerTestSupport.awaitCondition(
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
    void failureToBeginLaterRetirementStillFinalizesAlreadyDetachedWorkers() {
        ConfigService configService = mock(ConfigService.class);
        FlexlbConfig config = org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig();
        when(configService.loadBalanceConfig()).thenReturn(config);
        EndpointRegistry registry =
                org.mockito.Mockito.spy(RunnerTestSupport.endpointRegistry(configService));
        CacheAwareService cache = mock(CacheAwareService.class);
        WorkerStatus first = status("127.0.0.1", 8080);
        WorkerStatus second = status("127.0.0.2", 8080);
        registry.currentOrDiscover(RoleType.PREFILL, first.getLogicalIpPort(), () -> first);
        registry.currentOrDiscover(RoleType.PREFILL, second.getLogicalIpPort(), () -> second);
        first.markRemovedFromDiscovery();
        second.markRemovedFromDiscovery();
        var began = new java.util.concurrent.atomic.AtomicReference<WorkerStatus>();
        RuntimeException failure = new IllegalStateException("registry closing");
        org.mockito.Mockito.doAnswer(
                        invocation -> {
                            if (began.get() != null) {
                                throw failure;
                            }
                            EndpointRegistry.Retirement retirement =
                                    (EndpointRegistry.Retirement) invocation.callRealMethod();
                            began.set(retirement.status());
                            return retirement;
                        })
                .when(registry)
                .beginRetirement(
                        org.mockito.ArgumentMatchers.eq(RoleType.PREFILL),
                        org.mockito.ArgumentMatchers.anyString(),
                        org.mockito.ArgumentMatchers.any());

        org.junit.jupiter.api.Assertions.assertSame(
                failure,
                org.junit.jupiter.api.Assertions.assertThrows(
                        IllegalStateException.class,
                        () ->
                                new WorkerRetirementCleaner(cache, registry)
                                        .retireUnavailableWorkers()));
        org.flexlb.sync.runner.RunnerTestSupport.awaitCondition(
                () -> !registry.statusSnapshot(RoleType.PREFILL).containsValue(began.get()));
        assertEquals(1, registry.statusSnapshot(RoleType.PREFILL).size());
        assertTrue(
                registry.statusSnapshot(RoleType.PREFILL)
                        .values()
                        .iterator()
                        .next()
                        .isActiveGeneration());
        verify(cache).removeEngineBlockCache(began.get().getLogicalIpPort());
        registry.close();
    }

    private static WorkerStatus status(String ip, int port) {
        return WorkerStatus.createDiscovered(
                RoleType.PREFILL, null, ip, port, port + 1, "test-site");
    }
}
