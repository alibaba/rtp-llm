package org.flexlb.sync.schedule;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.sync.runner.RunnerTestSupport;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
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

class StaleWorkerCleanerTest {

    @Test
    void springSchedulesWorkerCleanupAtTheConfiguredInterval() {
        FlexlbConfig config = ConfigService.parse("""
                {"requestLifecycle":{"request":{"timeoutMs":60000}},
                 "workerRegistry":{"health":{"cleanupIntervalMs":1250}}}
                """);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        try (AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext()) {
            context.registerBean("configService", ConfigService.class, () -> configService);
            context.registerBean(CacheAwareService.class, () -> mock(CacheAwareService.class));
            context.registerBean(EndpointRegistry.class, () -> mock(EndpointRegistry.class));
            context.registerBean(TaskScheduler.class, () -> mock(TaskScheduler.class));
            context.registerBean(ScheduledAnnotationBeanPostProcessor.class);
            context.registerBean(StaleWorkerCleaner.class);
            context.refresh();
            var tasks = context.getBean(ScheduledAnnotationBeanPostProcessor.class).getScheduledTasks();
            assertEquals(1, tasks.size());
            FixedRateTask task = assertInstanceOf(FixedRateTask.class, tasks.iterator().next().getTask());
            assertEquals(1250L, task.getInterval());
            assertEquals(10000L, config.getWorkerRegistry().getHealth().getStatusStaleAfterMs());
        }
    }

    @Test
    void detachesEveryExpiredWorkerBeforeAwaitingAnyRetirement()
            throws Exception {
        EndpointRegistry registry = org.mockito.Mockito.spy(RunnerTestSupport.endpointRegistry(mock(ConfigService.class)));
        CacheAwareService cache = mock(CacheAwareService.class);
        WorkerStatus first = status("127.0.0.1", 8080);
        WorkerStatus second = status("127.0.0.2", 8080);
        EndpointRegistry directory = registry;
        directory.currentOrDiscover(
                RoleType.PREFILL, first.getIpPort(), () -> first);
        directory.currentOrDiscover(
                RoleType.PREFILL, second.getIpPort(), () -> second);

        CountDownLatch firstAwaitEntered = new CountDownLatch(1);
        CountDownLatch releaseFirstAwait = new CountDownLatch(1);
        org.mockito.Mockito.doAnswer(invocation -> {
            EndpointRegistry.Retirement retirement = org.mockito.Mockito.spy(
                    (EndpointRegistry.Retirement) invocation.callRealMethod());
            org.mockito.Mockito.doAnswer(completion -> {
                firstAwaitEntered.countDown();
                assertTrue(releaseFirstAwait.await(5, TimeUnit.SECONDS));
                return completion.callRealMethod();
            }).when(retirement).complete(org.mockito.ArgumentMatchers.same(cache), org.mockito.ArgumentMatchers.any());
            return retirement;
        }).when(registry).beginRetirement(RoleType.PREFILL, first.getIpPort(), first);

        ConfigService configService = mock(ConfigService.class);
        FlexlbConfig config = org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig();
        config.getWorkerRegistry().getHealth().setStatusStaleAfterMs(0L);
        when(configService.loadBalanceConfig()).thenReturn(config);
        StaleWorkerCleaner cleaner = new StaleWorkerCleaner(
                configService, cache, directory);
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<?> cleaning = executor.submit(
                    cleaner::cleanExpiredWorkers);

            assertTrue(firstAwaitEntered.await(2, TimeUnit.SECONDS));
            assertFalse(first.isActiveGeneration());
            assertFalse(second.isActiveGeneration(),
                    "the second routing gate must close before the first drain waits");
            releaseFirstAwait.countDown();
            cleaning.get(5, TimeUnit.SECONDS);
            assertTrue(directory.statusSnapshot(RoleType.PREFILL).isEmpty());
            verify(cache).removeEngineBlockCache(first.getIpPort());
            verify(cache).removeEngineBlockCache(second.getIpPort());
        } finally {
            releaseFirstAwait.countDown();
            executor.shutdownNow();
        }
    }

    @Test
    void failureToBeginLaterRetirementStillFinalizesAlreadyDetachedWorkers() {
        ConfigService configService = mock(ConfigService.class);
        FlexlbConfig config = org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig();
        config.getWorkerRegistry().getHealth().setStatusStaleAfterMs(0L);
        when(configService.loadBalanceConfig()).thenReturn(config);
        EndpointRegistry registry = org.mockito.Mockito.spy(RunnerTestSupport.endpointRegistry(configService));
        CacheAwareService cache = mock(CacheAwareService.class);
        WorkerStatus first = status("127.0.0.1", 8080);
        WorkerStatus second = status("127.0.0.2", 8080);
        registry.currentOrDiscover(RoleType.PREFILL, first.getIpPort(), () -> first);
        registry.currentOrDiscover(RoleType.PREFILL, second.getIpPort(), () -> second);
        var began = new java.util.concurrent.atomic.AtomicReference<WorkerStatus>();
        RuntimeException failure = new IllegalStateException("registry closing");
        org.mockito.Mockito.doAnswer(invocation -> {
            if (began.get() != null) { throw failure; }
            EndpointRegistry.Retirement retirement =
                    (EndpointRegistry.Retirement) invocation.callRealMethod();
            began.set(retirement.status());
            return retirement;
        }).when(registry).beginRetirement(org.mockito.ArgumentMatchers.eq(RoleType.PREFILL),
                org.mockito.ArgumentMatchers.anyString(), org.mockito.ArgumentMatchers.any());

        org.junit.jupiter.api.Assertions.assertSame(failure,
                org.junit.jupiter.api.Assertions.assertThrows(IllegalStateException.class,
                        () -> new StaleWorkerCleaner(configService, cache, registry).cleanExpiredWorkers()));
        assertFalse(registry.statusSnapshot(RoleType.PREFILL).containsValue(began.get()));
        assertEquals(1, registry.statusSnapshot(RoleType.PREFILL).size());
        assertTrue(registry.statusSnapshot(RoleType.PREFILL).values().iterator().next().isActiveGeneration());
        verify(cache).removeEngineBlockCache(began.get().getIpPort());
        registry.close();
    }

    private static WorkerStatus status(String ip, int port) {
        return WorkerStatus.createDiscovered(
                RoleType.PREFILL, null, ip, port, port + 1, "test-site");
    }
}
