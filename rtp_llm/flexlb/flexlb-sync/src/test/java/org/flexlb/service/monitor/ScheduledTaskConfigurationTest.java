package org.flexlb.service.monitor;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.sync.synchronizer.WorkerStatusSynchronizer;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.scheduling.annotation.ScheduledAnnotationBeanPostProcessor;
import org.springframework.scheduling.config.FixedRateTask;
import org.springframework.scheduling.config.ScheduledTaskRegistrar;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class ScheduledTaskConfigurationTest {

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void existingWorkerCleanupUsesTheFourThreadSchedulerWithOrWithoutLegacyMetricBean(
            boolean legacyMetricBean) throws Exception {
        var config =
                ConfigService.parse(
                        """
                {"requestLifecycle":{"request":{"timeoutMs":60000}},
                 "workerRegistry":{"health":{"cleanupIntervalMs":1250}}}
                """);
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var registry = mock(EndpointRegistry.class);
        var polled = new CountDownLatch(1);
        var worker = new AtomicReference<Thread>();
        when(registry.statusSnapshot(any()))
                .thenAnswer(
                        call -> {
                            worker.compareAndSet(null, Thread.currentThread());
                            polled.countDown();
                            return Map.of();
                        });
        ScheduledThreadPoolExecutor scheduler;
        try (var context = new AnnotationConfigApplicationContext()) {
            context.register(Scheduling.class);
            for (String name :
                    java.util.List.of(
                            "environmentConfigSource",
                            "nacosConfigSource",
                            "uniConfigConfigSource")) {
                context.registerBean(
                        name,
                        org.flexlb.service.config.ConfigSource.class,
                        () -> mock(org.flexlb.service.config.ConfigSource.class));
            }
            context.registerBean("configService", ConfigService.class, () -> service);
            context.registerBean(EndpointRegistry.class, () -> registry);
            context.registerBean(CacheAwareService.class, () -> mock(CacheAwareService.class));
            context.registerBean(
                    org.flexlb.sync.synchronizer.WorkerStatusSynchronizer.class,
                    () ->
                            org.flexlb.sync.synchronizer.WorkerSyncTestSupport.retirements(
                                    context.getBean(CacheAwareService.class),
                                    context.getBean(EndpointRegistry.class)));
            if (legacyMetricBean) {
                context.registerBean(
                        "taskMetricScheduler",
                        ScheduledThreadPoolExecutor.class,
                        () -> new ScheduledThreadPoolExecutor(1));
            }
            context.refresh();
            scheduler = context.getBean("taskScheduler", ScheduledThreadPoolExecutor.class);
            assertEquals(4, scheduler.getCorePoolSize());
            var processor = context.getBean(ScheduledAnnotationBeanPostProcessor.class);
            var registrar =
                    (ScheduledTaskRegistrar) ReflectionTestUtils.getField(processor, "registrar");
            assertSame(
                    scheduler,
                    ReflectionTestUtils.getField(registrar.getScheduler(), "scheduledExecutor"),
                    "Spring must use the named executor, without a single-thread fallback");
            var task =
                    assertInstanceOf(
                            FixedRateTask.class,
                            processor.getScheduledTasks().stream()
                                    .filter(
                                            item ->
                                                    ((org.springframework.scheduling.support
                                                                                    .ScheduledMethodRunnable)
                                                                            item.getTask()
                                                                                    .getRunnable())
                                                                    .getTarget()
                                                            instanceof WorkerStatusSynchronizer)
                                    .findFirst()
                                    .orElseThrow()
                                    .getTask());
            assertEquals(1250L, task.getInterval());
            assertTrue(polled.await(5, TimeUnit.SECONDS), "the existing cleanup task must execute");
            assertTrue(worker.get().getName().startsWith("task-scheduler"));
        }
        assertTrue(scheduler.isShutdown(), "Spring owns the selected executor lifecycle");
    }

    @org.springframework.context.annotation.Configuration
    @org.springframework.scheduling.annotation.EnableScheduling
    static class Scheduling {
        @org.springframework.context.annotation.Bean(name = "taskScheduler")
        ScheduledThreadPoolExecutor taskScheduler() {
            return new org.flexlb.balance.scheduler.PlacementConfiguration().taskScheduler();
        }
    }
}
