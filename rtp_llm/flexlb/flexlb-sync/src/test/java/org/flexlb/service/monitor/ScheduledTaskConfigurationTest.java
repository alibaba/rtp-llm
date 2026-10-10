package org.flexlb.service.monitor;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.sync.schedule.StaleWorkerCleaner;
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

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class ScheduledTaskConfigurationTest {

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void existingWorkerCleanupUsesTheFourThreadSchedulerWithOrWithoutLegacyMetricBean(
            boolean legacyMetricBean) throws Exception {
        var config = ConfigService.parse("""
                {"requestLifecycle":{"request":{"timeoutMs":60000}},
                 "workerRegistry":{"health":{"cleanupIntervalMs":1250}}}
                """);
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var registry = mock(EndpointRegistry.class);
        var polled = new CountDownLatch(1);
        var worker = new AtomicReference<Thread>();
        when(registry.statusSnapshot(any())).thenAnswer(call -> {
            worker.compareAndSet(null, Thread.currentThread());
            polled.countDown();
            return Map.of();
        });
        ScheduledThreadPoolExecutor scheduler;
        try (var context = new AnnotationConfigApplicationContext()) {
            context.register(ScheduledTaskConfiguration.class);
            context.registerBean("configService", ConfigService.class, () -> service);
            context.registerBean(EndpointRegistry.class, () -> registry);
            context.registerBean(CacheAwareService.class, () -> mock(CacheAwareService.class));
            context.registerBean(StaleWorkerCleaner.class);
            if (legacyMetricBean) {
                context.registerBean("taskMetricScheduler", ScheduledThreadPoolExecutor.class,
                        () -> new ScheduledThreadPoolExecutor(1));
            }
            context.refresh();
            scheduler = context.getBean("taskScheduler", ScheduledThreadPoolExecutor.class);
            assertEquals(4, scheduler.getCorePoolSize());
            var processor = context.getBean(ScheduledAnnotationBeanPostProcessor.class);
            var registrar = (ScheduledTaskRegistrar) ReflectionTestUtils.getField(processor, "registrar");
            assertSame(scheduler, ReflectionTestUtils.getField(registrar.getScheduler(), "scheduledExecutor"),
                    "Spring must use the named executor, without a single-thread fallback");
            var task = assertInstanceOf(FixedRateTask.class,
                    processor.getScheduledTasks().iterator().next().getTask());
            assertEquals(1250L, task.getInterval());
            assertTrue(polled.await(5, TimeUnit.SECONDS), "the existing cleanup task must execute");
            assertTrue(worker.get().getName().startsWith("task-scheduler"));
        }
        assertTrue(scheduler.isShutdown(), "Spring owns the selected executor lifecycle");
    }
}
