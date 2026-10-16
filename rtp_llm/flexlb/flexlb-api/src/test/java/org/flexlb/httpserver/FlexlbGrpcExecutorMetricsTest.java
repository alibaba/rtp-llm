package org.flexlb.httpserver;

import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.flexlb.Application;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.constant.MetricConstant;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.MicrometerFlexMonitor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.mock.env.MockEnvironment;
import org.springframework.scheduling.TaskScheduler;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.scheduling.concurrent.ThreadPoolTaskScheduler;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.RejectedExecutionHandler;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

@Timeout(10)
class FlexlbGrpcExecutorMetricsTest {
    private static final List<String> METRICS = List.of(
            MetricConstant.GRPC_SERVER_EXECUTOR_ACTIVE_THREADS,
            MetricConstant.GRPC_SERVER_EXECUTOR_QUEUE_SIZE,
            MetricConstant.GRPC_SERVER_EXECUTOR_POOL_SIZE,
            MetricConstant.GRPC_SERVER_EXECUTOR_MAX_POOL_SIZE,
            MetricConstant.GRPC_SERVER_EXECUTOR_REJECTED_TASKS);

    @Test
    void skipsReportingBeforeExecutorCreationAndRegistersExpectedMetricTypes() {
        FlexMonitor monitor = mock(FlexMonitor.class);
        FlexlbGrpcServer server = newServer(monitor);

        ReflectionTestUtils.invokeMethod(server, "reportExecutorMetrics");
        verifyNoInteractions(monitor);

        ReflectionTestUtils.invokeMethod(server, "registerMetrics");
        for (String metric : METRICS) {
            verify(monitor).register(metric, metric.equals(MetricConstant.GRPC_SERVER_EXECUTOR_REJECTED_TASKS)
                            ? FlexMetricType.COUNTER : FlexMetricType.GAUGE,
                    FlexPriorityType.PRECISE);
        }
        verifyNoMoreInteractions(monitor);
    }

    @Test
    @Timeout(15)
    void registersAndRunsExecutorMetricsThroughSpringScheduling() throws Exception {
        assertTrue(Application.class.isAnnotationPresent(EnableScheduling.class));
        assertEquals(2_000L, FlexlbGrpcServer.class.getDeclaredMethod("reportExecutorMetrics")
                .getAnnotation(Scheduled.class).fixedRate());
        try (Fixture fixture = new Fixture();
             AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext()) {
            FlexlbGrpcServer scheduledServer = spy(fixture.server);
            doNothing().when(scheduledServer).start();
            context.register(SchedulingConfiguration.class);
            context.registerBean(FlexlbGrpcServer.class, () -> scheduledServer);
            context.refresh();

            fixture.blockFirstTaskAndQueueSecond();
            long timeoutAt = System.nanoTime() + TimeUnit.SECONDS.toNanos(6);
            while (!(fixture.hasValue(MetricConstant.GRPC_SERVER_EXECUTOR_ACTIVE_THREADS, 1)
                    && fixture.hasValue(MetricConstant.GRPC_SERVER_EXECUTOR_QUEUE_SIZE, 1)
                    && fixture.hasValue(MetricConstant.GRPC_SERVER_EXECUTOR_POOL_SIZE, 1))
                    && System.nanoTime() < timeoutAt) {
                Thread.sleep(10);
            }

            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_ACTIVE_THREADS, 1);
            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_QUEUE_SIZE, 1);
            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_POOL_SIZE, 1);
            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_MAX_POOL_SIZE, 1);
            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_REJECTED_TASKS, 0);
        }
    }

    @Configuration
    @EnableScheduling
    static class SchedulingConfiguration {
        @Bean
        TaskScheduler taskScheduler() {
            ThreadPoolTaskScheduler scheduler = new ThreadPoolTaskScheduler();
            scheduler.setPoolSize(1);
            scheduler.setDaemon(true);
            return scheduler;
        }
    }

    @Test
    void reportsIdleBusyQueuedAndTerminatedExecutorState() throws Exception {
        try (Fixture fixture = new Fixture()) {
            fixture.report();
            for (String metric : METRICS) {
                fixture.assertValue(metric,
                        metric.equals(MetricConstant.GRPC_SERVER_EXECUTOR_MAX_POOL_SIZE) ? 1 : 0);
            }

            fixture.blockFirstTaskAndQueueSecond();
            fixture.report();
            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_ACTIVE_THREADS, 1);
            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_QUEUE_SIZE, 1);
            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_POOL_SIZE, 1);

            fixture.finishTasks();
            fixture.report();
            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_ACTIVE_THREADS, 0);
            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_QUEUE_SIZE, 0);
            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_POOL_SIZE, 0);
        }
    }

    @Test
    void reportsActualSaturationAndShutdownRejectionsWithoutRunningRejectedTasks() throws Exception {
        try (Fixture fixture = new Fixture()) {
            fixture.blockFirstTaskAndQueueSecond();
            Runnable rejectedTask = () -> {
                throw new AssertionError("Rejected tasks must not execute on the caller");
            };
            assertThrows(RejectedExecutionException.class, () -> fixture.executor.execute(rejectedTask));
            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_REJECTED_TASKS, 1);
            fixture.report();
            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_REJECTED_TASKS, 1);
            fixture.report();
            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_REJECTED_TASKS, 1);

            fixture.executor.shutdown();
            assertThrows(RejectedExecutionException.class, () -> fixture.executor.execute(rejectedTask));
            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_REJECTED_TASKS, 2);
            fixture.report();
            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_REJECTED_TASKS, 2);
            fixture.finishTasks();
            fixture.report();
            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_REJECTED_TASKS, 2);
        }
    }

    @Test
    void monitorFailureStillAbortsTheRejectedTask() {
        FlexMonitor monitor = mock(FlexMonitor.class);
        doThrow(new IllegalStateException("monitor unavailable")).when(monitor)
                .report(MetricConstant.GRPC_SERVER_EXECUTOR_REJECTED_TASKS, 1.0);
        FlexlbGrpcServer server = newServer(monitor);
        RejectedExecutionHandler handler =
                (RejectedExecutionHandler) ReflectionTestUtils.getField(server, "rejectionHandler");
        ThreadPoolExecutor executor = new ThreadPoolExecutor(1, 1, 1, TimeUnit.MINUTES,
                new LinkedBlockingQueue<>(1), handler);
        executor.shutdown();
        Runnable task = mock(Runnable.class);

        assertThrows(RejectedExecutionException.class, () -> executor.execute(task));

        verify(monitor).report(MetricConstant.GRPC_SERVER_EXECUTOR_REJECTED_TASKS, 1.0);
        verifyNoInteractions(task);
    }

    private static FlexlbGrpcServer newServer(FlexMonitor monitor) {
        FlexlbConfig loadBalanceConfig = ConfigService.parse("""
                {"requestLifecycle":{"request":{"timeoutMs":60000}}}
                """);
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(loadBalanceConfig);
        return new FlexlbGrpcServer(null, config, new MockEnvironment(), null, monitor, null, null);
    }

    private static class Fixture implements AutoCloseable {
        private final SimpleMeterRegistry registry = new SimpleMeterRegistry();
        private final FlexlbGrpcServer server = newServer(new MicrometerFlexMonitor(registry));
        private final CountDownLatch started = new CountDownLatch(1);
        private final CountDownLatch release = new CountDownLatch(1);
        private final ThreadPoolExecutor executor;

        Fixture() {
            RejectedExecutionHandler rejectionHandler =
                    (RejectedExecutionHandler) ReflectionTestUtils.getField(server, "rejectionHandler");
            executor = new ThreadPoolExecutor(1, 1, 1, TimeUnit.MINUTES,
                    new LinkedBlockingQueue<>(1), rejectionHandler);
            ReflectionTestUtils.setField(server, "grpcExecutor", executor);
            ReflectionTestUtils.invokeMethod(server, "registerMetrics");
        }

        void blockFirstTaskAndQueueSecond() throws InterruptedException {
            executor.execute(() -> {
                started.countDown();
                try {
                    release.await();
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                }
            });
            assertTrue(started.await(3, TimeUnit.SECONDS), "first task must be running");
            executor.execute(() -> { });
        }

        void finishTasks() throws InterruptedException {
            release.countDown();
            executor.shutdown();
            assertTrue(executor.awaitTermination(3, TimeUnit.SECONDS));
        }

        void report() {
            ReflectionTestUtils.invokeMethod(server, "reportExecutorMetrics");
        }

        boolean hasValue(String metric, double expected) {
            var gauge = registry.find("flexlb." + metric).gauge();
            return gauge != null && gauge.value() == expected;
        }

        void assertValue(String metric, double expected) {
            double value = metric.equals(MetricConstant.GRPC_SERVER_EXECUTOR_REJECTED_TASKS)
                    ? registry.get("flexlb." + metric).counter().count()
                    : registry.get("flexlb." + metric).gauge().value();
            assertEquals(expected, value, metric);
        }

        @Override
        public void close() throws InterruptedException {
            release.countDown();
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(3, TimeUnit.SECONDS));
            registry.close();
        }
    }
}
