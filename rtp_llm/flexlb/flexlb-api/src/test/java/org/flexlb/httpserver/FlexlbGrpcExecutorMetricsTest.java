package org.flexlb.httpserver;

import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.constant.MetricConstant;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.MicrometerFlexMonitor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.mock.env.MockEnvironment;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
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
            MetricConstant.GRPC_SERVER_EXECUTOR_COMPLETED_TASKS,
            MetricConstant.GRPC_SERVER_EXECUTOR_REJECTED_TASKS);

    @Test
    void skipsReportingBeforeExecutorCreationAndRegistersSixGauges() {
        FlexMonitor monitor = mock(FlexMonitor.class);
        FlexlbGrpcServer server = newServer(monitor);

        ReflectionTestUtils.invokeMethod(server, "reportExecutorMetrics");
        verifyNoInteractions(monitor);

        ReflectionTestUtils.invokeMethod(server, "registerMetrics");
        for (String metric : METRICS) {
            verify(monitor).register(metric, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        }
        verifyNoMoreInteractions(monitor);
    }

    @Test
    void reportsIdleZeroBusyQueueAndCompletedSnapshotsWithoutDoubleCounting() throws Exception {
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
            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_COMPLETED_TASKS, 0);

            fixture.finishTasks();
            fixture.report();
            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_ACTIVE_THREADS, 0);
            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_QUEUE_SIZE, 0);
            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_POOL_SIZE, 0);
            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_COMPLETED_TASKS, 2);
            fixture.report();
            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_COMPLETED_TASKS, 2);
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
            fixture.report();
            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_REJECTED_TASKS, 1);
            fixture.report();
            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_REJECTED_TASKS, 1);

            fixture.executor.shutdown();
            assertThrows(RejectedExecutionException.class, () -> fixture.executor.execute(rejectedTask));
            fixture.report();
            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_REJECTED_TASKS, 2);
            fixture.finishTasks();
            fixture.report();
            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_REJECTED_TASKS, 2);
            fixture.assertValue(MetricConstant.GRPC_SERVER_EXECUTOR_COMPLETED_TASKS, 2);
        }
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
            FlexlbGrpcServer.CountingAbortHandler rejectionHandler =
                    (FlexlbGrpcServer.CountingAbortHandler) ReflectionTestUtils.getField(server, "countingAbortHandler");
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

        void assertValue(String metric, double expected) {
            assertEquals(expected, registry.get("flexlb." + metric).gauge().value(), metric);
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
