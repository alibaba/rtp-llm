package org.flexlb.httpserver;

import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.constant.MetricConstant;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.metric.MicrometerFlexMonitor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.mock.env.MockEnvironment;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

@Timeout(10)
class FlexlbGrpcExecutorMetricsTest {
    @Test
    void reporterReadsIdleBusyAndTerminatedServerWithoutScheduledPush() throws Exception {
        try (Fixture fixture = new Fixture()) {
            fixture.assertGauge(MetricConstant.GRPC_SERVER_EXECUTOR_ACTIVE_THREADS, 0);
            fixture.assertGauge(MetricConstant.GRPC_SERVER_EXECUTOR_QUEUE_SIZE, 0);
            fixture.assertGauge(MetricConstant.GRPC_SERVER_EXECUTOR_POOL_SIZE, 0);
            fixture.assertGauge(MetricConstant.GRPC_SERVER_EXECUTOR_MAX_POOL_SIZE, 1);
            fixture.assertCounter(MetricConstant.GRPC_SERVER_EXECUTOR_COMPLETED_TASKS, 0);
            fixture.assertCounter(MetricConstant.GRPC_SERVER_EXECUTOR_CALLER_RUNS, 0);
            fixture.saturate();
            fixture.assertGauge(MetricConstant.GRPC_SERVER_EXECUTOR_ACTIVE_THREADS, 1);
            fixture.assertGauge(MetricConstant.GRPC_SERVER_EXECUTOR_QUEUE_SIZE, 1);
            fixture.assertGauge(MetricConstant.GRPC_SERVER_EXECUTOR_POOL_SIZE, 1);
            fixture.finish();
            fixture.assertGauge(MetricConstant.GRPC_SERVER_EXECUTOR_ACTIVE_THREADS, 0);
            fixture.assertGauge(MetricConstant.GRPC_SERVER_EXECUTOR_QUEUE_SIZE, 0);
            fixture.assertGauge(MetricConstant.GRPC_SERVER_EXECUTOR_POOL_SIZE, 0);
            fixture.assertCounter(MetricConstant.GRPC_SERVER_EXECUTOR_COMPLETED_TASKS, 2);
            fixture.assertCounter(MetricConstant.GRPC_SERVER_EXECUTOR_COMPLETED_TASKS, 2);
        }
    }

    @Test
    void cumulativeCallerRunsCountPreservesAcceptedCallbacksAtSaturationAndShutdown()
            throws Exception {
        try (Fixture fixture = new Fixture()) {
            fixture.saturate();
            var callbacks = new AtomicInteger();
            fixture.executor.execute(callbacks::incrementAndGet);
            assertEquals(1, callbacks.get());
            fixture.assertCounter(MetricConstant.GRPC_SERVER_EXECUTOR_CALLER_RUNS, 1);
            fixture.assertCounter(MetricConstant.GRPC_SERVER_EXECUTOR_CALLER_RUNS, 1);
            fixture.executor.shutdown();
            fixture.executor.execute(callbacks::incrementAndGet);
            assertEquals(2, callbacks.get());
            fixture.assertCounter(MetricConstant.GRPC_SERVER_EXECUTOR_CALLER_RUNS, 2);
            fixture.finish();
            fixture.assertCounter(MetricConstant.GRPC_SERVER_EXECUTOR_CALLER_RUNS, 2);
        }
    }

    private static final class Fixture implements AutoCloseable {
        final SimpleMeterRegistry registry = new SimpleMeterRegistry();
        final CountDownLatch release = new CountDownLatch(1);
        final ThreadPoolExecutor executor;
        final FlexlbGrpcServer server;

        Fixture() {
            ConfigService config = mock(ConfigService.class);
            when(config.loadBalanceConfig()).thenReturn(new FlexlbConfig());
            var reporter = new GrpcReporter(new MicrometerFlexMonitor(registry));
            server =
                    new FlexlbGrpcServer(
                            null, config, new MockEnvironment(), null, reporter, null, null);
            var handler = new FlexlbGrpcServer.CountingCallerRunsHandler();
            executor =
                    new ThreadPoolExecutor(
                            1, 1, 1, TimeUnit.MINUTES, new LinkedBlockingQueue<>(1), handler);
            ReflectionTestUtils.setField(server, "grpcExecutor", executor);
            reporter.registerServerExecutor(server, 1, handler.getCallerRunsCount()::get);
        }

        void saturate() throws InterruptedException {
            var started = new CountDownLatch(1);
            executor.execute(
                    () -> {
                        started.countDown();
                        try {
                            release.await();
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        }
                    });
            assertTrue(started.await(3, TimeUnit.SECONDS));
            executor.execute(() -> {});
        }

        void finish() throws InterruptedException {
            release.countDown();
            executor.shutdown();
            assertTrue(executor.awaitTermination(3, TimeUnit.SECONDS));
        }

        void assertGauge(String name, double value) {
            assertEquals(value, registry.get("flexlb." + name).gauge().value(), name);
        }

        void assertCounter(String name, double value) {
            assertEquals(value, registry.get("flexlb." + name).functionCounter().count(), name);
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
