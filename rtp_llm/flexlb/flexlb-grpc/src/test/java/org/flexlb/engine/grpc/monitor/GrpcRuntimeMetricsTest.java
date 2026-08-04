package org.flexlb.engine.grpc.monitor;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.flexlb.engine.grpc.config.GrpcCallbackThreadPoolExecutor;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

class GrpcRuntimeMetricsTest {

    @Test
    void exportsCurrentInFlightCallsAndZeroAfterCompletion() {
        FlexMonitor monitor = mock(FlexMonitor.class);
        ThreadPoolExecutor callbackExecutor = mock(ThreadPoolExecutor.class);
        GrpcRuntimeMetrics metrics = new GrpcRuntimeMetrics(monitor, callbackExecutor);
        metrics.init();

        GrpcRuntimeMetrics.GrpcCallHandle firstCall =
                metrics.recordCallStarted("engine", "GetWorkerStatus", TimeUnit.SECONDS.toMillis(1));
        GrpcRuntimeMetrics.GrpcCallHandle secondCall =
                metrics.recordCallStarted("engine", "GetWorkerStatus", TimeUnit.SECONDS.toMillis(1));
        metrics.reportRuntimeMetrics();

        verify(monitor).register("app.grpc.call.inflight", FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).report(
                "app.grpc.call.inflight",
                FlexMetricTags.of("client", "engine", "service", "GetWorkerStatus"),
                2.0);

        metrics.recordCallCompleted(firstCall);
        metrics.recordCallCompleted(secondCall);
        metrics.reportRuntimeMetrics();

        verify(monitor).report(
                "app.grpc.call.inflight",
                FlexMetricTags.of("client", "engine", "service", "GetWorkerStatus"),
                0.0);
    }

    @Test
    void expiresInFlightCallWhenNoTerminalCallbackArrivesBeforeDeadline() {
        FlexMonitor monitor = mock(FlexMonitor.class);
        ThreadPoolExecutor callbackExecutor = mock(ThreadPoolExecutor.class);
        GrpcRuntimeMetrics metrics = new GrpcRuntimeMetrics(monitor, callbackExecutor);

        metrics.recordCallStarted("engine", "GetWorkerStatus", 0);
        metrics.reportRuntimeMetrics();

        verify(monitor).report(
                "app.grpc.call.inflight",
                FlexMetricTags.of("client", "engine", "service", "GetWorkerStatus"),
                0.0);
    }

    @Test
    void doesNotDecrementOtherCallsWhenAnExpiredCallCompletesLate() {
        FlexMonitor monitor = mock(FlexMonitor.class);
        ThreadPoolExecutor callbackExecutor = mock(ThreadPoolExecutor.class);
        GrpcRuntimeMetrics metrics = new GrpcRuntimeMetrics(monitor, callbackExecutor);
        GrpcRuntimeMetrics.GrpcCallHandle expiredCall =
                metrics.recordCallStarted("engine", "GetWorkerStatus", 0);
        GrpcRuntimeMetrics.GrpcCallHandle activeCall =
                metrics.recordCallStarted("engine", "GetWorkerStatus", TimeUnit.SECONDS.toMillis(1));

        metrics.reportRuntimeMetrics();
        metrics.recordCallCompleted(expiredCall);
        metrics.reportRuntimeMetrics();

        verify(monitor, times(2)).report(
                "app.grpc.call.inflight",
                FlexMetricTags.of("client", "engine", "service", "GetWorkerStatus"),
                1.0);
        metrics.recordCallCompleted(activeCall);
    }

    @Test
    void exportsRejectedCallbackTasksAsCounter() throws InterruptedException {
        FlexMonitor monitor = mock(FlexMonitor.class);
        GrpcCallbackThreadPoolExecutor callbackExecutor = new GrpcCallbackThreadPoolExecutor(
                1, 1, 1, TimeUnit.MINUTES, new ArrayBlockingQueue<>(1),
                new NamedThreadFactory("grpc-callback-test"));
        GrpcRuntimeMetrics metrics = new GrpcRuntimeMetrics(monitor, callbackExecutor);
        CountDownLatch taskStarted = new CountDownLatch(1);
        CountDownLatch releaseTask = new CountDownLatch(1);
        try {
            callbackExecutor.execute(() -> {
                taskStarted.countDown();
                try {
                    releaseTask.await();
                } catch (InterruptedException interruptedException) {
                    Thread.currentThread().interrupt();
                }
            });
            assertTrue(taskStarted.await(1, TimeUnit.SECONDS));
            callbackExecutor.execute(() -> { });
            assertThrows(RejectedExecutionException.class, () -> callbackExecutor.execute(() -> { }));

            metrics.init();
            metrics.reportRuntimeMetrics();

            verify(monitor).register("app.grpc.callback.executor.rejected.total", FlexMetricType.COUNTER,
                    FlexPriorityType.PRECISE);
            verify(monitor).report(
                    "app.grpc.callback.executor.rejected.total",
                    FlexMetricTags.of("executor", "gRpcExecutor"),
                    1.0);
        } finally {
            releaseTask.countDown();
            callbackExecutor.shutdownNow();
        }
    }
}
