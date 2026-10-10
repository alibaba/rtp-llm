package org.flexlb.balance.scheduler;

import static org.flexlb.balance.scheduler.SchedulingTestConfig.freezeInputs;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/** Route-time queue capture regression on the final Prefill runtime boundary. */
@Tag("performance-regression")
class WorkerBatcherPerformanceTest {

    private static final int[] QUEUE_DEPTHS = {0, 1, 32, 128, 512};
    private static final int MEASUREMENT_ROUNDS = 5;

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void queueWaitSnapshotCaptureDoesNotScaleWithRequestCount() throws Throwable {
        var capture = MethodHandles.privateLookupIn(WorkerBatcher.class, MethodHandles.lookup())
                .findVirtual(WorkerBatcher.class, "recordQueueWait",
                        MethodType.methodType(void.class, RequestRoute.class, String.class));
        var allocationBean = ManagementFactory.getThreadMXBean() instanceof com.sun.management.ThreadMXBean bean
                && bean.isThreadAllocatedMemorySupported() ? bean : null;
        if (allocationBean != null) { allocationBean.setThreadAllocatedMemoryEnabled(true); }
        long threadId = Thread.currentThread().threadId();
        long shallowAllocation = 0;
        int operations = 2_000;
        // Both depths populate every priority bucket. Additional requests must not add capture work.
        for (int depth : new int[]{128, 512, 4096}) {
            WorkerBatcher runtime = runtimeWithDepth(depth);
            try {
                RequestRoute head = WorkerBatcherTestSupport.capture(runtime).items().getFirst();
                for (int i = 0; i < operations; i++) {
                    capture.invokeExact(runtime, head, "Prefill capacity exhausted");
                }
                long before = allocationBean == null ? 0 : allocationBean.getThreadAllocatedBytes(threadId);
                long started = System.nanoTime();
                for (int i = 0; i < operations; i++) {
                    capture.invokeExact(runtime, head, "Prefill capacity exhausted");
                }
                long nsPerCapture = (System.nanoTime() - started) / operations;
                long bytesPerCapture = allocationBean == null ? 0
                        : (allocationBean.getThreadAllocatedBytes(threadId) - before) / operations;
                assertEquals(depth, runtime.getLatestQueueWaitSnapshot().get("queueDepth"));
                System.out.printf("FlexLB wait capture: depth=%d ns_per_capture=%d bytes_per_capture=%d%n",
                        depth, nsPerCapture, bytesPerCapture);
                if (depth == 128) { shallowAllocation = bytesPerCapture; }
                if (allocationBean != null) {
                    assertTrue(bytesPerCapture < 1_024,
                            "the scheduling loop must retain raw counters without materializing PV maps");
                    assertTrue(bytesPerCapture <= shallowAllocation + 1_024,
                            "capture allocation must be bounded by priority levels, not request count");
                }
            } finally {
                runtime.stopAndAwait();
            }
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void timeoutDiagnosticsReadDoesNotGrowWithQueueDepth() throws Throwable {
        var allocationBean = ManagementFactory.getThreadMXBean() instanceof com.sun.management.ThreadMXBean bean
                && bean.isThreadAllocatedMemorySupported() ? bean : null;
        if (allocationBean != null) { allocationBean.setThreadAllocatedMemoryEnabled(true); }
        long threadId = Thread.currentThread().threadId();
        int operations = 100_000;
        for (int depth : QUEUE_DEPTHS) {
            WorkerBatcher runtime = runtimeWithDepth(depth);
            try {
                if (depth > 0) {
                    var capture = MethodHandles.privateLookupIn(WorkerBatcher.class, MethodHandles.lookup())
                            .findVirtual(WorkerBatcher.class, "recordQueueWait",
                                    MethodType.methodType(void.class, RequestRoute.class, String.class));
                    RequestRoute head = WorkerBatcherTestSupport.capture(runtime).items().getFirst();
                    capture.invokeExact(runtime, head, "Prefill capacity exhausted");
                    assertEquals(depth, runtime.getLatestQueueWaitSnapshot().get("queueDepth"));
                }
                long checksum = 0;
                for (int warmup = 0; warmup < operations; warmup++) {
                    checksum += runtime.getLatestQueueWaitSnapshot().size();
                }
                long allocatedBefore = allocationBean == null ? 0 : allocationBean.getThreadAllocatedBytes(threadId);
                long started = System.nanoTime();
                for (int operation = 0; operation < operations; operation++) {
                    checksum += runtime.getLatestQueueWaitSnapshot().size();
                }
                long nsPerRead = (System.nanoTime() - started) / operations;
                long bytesPerRead = allocationBean == null ? 0
                        : (allocationBean.getThreadAllocatedBytes(threadId) - allocatedBefore) / operations;
                System.out.printf("FlexLB timeout diagnostics: depth=%d ns_per_read=%d bytes_per_read=%d checksum=%d%n",
                        depth, nsPerRead, bytesPerRead, checksum);
                assertTrue(nsPerRead < 10_000L, "timeout diagnostics must not traverse the request queue");
                if (allocationBean != null) {
                    assertEquals(0L, bytesPerRead, "timeout reads must reuse the queue's published snapshot");
                }
            } finally {
                runtime.stopAndAwait();
            }
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void immutableProjectionCaptureRemainsBoundedAtDeepQueueDepth()
            throws Exception {
        int operations = Integer.getInteger(
                "flexlb.perf.queue-capture.operations-per-round", 500);
        long maxNsAtDepth512 = Long.getLong(
                "flexlb.perf.queue-capture.max-ns-at-depth-512",
                2_000_000L);
        long maxAllocatedBytesAtDepth512 = Long.getLong(
                "flexlb.perf.queue-capture.max-allocated-bytes-at-depth-512",
                256L * 1_024L);
        java.lang.management.ThreadMXBean baseThreadBean =
                ManagementFactory.getThreadMXBean();
        com.sun.management.ThreadMXBean allocationBean =
                baseThreadBean instanceof com.sun.management.ThreadMXBean bean
                        ? bean : null;
        if (allocationBean != null
                && allocationBean.isThreadAllocatedMemorySupported()
                && !allocationBean.isThreadAllocatedMemoryEnabled()) {
            allocationBean.setThreadAllocatedMemoryEnabled(true);
        }
        boolean measuresAllocations = allocationBean != null
                && allocationBean.isThreadAllocatedMemoryEnabled();
        long threadId = Thread.currentThread().threadId();

        for (int depth : QUEUE_DEPTHS) {
            WorkerBatcher runtime = runtimeWithDepth(depth);
            PrefillEndpoint endpoint = (PrefillEndpoint) org.springframework.test.util.ReflectionTestUtils
                    .getField(runtime, "prefillEndpoint");
            try {
                for (int warmup = 0; warmup < 100; warmup++) {
                    assertEquals(depth, endpoint.captureRouteProjectionInputs()
                            .queue().activeItems().size());
                }

                long[] elapsedRounds = new long[MEASUREMENT_ROUNDS];
                long[] allocatedRounds = new long[MEASUREMENT_ROUNDS];
                long checksum = 0L;
                for (int round = 0; round < MEASUREMENT_ROUNDS; round++) {
                    long allocatedBefore = measuresAllocations
                            ? allocationBean.getThreadAllocatedBytes(threadId)
                            : 0L;
                    long started = System.nanoTime();
                    for (int operation = 0;
                         operation < operations;
                         operation++) {
                        RouteProjection.Inputs inputs =
                                endpoint.captureRouteProjectionInputs();
                        checksum += inputs.queue().activeItems().size();
                        checksum += inputs.ownershipVersion();
                    }
                    elapsedRounds[round] =
                            (System.nanoTime() - started) / operations;
                    if (measuresAllocations) {
                        allocatedRounds[round] = Math.max(
                                0L,
                                allocationBean.getThreadAllocatedBytes(threadId)
                                        - allocatedBefore) / operations;
                    }
                }
                Arrays.sort(elapsedRounds);
                Arrays.sort(allocatedRounds);
                long medianNs = elapsedRounds[MEASUREMENT_ROUNDS / 2];
                long medianAllocatedBytes =
                        allocatedRounds[MEASUREMENT_ROUNDS / 2];
                System.out.printf(
                        "FlexLB queue-capture performance: depth=%d "
                                + "ns_per_op=%d allocated_bytes_per_op=%d "
                                + "checksum=%d%n",
                        depth,
                        medianNs,
                        medianAllocatedBytes,
                        checksum);
                if (depth == 512) {
                    assertTrue(medianNs <= maxNsAtDepth512,
                            () -> "depth-512 immutable queue capture took "
                                    + medianNs + " ns/op, above ceiling "
                                    + maxNsAtDepth512);
                    if (measuresAllocations) {
                        assertTrue(
                                medianAllocatedBytes
                                        <= maxAllocatedBytesAtDepth512,
                                () -> "depth-512 immutable queue capture "
                                        + "allocated " + medianAllocatedBytes
                                        + " bytes/op, above ceiling "
                                        + maxAllocatedBytesAtDepth512);
                    }
                }
            } finally {
                runtime.stopAndAwait();
            }
        }
    }

    private static WorkerBatcher runtimeWithDepth(int depth)
            throws InterruptedException {
        FlexlbConfig config = org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useSingleDecision(config);
        WorkerStatus status = WorkerStatus.createDiscovered(RoleType.PREFILL,
                "perf", "127.0.0.1", 8080, 8090, "perf-site");
        BlockingDeliveryStrategy delivery = new BlockingDeliveryStrategy();
        AbstractRequestScheduler scheduler = mock(AbstractRequestScheduler.class);
        PrefillEndpoint endpoint = org.flexlb.balance.endpoint.EndpointTestSupport.prefill(
                status, config, delivery, SchedulerTestSupport.repository(scheduler),
                mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class));
        endpoint.enableQueueRuntime(QueueExecutionSettings.capture(config));
        WorkerBatcher runtime = org.flexlb.balance.endpoint.EndpointTestSupport.batcher(endpoint);
        long now = System.currentTimeMillis();
        List<RequestRoute> items = new ArrayList<>(depth);
        for (int index = 0; index < depth; index++) {
            items.add(item(
                    config,
                    endpoint,
                    index + 1L,
                    1 + (index * 37 % 100),
                    now - depth + index,
                    256L + (index % 32)));
        }
        for (RequestRoute item : items) {
            SchedulerTestSupport.bindOwner(item.ctx(), scheduler);
            assertTrue(runtime.offer(item));
        }
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (WorkerBatcherTestSupport.state(runtime).queueDepth() != depth && System.nanoTime() < deadline) {
            TimeUnit.MILLISECONDS.sleep(1L);
        }
        assertEquals(depth, WorkerBatcherTestSupport.state(runtime).queueDepth());
        return runtime;
    }

    private static RequestRoute item(
            FlexlbConfig config,
            PrefillEndpoint endpoint,
            long requestId,
            int priority,
            long enqueuedAtMs,
            long seqLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        request.setPriority(priority);
        RequestContext context = new RequestContext(config);
        context.setRequest(request);
        context.setSchedulingMetadata(
                SchedulingMetadata.explicit(priority, Long.MAX_VALUE));
        context.setFuture(new CompletableFuture<Response>());
        return org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context),
                null,
                null,
                null,
                endpoint,
                null,
                null,
                enqueuedAtMs);
    }

    private static final class BlockingDeliveryStrategy
            implements DeliveryStrategy {
        @Override
        public void deliver(DeliveryTransaction transaction, String reason, int queueDepth,
                org.flexlb.balance.projection.WorkSnapshot precedingWork, PrefillTimePredictor.Evaluator evaluator) {
            throw new AssertionError("boundary-only strategy cannot deliver");
        }


        private final CapacityBoundary.Availability availability =
                new CapacityBoundary.Availability() {
                    @Override
                    public boolean isAvailable() {
                        return false;
                    }

                    @Override
                    public void addListener(Runnable listener) {
                    }

                    @Override
                    public void removeListener(Runnable listener) {
                    }
                };

        @Override
        public DeliveryTransaction prepare(
                List<RequestRoute> candidates,
                PrefillTimePredictor.Evaluator evaluator,
                OptionalLong plannedPrediction) {
            return WorkerBatcherTestSupport.boundaryOnly(
                    candidates.getFirst(),
                    CapacityBoundary.unavailable(
                            availability,
                            new RouteProjection.AdmissionBlockSemantics(
                                    "PERF_BLOCK",
                                    RouteProjection.AfterProbeAdmission.BLOCKED,
                                    "PERF_BLOCK",
                                    RoleType.PREFILL)));
        }

        @Override
        public GroupPlanner.PrefixPrediction<RequestRoute> newGroupPredictor(
                PrefillTimePredictor.Evaluator evaluator) {
            return (added, items) -> {
                return 0.0;
            };
        }

        @Override
        public RouteProjection.DeliveryProjection projectionPolicy() {
            return mock(RouteProjection.DeliveryProjection.class);
        }

    }
}
