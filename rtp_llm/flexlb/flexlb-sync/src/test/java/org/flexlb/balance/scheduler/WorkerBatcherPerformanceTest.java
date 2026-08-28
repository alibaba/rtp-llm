package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.balance.endpoint.EndpointEventSink;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

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
import static org.mockito.Mockito.when;

/** Route-time queue capture regression on the final Prefill runtime boundary. */
@Tag("performance-regression")
class WorkerBatcherPerformanceTest {

    private static final int[] QUEUE_DEPTHS = {0, 1, 32, 128, 512};
    private static final int MEASUREMENT_ROUNDS = 5;

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
            try {
                for (int warmup = 0; warmup < 100; warmup++) {
                    assertEquals(depth, runtime.captureRouteProjectionInputs()
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
                                runtime.captureRouteProjectionInputs();
                        checksum += inputs.queue().activeItems().size();
                        checksum += inputs.pendingRequestCount();
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
        FlexlbConfig config = new FlexlbConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useSingleDecision(config);
        SchedulingTestConfig.useQueueCapacity(config)
                .setMaxWaitingRequestsPerPrefillWorker(1_024);
        config.getRouter().getRoles().getPrefill().getAvailability()
                .setMaxPendingRequests(1_024);
        PrefillEndpoint endpoint = stablePrefillEndpoint();
        BlockingDeliveryStrategy delivery = new BlockingDeliveryStrategy();
        WorkerBatcher runtime = new WorkerBatcher(
                "perf-worker-" + depth,
                endpoint,
                config,
                delivery,
                mock(EndpointEventSink.class));
        runtime.start();
        long now = System.currentTimeMillis();
        List<ScheduledRequest> items = new ArrayList<>(depth);
        for (int index = 0; index < depth; index++) {
            items.add(item(
                    config,
                    endpoint,
                    index + 1L,
                    1 + (index * 37 % 100),
                    now - depth + index,
                    256L + (index % 32)));
        }
        for (ScheduledRequest item : items) {
            assertTrue(runtime.offer(item));
        }
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (runtime.queueSize() != depth && System.nanoTime() < deadline) {
            TimeUnit.MILLISECONDS.sleep(1L);
        }
        assertEquals(depth, runtime.queueSize());
        return runtime;
    }

    private static ScheduledRequest item(
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
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(config);
        context.setSchedulingMetadata(
                SchedulingMetadata.explicit(priority, Long.MAX_VALUE));
        return new ScheduledRequest(
                context,
                new CompletableFuture<Response>(),
                null,
                null,
                null,
                endpoint,
                null,
                null,
                enqueuedAtMs);
    }

    private static PrefillEndpoint stablePrefillEndpoint() {
        PrefillTimePredictor.Evaluator evaluator =
                mock(PrefillTimePredictor.Evaluator.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(predictor.evaluator()).thenReturn(evaluator);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.getStatus()).thenReturn(WorkerStatus.createDiscovered(
                RoleType.PREFILL,
                "perf",
                "127.0.0.1",
                8080,
                8090,
                "perf-site"));
        return endpoint;
    }

    private static final class BlockingDeliveryStrategy
            implements DeliveryStrategy {

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
        public Transaction prepare(
                List<ScheduledRequest> candidates,
                PrefillTimePredictor.Evaluator evaluator,
                OptionalLong plannedPrediction) {
            return GroupPolicyTestSupport.boundaryOnly(
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
        public double projectGroupDurationMs(
                List<ScheduledRequest> items,
                PrefillTimePredictor.Evaluator evaluator) {
            return 0.0;
        }

        @Override
        public RouteProjection.DeliveryProjection projectionPolicy() {
            return mock(RouteProjection.DeliveryProjection.class);
        }

    }
}
