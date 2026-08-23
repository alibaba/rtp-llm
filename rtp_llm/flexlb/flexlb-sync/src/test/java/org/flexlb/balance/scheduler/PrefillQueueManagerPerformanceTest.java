package org.flexlb.balance.scheduler;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/**
 * Allocation/sort regression coverage for the Auto-TPM queue-wait hot path.
 *
 * <p>Cost-based routing evaluates this method once per candidate endpoint, so
 * queue depth and endpoint count multiply the per-call cost in production.
 */
@Tag("performance-regression")
class PrefillQueueManagerPerformanceTest {

    private static final int[] QUEUE_DEPTHS = {0, 1, 32, 128, 512};
    private static final int MEASUREMENT_ROUNDS = 5;

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void estimateWaitScalesLinearlyAcrossDeepQueues() {
        int operations = Integer.getInteger(
                "flexlb.perf.queue-wait.operations-per-round", 2_000);
        long maxNsAtDepth512 = Long.getLong(
                "flexlb.perf.queue-wait.max-ns-at-depth-512", 10_000L);
        long maxAllocatedBytesPerOperation = Long.getLong(
                "flexlb.perf.queue-wait.max-allocated-bytes-per-operation", 64L);
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
            WorkerBatcher batcher = batcherWithDepth(depth);
            PrefillQueueManager manager = batcher.queueManager();
            for (int warmup = 0; warmup < 1_000; warmup++) {
                manager.estimateWaitMs(50, 1_000_000L + warmup);
            }

            long[] rounds = new long[MEASUREMENT_ROUNDS];
            long[] allocatedRounds = new long[MEASUREMENT_ROUNDS];
            long checksum = 0L;
            for (int round = 0; round < MEASUREMENT_ROUNDS; round++) {
                long allocatedBefore = measuresAllocations
                        ? allocationBean.getThreadAllocatedBytes(threadId) : 0L;
                long started = System.nanoTime();
                for (int operation = 0; operation < operations; operation++) {
                    checksum += manager.estimateWaitMs(50, 2_000_000L + operation);
                }
                rounds[round] = (System.nanoTime() - started) / operations;
                if (measuresAllocations) {
                    allocatedRounds[round] = Math.max(0L,
                            allocationBean.getThreadAllocatedBytes(threadId) - allocatedBefore)
                            / operations;
                }
            }
            Arrays.sort(rounds);
            Arrays.sort(allocatedRounds);
            long medianNs = rounds[MEASUREMENT_ROUNDS / 2];
            long medianAllocatedBytes = allocatedRounds[MEASUREMENT_ROUNDS / 2];
            System.out.printf(
                    "FlexLB queue-wait performance: depth=%d ns_per_op=%d "
                            + "allocated_bytes_per_op=%d checksum=%d%n",
                    depth, medianNs, medianAllocatedBytes, checksum);
            if (depth == 512) {
                assertTrue(medianNs <= maxNsAtDepth512,
                        () -> "depth-512 queue wait estimate took " + medianNs
                                + " ns/op, above regression ceiling " + maxNsAtDepth512);
                if (measuresAllocations) {
                    assertTrue(medianAllocatedBytes <= maxAllocatedBytesPerOperation,
                            () -> "depth-512 queue wait estimate allocated "
                                    + medianAllocatedBytes + " bytes/op, above ceiling "
                                    + maxAllocatedBytesPerOperation);
                }
            }
            batcher.shutdown();
        }
    }

    private static WorkerBatcher batcherWithDepth(int depth) {
        FlexlbConfig config = new FlexlbConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useBatchDispatcher(config);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(200);
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(32);
        SchedulingTestConfig.useQueueCapacity(config)
                .setMaxWaitingRequestsPerPrefillWorker(1_024);
        WorkerBatcher batcher = new WorkerBatcher(
                "perf-worker-" + depth,
                null,
                config,
                mock(DecisionGroupHandler.class),
                TestCapacityAdmission.alwaysAvailable(),
                mock(BatchSchedulerReporter.class));
        long now = System.currentTimeMillis();
        List<BatchItem> items = new ArrayList<>(depth);
        for (int index = 0; index < depth; index++) {
            items.add(item(
                    config,
                    index + 1L,
                    1 + (index * 37 % 100),
                    now - depth + index,
                    256L + (index % 32)));
        }
        for (BatchItem item : items) {
            assertTrue(batcher.tryOffer(item));
        }
        return batcher;
    }

    private static BatchItem item(FlexlbConfig config, long requestId, int priority,
                                  long enqueuedAtMs, long seqLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        request.setPriority(priority);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(config);
        context.setSchedulingMetadata(
                SchedulingMetadata.explicit(priority, Long.MAX_VALUE));
        return new BatchItem(
                context,
                new CompletableFuture<>(),
                null,
                null,
                null,
                null,
                null,
                enqueuedAtMs);
    }
}
