package org.flexlb.balance.projection;

import org.flexlb.balance.planner.BuiltinGroupPlanner;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.prediction.PrefillTimePredictor.PrefillBatchFeatures;
import org.flexlb.balance.projection.RequestTimeEstimation.QueueSnapshot;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot;
import org.flexlb.config.DispatcherConfig;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.lang.management.ManagementFactory;
import java.util.Comparator;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Tag("performance-regression")
class ProjectionPerformanceTest {
    @Test
    void repeatedNonemptyQueueProjectionHasBoundedAllocation() {
        var evaluator =
                new PrefillTimePredictor.Evaluator() {
                    public long estimateMs(long tokens, long hit) {
                        return tokens - hit;
                    }

                    public double predictBatchMs(PrefillBatchFeatures features) {
                        long sum = 0;
                        for (int index = 0; index < features.items().size(); index++) {
                            var item = features.items().get(index);
                            sum += item.seqLen() - item.hitCache();
                        }
                        return sum;
                    }
                };
        var queue =
                new QueueSnapshot(
                        1000,
                        true,
                        BuiltinGroupPlanner.FIXED_WINDOW,
                        Comparator.comparingLong(GroupPlanner.Item::requestId),
                        new GroupPlanner.Constraints(4, 1000, 1000, 0, 30),
                        List.of(new GroupPlanner.Item(1, 50, 1, 1000, Long.MAX_VALUE, 10, 0)),
                        null);
        var inputs =
                new RequestTimeEstimation.Inputs(
                        queue,
                        new WorkSnapshot(1000, List.of(), List.of(), 0),
                        0,
                        evaluator,
                        DispatcherConfig.Type.BATCH);
        var bean = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
        assertTrue(bean.isThreadAllocatedMemorySupported());
        bean.setThreadAllocatedMemoryEnabled(true);
        long thread = Thread.currentThread().threadId();
        int operations = 100000;
        var estimator = RequestTimeEstimation.current();
        for (int round = 0; round < 4; round++) {
            long before = bean.getThreadAllocatedBytes(thread);
            long start = System.nanoTime();
            long checksum = 0;
            for (int index = 0; index < operations; index++) {
                var result =
                        estimator.projectRankingView(
                                inputs, 2 + index / 750, 50, 1000, Long.MAX_VALUE, 20, 0, 0, 1000);
                checksum += result.projectedTtftMsValue();
            }
            long elapsed = System.nanoTime() - start;
            long bytes = (bean.getThreadAllocatedBytes(thread) - before) / operations;
            assertEquals(30L * operations, checksum);
            if (round > 0) {
                System.out.printf(
                        "Nonempty projection: ns/op=%d bytes/op=%d%n", elapsed / operations, bytes);
                assertTrue(
                        bytes <= 96,
                        "fleet projection must reuse the same incoming value without copying the queue");
            }
        }
    }
}
