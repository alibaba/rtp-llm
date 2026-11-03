import com.sun.management.ThreadMXBean;
import java.lang.management.ManagementFactory;
import java.nio.file.*;
import java.util.*;
import org.flexlb.balance.delivery.*;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.*;
import org.flexlb.balance.projection.*;
import org.flexlb.balance.scheduler.*;
import static org.mockito.Mockito.mock;

/** Same full-fleet projection workload on both revisions; no altered baseline algorithm. */
public class CacheBench {
    static volatile long sink;
    static final ThreadMXBean CPU = (ThreadMXBean) ManagementFactory.getThreadMXBean();
    static void bench(String name, FormulaPredictor model, RouteProjection.DeliveryProjection policy,
                      int depth, boolean churn, long predictionBudgetMs) {
        var snapshots = new ArrayList<RouteProjection.Inputs>();
        for (int e = 0; e < 64; e++) {
            var items = new ArrayList<GroupPlanner.Item>();
            for (int i = 0; i < depth; i++) {
                long shape = churn ? (e * depth + i) : i % 64;
                items.add(new GroupPlanner.Item(i, 0, i, 1000, 1_000_000,
                        1000 + shape, shape % 100));
            }
            snapshots.add(new RouteProjection.Inputs(new QueueSnapshot(1000, true,
                    Comparator.comparingLong(GroupPlanner.Item::enqueueSeq),
                    new GroupPlanner.Constraints(64, 10_000_000, 20_000_000, predictionBudgetMs, 0), items, null),
                    new WorkSnapshot(1000, List.of(), List.of(), 0)));
        }
        long sequence = 0;
        for (int round = 0; round < 8; round++) {
            long start = System.nanoTime(), cpu = CPU.getCurrentThreadCpuTime();
            long bytes = CPU.getThreadAllocatedBytes(Thread.currentThread().threadId()), count = 0, checksum = 0;
            do {
                // Visit the exact same 64 probes in every measured operation.
                for (int p = 0; p < 64; p++) {
                    var probe = new RouteProjection.Probe(999999, 0, 1000, 1_000_000,
                            1024 + p * 32, p * 8, p * 8);
                    for (int e = 0; e < 5; e++) {
                        var input = snapshots.get((p + e) & 63);
                        if (name.endsWith("_invalidated")) {
                            input = new RouteProjection.Inputs(input.queue(), input.work(), p + 1);
                        }
                        var result = RouteProjection.project(input, probe, model, policy);
                        checksum += result.requiredProjectedTtftMs();
                    }
                    count++;
                }
            } while (System.nanoTime() - start < 300_000_000L);
            long wall = System.nanoTime() - start, used = CPU.getCurrentThreadCpuTime() - cpu;
            long allocated = CPU.getThreadAllocatedBytes(Thread.currentThread().threadId()) - bytes;
            sink = checksum + sequence++;
            if (round >= 5) System.out.printf(Locale.ROOT,
                    "%s ns=%.2f cpu=%.2f bytes=%.2f checksum=%.9f%n", name,
                    (double)wall/count, (double)used/count, (double)allocated/count, (double)checksum/count);
        }
    }
    public static void main(String[] args) throws Exception {
        var model = new FormulaPredictor(Files.readString(Path.of(args[0])));
        var registry = mock(RequestRegistry.class);
        var metrics = mock(DeliveryMetrics.class);
        var route = new RouteDeliveryStrategy(registry, metrics).projectionPolicy();
        var batch = new BatchDeliveryStrategy(
                () -> CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST),
                () -> 1L, registry, metrics).projectionPolicy();
        bench("batch128", model, batch, 128, false, 700);
        bench("route128", model, route, 128, false, 700);
        bench("route512", model, route, 512, false, 700);
        bench("route512_churn", model, route, 512, true, 700);
        bench("route512_wide", model, route, 512, false, 1_000_000);
        bench("route512_no_budget", model, route, 512, false, 0);
        bench("batch128_invalidated", model, batch, 128, false, 700);
    }
}
