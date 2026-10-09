import com.sun.management.ThreadMXBean;
import java.lang.management.ManagementFactory;
import java.nio.file.*;
import java.util.*;
import java.util.function.DoubleSupplier;
import org.flexlb.balance.delivery.*;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.*;
import org.flexlb.balance.projection.*;
import org.flexlb.balance.scheduler.*;
import static org.mockito.Mockito.mock;

/** Forked benchmark of whole prefix selection and five-candidate frozen queue projection. */
public class PlanningBench {
    static volatile double sink;
    static final ThreadMXBean CPU = (ThreadMXBean) ManagementFactory.getThreadMXBean();
    static final long THREAD = Thread.currentThread().threadId();
    static final Comparator<GroupPlanner.Item> ORDER = Comparator.comparingInt(GroupPlanner.Item::priority)
            .thenComparingLong(GroupPlanner.Item::enqueueSeq);
    static FormulaPredictor model;
    static RouteProjection.DeliveryProjection policy;
    static GroupPlanner.Constraints constraints(int size) {
        return new GroupPlanner.Constraints(size, 1_000_000, 2_000_000, 700, 700);
    }
    static List<GroupPlanner.Item> items(int count, int seed) {
        var random = new Random(seed);
        var items = new ArrayList<GroupPlanner.Item>();
        for (int i = 0; i < count; i++) {
            long compute = 100 + random.nextInt(4096), hit = random.nextInt(4096);
            items.add(new GroupPlanner.Item(i, 0, i, 1000, 61_000, compute + hit, hit));
        }
        return items;
    }
    static double select(List<GroupPlanner.Item> items, int size) {
        // INCREMENTAL_BEGIN
        var batch = model.newBatchPrediction();
        var selected = GroupPlanner.selectWithPrediction(items, GroupPlanner.itemAccess(), constraints(size),
                (added, prefix) -> batch.append(added.seqLen(), added.hitCache()));
        // INCREMENTAL_END
        return selected.items().size() + selected.selectedPredictionMs().orElse(0);
    }
    static void bench(String name, DoubleSupplier task) {
        for (int round = 0; round < 6; round++) {
            long start = System.nanoTime(), cpu = CPU.getCurrentThreadCpuTime();
            long bytes = CPU.getThreadAllocatedBytes(THREAD), count = 0;
            double total = 0;
            do {
                for (int j = 0; j < 64; j++) total += task.getAsDouble();
                count += 64;
            } while (System.nanoTime() - start < 400_000_000L);
            long elapsed = System.nanoTime() - start;
            long usedCpu = CPU.getCurrentThreadCpuTime() - cpu;
            long allocated = CPU.getThreadAllocatedBytes(THREAD) - bytes;
            sink = total;
            if (round >= 3) System.out.printf(Locale.ROOT,
                    "%s round=%d ns/op=%.2f cpu_ns/op=%.2f bytes/op=%.2f checksum=%.9f%n",
                    name, round, (double) elapsed / count, (double) usedCpu / count,
                    (double) allocated / count, total / count);
        }
    }
    public static void main(String[] args) throws Exception {
        model = new FormulaPredictor(Files.readString(Path.of(args[0])));
        policy = new BatchDeliveryStrategy(() -> CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST),
                () -> 1L, mock(RequestRegistry.class), mock(DeliveryMetrics.class)).projectionPolicy();
        for (int size : new int[]{1, 8, 32, 64}) {
            var inputs = new ArrayList<List<GroupPlanner.Item>>();
            for (int i = 0; i < 64; i++) inputs.add(items(size, i));
            int[] index = {0};
            bench("select_" + size, () -> select(inputs.get(index[0]++ & 63), size));
        }
        for (int depth : new int[]{0, 32, 128, 512}) {
            var endpoints = new ArrayList<RouteProjection.Inputs>();
            for (int i = 0; i < 5; i++) {
                endpoints.add(new RouteProjection.Inputs(new QueueSnapshot(1000, true, ORDER,
                        constraints(64), items(depth, i), null),
                        new WorkSnapshot(1000, List.of(), List.of(), 0)));
            }
            var probes = new ArrayList<RouteProjection.Probe>();
            for (int i = 0; i < 64; i++) probes.add(new RouteProjection.Probe(99999, 0, 1000, 61_000,
                    1024 + i * 32, i * 8, i * 8));
            int[] index = {0};
            bench("fleet5_depth_" + depth, () -> {
                var probe = probes.get(index[0]++ & 63);
                double total = 0;
                for (var endpoint : endpoints) {
                    var candidate = RouteProjection.project(endpoint, probe, model, policy);
                    total += candidate.requiredProjectedTtftMs();
                }
                return total;
            });
        }
    }
}
