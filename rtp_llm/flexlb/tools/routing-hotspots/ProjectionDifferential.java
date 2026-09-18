import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.*;
import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.projection.*;
import org.flexlb.balance.scheduler.RouteDeliveryStrategy;
import org.flexlb.service.monitor.BatchSchedulerReporter;

/** Run identically against both revisions; compare complete selections and candidates. */
public class ProjectionDifferential {
    static void add(MessageDigest digest, Object value) {
        digest.update(value.toString().getBytes(StandardCharsets.UTF_8));
        digest.update((byte) '\n');
    }

    public static void main(String[] args) throws Exception {
        SnapshotBench.initialize(Path.of(args[0]));
        var model = SnapshotBench.model;
        var batchPolicy = SnapshotBench.strategy.projectionPolicy();
        var routePolicy = new RouteDeliveryStrategy(SnapshotBench.registry,
                new DeliveryMetrics(new BatchSchedulerReporter(null))).projectionPolicy();
        var random = new Random(1729);
        var digest = MessageDigest.getInstance("SHA-256");
        var states = new TreeMap<String, Integer>();
        for (int trial = 0; trial < 2000; trial++) {
            var items = new ArrayList<GroupPlanner.Item>();
            int count = trial % 11 == 0 ? 1024 : random.nextInt(100);
            for (int i = 0; i < count; i++) {
                long tokens = trial % 17 == 0 ? Long.MAX_VALUE / 2 + 1 : random.nextInt(32768);
                long hit = tokens > 32768 ? 0 : random.nextInt((int) tokens + 1);
                long expiry = switch (i % 7) {
                    case 0 -> 999;
                    case 1 -> 1010;
                    case 2 -> 1020;
                    default -> Long.MAX_VALUE;
                };
                items.add(new GroupPlanner.Item(i, random.nextInt(8), i,
                        980 + i % 15, expiry, tokens, hit));
            }
            Comparator<GroupPlanner.Item> order = Comparator.comparingLong(GroupPlanner.Item::enqueueSeq);
            if (trial % 2 == 0) order = Comparator.comparingInt(GroupPlanner.Item::priority)
                    .reversed().thenComparing(order);
            items.sort(order);
            var limits = new GroupPlanner.Constraints(
                    new int[]{1, 4, 64, 1024}[trial % 4],
                    trial % 3 == 0 ? 200_000 : Long.MAX_VALUE,
                    trial % 5 == 0 ? 300_000 : Long.MAX_VALUE,
                    new long[]{0, 700, 1_000_000_000}[trial % 3],
                    new long[]{0, 30, 700}[trial % 3]);
            var predictor = model.newBatchPrediction();
            var selection = GroupPlanner.selectWithPrediction(items, GroupPlanner.itemAccess(), limits,
                    (added, prefix) -> predictor.append(added.seqLen(), added.hitCache()));
            add(digest, selection);
            add(digest, GroupPlanner.evaluateReadiness(selection, limits, 1000));
            var committed = new WorkSnapshot(1000, trial % 4 == 0
                    ? List.of(new WorkSnapshot.RequestWork(9999, WorkSnapshot.Phase.ENGINE_RUNNING, 500))
                    : List.of(), List.of(), 0);
            var inputs = new RouteProjection.Inputs(new QueueSnapshot(1000, true, order, limits, items, null), committed);
            var probe = new RouteProjection.Probe(trial % 50 == 0 ? 1 : 999999,
                    random.nextInt(10), 1000, trial % 7 == 0 ? 1015 : Long.MAX_VALUE,
                    2048, 256, 256);
            for (var policy : List.of(batchPolicy, routePolicy)) {
                var candidate = RouteProjection.project(inputs, probe, model, policy);
                add(digest, candidate);
                states.merge(candidate.state().name(), 1, Integer::sum);
            }
        }
        System.out.println("VERIFY selections=2000 plans=2000 candidates=4000 states=" + states
                + " sha256=" + HexFormat.of().formatHex(digest.digest()));
    }
}
