import com.sun.management.ThreadMXBean;
import java.lang.management.ManagementFactory;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.locks.ReentrantLock;
import org.flexlb.balance.delivery.*;
import org.flexlb.balance.endpoint.PrefillActiveIndex;
import org.flexlb.balance.prediction.FormulaPredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.scheduler.*;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.*;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.config.ConfigService;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;

/** Same source on both revisions; includes real WorkerBatcher snapshot capture. */
public class SnapshotBench {
    static final ThreadMXBean CPU = (ThreadMXBean) ManagementFactory.getThreadMXBean();
    static volatile long sink;
    static FormulaPredictor model;
    static BatchDeliveryStrategy strategy;
    static RequestRegistry registry;
    static final RouteProjection.Probe PROBE = new RouteProjection.Probe(
            Long.MAX_VALUE, 0, 1, Long.MAX_VALUE, 2048, 0, 0);

    record Endpoint(WorkerBatcher runtime, ReentrantLock lock, ScheduledRequest member) {
        void changeMembership() {
            lock.lock();
            try {
                if (!runtime.ownedState().terminalizeActiveUnderLock(member)
                        || !runtime.ownedState().enqueueActiveUnderLock(member, 0)) {
                    throw new AssertionError("membership update failed");
                }
            } finally {
                lock.unlock();
            }
        }
    }
    record Sample(long ns, long cpu, long bytes, long checksum) {}

    static ScheduledRequest request(FlexlbConfig config, long id, int priority, long tokens) {
        var request = new Request();
        request.setRequestId(id);
        request.setSeqLen(tokens);
        request.setPriority(priority);
        var context = new BalanceContext(config);
        context.setRequest(request);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(priority, Long.MAX_VALUE));
        return new ScheduledRequest(context, new CompletableFuture<>(), null,
                null, null, null, null, null, 1L);
    }

    static FlexlbConfig config() {
        var config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        var decision = SchedulingTestConfig.useFixedWindowDecision(config);
        decision.setMaxRequests(1024);
        decision.setMaxCollectionWaitMs(0L);
        decision.setMaxPredictedExecutionMs(700L);
        return config;
    }

    static Endpoint endpoint(int depth, int seed) throws Exception {
        var config = config();
        // No scheduler thread or RPCs. Mutations still use the real ownership ledger and lock.
        var runtime = new WorkerBatcher("bench-" + seed, null, config, strategy,
                new EndpointEventProjector(registry));
        Field lockField = WorkerBatcher.class.getDeclaredField("queueLock");
        lockField.setAccessible(true);
        var lock = (ReentrantLock) lockField.get(runtime);
        var random = new Random(seed);
        ScheduledRequest last = null;
        lock.lock();
        try {
            for (int i = 0; i < depth; i++) {
                last = request(config, i, 1 + i % 4, 100 + random.nextInt(4096));
                if (!runtime.ownedState().enqueueActiveUnderLock(last, 0)) throw new AssertionError();
            }
        } finally {
            lock.unlock();
        }
        return new Endpoint(runtime, lock, last);
    }

    static Sample read(List<Endpoint> endpoints, boolean project, int repetitions) {
        long id = Thread.currentThread().threadId();
        long bytes = CPU.getThreadAllocatedBytes(id), cpu = CPU.getCurrentThreadCpuTime();
        long started = System.nanoTime(), checksum = 0;
        for (int i = 0; i < repetitions; i++) {
            for (var endpoint : endpoints) {
                var inputs = endpoint.runtime.captureRouteProjectionInputs();
                checksum += inputs.queue().activeItems().size();
                if (project) checksum += RouteProjection.project(inputs, PROBE, model,
                        strategy.projectionPolicy()).requiredProjectedTtftMs();
            }
        }
        return new Sample(System.nanoTime() - started, CPU.getCurrentThreadCpuTime() - cpu,
                CPU.getThreadAllocatedBytes(id) - bytes, checksum);
    }

    static void fleet(int count, int depth, int planners, String mode, boolean project) throws Exception {
        var endpoints = new ArrayList<Endpoint>();
        for (int i = 0; i < count; i++) endpoints.add(endpoint(depth, i));
        int repetitions = project ? 2 : 8;
        long expected = read(endpoints, project, repetitions).checksum;
        String name = "fleet" + count + "_depth" + depth + "_p" + planners + "_" + mode
                + (project ? "_project" : "_capture");
        System.out.println("VERIFY " + name + " checksum=" + expected);
        try (var pool = Executors.newFixedThreadPool(planners)) {
            for (int round = 0; round < 5; round++) {
                long allocated = CPU.getThreadAllocatedBytes(Thread.currentThread().threadId());
                long mainCpu = CPU.getCurrentThreadCpuTime();
                long started = System.nanoTime();
                long cpu = 0, bytes = 0, checksum = 0, operations = 0;
                do {
                    for (var endpoint : endpoints) {
                        if (mode.equals("status")) endpoint.runtime.signalSchedulingInputsChanged();
                        if (mode.equals("membership")) endpoint.changeMembership();
                    }
                    var gate = new CountDownLatch(1);
                    var tasks = new ArrayList<Future<Sample>>();
                    for (int i = 0; i < planners; i++) {
                        tasks.add(pool.submit(() -> {
                            gate.await();
                            return read(endpoints, project, repetitions);
                        }));
                    }
                    gate.countDown();
                    for (var task : tasks) {
                        Sample sample = task.get();
                        if (sample.checksum != expected) throw new AssertionError(name);
                        cpu += sample.cpu;
                        bytes += sample.bytes;
                        checksum += sample.checksum;
                    }
                    operations += (long) planners * repetitions;
                } while (System.nanoTime() - started < 200_000_000L);
                long elapsed = System.nanoTime() - started;
                cpu += CPU.getCurrentThreadCpuTime() - mainCpu;
                bytes += CPU.getThreadAllocatedBytes(Thread.currentThread().threadId()) - allocated;
                sink = checksum;
                if (round >= 2) System.out.printf(Locale.ROOT,
                        "%s round=%d ns/op=%.2f cpu_ns/op=%.2f bytes/op=%.2f%n",
                        name, round, (double) elapsed / operations, (double) cpu / operations,
                        (double) bytes / operations);
            }
        }
    }

    static void writes(int depth) {
        var config = config();
        var index = PrefillActiveIndex.ordered(depth,
                Comparator.comparingInt(ScheduledRequest::priority).reversed()
                        .thenComparingLong(ScheduledRequest::enqueueSeq));
        var members = new ArrayList<ScheduledRequest>();
        for (int i = 0; i < depth; i++) {
            var item = request(config, i, i % 4, 100);
            members.add(item);
            index.add(item);
        }
        for (int round = 0; round < 5; round++) {
            long bytes = CPU.getThreadAllocatedBytes(Thread.currentThread().threadId());
            long cpu = CPU.getCurrentThreadCpuTime(), started = System.nanoTime();
            int operations = 100_000;
            for (int i = 0; i < operations; i++) {
                var member = members.get(i % depth);
                if (!index.remove(member) || !index.add(member)) throw new AssertionError();
            }
            long elapsed = System.nanoTime() - started;
            cpu = CPU.getCurrentThreadCpuTime() - cpu;
            bytes = CPU.getThreadAllocatedBytes(Thread.currentThread().threadId()) - bytes;
            if (round >= 2) System.out.printf(Locale.ROOT,
                    "index_remove_add_depth%d round=%d ns/op=%.2f cpu_ns/op=%.2f bytes/op=%.2f%n",
                    depth, round, (double) elapsed / operations, (double) cpu / operations,
                    (double) bytes / operations);
        }
    }

    static void initialize(Path formula) throws Exception {
        model = new FormulaPredictor(Files.readString(formula));
        var constructor = ConfigService.class.getDeclaredConstructor(String.class);
        constructor.setAccessible(true);
        var service = constructor.newInstance(
                "{\"requestLifecycle\":{\"request\":{\"timeoutMs\":60000},\"decision\":{\"lifetime\":2.0}}}");
        var reporter = new BatchSchedulerReporter(null);
        registry = new RequestRegistry(service, reporter, new RequestSchedulerReporter(null));
        strategy = new BatchDeliveryStrategy(
                () -> CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST),
                () -> 1L, registry, new DeliveryMetrics(reporter));
    }

    public static void main(String[] args) throws Exception {
        initialize(Path.of(args[0]));
        for (int depth : new int[]{32, 1024}) {
            writes(depth);
            for (int count : new int[]{10, 100}) {
                for (int planners : new int[]{1, 64}) {
                    fleet(count, depth, planners, "status", false);
                    fleet(count, depth, planners, "membership", false);
                    fleet(count, depth, planners, "status", true);
                }
            }
        }
    }
}
