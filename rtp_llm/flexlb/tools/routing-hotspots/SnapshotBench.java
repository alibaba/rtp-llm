import org.flexlb.balance.scheduler.RouteProjectionTestSupport;
import static org.mockito.Mockito.mock;
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
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.balance.prediction.FormulaPredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.scheduler.*;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.*;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.service.monitor.DeliveryMetricsReporter;

/** Same source on both revisions; includes real WorkerBatcher snapshot capture. */
public class SnapshotBench {
    static final ThreadMXBean CPU = (ThreadMXBean) ManagementFactory.getThreadMXBean();
    static volatile long sink;
    static FormulaPredictor model;
    static BatchDeliveryStrategy strategy;
    static final RouteProjectionTestSupport.Probe PROBE = new RouteProjectionTestSupport.Probe(
            Long.MAX_VALUE, 0, 1, Long.MAX_VALUE, 2048, 0, 0);

    record Endpoint(PrefillEndpoint endpoint, PrefillState state, ReentrantLock lock, RequestRoute member) {
        void changeMembership() {
            lock.lock();
            try {
                if (!state.removeQueuedLocked(member)
                        || !state.enqueueActiveLocked(member, 0)) {
                    throw new AssertionError("membership update failed");
                }
            } finally {
                lock.unlock();
            }
        }
    }
    record Sample(long ns, long cpu, long bytes, long checksum) {}

    static RequestRoute request(FlexlbConfig config, long id, int priority, long tokens) {
        var request = new Request();
        request.setRequestId(id);
        request.setSeqLen(tokens);
        request.setPriority(priority);
        var context = new RequestContext(config);
        context.setRequest(request);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(priority, Long.MAX_VALUE));
        context.setFuture(new CompletableFuture<>());
        return org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(SchedulingTestConfig.freezeInputs(context), null,
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
        var constructor = PrefillEndpoint.class.getDeclaredConstructor(WorkerStatus.class,
                FlexlbConfig.class, DeliveryStrategy.class, DeliveryMetricsReporter.class, PlacementAvailability.class);
        constructor.setAccessible(true);
        var endpoint = constructor.newInstance(
                WorkerStatus.createDiscovered(RoleType.PREFILL, null, "127.0.0.1", 8080, 9090, null),
                config, strategy, mock(DeliveryMetricsReporter.class), new PlacementAvailability());
        Field stateField = PrefillEndpoint.class.getDeclaredField("prefillState");
        stateField.setAccessible(true);
        var state = (PrefillState) stateField.get(endpoint);
        Field lockField = PrefillState.class.getDeclaredField("lock");
        lockField.setAccessible(true);
        var lock = (ReentrantLock) lockField.get(state);
        var enableQueue = PrefillState.class.getDeclaredMethod("enableQueueLocked", Comparator.class);
        enableQueue.setAccessible(true);
        lock.lock();
        try {
            enableQueue.invoke(state, WorkerBatcher.PRIORITY_QUEUE_ORDER);
        } finally {
            lock.unlock();
        }
        var worker = new WorkerBatcher(endpoint.ipPort(), endpoint,
                QueueExecutionSettings.capture(config), strategy, state);
        Field batcherField = PrefillEndpoint.class.getDeclaredField("batcher");
        batcherField.setAccessible(true);
        batcherField.set(endpoint, worker);
        var random = new Random(seed);
        RequestRoute last = null;
        lock.lock();
        try {
            for (int i = 0; i < depth; i++) {
                last = request(config, i, 1 + i % 4, 100 + random.nextInt(4096));
                if (!state.enqueueActiveLocked(last, 0)) throw new AssertionError();
            }
        } finally {
            lock.unlock();
        }
        return new Endpoint(endpoint, state, lock, last);
    }

    static Sample read(List<Endpoint> endpoints, boolean project, int repetitions) {
        long id = Thread.currentThread().threadId();
        long bytes = CPU.getThreadAllocatedBytes(id), cpu = CPU.getCurrentThreadCpuTime();
        long started = System.nanoTime(), checksum = 0;
        for (int i = 0; i < repetitions; i++) {
            for (var endpoint : endpoints) {
                var inputs = endpoint.endpoint.captureRouteProjectionInputs();
                checksum += inputs.queue().activeItems().size();
                if (project) checksum += RouteProjectionTestSupport.project(inputs, PROBE, model,
                        strategy.projectionPolicy()).projectedTtftMsValue();
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
                        if (mode.equals("status")) endpoint.endpoint.signalRouteReady();
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
                Comparator.comparingInt(RequestRoute::priority).reversed()
                        .thenComparingLong(RequestRoute::enqueueSeq));
        var members = new ArrayList<RequestRoute>();
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
        var reporter = new DeliveryMetricsReporter(null);
        strategy = new BatchDeliveryStrategy(
                () -> CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST),
                () -> 1L, reporter);
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
