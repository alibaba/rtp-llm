package org.flexlb.balance.strategy;

import org.flexlb.balance.resource.DecodeResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskStateEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Matched synthetic lifecycle simulation; this measures placement, not GPU throughput.
 * Workload and virtual time are fixed. WeightedCacheLoadBalancer retains its native
 * ThreadLocalRandom sampling, so its metrics are observations rather than seeded results.
 */
class LeastLoadDecodeSimulationTest {
    private static final int SNAPSHOT_INTERVAL_MS = 20;
    private static final int COMPLETION_REPORT_LAG_MS = 40;
    private static final int TRANSIT_MS = 10;
    private static final long KV_CAPACITY = 1_000_000;
    private static final String FIRST_IP = "127.0.0.1";
    private static final String SECOND_IP = "127.0.0.2";

    @AfterEach
    void tearDown() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
    }

    @Test
    void mixed_length_lifecycles_should_recover_initial_skew_and_drain_with_delayed_reports() throws Exception {
        List<Arrival> workload = workload();
        List<String> trace = new ArrayList<>();
        trace.add("strategy,virtual_ms,actual_dp0,actual_dp1,estimated_dp0,estimated_dp1,local_dp0,local_dp1");
        Result leastLoad = simulate("least-load", workload, trace);
        Result weighted = simulate("weighted-cache", workload, trace);
        Result roundRobin = simulate("round-robin", workload, trace);

        // Initially dp0/dp1 contain 12/4 active requests. No job completes during
        // these eight arrivals; reservations must immediately repair that skew.
        assertEquals(0, leastLoad.firstEightDp0());
        assertEquals(8, leastLoad.firstEightDp1());
        assertEquals(0, leastLoad.gapAfterFirstEight());
        assertEquals(4, roundRobin.firstEightDp0());
        assertEquals(4, roundRobin.firstEightDp1());
        assertEquals(8, roundRobin.gapAfterFirstEight());

        for (Result result : List.of(leastLoad, weighted, roundRobin)) {
            assertEquals(workload.size(), result.assignedDp0() + result.assignedDp1());
            assertTrue(result.sawInTransit(), "A local reservation must precede the next engine report");
            assertTrue(result.sawCompletionLag(), "Completed requests must remain estimated until reported");
            assertEquals(0, result.finalActual());
            assertEquals(0, result.finalEstimated());
            assertEquals(0, result.finalLocal());
            System.out.printf(java.util.Locale.ROOT,
                    "SYNTHETIC_NON_GPU strategy=%s assignments=[%d,%d] first8=[%d,%d] "
                            + "gap_after_first8=%d mean_actual_inventory_gap=%.3f max_actual_inventory_gap=%d "
                            + "drain_actual=%d drain_estimated=%d drain_local=%d%n",
                    result.name(), result.assignedDp0(), result.assignedDp1(),
                    result.firstEightDp0(), result.firstEightDp1(), result.gapAfterFirstEight(),
                    result.meanActualGap(), result.maximumActualGap(), result.finalActual(),
                    result.finalEstimated(), result.finalLocal());
        }
        Path output = Path.of("target", "least-load-decode-simulation.csv");
        Files.createDirectories(output.getParent());
        Files.write(output, trace);
        System.out.println("SYNTHETIC_NON_GPU lifecycle trace: " + output.toAbsolutePath()
                + "; fixed virtual timeline; weighted-cache uses native random sampling");
    }

    private Result simulate(String name, List<Arrival> workload, List<String> trace) {
        Map<String, WorkerStatus> workerMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap();
        workerMap.clear();
        List<WorkerStatus> workers = List.of(worker(FIRST_IP), worker(SECOND_IP));
        workers.forEach(worker -> workerMap.put(worker.getIpPort(), worker));
        FlexlbConfig config = new FlexlbConfig();
        config.setDecodeConcurrencyLimit(1000);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService);
        ResourceMeasureFactory factory = mock(ResourceMeasureFactory.class);
        when(factory.getMeasure(any())).thenReturn(measure);
        EngineWorkerStatus status = new EngineWorkerStatus(new ModelMetaConfig());
        Router router;
        if (name.equals("least-load")) {
            LeastLoadDecodeStrategy strategy = new LeastLoadDecodeStrategy(configService, status, factory);
            router = context -> strategy.select(context, RoleType.DECODE, null);
        } else if (name.equals("weighted-cache")) {
            WeightedCacheLoadBalancer strategy = new WeightedCacheLoadBalancer(configService, status, factory);
            router = context -> strategy.select(context, RoleType.DECODE, null);
        } else {
            router = new RoundRobinRouter(workers, measure);
        }

        List<Job> jobs = new ArrayList<>();
        // Unequal initial request counts conflict with unequal KV use: dp0 has
        // twelve 64-token jobs, dp1 has four 1024-token jobs. All end at t=1000.
        for (int index = 0; index < 16; index++) {
            int dp = index < 12 ? 0 : 1;
            jobs.add(new Job(index + 1L, dp, -TRANSIT_MS, 1000, dp == 0 ? 64 : 1024));
        }
        int[] assigned = new int[2];
        int[] firstEight = new int[2];
        int afterEightGap = -1;
        long gapSum = 0;
        int maxGap = 0;
        boolean sawInTransit = false;
        boolean sawLag = false;
        int nextArrival = 0;
        int finalTime = 1400;
        for (int now = 0; now <= finalTime; now++) {
            if (now % SNAPSHOT_INTERVAL_MS == 0) {
                report(workers, jobs, now);
            }
            while (nextArrival < workload.size() && workload.get(nextArrival).atMs() == now) {
                Arrival arrival = workload.get(nextArrival);
                BalanceContext context = context(arrival, config);
                ServerStatus selected = router.select(context);
                assertTrue(selected.isSuccess(), name + " unexpectedly rejected request " + arrival.id());
                int dp = selected.getServerIp().equals(FIRST_IP) ? 0 : 1;
                WorkerStatus worker = workers.get(dp);
                TaskInfo reservation = worker.getLocalTaskMap().get(arrival.id());
                assertEquals(TaskStateEnum.IN_TRANSIT, reservation.getTaskState());
                sawInTransit = true;
                assigned[dp]++;
                jobs.add(new Job(arrival.id(), dp, now, now + TRANSIT_MS + arrival.durationMs(), arrival.tokens()));
                if (nextArrival < 8) {
                    firstEight[dp]++;
                    if (nextArrival == 7) {
                        afterEightGap = Math.abs(actual(jobs, 0, now) - actual(jobs, 1, now));
                    }
                }
                nextArrival++;
            }
            int actual0 = actual(jobs, 0, now);
            int actual1 = actual(jobs, 1, now);
            long estimated0 = workers.get(0).getDecodeConcurrency();
            long estimated1 = workers.get(1).getDecodeConcurrency();
            if (estimated0 + estimated1 > actual0 + actual1) {
                sawLag = true;
            }
            int gap = Math.abs(actual0 - actual1);
            gapSum += gap;
            maxGap = Math.max(maxGap, gap);
            if (now % SNAPSHOT_INTERVAL_MS == 0) {
                trace.add(name + "," + now + "," + actual0 + "," + actual1 + ","
                        + estimated0 + "," + estimated1 + "," + workers.get(0).getLocalTaskMap().size()
                        + "," + workers.get(1).getLocalTaskMap().size());
            }
        }
        assertEquals(workload.size(), nextArrival);
        return new Result(name, assigned[0], assigned[1], firstEight[0], firstEight[1], afterEightGap,
                gapSum / (double) (finalTime + 1), maxGap, sawInTransit, sawLag,
                actual(jobs, 0, finalTime) + actual(jobs, 1, finalTime),
                workers.stream().mapToLong(WorkerStatus::getDecodeConcurrency).sum(),
                workers.stream().mapToInt(worker -> worker.getLocalTaskMap().size()).sum());
    }

    private void report(List<WorkerStatus> workers, List<Job> jobs, int now) {
        for (int dp = 0; dp < workers.size(); dp++) {
            Map<String, TaskInfo> waiting = new HashMap<>();
            Map<String, TaskInfo> running = new HashMap<>();
            Map<String, TaskInfo> finished = new HashMap<>();
            long used = 0;
            for (Job job : jobs) {
                if (job.dp() != dp || now < job.arrivedMs() + TRANSIT_MS) {
                    continue;
                }
                TaskInfo task = task(job.id(), job.tokens());
                if (now >= job.endsMs() + COMPLETION_REPORT_LAG_MS) {
                    finished.put(String.valueOf(job.id()), task);
                } else if (now < job.arrivedMs() + TRANSIT_MS + SNAPSHOT_INTERVAL_MS) {
                    waiting.put(String.valueOf(job.id()), task);
                } else {
                    // Keep the last running state until delayed completion arrives.
                    running.put(String.valueOf(job.id()), task);
                }
                if (now < job.endsMs()) {
                    used += job.tokens();
                }
            }
            WorkerStatus worker = workers.get(dp);
            worker.setWaitingTaskList(waiting);
            worker.setRunningTaskList(running);
            worker.updateTaskStates(waiting, running, finished);
            worker.updateKvCacheTokens(used, KV_CAPACITY - used);
        }
    }

    private int actual(List<Job> jobs, int dp, int now) {
        return (int) jobs.stream().filter(job -> job.dp() == dp && now < job.endsMs()).count();
    }

    private List<Arrival> workload() {
        List<Arrival> arrivals = new ArrayList<>();
        for (int index = 0; index < 8; index++) {
            arrivals.add(new Arrival(1000L + index, index, index % 2 == 0 ? 64 : 1024,
                    index % 2 == 0 ? 60 : 320));
        }
        for (int index = 0; index < 160; index++) {
            boolean longRequest = index % 4 == 0;
            arrivals.add(new Arrival(2000L + index, 40 + index * 5, longRequest ? 1024 : 64,
                    longRequest ? 320 : 60));
        }
        return arrivals;
    }

    private WorkerStatus worker(String ip) {
        WorkerStatus worker = new WorkerStatus();
        worker.setIp(ip);
        worker.setPort(8080);
        worker.setRole(RoleType.DECODE.getCode());
        worker.setAlive(true);
        worker.getAvailableKvCacheTokens().set(KV_CAPACITY);
        return worker;
    }

    private BalanceContext context(Arrival arrival, FlexlbConfig config) {
        Request request = new Request();
        request.setRequestId(arrival.id());
        request.setSeqLen(arrival.tokens());
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(config);
        return context;
    }

    private static TaskInfo task(long id, long tokens) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(id);
        task.setInputLength(tokens);
        return task;
    }

    private interface Router {
        ServerStatus select(BalanceContext context);
    }

    /** Pure RR uses the same availability gate and local reservation as the real strategies. */
    private static class RoundRobinRouter implements Router {
        private final List<WorkerStatus> workers;
        private final DecodeResourceMeasure measure;
        private int cursor;

        RoundRobinRouter(List<WorkerStatus> workers, DecodeResourceMeasure measure) {
            this.workers = workers;
            this.measure = measure;
        }

        @Override
        public ServerStatus select(BalanceContext context) {
            WorkerStatus selected = workers.get(cursor++ % workers.size());
            assertTrue(measure.isResourceAvailable(selected));
            selected.putLocalTask(context.getRequestId(), task(context.getRequestId(), context.getRequest().getSeqLen()));
            ServerStatus status = new ServerStatus();
            status.setSuccess(true);
            status.setServerIp(selected.getIp());
            return status;
        }
    }

    private record Arrival(long id, int atMs, long tokens, int durationMs) { }
    private record Job(long id, int dp, int arrivedMs, int endsMs, long tokens) { }
    private record Result(String name, int assignedDp0, int assignedDp1,
                          int firstEightDp0, int firstEightDp1, int gapAfterFirstEight,
                          double meanActualGap, int maximumActualGap, boolean sawInTransit,
                          boolean sawCompletionLag, int finalActual, long finalEstimated, int finalLocal) { }
}
