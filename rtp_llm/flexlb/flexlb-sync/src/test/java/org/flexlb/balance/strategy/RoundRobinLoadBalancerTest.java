package org.flexlb.balance.strategy;

import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.EngineType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

class RoundRobinLoadBalancerTest {

    private EngineWorkerStatus engineWorkerStatus;
    private ConfigService configService;
    private RoundRobinLoadBalancer rr;

    @BeforeEach
    void setUp() {
        clearWorkerMaps();
        engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
        configService = new ConfigService();
        rr = new RoundRobinLoadBalancer(engineWorkerStatus, configService);
        populatePdFusion(4);
    }

    @AfterEach
    void tearDown() {
        clearWorkerMaps();
    }

    @Test
    void registers_under_round_robin_strategy() {
        LoadBalancer registered = LoadBalanceStrategyFactory.getLoadBalancer(LoadBalanceStrategyEnum.ROUND_ROBIN);
        Assertions.assertSame(rr, registered);
    }

    @Test
    void select_cycles_through_alive_workers() {
        Set<String> seen = new HashSet<>();
        for (int i = 0; i < 4; i++) {
            BalanceContext ctx = newSingleContext(1000L + i);
            ServerStatus s = rr.select(ctx, RoleType.PDFUSION, null);
            Assertions.assertTrue(s.isSuccess(), "select should succeed");
            seen.add(s.getServerIp() + ":" + s.getHttpPort());
        }
        Assertions.assertEquals(4, seen.size(), "4 sequential selects should hit all 4 workers");
    }

    @Test
    void selectBatch_assigns_one_per_worker_when_count_equals_pool_size() {
        List<BatchScheduleTarget> targets = rr.selectBatch(4, RoleType.PDFUSION, null);

        Assertions.assertEquals(4, targets.size());
        Set<String> workers = new HashSet<>();
        for (BatchScheduleTarget t : targets) {
            workers.add(t.getServerIp() + ":" + t.getHttpPort());
            Assertions.assertEquals(t.getHttpPort() + 1, t.getGrpcPort().intValue(),
                    "grpc_port should be http_port + 1");
            Assertions.assertNull(t.getArpcPort(), "LLM targets must not carry an arpc slot");
        }
        Assertions.assertEquals(4, workers.size(), "4 slots and 4 workers must hit all 4");
    }

    @Test
    void selectBatch_embedding_engine_fills_arpc_slot_only() {
        configService.loadBalanceConfig().setEngineType(EngineType.EMBEDDING);

        List<BatchScheduleTarget> targets = rr.selectBatch(4, RoleType.PDFUSION, null);

        Assertions.assertEquals(4, targets.size());
        for (BatchScheduleTarget t : targets) {
            Assertions.assertEquals(t.getHttpPort() + 1, t.getArpcPort().intValue(),
                    "arpc_port should be http_port + 1");
            Assertions.assertNull(t.getGrpcPort(), "embedding targets must not carry a grpc slot");
        }
    }

    @Test
    void selectBatch_wraps_cursor_when_count_exceeds_pool_size() {
        List<BatchScheduleTarget> targets = rr.selectBatch(8, RoleType.PDFUSION, null);

        Assertions.assertEquals(8, targets.size());
        Set<String> workers = new HashSet<>();
        for (BatchScheduleTarget t : targets) {
            workers.add(t.getServerIp() + ":" + t.getHttpPort());
        }
        Assertions.assertEquals(4, workers.size(),
                "8 slots over 4 workers should hit all 4 (cursor wraps)");
    }

    @Test
    void selectBatch_returns_empty_list_when_no_alive_workers() {
        clearWorkerMaps();

        List<BatchScheduleTarget> targets = rr.selectBatch(3, RoleType.PDFUSION, null);

        Assertions.assertTrue(targets.isEmpty(),
                "no alive workers should yield an empty target list");
    }

    @Test
    void selectBatch_order_matches_cursor_progression() {
        List<BatchScheduleTarget> first = rr.selectBatch(2, RoleType.PDFUSION, null);
        List<BatchScheduleTarget> second = rr.selectBatch(2, RoleType.PDFUSION, null);

        Assertions.assertEquals(2, first.size());
        Assertions.assertEquals(2, second.size());

        Set<String> firstSet = new HashSet<>();
        Set<String> secondSet = new HashSet<>();
        for (BatchScheduleTarget t : first) {
            firstSet.add(t.getServerIp() + ":" + t.getHttpPort());
        }
        for (BatchScheduleTarget t : second) {
            secondSet.add(t.getServerIp() + ":" + t.getHttpPort());
        }
        Set<String> overlap = new HashSet<>(firstSet);
        overlap.retainAll(secondSet);
        Assertions.assertTrue(overlap.isEmpty(),
                "consecutive batch_size=2 calls on a 4-worker pool should not overlap");
    }

    @Test
    void selectBatch_atomic_range_does_not_overlap_with_concurrent_selects() throws Exception {
        // Verifies the getAndAdd(count) optimization: a batch call must occupy a
        // contiguous cursor range so that concurrent select() calls do not land
        // on indices already claimed by the batch (and vice versa).
        // 4 workers, batch=4 -> batch claims one full cycle; concurrent singles
        // proceed on the next cycle. Across many trials, the union of slots
        // assigned in one (batch, single) pair should always cover 4 distinct
        // index positions modulo 4 (i.e. the batch picks 4 distinct workers and
        // each single pick is one of those 4 -- which is exactly the RR contract
        // when cursor advances atomically).

        int trials = 200;
        int concurrentSingles = 16;
        ExecutorService exec = Executors.newFixedThreadPool(concurrentSingles + 1);
        AtomicInteger violations = new AtomicInteger(0);

        try {
            for (int t = 0; t < trials; t++) {
                CountDownLatch ready = new CountDownLatch(concurrentSingles + 1);
                CountDownLatch start = new CountDownLatch(1);
                CountDownLatch done = new CountDownLatch(concurrentSingles + 1);

                List<List<BatchScheduleTarget>> batchOut = new ArrayList<>();
                batchOut.add(null);

                exec.submit(() -> {
                    try {
                        ready.countDown();
                        start.await();
                        batchOut.set(0, rr.selectBatch(4, RoleType.PDFUSION, null));
                    } catch (InterruptedException ignored) {
                    } finally {
                        done.countDown();
                    }
                });
                List<ServerStatus> singleOut = new ArrayList<>();
                for (int i = 0; i < concurrentSingles; i++) {
                    final long rid = (long) t * 1000 + i;
                    exec.submit(() -> {
                        try {
                            ready.countDown();
                            start.await();
                            BalanceContext ctx = newSingleContext(rid);
                            ServerStatus s = rr.select(ctx, RoleType.PDFUSION, null);
                            synchronized (singleOut) {
                                singleOut.add(s);
                            }
                        } catch (InterruptedException ignored) {
                        } finally {
                            done.countDown();
                        }
                    });
                }
                ready.await();
                start.countDown();
                Assertions.assertTrue(done.await(5, TimeUnit.SECONDS),
                        "trial " + t + " did not finish in time");

                List<BatchScheduleTarget> b = batchOut.get(0);
                Set<String> batchWorkers = new HashSet<>();
                for (BatchScheduleTarget bt : b) {
                    batchWorkers.add(bt.getServerIp() + ":" + bt.getHttpPort());
                }
                if (batchWorkers.size() != 4) {
                    violations.incrementAndGet();
                }
                Set<String> poolWorkers =
                        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().keySet();
                synchronized (singleOut) {
                    Assertions.assertEquals(concurrentSingles, singleOut.size(),
                            "every concurrent select() must produce a result");
                    for (ServerStatus s : singleOut) {
                        Assertions.assertTrue(s.isSuccess(),
                                "concurrent select() must succeed under batch pressure: " + s.getMessage());
                        Assertions.assertTrue(poolWorkers.contains(s.getServerIp() + ":" + s.getHttpPort()),
                                "single pick must be a pool worker: " + s.getServerIp() + ":" + s.getHttpPort());
                    }
                }
            }
        } finally {
            exec.shutdown();
            exec.awaitTermination(5, TimeUnit.SECONDS);
        }

        Assertions.assertEquals(0, violations.get(),
                "batch=4 must always pick all 4 workers regardless of concurrent select() pressure "
                        + "(verifies getAndAdd is atomic over the whole range)");
    }

    @Test
    void selectBatch_does_not_record_local_task() {
        rr.selectBatch(4, RoleType.PDFUSION, null);

        int totalLocalTasks = 0;
        for (WorkerStatus w : EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().values()) {
            totalLocalTasks += w.getLocalTaskMap().size();
        }
        Assertions.assertEquals(0, totalLocalTasks,
                "batch path must not write to localTaskMap (no bookkeeping)");
    }

    @Test
    void select_still_records_local_task() {
        BalanceContext ctx = newSingleContext(5000L);
        ServerStatus assigned = rr.select(ctx, RoleType.PDFUSION, null);
        Assertions.assertTrue(assigned.isSuccess());

        String ipPort = assigned.getServerIp() + ":" + assigned.getHttpPort();
        WorkerStatus worker = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().get(ipPort);
        Assertions.assertNotNull(worker.getLocalTaskMap().get(5000L),
                "select() path still records to localTaskMap for /schedule use");
    }

    @Test
    void rollBack_removes_local_task() {
        BalanceContext ctx = newSingleContext(5000L);
        ServerStatus assigned = rr.select(ctx, RoleType.PDFUSION, null);
        Assertions.assertTrue(assigned.isSuccess());

        String ipPort = assigned.getServerIp() + ":" + assigned.getHttpPort();
        WorkerStatus worker = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().get(ipPort);
        Assertions.assertNotNull(worker.getLocalTaskMap().get(5000L), "localTaskMap must contain the assignment");

        rr.rollBack(ipPort, 5000L);

        Assertions.assertNull(worker.getLocalTaskMap().get(5000L), "rollBack must remove the local task");
    }

    private BalanceContext newSingleContext(long requestId) {
        BalanceContext ctx = new BalanceContext();
        Request req = new Request();
        req.setRequestId(requestId);
        req.setSeqLen(128);
        req.setBlockCacheKeys(new ArrayList<>());
        ctx.setRequest(req);
        return ctx;
    }

    private void populatePdFusion(int count) {
        for (int i = 0; i < count; i++) {
            WorkerStatus w = new WorkerStatus();
            w.setIp("10.0.0." + i);
            w.setPort(8080);
            w.setRole(RoleType.PDFUSION.getCode());
            w.setAlive(true);
            w.setAvailableConcurrency(1L);
            w.setWaitingTaskList(new HashMap<>());
            w.setRunningTaskList(new HashMap<>());
            w.updateTaskStates(new HashMap<>(), new HashMap<>(), new HashMap<>());
            w.setLocalTaskMap(new ConcurrentHashMap<>());
            CacheStatus cs = new CacheStatus();
            cs.setAvailableKvCache(100_000);
            cs.setBlockSize(16);
            w.setCacheStatus(cs);
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().put(w.getIpPort(), w);
        }
    }

    private void clearWorkerMaps() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().clear();
    }
}
