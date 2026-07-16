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

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
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
        LoadBalanceStrategyFactory.resetForTesting();
        clearWorkerMaps();
        engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
        configService = new ConfigService();
        rr = new RoundRobinLoadBalancer(engineWorkerStatus, configService);
        populatePdFusion(4);
    }

    @AfterEach
    void tearDown() {
        clearWorkerMaps();
        LoadBalanceStrategyFactory.resetForTesting();
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
        // Engine type is fixed at boot, so the balancer resolves it at construction — build a
        // fresh instance against the EMBEDDING config rather than flipping it mid-flight.
        configService.loadBalanceConfig().setEngineType(EngineType.EMBEDDING);
        RoundRobinLoadBalancer embeddingRr = new RoundRobinLoadBalancer(engineWorkerStatus, configService);

        List<BatchScheduleTarget> targets = embeddingRr.selectBatch(4, RoleType.PDFUSION, null);

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
    void selectBatch_stays_in_range_after_cursor_overflows_past_max_value() throws Exception {
        // The cursor is an AtomicInteger that getAndAdd()s without bound; once it crosses
        // Integer.MAX_VALUE it goes negative. Math.floorMod (not %) is what keeps the index
        // non-negative through that wrap — seed the cursor at the boundary to exercise it.
        seedCursor(RoleType.PDFUSION, Integer.MAX_VALUE - 1);

        List<BatchScheduleTarget> targets = rr.selectBatch(8, RoleType.PDFUSION, null);

        Assertions.assertEquals(8, targets.size());
        Set<String> poolWorkers =
                EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().keySet();
        Set<String> picked = new HashSet<>();
        for (BatchScheduleTarget t : targets) {
            String ipPort = t.getServerIp() + ":" + t.getHttpPort();
            Assertions.assertTrue(poolWorkers.contains(ipPort),
                    "index must stay valid after the cursor overflows to negative: " + ipPort);
            picked.add(ipPort);
        }
        Assertions.assertEquals(4, picked.size(),
                "8 slots over 4 workers must still hit all 4 across the overflow boundary");
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

    @SuppressWarnings("unchecked")
    private void seedCursor(RoleType role, int value) throws Exception {
        Field f = RoundRobinLoadBalancer.class.getDeclaredField("cursors");
        f.setAccessible(true);
        // Cursors are keyed "<role>" for group=null and "<role>|<group>" otherwise.
        Map<String, AtomicInteger> cursors = (Map<String, AtomicInteger>) f.get(rr);
        cursors.computeIfAbsent(role.name(), k -> new AtomicInteger(0)).set(value);
    }

    @SuppressWarnings("unchecked")
    private Map<String, AtomicInteger> cursorMap() throws Exception {
        Field f = RoundRobinLoadBalancer.class.getDeclaredField("cursors");
        f.setAccessible(true);
        return (Map<String, AtomicInteger>) f.get(rr);
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

    @Test
    void select_and_selectBatch_skip_dead_workers() {
        // The isAlive filter is the only gate keeping traffic off dead workers — and the
        // consumer of EMBEDDING's markDeadFromDiscovery. Kill 2 of 4 and require that neither
        // path ever picks them.
        Map<String, WorkerStatus> map = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap();
        map.get("10.0.0.1:8080").setAlive(false);
        map.get("10.0.0.3:8080").setAlive(false);

        Set<String> picked = new HashSet<>();
        for (int i = 0; i < 8; i++) {
            ServerStatus s = rr.select(newSingleContext(2000L + i), RoleType.PDFUSION, null);
            Assertions.assertTrue(s.isSuccess());
            picked.add(s.getServerIp());
        }
        for (BatchScheduleTarget t : rr.selectBatch(8, RoleType.PDFUSION, null)) {
            picked.add(t.getServerIp());
        }
        Assertions.assertEquals(Set.of("10.0.0.0", "10.0.0.2"), picked,
                "dead workers must never be selected by either path");
    }

    @Test
    void all_dead_workers_yield_error_and_empty_batch() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap()
                .values().forEach(w -> w.setAlive(false));

        ServerStatus single = rr.select(newSingleContext(3000L), RoleType.PDFUSION, null);
        Assertions.assertFalse(single.isSuccess(),
                "a populated-but-all-dead map must fail the single select, not gamble");

        Assertions.assertTrue(rr.selectBatch(4, RoleType.PDFUSION, null).isEmpty(),
                "a populated-but-all-dead map must yield no batch targets");
    }

    @Test
    void select_keeps_independent_cursor_per_group() throws Exception {
        // Cursors are keyed per (role, group). With a single shared per-role cursor,
        // alternating g1/g2 traffic would advance it twice per group pick, so each
        // 2-worker group would floorMod onto the same worker every time.
        Map<String, WorkerStatus> map = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap();
        map.get("10.0.0.0:8080").setGroup("g1");
        map.get("10.0.0.1:8080").setGroup("g1");
        map.get("10.0.0.2:8080").setGroup("g2");
        map.get("10.0.0.3:8080").setGroup("g2");

        Map<String, Integer> g1Picks = new HashMap<>();
        Map<String, Integer> g2Picks = new HashMap<>();
        for (int i = 0; i < 4; i++) {
            ServerStatus s1 = rr.select(newSingleContext(6000L + 2 * i), RoleType.PDFUSION, "g1");
            Assertions.assertTrue(s1.isSuccess(), "g1 select should succeed");
            g1Picks.merge(s1.getServerIp(), 1, Integer::sum);
            ServerStatus s2 = rr.select(newSingleContext(6001L + 2 * i), RoleType.PDFUSION, "g2");
            Assertions.assertTrue(s2.isSuccess(), "g2 select should succeed");
            g2Picks.merge(s2.getServerIp(), 1, Integer::sum);
        }

        Assertions.assertEquals(Map.of("10.0.0.0", 2, "10.0.0.1", 2), g1Picks,
                "g1 picks must rotate uniformly over g1's own subset despite interleaved g2 traffic");
        Assertions.assertEquals(Map.of("10.0.0.2", 2, "10.0.0.3", 2), g2Picks,
                "g2 picks must rotate uniformly over g2's own subset despite interleaved g1 traffic");

        Map<String, AtomicInteger> cursors = cursorMap();
        Assertions.assertEquals(4, cursors.get(RoleType.PDFUSION.name() + "|g1").get(),
                "g1 cursor must advance once per g1 pick only");
        Assertions.assertEquals(4, cursors.get(RoleType.PDFUSION.name() + "|g2").get(),
                "g2 cursor must advance once per g2 pick only");
        Assertions.assertNull(cursors.get(RoleType.PDFUSION.name()),
                "group traffic must not touch the null-group cursor");
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
