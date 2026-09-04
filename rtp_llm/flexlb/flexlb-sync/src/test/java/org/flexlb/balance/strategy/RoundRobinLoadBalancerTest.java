package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.EngineType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

class RoundRobinLoadBalancerTest {

    private EngineWorkerStatus engineWorkerStatus;
    private ConfigService configService;
    private EndpointRegistry endpointRegistry;
    private RoundRobinLoadBalancer rr;

    @BeforeEach
    void setUp() {
        LoadBalanceStrategyFactory.resetForTesting();
        clearWorkerMaps();
        FlexlbConfig config = new FlexlbConfig();
        configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(config);
        endpointRegistry = new EndpointRegistry(
                configService,
                () -> null,
                Mockito.mock(BatchSchedulerReporter.class));
        engineWorkerStatus = new EngineWorkerStatus(endpointRegistry);
        rr = new RoundRobinLoadBalancer(engineWorkerStatus, configService);
        populatePdFusion(4);
    }

    @AfterEach
    void tearDown() {
        clearWorkerMaps();
        LoadBalanceStrategyFactory.resetForTesting();
        endpointRegistry.close();
    }

    @Test
    void registers_under_round_robin_strategy() {
        LoadBalanceStrategy registered = LoadBalanceStrategyFactory.getLoadBalanceStrategy(
                LoadBalanceStrategyEnum.ROUND_ROBIN);
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
        // The cursor is an AtomicLong that getAndAdd()s without bound; once it crosses
        // Long.MAX_VALUE it goes negative. Math.floorMod (not %) is what keeps the index
        // non-negative through that wrap — seed the cursor at the boundary to exercise it.
        seedCursor(RoleType.PDFUSION, Long.MAX_VALUE - 1);

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
        // Verifies getAndAdd(count) atomically reserves a CONTIGUOUS cursor range: a batch call
        // and the concurrent select() calls must never draw the same cursor value.
        //
        // The earlier version of this test used pool==batch==4, which made the check vacuous:
        // 4 consecutive integers mod 4 always cover {0,1,2,3} for ANY start, atomic or not, so the
        // only assertion (batchWorkers.size()==4) held even under a non-atomic cursor. Fix: make
        // the pool LARGER than the total picks per trial so each consumed cursor value maps to a
        // DISTINCT worker, then require every pick (batch targets + concurrent singles) to be
        // distinct. Atomic operations partition the consecutive integers [base, base+perTrialPicks)
        // one-per-op, so under a correct cursor the union is exactly perTrialPicks distinct workers.
        // A non-atomic reservation (e.g. get()+set() instead of getAndAdd) lets a concurrent
        // select() read the un-advanced cursor and collide with the batch's range -> a duplicate
        // worker -> a detected violation, and also loses updates -> a wrong final cursor value.
        int pool = 32;
        int batch = 4;
        int concurrentSingles = 16; // batch + singles = 20 < pool, so distinct cursor -> distinct worker
        int perTrialPicks = batch + concurrentSingles;
        clearWorkerMaps();
        populatePdFusion(pool);

        int trials = 200;
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
                        batchOut.set(0, rr.selectBatch(batch, RoleType.PDFUSION, null));
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

                // Union of every worker picked in this trial: batch targets + concurrent singles.
                // With pool > perTrialPicks, atomic cursor reservation => all consumed cursor values
                // are distinct consecutive integers => all workers distinct.
                Set<String> poolWorkers =
                        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().keySet();
                Set<String> allPicks = new HashSet<>();
                for (BatchScheduleTarget bt : batchOut.get(0)) {
                    allPicks.add(bt.getServerIp() + ":" + bt.getHttpPort());
                }
                synchronized (singleOut) {
                    Assertions.assertEquals(concurrentSingles, singleOut.size(),
                            "every concurrent select() must produce a result");
                    for (ServerStatus s : singleOut) {
                        Assertions.assertTrue(s.isSuccess(),
                                "concurrent select() must succeed under batch pressure: " + s.getMessage());
                        Assertions.assertTrue(poolWorkers.contains(s.getServerIp() + ":" + s.getHttpPort()),
                                "single pick must be a pool worker: " + s.getServerIp() + ":" + s.getHttpPort());
                        allPicks.add(s.getServerIp() + ":" + s.getHttpPort());
                    }
                }
                if (allPicks.size() != perTrialPicks) {
                    // A collision means two picks shared a cursor value -> the batch range was not
                    // reserved atomically against the concurrent singles.
                    violations.incrementAndGet();
                }
            }
        } finally {
            exec.shutdown();
            exec.awaitTermination(5, TimeUnit.SECONDS);
        }

        Assertions.assertEquals(0, violations.get(),
                "batch + concurrent singles must never share a cursor value "
                        + "(verifies getAndAdd reserves a contiguous range atomically)");
        Assertions.assertEquals((long) trials * perTrialPicks,
                cursorMap().get(RoleType.PDFUSION.name()).get(),
                "cursor must advance by exactly one per single + count per batch, with no lost updates");
    }

    @Test
    void selectBatch_does_not_change_endpoint_load() {
        WorkerEndpoint endpoint = endpointRegistry.get(
                RoleType.PDFUSION, "10.0.0.0:8080");
        long before = endpoint.getLoadMetric();

        rr.selectBatch(4, RoleType.PDFUSION, null);

        Assertions.assertEquals(before, endpoint.getLoadMetric(),
                "round-robin batch selection must stay reservation-free");
    }

    @Test
    void select_and_rollback_are_stateless() {
        Map<String, Long> before = new HashMap<>();
        endpointRegistry.getEndpoints(RoleType.PDFUSION)
                .forEach((address, endpoint) -> before.put(address, endpoint.getLoadMetric()));
        BalanceContext ctx = newSingleContext(5000L);
        ServerStatus assigned = rr.select(ctx, RoleType.PDFUSION, null);
        Assertions.assertTrue(assigned.isSuccess());

        String ipPort = assigned.getServerIp() + ":" + assigned.getHttpPort();
        WorkerEndpoint selected = endpointRegistry.get(RoleType.PDFUSION, ipPort);
        rr.rollBack(selected, 5000L);

        endpointRegistry.getEndpoints(RoleType.PDFUSION).forEach((address, endpoint) ->
                Assertions.assertEquals(before.get(address), endpoint.getLoadMetric(),
                        "stateless strategy must not change endpoint load: " + address));
    }

    @SuppressWarnings("unchecked")
    private void seedCursor(RoleType role, long value) throws Exception {
        Field f = RoundRobinLoadBalancer.class.getDeclaredField("cursors");
        f.setAccessible(true);
        // Cursors are keyed "<role>" for group=null and "<role>|<group>" otherwise.
        Map<String, AtomicLong> cursors = (Map<String, AtomicLong>) f.get(rr);
        cursors.computeIfAbsent(role.name(), k -> new AtomicLong(0)).set(value);
    }

    @SuppressWarnings("unchecked")
    private Map<String, AtomicLong> cursorMap() throws Exception {
        Field f = RoundRobinLoadBalancer.class.getDeclaredField("cursors");
        f.setAccessible(true);
        return (Map<String, AtomicLong>) f.get(rr);
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

        Map<String, AtomicLong> cursors = cursorMap();
        Assertions.assertEquals(4L, cursors.get(RoleType.PDFUSION.name() + "|g1").get(),
                "g1 cursor must advance once per g1 pick only");
        Assertions.assertEquals(4L, cursors.get(RoleType.PDFUSION.name() + "|g2").get(),
                "g2 cursor must advance once per g2 pick only");
        Assertions.assertNull(cursors.get(RoleType.PDFUSION.name()),
                "group traffic must not touch the null-group cursor");
    }

    private void populatePdFusion(int count) {
        for (int i = 0; i < count; i++) {
            WorkerStatus w = new WorkerStatus();
            w.setIp("10.0.0." + i);
            w.setPort(8080);
            w.setRole(RoleType.PDFUSION);
            w.setAlive(true);
            w.setAvailableConcurrency(1L);
            w.setRunningTaskList(new HashMap<>());
            CacheStatus cs = new CacheStatus();
            cs.setAvailableKvCache(100_000);
            cs.setBlockSize(16);
            w.setCacheStatus(cs);
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().put(w.getIpPort(), w);
            endpointRegistry.ensureEndpoint(RoleType.PDFUSION, w.getIpPort(), w);
        }
    }

    private void clearWorkerMaps() {
        if (endpointRegistry != null) {
            for (RoleType roleType : RoleType.values()) {
                new ArrayList<>(endpointRegistry.getEndpoints(roleType).entrySet())
                        .forEach(entry -> endpointRegistry.remove(
                                roleType, entry.getKey(), entry.getValue().getStatus()));
            }
        }
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().clear();
    }
}
