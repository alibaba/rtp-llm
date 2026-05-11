package org.flexlb.balance.strategy;

import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
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

class RoundRobinLoadBalancerTest {

    private EngineWorkerStatus engineWorkerStatus;
    private RoundRobinLoadBalancer rr;

    @BeforeEach
    void setUp() {
        clearWorkerMaps();
        engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
        rr = new RoundRobinLoadBalancer(engineWorkerStatus);
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
            Assertions.assertEquals(t.getHttpPort() + 1, t.getGrpcPort(),
                    "grpc_port should be http_port + 1");
        }
        Assertions.assertEquals(4, workers.size(), "4 slots and 4 workers must hit all 4");
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
