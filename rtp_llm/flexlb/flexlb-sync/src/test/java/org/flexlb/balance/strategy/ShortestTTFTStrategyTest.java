package org.flexlb.balance.strategy;

import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author zjw
 * description:
 * date: 2025/3/11
 */
class ShortestTTFTStrategyTest {

    @Test
    void test() {

        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
        Map<String/*ip*/, WorkerStatus> prefillStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        Map<String, TaskInfo> waitingTaskList = new HashMap<>();
        Map<String, TaskInfo> runningTaskList = new HashMap<>();
        Map<String, TaskInfo> finishedTaskList = new HashMap<>();
        ConcurrentHashMap<String, TaskInfo> localTaskList = new ConcurrentHashMap<>();
        WorkerStatus workerStatus = createWorkerStatus("127.0.0.1", 200, waitingTaskList, runningTaskList, finishedTaskList, localTaskList);

        Map<String, TaskInfo> waitingTaskList1 = new HashMap<>();
        Map<String, TaskInfo> runningTaskList1 = new HashMap<>();
        Map<String, TaskInfo> finishedTaskList1 = new HashMap<>();
        ConcurrentHashMap<String, TaskInfo> localTaskList1 = new ConcurrentHashMap<>();
        WorkerStatus workerStatus1 = createWorkerStatus("127.0.0.2", 100, waitingTaskList1, runningTaskList1, finishedTaskList1, localTaskList1);

        prefillStatusMap.put("127.0.0.1:8080", workerStatus);
        prefillStatusMap.put("127.0.0.2:8080", workerStatus1);
        Request req = new Request();
        req.setSeqLen(1000);
        req.setRequestId(12345L);
        List<Long> blockCacheKeys = new ArrayList<>();
        blockCacheKeys.add(1L);
        blockCacheKeys.add(2L);
        req.setBlockCacheKeys(blockCacheKeys);

        EngineHealthReporter engineHealthReporter = Mockito.mock(EngineHealthReporter.class);
        CacheAwareService cacheAwareService = Mockito.mock(CacheAwareService.class);
        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        org.flexlb.balance.resource.ResourceMeasure resourceMeasure = Mockito.mock(org.flexlb.balance.resource.ResourceMeasure.class);
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(resourceMeasure);
        Mockito.when(resourceMeasure.isResourceAvailable(Mockito.any())).thenReturn(true);
        Mockito.when(cacheAwareService.findMatchingEngines(Mockito.anyList(), Mockito.any(), Mockito.any())).thenReturn(new HashMap<>());

        ShortestTTFTStrategy staticCacheLoadBalancer =
                new ShortestTTFTStrategy(engineWorkerStatus, engineHealthReporter, cacheAwareService, resourceMeasureFactory);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setConfig(new FlexlbConfig());
        balanceContext.setRequest(req);
        ServerStatus result = staticCacheLoadBalancer.select(balanceContext, RoleType.PREFILL, null);
        if (!result.isSuccess()) {
            System.out.println("Result not successful - code: " + result.getCode() + ", message: " + result.getMessage());
        }
        Assertions.assertTrue(result.isSuccess(), "Result should be successful but got: " + result.getMessage());
        Assertions.assertEquals("127.0.0.2", result.getServerIp());
    }

    @Test
    void dp_enabled_worker_is_scored_alongside_single_rank_workers() {
        // Backward-compat path: legacy single-shot requests (max_new_tokens=1,
        // beam search, SP-disabled, or dpBalanceEnabled=false) bypass
        // DpBatchScheduler and route via ShortestTTFT. DP-enabled workers must
        // be scored, not filtered out — KvCacheManager.findMatchingEngines now
        // returns MAX-per-rank for DP engines, so the score is honest.
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
        Map<String, WorkerStatus> prefillStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillStatusMap.clear();

        WorkerStatus dpWorker = createWorkerStatus("10.99.0.1", 50,
                new HashMap<>(), new HashMap<>(), new HashMap<>(), new ConcurrentHashMap<>());
        dpWorker.setDpSize(4);
        prefillStatusMap.put("10.99.0.1:8080", dpWorker);

        Request req = new Request();
        req.setSeqLen(1000);
        req.setRequestId(424242L);
        req.setBlockCacheKeys(new ArrayList<>());

        EngineHealthReporter reporter = Mockito.mock(EngineHealthReporter.class);
        CacheAwareService cacheAwareService = Mockito.mock(CacheAwareService.class);
        ResourceMeasureFactory rf = Mockito.mock(ResourceMeasureFactory.class);
        org.flexlb.balance.resource.ResourceMeasure rm = Mockito.mock(org.flexlb.balance.resource.ResourceMeasure.class);
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        Mockito.when(rf.getMeasure(Mockito.any())).thenReturn(rm);
        Mockito.when(rm.isResourceAvailable(Mockito.any())).thenReturn(true);
        Mockito.when(cacheAwareService.findMatchingEngines(Mockito.anyList(), Mockito.any(), Mockito.any()))
                .thenReturn(new HashMap<>());

        ShortestTTFTStrategy strategy = new ShortestTTFTStrategy(engineWorkerStatus, reporter, cacheAwareService, rf);
        BalanceContext bc = new BalanceContext();
        bc.setConfig(new FlexlbConfig());
        bc.setRequest(req);

        ServerStatus result = strategy.select(bc, RoleType.PREFILL, null);
        Assertions.assertTrue(result.isSuccess(),
                "DP worker must be a valid candidate when it's the only one available");
        Assertions.assertEquals("10.99.0.1", result.getServerIp());
    }

    @Test
    void select_returns_no_available_worker_when_status_map_is_empty() {
        // Empty PREFILL map means no candidate at all. Strategy must surface
        // this as NO_AVAILABLE_WORKER so the caller can fail fast rather than
        // looping on an empty selection. (Note: NO_PREFILL_WORKER is the
        // role-specific mapping applied a level up by RouteService, not at
        // the strategy boundary.)
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();

        ShortestTTFTStrategy strategy = newStrategy(engineWorkerStatus);
        ServerStatus result = strategy.select(newCtx(1L, List.of()), RoleType.PREFILL, null);

        Assertions.assertFalse(result.isSuccess());
        Assertions.assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), result.getCode());
    }

    @Test
    void dead_worker_is_filtered_out_of_candidates() {
        // A worker reported alive=false must not be picked even when it would
        // otherwise win on queue time — the routing layer trusts the
        // synchronizer's liveness signal as the sole truth.
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
        Map<String, WorkerStatus> prefillStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillStatusMap.clear();

        WorkerStatus dead = createWorkerStatus("10.0.0.1", 10,
                new HashMap<>(), new HashMap<>(), new HashMap<>(), new ConcurrentHashMap<>());
        dead.setAlive(false);
        WorkerStatus alive = createWorkerStatus("10.0.0.2", 200,
                new HashMap<>(), new HashMap<>(), new HashMap<>(), new ConcurrentHashMap<>());
        prefillStatusMap.put("10.0.0.1:8080", dead);
        prefillStatusMap.put("10.0.0.2:8080", alive);

        ShortestTTFTStrategy strategy = newStrategy(engineWorkerStatus);
        ServerStatus result = strategy.select(newCtx(2L, List.of()), RoleType.PREFILL, null);

        Assertions.assertTrue(result.isSuccess());
        Assertions.assertEquals("10.0.0.2", result.getServerIp(),
                "dead worker must be filtered even with the better score");
    }

    @Test
    void resource_unavailable_worker_is_filtered_out() {
        // ResourceMeasure gate (e.g. queue time over upper bound) is the
        // dynamic equivalent of alive=false: candidates that fail the gate
        // are skipped regardless of TTFT score.
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
        Map<String, WorkerStatus> prefillStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillStatusMap.clear();

        WorkerStatus saturated = createWorkerStatus("10.0.0.1", 10,
                new HashMap<>(), new HashMap<>(), new HashMap<>(), new ConcurrentHashMap<>());
        WorkerStatus available = createWorkerStatus("10.0.0.2", 500,
                new HashMap<>(), new HashMap<>(), new HashMap<>(), new ConcurrentHashMap<>());
        prefillStatusMap.put("10.0.0.1:8080", saturated);
        prefillStatusMap.put("10.0.0.2:8080", available);

        EngineHealthReporter reporter = Mockito.mock(EngineHealthReporter.class);
        CacheAwareService cacheAwareService = Mockito.mock(CacheAwareService.class);
        ResourceMeasureFactory rf = Mockito.mock(ResourceMeasureFactory.class);
        org.flexlb.balance.resource.ResourceMeasure rm = Mockito.mock(org.flexlb.balance.resource.ResourceMeasure.class);
        Mockito.when(rf.getMeasure(Mockito.any())).thenReturn(rm);
        Mockito.when(rm.isResourceAvailable(saturated)).thenReturn(false);
        Mockito.when(rm.isResourceAvailable(available)).thenReturn(true);
        Mockito.when(cacheAwareService.findMatchingEngines(Mockito.anyList(), Mockito.any(), Mockito.any()))
                .thenReturn(new HashMap<>());

        ShortestTTFTStrategy strategy = new ShortestTTFTStrategy(
                engineWorkerStatus, reporter, cacheAwareService, rf);
        ServerStatus result = strategy.select(newCtx(3L, List.of()), RoleType.PREFILL, null);

        Assertions.assertTrue(result.isSuccess());
        Assertions.assertEquals("10.0.0.2", result.getServerIp(),
                "saturated worker filtered by ResourceMeasure gate");
    }

    private static ShortestTTFTStrategy newStrategy(EngineWorkerStatus engineWorkerStatus) {
        EngineHealthReporter reporter = Mockito.mock(EngineHealthReporter.class);
        CacheAwareService cacheAwareService = Mockito.mock(CacheAwareService.class);
        ResourceMeasureFactory rf = Mockito.mock(ResourceMeasureFactory.class);
        org.flexlb.balance.resource.ResourceMeasure rm = Mockito.mock(org.flexlb.balance.resource.ResourceMeasure.class);
        Mockito.when(rf.getMeasure(Mockito.any())).thenReturn(rm);
        Mockito.when(rm.isResourceAvailable(Mockito.any())).thenReturn(true);
        Mockito.when(cacheAwareService.findMatchingEngines(Mockito.anyList(), Mockito.any(), Mockito.any()))
                .thenReturn(new HashMap<>());
        return new ShortestTTFTStrategy(engineWorkerStatus, reporter, cacheAwareService, rf);
    }

    private static BalanceContext newCtx(long requestId, List<Long> blockCacheKeys) {
        Request req = new Request();
        req.setSeqLen(1000);
        req.setRequestId(requestId);
        req.setBlockCacheKeys(blockCacheKeys);
        BalanceContext ctx = new BalanceContext();
        ctx.setConfig(new FlexlbConfig());
        ctx.setRequest(req);
        return ctx;
    }

    WorkerStatus createWorkerStatus(String ip,
                                    long runningQueueTime,
                                    Map<String, TaskInfo> waitingTaskInfo,
                                    Map<String, TaskInfo> finishedTaskList,
                                    Map<String, TaskInfo> runningTaslList,
                                    ConcurrentHashMap<String, TaskInfo> localTaskList) {
        WorkerStatus workerStatus = new WorkerStatus();

        workerStatus.setIp(ip);
        workerStatus.setPort(8080);
        workerStatus.setSite("na61");
        workerStatus.setAlive(true);
        workerStatus.setRole(RoleType.PREFILL.getCode());
        CacheStatus cacheStatus = new CacheStatus();
        cacheStatus.setAvailableKvCache(10000);
        cacheStatus.setBlockSize(256);
        workerStatus.setCacheStatus(cacheStatus);
        workerStatus.getRunningQueueTime().getAndSet(runningQueueTime);
        workerStatus.setWaitingTaskList(waitingTaskInfo);
        workerStatus.updateTaskStates(waitingTaskInfo, runningTaslList, finishedTaskList);
        workerStatus.setRunningTaskList(runningTaslList);
        return workerStatus;
    }

}