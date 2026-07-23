package org.flexlb.balance.strategy;

import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
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
    void selects_the_lower_queue_time_worker_when_no_cache_hit() {
        // Baseline dimension: with no cache matches and both workers resource-available, the winner
        // is decided purely by queue time — 127.0.0.2 (queue 100) beats 127.0.0.1 (queue 200). The
        // tests below add the cache-aware and resource-gate dimensions this one holds flat, so a
        // regression in either is caught rather than masked by the over-mocked baseline.
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

    @AfterEach
    void tearDown() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().clear();
    }

    @Test
    void cache_hit_lets_a_busier_worker_win_over_a_lower_queue_one() {
        // 127.0.0.1 carries the HIGHER queue time (200 vs 100), so on queue alone it loses. But it
        // holds a large prefix-cache match. TTFT = prefillTime + queueTime, with
        // prefillTime = seqLen - hitCacheTokens*0.7 (TaskInfo.estimatePrefillTimeMs). With
        // seqLen=1000, blockSize=256 and 2 matched blocks -> hitCacheTokens=512 shaves 358:
        //   TTFT(.1) = (1000-358)+200 = 842   vs   TTFT(.2) = (1000-0)+100 = 1100.
        // The cache hit flips the winner to 127.0.0.1. If the strategy ignored cacheMatchResults,
        // .2 (1100 < 1200) would win — so this pins the cache-aware path, not the queue tiebreak.
        Map<String, WorkerStatus> prefillStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillStatusMap.put("127.0.0.1:8080", newWorker("127.0.0.1", 200));
        prefillStatusMap.put("127.0.0.2:8080", newWorker("127.0.0.2", 100));

        CacheAwareService cacheAwareService = Mockito.mock(CacheAwareService.class);
        Map<String, Integer> matches = new HashMap<>();
        matches.put("127.0.0.1:8080", 2); // two matched cache blocks on the busier worker
        Mockito.when(cacheAwareService.findMatchingEngines(Mockito.anyList(), Mockito.any(), Mockito.any()))
                .thenReturn(matches);
        ResourceMeasure resourceMeasure = Mockito.mock(ResourceMeasure.class);
        Mockito.when(resourceMeasure.isResourceAvailable(Mockito.any())).thenReturn(true);

        ServerStatus result = newStrategy(cacheAwareService, resourceMeasure)
                .select(newContext(1000, List.of(1L, 2L)), RoleType.PREFILL, null);

        Assertions.assertTrue(result.isSuccess(), result.getMessage());
        Assertions.assertEquals("127.0.0.1", result.getServerIp(),
                "the cached worker must win despite its higher queue time");
    }

    @Test
    void resource_unavailable_worker_is_excluded_even_with_the_lower_ttft() {
        // 127.0.0.2 has the lower queue time (100 vs 200) and would win on TTFT, but its resource
        // gate reports unavailable. getAvailableWorkers filters it out, leaving only 127.0.0.1.
        // If the strategy skipped the isResourceAvailable gate, .2 would win — so this pins the
        // resource filter, not the queue-time comparison.
        Map<String, WorkerStatus> prefillStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        WorkerStatus w1 = newWorker("127.0.0.1", 200);
        WorkerStatus w2 = newWorker("127.0.0.2", 100);
        prefillStatusMap.put("127.0.0.1:8080", w1);
        prefillStatusMap.put("127.0.0.2:8080", w2);

        CacheAwareService cacheAwareService = Mockito.mock(CacheAwareService.class);
        Mockito.when(cacheAwareService.findMatchingEngines(Mockito.anyList(), Mockito.any(), Mockito.any()))
                .thenReturn(new HashMap<>());
        ResourceMeasure resourceMeasure = Mockito.mock(ResourceMeasure.class);
        Mockito.when(resourceMeasure.isResourceAvailable(w1)).thenReturn(true);
        Mockito.when(resourceMeasure.isResourceAvailable(w2)).thenReturn(false);

        ServerStatus result = newStrategy(cacheAwareService, resourceMeasure)
                .select(newContext(1000, List.of(1L, 2L)), RoleType.PREFILL, null);

        Assertions.assertTrue(result.isSuccess(), result.getMessage());
        Assertions.assertEquals("127.0.0.1", result.getServerIp(),
                "the resource-unavailable worker must be excluded from selection");
    }

    @Test
    void returns_a_valid_candidate_when_workers_tie_on_ttft() {
        // Two identical workers (same queue time, no cache) tie on TTFT. The strategy must still
        // return one of them successfully rather than crash or hand back an empty result.
        Map<String, WorkerStatus> prefillStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillStatusMap.put("127.0.0.1:8080", newWorker("127.0.0.1", 100));
        prefillStatusMap.put("127.0.0.2:8080", newWorker("127.0.0.2", 100));

        CacheAwareService cacheAwareService = Mockito.mock(CacheAwareService.class);
        Mockito.when(cacheAwareService.findMatchingEngines(Mockito.anyList(), Mockito.any(), Mockito.any()))
                .thenReturn(new HashMap<>());
        ResourceMeasure resourceMeasure = Mockito.mock(ResourceMeasure.class);
        Mockito.when(resourceMeasure.isResourceAvailable(Mockito.any())).thenReturn(true);

        ServerStatus result = newStrategy(cacheAwareService, resourceMeasure)
                .select(newContext(1000, List.of()), RoleType.PREFILL, null);

        Assertions.assertTrue(result.isSuccess(), result.getMessage());
        Assertions.assertTrue(
                "127.0.0.1".equals(result.getServerIp()) || "127.0.0.2".equals(result.getServerIp()),
                "tie must resolve to one of the tied workers, got " + result.getServerIp());
    }

    private ShortestTTFTStrategy newStrategy(CacheAwareService cacheAwareService, ResourceMeasure resourceMeasure) {
        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(resourceMeasure);
        return new ShortestTTFTStrategy(
                new EngineWorkerStatus(new ModelMetaConfig()),
                Mockito.mock(EngineHealthReporter.class),
                cacheAwareService,
                resourceMeasureFactory);
    }

    private WorkerStatus newWorker(String ip, long runningQueueTime) {
        return createWorkerStatus(ip, runningQueueTime, new HashMap<>(), new HashMap<>(), new HashMap<>(),
                new ConcurrentHashMap<>());
    }

    private static BalanceContext newContext(long seqLen, List<Long> blockCacheKeys) {
        Request req = new Request();
        req.setSeqLen(seqLen);
        req.setRequestId(12345L);
        req.setBlockCacheKeys(new ArrayList<>(blockCacheKeys));
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