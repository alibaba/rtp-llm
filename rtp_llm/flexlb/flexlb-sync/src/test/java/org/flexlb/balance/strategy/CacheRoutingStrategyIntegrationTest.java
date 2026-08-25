package org.flexlb.balance.strategy;

import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class CacheRoutingStrategyIntegrationTest {

    private static final long BLOCK_SIZE = 2_000;
    private static final long INPUT_TOKENS = 50_000;

    @BeforeEach
    @AfterEach
    void clearWorkerStatuses() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().clear();
    }

    @Test
    void shortestTtftDoesNotUseCacheAffinityExtraWorkLimit() {
        FlexlbConfig config = new FlexlbConfig();
        config.setOutstandingUncachedTokensThreshold(1_000_000L);
        config.setCacheAffinityFirstMaxExtraWorkTokens(0);

        SelectionResult shortestSelection = select(
                LoadBalanceStrategyEnum.SHORTEST_TTFT,
                List.of(worker("127.0.0.1", 0), worker("127.0.0.2", 2_500)),
                Map.of("127.0.0.1:8080@0", 15, "127.0.0.2:8080@0", 16),
                config,
                "shortest-ignores-cache-limit");
        SelectionResult cacheAffinitySelection = select(
                LoadBalanceStrategyEnum.CACHE_AFFINITY_FIRST,
                List.of(worker("127.0.0.1", 0), worker("127.0.0.2", 2_500)),
                Map.of("127.0.0.1:8080@0", 15, "127.0.0.2:8080@0", 16),
                config,
                "cache-affinity-uses-cache-limit");

        assertThat(shortestSelection.serverStatus().getServerIp()).isEqualTo("127.0.0.2");
        assertThat(cacheAffinitySelection.serverStatus().getServerIp()).isEqualTo("127.0.0.1");
    }

    @Test
    void outstandingThresholdRejectsOverloadedCacheLeaderForBothStrategies() {
        FlexlbConfig config = new FlexlbConfig();
        config.setOutstandingUncachedTokensThreshold(1_000_000L);
        config.setCacheAffinityFirstMaxExtraWorkTokens(2_000_000);

        WorkerStatus shortestWorker = worker("127.0.0.1", 0);
        WorkerStatus overloadedCacheLeader = worker("127.0.0.2", 0);
        addUncachedPendingWork(overloadedCacheLeader, 990_000);
        SelectionResult shortestSelection = select(
                LoadBalanceStrategyEnum.SHORTEST_TTFT,
                List.of(shortestWorker, overloadedCacheLeader),
                Map.of(shortestWorker.getLogicalIpPort(), 15, overloadedCacheLeader.getLogicalIpPort(), 17),
                config,
                "shortest-outstanding-threshold");

        WorkerStatus cacheAffinityShortestWorker = worker("127.0.0.1", 0);
        WorkerStatus cacheAffinityOverloadedLeader = worker("127.0.0.2", 0);
        addUncachedPendingWork(cacheAffinityOverloadedLeader, 990_000);
        SelectionResult cacheAffinitySelection = select(
                LoadBalanceStrategyEnum.CACHE_AFFINITY_FIRST,
                List.of(cacheAffinityShortestWorker, cacheAffinityOverloadedLeader),
                Map.of(
                        cacheAffinityShortestWorker.getLogicalIpPort(), 15,
                        cacheAffinityOverloadedLeader.getLogicalIpPort(), 17),
                config,
                "cache-affinity-outstanding-threshold");

        assertThat(shortestSelection.serverStatus().getServerIp()).isEqualTo(shortestWorker.getIp());
        assertThat(cacheAffinitySelection.serverStatus().getServerIp()).isEqualTo(cacheAffinityShortestWorker.getIp());
        assertThat(outstandingGuardEligible(shortestSelection, overloadedCacheLeader.getIp())).isFalse();
        assertThat(outstandingGuardEligible(cacheAffinitySelection, cacheAffinityOverloadedLeader.getIp())).isFalse();
    }

    @Test
    void cacheAffinityEnforcesMinimumHitRateWhileShortestTtftKeepsSimilarCachePreference() {
        FlexlbConfig config = new FlexlbConfig();
        config.setCacheAffinityFirstMaxExtraWorkTokens(10_000);
        config.setCacheAffinityFirstMinHitRate(5);

        SelectionResult shortestSelection = select(
                LoadBalanceStrategyEnum.SHORTEST_TTFT,
                List.of(worker("127.0.0.1", 0), worker("127.0.0.2", 5_000)),
                Map.of("127.0.0.2:8080@0", 1),
                config,
                "shortest-low-hit-cache-preference");
        SelectionResult cacheAffinitySelection = select(
                LoadBalanceStrategyEnum.CACHE_AFFINITY_FIRST,
                List.of(worker("127.0.0.1", 0), worker("127.0.0.2", 5_000)),
                Map.of("127.0.0.2:8080@0", 1),
                config,
                "cache-affinity-low-hit-rejected");

        assertThat(shortestSelection.serverStatus().getServerIp()).isEqualTo("127.0.0.2");
        assertThat(cacheAffinitySelection.serverStatus().getServerIp()).isEqualTo("127.0.0.1");
        assertThat(cacheAffinitySelection.balanceContext().getSelectionReasonByRole().get(RoleType.PREFILL))
                .isEqualTo("SHORTEST_TTFT_LOW_CACHE_HIT");
    }

    private SelectionResult select(LoadBalanceStrategyEnum strategyType,
                                   List<WorkerStatus> workers,
                                   Map<String, Integer> cacheMatches,
                                   FlexlbConfig config,
                                   String requestId) {
        workers.forEach(worker -> EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS
                .getPrefillStatusMap()
                .put(worker.getLogicalIpPort(), worker));

        ResourceMeasure resourceMeasure = Mockito.mock(ResourceMeasure.class);
        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(resourceMeasure);
        Mockito.when(resourceMeasure.isResourceAvailable(Mockito.any())).thenReturn(true);

        CacheAwareService cacheAwareService = Mockito.mock(CacheAwareService.class);
        Mockito.when(cacheAwareService.findMatchingEngines(Mockito.any(CacheMatchQuery.class)))
                .thenReturn(new CacheMatchResult(
                        HostCacheMatch.fromLocalMatches(cacheMatches), CacheMatchSource.KVCM, 1, BLOCK_SIZE));

        LoadBalancer strategy = strategyType == LoadBalanceStrategyEnum.SHORTEST_TTFT
                ? new ShortestTTFTStrategy(
                        new EngineWorkerStatus(new ModelMetaConfig()),
                        Mockito.mock(EngineHealthReporter.class),
                        cacheAwareService,
                        resourceMeasureFactory)
                : new CacheAffinityFirstStrategy(
                        new EngineWorkerStatus(new ModelMetaConfig()),
                        Mockito.mock(EngineHealthReporter.class),
                        cacheAwareService,
                        resourceMeasureFactory);
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(INPUT_TOKENS);
        request.setBlockSize(BLOCK_SIZE);
        request.setBlockCacheKeys(List.of(1L));
        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setConfig(config);
        balanceContext.setRequest(request);
        return new SelectionResult(strategy.select(balanceContext, RoleType.PREFILL, null), balanceContext);
    }

    private WorkerStatus worker(String ip, long queueWorkTokens) {
        WorkerStatus worker = new WorkerStatus();
        worker.setIp(ip);
        worker.setPort(8080);
        worker.setAlive(true);
        worker.setRole(RoleType.PREFILL.getCode());
        worker.getRunningQueueTime().set(queueWorkTokens);
        CacheStatus cacheStatus = new CacheStatus();
        cacheStatus.setBlockSize(BLOCK_SIZE);
        cacheStatus.setAvailableKvCache(1_000_000);
        worker.setCacheStatus(cacheStatus);
        return worker;
    }

    private void addUncachedPendingWork(WorkerStatus worker, long tokens) {
        TaskInfo task = new TaskInfo();
        task.setRequestId("pending-" + worker.getIp());
        task.setInputLength(tokens);
        task.setPrefixLength(0);
        task.setPredictedPrefixLength(0);
        worker.putLocalTask(task.getRequestId(), task);
    }

    private boolean outstandingGuardEligible(SelectionResult selection, String workerIp) {
        return selection.balanceContext()
                .getShortestTtftDecisionByRole()
                .get(RoleType.PREFILL)
                .workers()
                .stream()
                .filter(worker -> worker.ip().equals(workerIp))
                .findFirst()
                .orElseThrow()
                .outstandingGuardEligible();
    }

    private record SelectionResult(ServerStatus serverStatus, BalanceContext balanceContext) {}
}
