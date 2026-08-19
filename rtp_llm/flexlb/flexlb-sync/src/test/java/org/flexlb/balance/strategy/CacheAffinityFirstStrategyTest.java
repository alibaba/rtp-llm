package org.flexlb.balance.strategy;

import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.StrategyConfigs;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.worker.ScoredWorker;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class CacheAffinityFirstStrategyTest {

    private static final long SEQ_LEN = 1_000;
    private EngineHealthReporter engineHealthReporter;

    @BeforeEach
    void setUp() {
        clearWorkerStatuses();
        engineHealthReporter = Mockito.mock(EngineHealthReporter.class);
    }

    @AfterEach
    void tearDown() {
        clearWorkerStatuses();
    }

    @Test
    void should_select_global_cache_leader_within_extra_work_cap_and_register_factory() {
        WorkerStatus shortest = worker("127.0.0.1", 1_000);
        WorkerStatus secondShortest = worker("127.0.0.2", 500);
        WorkerStatus cacheLeader = worker("127.0.0.3", 100);
        CacheAffinityFirstStrategy strategy = createStrategy(
                List.of(shortest, secondShortest, cacheLeader), Map.of(), ignored -> true);
        FlexlbConfig config = cacheAffinityConfig(50, 5);

        ScoredWorker selected = selectDirectly(
                strategy,
                List.of(
                        scored(shortest, 100, 0),
                        scored(secondShortest, 120, 100),
                        scored(cacheLeader, 150, 500)),
                config,
                fixedCandidatePool(1));

        assertSame(cacheLeader, selected.worker());
        assertSame(
                strategy,
                LoadBalanceStrategyFactory.getLoadBalancer(
                        LoadBalanceStrategyEnum.CACHE_AFFINITY_FIRST));
        verify(engineHealthReporter).reportCacheAffinityDecision(
                RoleType.PREFILL, cacheLeader.getIp(), "CACHE_LEADER");
    }

    @Test
    void should_allow_cache_leader_at_extra_work_cap_boundary() {
        WorkerStatus shortest = worker("127.0.0.1", 1_000);
        WorkerStatus cacheLeader = worker("127.0.0.2", 100);
        CacheAffinityFirstStrategy strategy = createStrategy(
                List.of(shortest, cacheLeader), Map.of(), ignored -> true);

        ScoredWorker selected = selectDirectly(
                strategy,
                List.of(
                        scored(shortest, 100, 0),
                        scored(cacheLeader, 150, 500)),
                cacheAffinityConfig(50, 5),
                fixedCandidatePool(1));

        assertSame(cacheLeader, selected.worker());
    }

    @Test
    void should_preserve_exact_baseline_selection_when_cache_leader_exceeds_cap() {
        WorkerStatus shortest = worker("127.0.0.1", 1_000);
        WorkerStatus olderSimilarWorker = worker("127.0.0.2", 1);
        WorkerStatus cacheLeader = worker("127.0.0.3", 100);
        CacheAffinityFirstStrategy strategy = createStrategy(
                List.of(shortest, olderSimilarWorker, cacheLeader), Map.of(), ignored -> true);

        ScoredWorker selected = selectDirectly(
                strategy,
                List.of(
                        scored(shortest, 100, 0),
                        scored(olderSimilarWorker, 105, 0),
                        scored(cacheLeader, 151, 500)),
                cacheAffinityConfig(50, 5),
                fixedCandidatePool(2));

        // The legacy baseline applies LRU fairness within its similar-TTFT pool; it does not
        // blindly return the absolute shortest worker.
        assertSame(olderSimilarWorker, selected.worker());
        verify(engineHealthReporter).reportCacheAffinityDecision(
                RoleType.PREFILL, olderSimilarWorker.getIp(), "OVER_CAP");
    }

    @Test
    void should_fall_back_to_baseline_when_global_leader_is_over_cap_even_if_another_cache_worker_fits() {
        WorkerStatus shortest = worker("127.0.0.1", 1_000);
        WorkerStatus admissibleCacheWorker = worker("127.0.0.2", 100);
        WorkerStatus globalCacheLeader = worker("127.0.0.3", 200);
        CacheAffinityFirstStrategy strategy = createStrategy(
                List.of(shortest, admissibleCacheWorker, globalCacheLeader), Map.of(), ignored -> true);

        ScoredWorker selected = selectDirectly(
                strategy,
                List.of(
                        scored(shortest, 100, 0),
                        scored(admissibleCacheWorker, 140, 400),
                        scored(globalCacheLeader, 151, 500)),
                cacheAffinityConfig(50, 5),
                fixedCandidatePool(1));

        // This strategy intentionally mirrors feature/flexlb-kvcm: once the global leader is
        // rejected, it returns to the legacy baseline instead of inventing a second-best policy.
        assertSame(shortest, selected.worker());
    }

    @Test
    void should_preserve_exact_baseline_selection_when_cache_hit_rate_is_too_low() {
        WorkerStatus shortest = worker("127.0.0.1", 1_000);
        WorkerStatus olderSimilarWorker = worker("127.0.0.2", 1);
        WorkerStatus lowHitCacheLeader = worker("127.0.0.3", 100);
        CacheAffinityFirstStrategy strategy = createStrategy(
                List.of(shortest, olderSimilarWorker, lowHitCacheLeader), Map.of(), ignored -> true);

        ScoredWorker selected = selectDirectly(
                strategy,
                List.of(
                        scored(shortest, 100, 0),
                        scored(olderSimilarWorker, 105, 0),
                        scored(lowHitCacheLeader, 120, 49)),
                cacheAffinityConfig(100, 5),
                fixedCandidatePool(2));

        assertSame(olderSimilarWorker, selected.worker());
    }

    @Test
    void should_keep_cold_request_fairness() {
        WorkerStatus recentlySelected = worker("127.0.0.1", 1_000);
        WorkerStatus leastRecentlySelected = worker("127.0.0.2", 1);
        CacheAffinityFirstStrategy strategy = createStrategy(
                List.of(recentlySelected, leastRecentlySelected), Map.of(), ignored -> true);

        ScoredWorker selected = selectDirectly(
                strategy,
                List.of(
                        scored(recentlySelected, 100, 0),
                        scored(leastRecentlySelected, 105, 0)),
                cacheAffinityConfig(1_000, 0),
                fixedCandidatePool(2));

        assertSame(leastRecentlySelected, selected.worker());
    }

    @Test
    void should_keep_baseline_fairness_when_all_workers_have_equal_cache_hits() {
        WorkerStatus recentlySelected = worker("127.0.0.1", 1_000);
        WorkerStatus leastRecentlySelected = worker("127.0.0.2", 1);
        CacheAffinityFirstStrategy strategy = createStrategy(
                List.of(recentlySelected, leastRecentlySelected), Map.of(), ignored -> true);

        ScoredWorker selected = selectDirectly(
                strategy,
                List.of(
                        scored(recentlySelected, 100, 200),
                        scored(leastRecentlySelected, 105, 200)),
                cacheAffinityConfig(1_000, 0),
                fixedCandidatePool(2));

        assertSame(leastRecentlySelected, selected.worker());
    }

    @Test
    void should_keep_cache_leader_when_it_is_also_the_shortest_worker() {
        WorkerStatus cacheLeader = worker("127.0.0.1", 100);
        WorkerStatus otherWorker = worker("127.0.0.2", 1);
        CacheAffinityFirstStrategy strategy = createStrategy(
                List.of(cacheLeader, otherWorker), Map.of(), ignored -> true);

        ScoredWorker selected = selectDirectly(
                strategy,
                List.of(
                        scored(cacheLeader, 100, 500),
                        scored(otherWorker, 120, 0)),
                cacheAffinityConfig(0, 5),
                fixedCandidatePool(2));

        assertSame(cacheLeader, selected.worker());
    }

    @Test
    void should_not_revive_resource_unavailable_cache_leader() {
        WorkerStatus unavailableCacheLeader = worker("127.0.0.1", 1);
        WorkerStatus availableWorker = worker("127.0.0.2", 2);
        CacheAffinityFirstStrategy strategy = createStrategy(
                List.of(unavailableCacheLeader, availableWorker),
                Map.of(unavailableCacheLeader.getIpPort(), 9),
                worker -> worker != unavailableCacheLeader);
        FlexlbConfig config = cacheAffinityConfig(10_000, 0);

        ServerStatus selected = strategy.select(balanceContext(config), RoleType.PREFILL, null);

        assertTrue(selected.isSuccess(), selected.getMessage());
        assertEquals(availableWorker.getIp(), selected.getServerIp());
    }

    @Test
    void should_try_another_qualifying_cache_worker_when_leader_claim_is_stale() {
        WorkerStatus cacheLeader = worker("127.0.0.1", 101);
        WorkerStatus cacheFallback = worker("127.0.0.2", 200);
        WorkerStatus shortest = worker("127.0.0.3", 300);
        CacheAffinityFirstStrategy strategy = createStrategy(
                List.of(cacheLeader, cacheFallback, shortest), Map.of(), ignored -> true);

        ScoredWorker selected = selectDirectly(
                strategy,
                List.of(
                        // The scored timestamp is deliberately stale, forcing the leader's CAS
                        // claim to fail without introducing a timing-dependent concurrent thread.
                        new ScoredWorker(cacheLeader, 150, 500, 100),
                        scored(cacheFallback, 140, 400),
                        scored(shortest, 100, 0)),
                cacheAffinityConfig(50, 5),
                fixedCandidatePool(1));

        assertSame(cacheFallback, selected.worker());
    }

    @Test
    void should_preserve_equal_max_cache_hit_when_shortest_leader_claim_is_stale() {
        WorkerStatus shortestCacheLeader = worker("127.0.0.1", 101);
        WorkerStatus lowerHitWorker = worker("127.0.0.2", 200);
        WorkerStatus equalHitFallback = worker("127.0.0.3", 300);
        CacheAffinityFirstStrategy strategy = createStrategy(
                List.of(shortestCacheLeader, lowerHitWorker, equalHitFallback), Map.of(), ignored -> true);

        ScoredWorker selected = selectDirectly(
                strategy,
                List.of(
                        new ScoredWorker(shortestCacheLeader, 100, 500, 100),
                        scored(lowerHitWorker, 105, 0),
                        scored(equalHitFallback, 110, 500)),
                cacheAffinityConfig(10, 5),
                fixedCandidatePool(1));

        assertSame(equalHitFallback, selected.worker());
    }

    @Test
    void should_fall_back_to_shortest_when_all_cache_affinity_claims_are_stale() {
        WorkerStatus cacheLeader = worker("127.0.0.1", 101);
        WorkerStatus cacheFallback = worker("127.0.0.2", 201);
        WorkerStatus shortest = worker("127.0.0.3", 301);
        CacheAffinityFirstStrategy strategy = createStrategy(
                List.of(cacheLeader, cacheFallback, shortest), Map.of(), ignored -> true);

        ScoredWorker selected = selectDirectly(
                strategy,
                List.of(
                        new ScoredWorker(cacheLeader, 150, 500, 100),
                        new ScoredWorker(cacheFallback, 140, 400, 200),
                        new ScoredWorker(shortest, 100, 0, 300)),
                cacheAffinityConfig(50, 5),
                fixedCandidatePool(1));

        assertSame(shortest, selected.worker());
    }

    private CacheAffinityFirstStrategy createStrategy(
            List<WorkerStatus> workers,
            Map<String, Integer> cacheMatches,
            Predicate<WorkerStatus> resourceAvailable) {
        Map<String, WorkerStatus> prefillWorkers =
                EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        workers.forEach(worker -> prefillWorkers.put(worker.getIpPort(), worker));

        ResourceMeasure resourceMeasure = Mockito.mock(ResourceMeasure.class);
        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        when(resourceMeasureFactory.getMeasure(any())).thenReturn(resourceMeasure);
        when(resourceMeasure.isResourceAvailable(any()))
                .thenAnswer(invocation -> resourceAvailable.test(invocation.getArgument(0)));

        CacheAwareService cacheAwareService = Mockito.mock(CacheAwareService.class);
        when(cacheAwareService.findMatchingEngines(anyList(), any(), any()))
                .thenReturn(cacheMatches);

        ConfigService configService = Mockito.mock(ConfigService.class);
        StrategyConfigs strategyConfigs = new StrategyConfigs();
        strategyConfigs.normalize();
        when(configService.getStrategyConfigs()).thenReturn(strategyConfigs);

        return new CacheAffinityFirstStrategy(
                new EngineWorkerStatus(new ModelMetaConfig()),
                engineHealthReporter,
                cacheAwareService,
                resourceMeasureFactory,
                configService);
    }

    private ScoredWorker selectDirectly(
            CacheAffinityFirstStrategy strategy,
            List<ScoredWorker> workers,
            FlexlbConfig config,
            StrategyConfigs.CandidatePoolConfig candidatePool) {
        return strategy.selectBestWorker(
                workers,
                new BalanceContext(),
                RoleType.PREFILL,
                null,
                SEQ_LEN,
                config,
                candidatePool);
    }

    private ScoredWorker scored(WorkerStatus worker, long ttft, long hitCacheTokens) {
        return new ScoredWorker(
                worker,
                ttft,
                hitCacheTokens,
                worker.getLastSelectedTime().get());
    }

    private WorkerStatus worker(String ip, long lastSelectedTime) {
        WorkerStatus worker = new WorkerStatus();
        worker.setIp(ip);
        worker.setPort(8080);
        worker.setRole(RoleType.PREFILL.getCode());
        worker.setGroup("default");
        worker.setAlive(true);
        worker.getLastSelectedTime().set(lastSelectedTime);

        CacheStatus cacheStatus = new CacheStatus();
        cacheStatus.setBlockSize(100);
        cacheStatus.setAvailableKvCache(1_000_000);
        worker.setCacheStatus(cacheStatus);
        return worker;
    }

    private FlexlbConfig cacheAffinityConfig(long maxExtraWorkTokens, double minHitRate) {
        FlexlbConfig config = new FlexlbConfig();
        config.setCacheAffinityFirstMaxExtraWorkTokens(maxExtraWorkTokens);
        config.setCacheAffinityFirstMinHitRate(minHitRate);
        return config;
    }

    private StrategyConfigs.CandidatePoolConfig fixedCandidatePool(int size) {
        StrategyConfigs strategyConfigs = new StrategyConfigs();
        StrategyConfigs.CandidatePoolConfig candidatePool =
                strategyConfigs.getShortestTtft().getCandidatePool();
        candidatePool.setMode(StrategyConfigs.CandidatePoolMode.FIXED);
        candidatePool.setSize(size);
        strategyConfigs.normalize();
        return candidatePool;
    }

    private BalanceContext balanceContext(FlexlbConfig config) {
        Request request = new Request();
        request.setRequestId(12345L);
        request.setSeqLen(SEQ_LEN);
        request.setBlockCacheKeys(List.of(1L));

        BalanceContext context = new BalanceContext();
        context.setConfig(config);
        context.setRequest(request);
        return context;
    }

    private void clearWorkerStatuses() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().clear();
    }
}
