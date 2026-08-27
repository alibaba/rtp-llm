package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillWorkLedger;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.resource.PrefillResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.domain.CacheMatch;
import org.flexlb.cache.domain.EngineGeneration;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DirectSchedulerConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.config.SingleDecisionConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.clearInvocations;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class CostBasedPrefillStrategyTest {

    private static final String CACHE_H1 = "127.0.0.1:8080";
    private static final String CACHE_H2 = "127.0.0.2:8080";
    private static final String BASELINE = "127.0.0.3:8080";

    private FlexlbConfig config;
    private EndpointRegistry endpointRegistry;
    private CacheAwareService cacheAwareService;
    private ResourceMeasureFactory resourceMeasureFactory;
    private CostBasedPrefillStrategy strategy;

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
        RoutingConfig.PrefillConfig prefill =
                config.getRouter().getRoles().getPrefill();
        prefill.getAvailability().setMaxPendingRequests(2L);
        ((RoutingConfig.FormulaEstimatorConfig)
                prefill.getExecutionTimeEstimator()).setExpression("100");
        RoutingConfig.EstimatedTtftSelectorConfig selector =
                new RoutingConfig.EstimatedTtftSelectorConfig();
        selector.setCandidateChoice(new RoutingConfig.BestOnlyConfig());
        prefill.setSelector(selector);
        config.queueScheduler().setDecision(new SingleDecisionConfig());

        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        endpointRegistry = StrategyTestSupport.endpointRegistry(configService);

        PrefillResourceMeasure resourceMeasure =
                new PrefillResourceMeasure(configService);
        resourceMeasureFactory = mock(ResourceMeasureFactory.class);
        when(resourceMeasureFactory.getMeasure(config.resourceMeasureFor(RoleType.PREFILL)))
                .thenReturn(resourceMeasure);

        cacheAwareService = mock(CacheAwareService.class);
        when(cacheAwareService.findMatchingEngines(
                anyList(), eq(RoleType.PREFILL), anyList()))
                .thenAnswer(call -> cacheMatches(call.getArgument(2)));
        strategy = new CostBasedPrefillStrategy(
                new EngineWorkerStatus(endpointRegistry),
                cacheAwareService,
                resourceMeasureFactory,
                mock(EngineHealthReporter.class));
    }

    @AfterEach
    void tearDown() {
        endpointRegistry.close();
    }

    @Test
    void queueKeepsAFullCacheLeaderInsideItsExtraTtftBudget() {
        assertSelected(100L, CACHE_H1);
    }

    @Test
    void fullCacheLeaderCarriesOnlyItsAbsoluteReselectBoundary() {
        configureAffinity(200L);
        registerCluster();
        BalanceContext context = requestContext();
        long startedAtMs = System.currentTimeMillis();
        context.setEnqueueTime(startedAtMs);

        try (SelectedRole selected = strategy.select(
                context, RoleType.PREFILL, null)) {
            assertNotNull(selected);
            assertEquals(CACHE_H1,
                    selected.serverStatus().getServerIp()
                            + ":" + selected.serverStatus().getHttpPort());
            assertEquals(startedAtMs + 200L,
                    selected.reselectNotAfterMs());
        }
    }

    @Test
    void availableCacheLeaderCarriesTheBoundaryForACommitTimeFullRace() {
        config.getRouter().getRoles().getPrefill()
                .getAvailability().setMaxPendingRequests(3L);
        configureAffinity(200L);
        registerCluster();
        BalanceContext context = requestContext();
        long startedAtMs = System.currentTimeMillis();
        context.setEnqueueTime(startedAtMs);

        try (SelectedRole selected = strategy.select(
                context, RoleType.PREFILL, null)) {
            assertNotNull(selected);
            assertEquals(CACHE_H1,
                    selected.serverStatus().getServerIp()
                            + ":" + selected.serverStatus().getHttpPort());
            assertEquals(startedAtMs + 200L,
                    selected.reselectNotAfterMs());
        }
    }

    @Test
    void expiredAffinityBudgetSelectsAvailableBaselineWithoutAnotherWake() {
        configureAffinity(100L);
        registerCluster();
        BalanceContext context = requestContext();
        context.setEnqueueTime(System.currentTimeMillis() - 101L);

        try (SelectedRole selected = strategy.select(
                context, RoleType.PREFILL, null)) {
            assertNotNull(selected);
            assertEquals(BASELINE,
                    selected.serverStatus().getServerIp()
                            + ":" + selected.serverStatus().getHttpPort());
            assertEquals(Long.MAX_VALUE, selected.reselectNotAfterMs());
        }
    }

    @Test
    void expiredBudgetExcludesUnavailableCacheLeaderButKeepsAvailableCacheHit() {
        assertExpiredBudgetSelectsAvailableCacheHit();
    }

    @Test
    void shortestTtftAlsoExcludesUnavailableCacheLeaderAfterBudgetExpires() {
        RoutingConfig.EstimatedTtftSelectorConfig selector =
                new RoutingConfig.EstimatedTtftSelectorConfig();
        selector.setCandidateChoice(
                new RoutingConfig.LeastRecentlyUsedInPoolConfig());
        config.getRouter().getRoles().getPrefill().setSelector(selector);
        strategy = new ShortestTTFTStrategy(
                new EngineWorkerStatus(endpointRegistry),
                cacheAwareService,
                resourceMeasureFactory,
                mock(EngineHealthReporter.class));

        assertExpiredBudgetSelectsAvailableCacheHit();
    }

    @Test
    void expiredAffinityBudgetDoesNotScheduleAnotherWakeWhenAllAreFull() {
        configureAffinity(100L);
        registerCluster();
        PrefillEndpoint h2 = (PrefillEndpoint) endpointRegistry.get(
                RoleType.PREFILL, CACHE_H2);
        PrefillEndpoint baseline = (PrefillEndpoint) endpointRegistry.get(
                RoleType.PREFILL, BASELINE);
        addCommittedWork(h2, 20_002L, 20L);
        addCommittedWork(baseline, 30_001L, 20L);
        addCommittedWork(baseline, 30_002L, 20L);
        BalanceContext context = requestContext();
        context.setEnqueueTime(System.currentTimeMillis() - 101L);

        try (SelectedRole selected = strategy.select(
                context, RoleType.PREFILL, null)) {
            assertNotNull(selected);
            assertEquals(Long.MAX_VALUE, selected.reselectNotAfterMs());
        }
    }

    @Test
    void queueWithoutCacheAffinityPrefersAvailableCapacity() {
        registerCluster();
        assertSelected(requestContext(), BASELINE);
    }

    @Test
    void queueUsesTheNextCacheHitCandidateWhenH1ExceedsTheBudget() {
        assertSelected(50L, CACHE_H2);
    }

    @Test
    void queueUsesTheOrdinaryBaselineWhenEveryCacheHitExceedsTheBudget() {
        assertSelected(10L, BASELINE);
    }

    @Test
    void directKeepsItsSynchronousFullWorkerFilter() {
        configureAffinity(100L);
        registerCluster();
        config.setScheduler(new DirectSchedulerConfig());
        assertSelected(requestContext(), CACHE_H2);
    }

    @Test
    void emptyCacheKeysDoNotEnterCacheLookup() {
        register(BASELINE);
        BalanceContext context = requestContext();
        context.getRequest().setBlockCacheKeys(List.of());
        clearInvocations(cacheAwareService);

        assertSelected(context, BASELINE);

        verifyNoInteractions(cacheAwareService);
    }

    private void assertSelected(long maxExtraTtftMs, String expectedAddress) {
        configureAffinity(maxExtraTtftMs);
        registerCluster();
        assertSelected(requestContext(), expectedAddress);
    }

    private void assertSelected(
            BalanceContext context, String expectedAddress) {
        try (SelectedRole selected = strategy.select(
                context, RoleType.PREFILL, null)) {
            assertNotNull(selected);
            assertEquals(expectedAddress,
                    selected.serverStatus().getServerIp()
                            + ":" + selected.serverStatus().getHttpPort());
        }
    }

    private void registerCluster() {
        PrefillEndpoint h1 = register(CACHE_H1);
        PrefillEndpoint h2 = register(CACHE_H2);
        register(BASELINE);

        addCommittedWork(h1, 10_001L, 50L);
        addCommittedWork(h1, 10_002L, 50L);
        addCommittedWork(h2, 20_001L, 20L);
        assertEquals(2L, h1.admissionPendingRequestCount());
        assertEquals(1L, h2.admissionPendingRequestCount());
    }

    private void assertExpiredBudgetSelectsAvailableCacheHit() {
        configureAffinity(1L);
        PrefillEndpoint h1 = register(CACHE_H1);
        PrefillEndpoint h2 = register(CACHE_H2);
        register(BASELINE);
        // Fill H1's binding seats without putting extra modeled work ahead of
        // the probe, so its projected TTFT remains tied with the cold baseline.
        addCommittedWork(h1, 10_001L, 0L);
        addCommittedWork(h1, 10_002L, 0L);
        addCommittedWork(h2, 20_001L, 0L);
        BalanceContext context = requestContext();
        context.setEnqueueTime(System.currentTimeMillis() - 1_000L);

        try (SelectedRole selected = strategy.select(
                context, RoleType.PREFILL, null)) {
            assertNotNull(selected);
            assertEquals(CACHE_H2,
                    selected.serverStatus().getServerIp()
                            + ":" + selected.serverStatus().getHttpPort());
            assertEquals(Long.MAX_VALUE, selected.reselectNotAfterMs());
        }
    }

    private PrefillEndpoint register(String address) {
        String ip = address.substring(0, address.indexOf(':'));
        WorkerStatus status = StrategyTestSupport.workerStatus(
                RoleType.PREFILL, null, ip, 8080, 8081,
                true, 0L, 0L);
        return (PrefillEndpoint) endpointRegistry.registerPreinitializedEndpoint(
                RoleType.PREFILL, address, status);
    }

    private static void addCommittedWork(
            PrefillEndpoint endpoint, long requestId, long predictedMs) {
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration();
             PrefillWorkLedger.DirectRegistration registration =
                     endpoint.registerDirectRequest(pin, requestId, predictedMs)) {
            registration.commit();
        }
    }

    private void configureAffinity(long maxExtraTtftMs) {
        RoutingConfig.CacheAffinityConfig affinity =
                new RoutingConfig.CacheAffinityConfig();
        affinity.setMaxExtraTtftMs(maxExtraTtftMs);
        affinity.setMinPrefixHitPercent(5.0);
        config.getRouter().getRoles().getPrefill().setCacheAffinity(affinity);
    }

    private BalanceContext requestContext() {
        Request request = new Request();
        request.setRequestId(1L);
        request.setSeqLen(1_000L);
        request.setCacheKeyBlockSize(100L);
        request.setBlockCacheKeys(List.of(1L, 2L, 3L));
        BalanceContext context = new BalanceContext();
        context.setConfig(config);
        context.setRequest(request);
        return context;
    }

    private static Map<EngineGeneration, CacheMatch> cacheMatches(
            List<EngineGeneration> candidates) {
        Map<EngineGeneration, CacheMatch> result = new HashMap<>();
        for (EngineGeneration candidate : candidates) {
            int prefixBlocks = switch (candidate.address()) {
                case CACHE_H1 -> 9;
                case CACHE_H2 -> 8;
                default -> 0;
            };
            result.put(candidate, new CacheMatch(prefixBlocks));
        }
        return result;
    }
}
