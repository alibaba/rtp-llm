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
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class CostBasedPrefillStrategyTest {

    private static final String CACHE_H1 = "127.0.0.1:8080";
    private static final String CACHE_H2 = "127.0.0.2:8080";
    private static final String BASELINE = "127.0.0.3:8080";

    private FlexlbConfig config;
    private EndpointRegistry endpointRegistry;
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
        ResourceMeasureFactory resourceMeasureFactory = mock(ResourceMeasureFactory.class);
        when(resourceMeasureFactory.getMeasure(config.resourceMeasureFor(RoleType.PREFILL)))
                .thenReturn(resourceMeasure);

        CacheAwareService cacheAwareService = mock(CacheAwareService.class);
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
