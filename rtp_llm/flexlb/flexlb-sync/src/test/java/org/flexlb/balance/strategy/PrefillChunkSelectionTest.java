package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.grpc.EngineStatusConverter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Exercises chunk selection through both routing entry points using real endpoint generations.
 */
class PrefillChunkSelectionTest {
    private EndpointRegistry registry;
    private CostBasedPrefillStrategy strategy;
    private BalanceContext context;
    private long dispatchDeadline;

    @BeforeEach
    void setUp() {
        FlexlbConfig config = StrategyTestSupport.config();
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.getRouter().getRoles().getPrefill().setChunkBatchingEnabled(true);
        config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator().setExpression("sum(computeTokens)");
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        registry = StrategyTestSupport.endpointRegistry(configService);
        CacheAwareService cache = mock(CacheAwareService.class);
        when(cache.findMatchingEngines(any())).thenReturn(CacheMatchResult.empty(CacheMatchSource.LOCAL_SYNC));
        strategy = new CostBasedPrefillStrategy(new WorkerDirectory(registry), cache, mock(EngineHealthReporter.class));
        long now = System.currentTimeMillis();
        dispatchDeadline = now + 30_000L;
        Request request = new Request();
        request.setRequestId("chunk-selection");
        request.setSeqLen(100L);
        request.setRequestTimeMs(now);
        request.setGenerateTimeout(60_000);
        request.setBlockCacheKeys(List.of());
        context = new BalanceContext(config);
        context.setRequest(request);
    }

    @AfterEach
    void tearDown() {
        registry.close();
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void selectsBestFitAndRefundsAfterEndpointHoldTransfer(boolean prepared) {
        publish("10.0.0.1", 500L, 3, dispatchDeadline);
        PrefillEndpoint tight = publish("10.0.0.2", 150L, 1, dispatchDeadline);
        SelectedRole selected = select(prepared).value();
        assertNotNull(selected);
        assertEquals("10.0.0.2", selected.serverStatus().getServerIp());
        assertEquals(1L, selected.serverStatus().getPrefillChunk().chunkId());
        assertEquals(Long.MAX_VALUE, remaining(tight));
        try (var hold = selected.takeEndpointHold()) {
            selected.close();
            selected.close();
            assertEquals(150L, remaining(tight));
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void selectsEarlierDeadlineBeforeTighterCapacity(boolean prepared) {
        publish("10.0.0.1", 150L, 3, dispatchDeadline);
        publish("10.0.0.2", 500L, 3, dispatchDeadline - 10_000L);
        try (SelectedRole selected = select(prepared).value()) {
            assertNotNull(selected);
            assertEquals("10.0.0.2", selected.serverStatus().getServerIp());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void skipsEarlierChunksWithoutEnoughTokensOrSlots(boolean prepared) {
        publish("10.0.0.1", 99L, 3, dispatchDeadline - 20_000L);
        publish("10.0.0.2", 500L, 0, dispatchDeadline - 15_000L);
        publish("10.0.0.3", 500L, 3, dispatchDeadline - 10_000L);
        publish("10.0.0.4", 150L, 3, dispatchDeadline);
        try (SelectedRole selected = select(prepared).value()) {
            assertNotNull(selected);
            assertEquals("10.0.0.3", selected.serverStatus().getServerIp());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void movesToNextDeadlineAfterEarlierChunkIsConsumed(boolean prepared) {
        publish("10.0.0.1", 200L, 2, dispatchDeadline - 10_000L);
        publish("10.0.0.2", 150L, 1, dispatchDeadline);
        for (int requestIndex = 0; requestIndex < 3; requestIndex++) {
            context.getRequest().setRequestId("chunk-selection-" + requestIndex);
            try (SelectedRole selected = select(prepared).value()) {
                assertNotNull(selected);
                assertEquals(requestIndex < 2 ? "10.0.0.1" : "10.0.0.2", selected.serverStatus().getServerIp());
                selected.takeUnpublishedChunkRefund();
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void repeatedStatusDoesNotReplenishCommittedGrant(boolean prepared) {
        PrefillEndpoint endpoint = publish("10.0.0.1", 150L, 1, dispatchDeadline);
        try (SelectedRole selected = select(prepared).value()) {
            assertNotNull(selected);
            // Queue admission takes responsibility for the grant and disables local rollback.
            selected.takeUnpublishedChunkRefund();
        }
        report(endpoint, 150L, 1, dispatchDeadline, 3L);
        var fallback = select(prepared);
        assertEquals(PlacementResult.Status.REJECTED, fallback.status());
        assertEquals(8600, fallback.failure().getCode());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void expiredGrantReturnsExplicitFallback(boolean prepared) {
        publish("10.0.0.1", 500L, 3, System.currentTimeMillis() - 1L);
        var fallback = select(prepared);
        assertEquals(PlacementResult.Status.REJECTED, fallback.status());
        assertEquals(8600, fallback.failure().getCode());
    }

    private PlacementResult<SelectedRole, RoleType> select(boolean prepared) {
        return prepared
                ? strategy.selectPrepared(context, RoleType.PREFILL, null,
                        strategy.prepare(context, RoleType.PREFILL, null), candidate -> candidate.predictedMs())
                : strategy.select(context, RoleType.PREFILL, null);
    }

    private PrefillEndpoint publish(String ip, long tokens, int slots, long deadline) {
        WorkerStatus source =
                StrategyTestSupport.workerStatus(RoleType.PREFILL, null, ip, 8080, 8081, true, 1_000_000L, 1_000_000L);
        PrefillEndpoint endpoint = (PrefillEndpoint) StrategyTestSupport.publishEndpoint(registry, RoleType.PREFILL,
                source.getLogicalIpPort(), source);
        report(endpoint, tokens, slots, deadline, 2L);
        return endpoint;
    }

    private void report(PrefillEndpoint endpoint, long tokens, int slots, long deadline, long version) {
        WorkerStatus worker = endpoint.getStatus();
        var status = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole("PREFILL")
                .setAlive(true)
                .setStatusVersion(version)
                .setAvailableKvCache(1_000_000L)
                .setTotalKvCache(1_000_000L)
                .setPrefillStepTokenBudget(1_000L)
                .setPrefillChunk(EngineRpcService.PrefillChunkPB.newBuilder()
                        .setEngineEpoch("epoch-1")
                        .setChunkId(1L)
                        .setInputTokenCapacity(tokens)
                        .setRequestCapacity(slots)
                        .setDispatchDeadlineMs(deadline))
                .build();
        Runnable notify;
        worker.lock.lock();
        try {
            notify = endpoint.applyPreparedStatus(worker,
                    worker.prepareNewStatus(EngineStatusConverter.convertToStatusObservation(worker, status)));
        } finally {
            worker.lock.unlock();
        }
        notify.run();
    }

    private long remaining(PrefillEndpoint endpoint) {
        WorkerStatus worker = endpoint.getStatus();
        worker.lock.lock();
        try {
            return worker.prefillChunkCapacity().remainingTokens(100L, System.currentTimeMillis(), Long.MAX_VALUE);
        } finally {
            worker.lock.unlock();
        }
    }
}
