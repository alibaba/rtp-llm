package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Same-host DP ranks are separate scheduling candidates, including rank-local running work. */
class CostBasedPrefillDpRankTest {
    @ParameterizedTest
    @ValueSource(ints = {2, 4})
    void selectsEveryDpRankAndPreservesItsPort(int dpSize) {
        for (DispatcherConfig.Type mode : DispatcherConfig.Type.values()) {
            exerciseRanks(dpSize, mode, false);
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {2, 4})
    void prefersIdleRanksOverTheRankWithRunningWork(int dpSize) {
        for (DispatcherConfig.Type mode : DispatcherConfig.Type.values()) {
            exerciseRanks(dpSize, mode, true);
        }
    }

    private void exerciseRanks(int dpSize, DispatcherConfig.Type mode, boolean busyRankZero) {
        FlexlbConfig config = new FlexlbConfig();
        config.getDispatcher().setType(mode);
        config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator()
                .setExpression("sum(computeTokens)");
        RoutingConfig.CandidateChoiceConfig choice = config.getRouter()
                .getRoles().getPrefill().getCandidateChoice();
        choice.setType(RoutingConfig.CandidateChoiceType.LEAST_RECENTLY_USED_IN_POOL);
        choice.getPool().setType(RoutingConfig.CandidatePoolType.FIXED);
        choice.getPool().setWorkers(dpSize);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        EndpointRegistry registry = StrategyTestSupport.endpointRegistry(configService);
        try {
            for (int rank = 0; rank < dpSize; rank++) {
                int port = 8000 + rank * 10;
                WorkerStatus status = WorkerStatus.createDiscovered(
                        RoleType.PREFILL, "dp-group", "127.0.0.1", port, port + 1, "test-site");
                WorkerStatusResponse response = StrategyTestSupport.response(
                        RoleType.PREFILL, true,
                        1_000_000L,
                        1_000_000L, 1L);
                response.setDpSize(dpSize);
                response.setDpRank(rank);
                response.setTpSize(1);
                if (busyRankZero && rank == 0) {
                    TaskInfo running = new TaskInfo();
                    running.setRequestId(999L);
                    running.setDpRank(rank);
                    running.setInputLength(1000L);
                    running.setPhase(TaskPhase.RUNNING);
                    response.setRunningTaskInfo(Map.of("999", running));
                    response.setRunningQueryLen(1L);
                }
                StrategyTestSupport.publish(status, response);
                StrategyTestSupport.publishEndpoint(
                        registry, RoleType.PREFILL, "127.0.0.1:" + port, status);
            }
            CacheAwareService cache = mock(CacheAwareService.class);
            when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of());
            CostBasedPrefillStrategy strategy = new CostBasedPrefillStrategy(
                    new WorkerDirectory(registry), cache, mock(EngineHealthReporter.class));
            Request request = new Request();
            request.setSeqLen(1000L);
            request.setPriority(50);
            BalanceContext context = new BalanceContext();
            context.setRequest(request);
            context.setConfig(config);
            context.setSchedulingMetadata(SchedulingMetadata.explicit(
                    50, System.currentTimeMillis() + 60_000L));
            int firstRank = busyRankZero ? 1 : 0;
            Map<Long, Integer> expected = new HashMap<>();
            for (int rank = firstRank; rank < dpSize; rank++) {
                expected.put((long) rank, 2);
            }
            Map<Long, Integer> selectedCounts = new HashMap<>();
            for (int index = 0; index < (dpSize - firstRank) * 2; index++) {
                request.setRequestId(10_000L + index);
                PlacementResult<SelectedRole, RoleType> result =
                        strategy.select(context, RoleType.PREFILL, "dp-group");
                assertEquals(PlacementResult.Status.SUCCESS, result.status());
                try (SelectedRole selected = result.value()) {
                    ServerStatus route = selected.serverStatus();
                    long rank = route.getDpRank();
                    assertEquals("127.0.0.1", route.getServerIp());
                    assertEquals(8000 + rank * 10, route.getHttpPort());
                    assertEquals(8001 + rank * 10, route.getGrpcPort());
                    selectedCounts.merge(rank, 1, Integer::sum);
                }
            }
            assertEquals(expected, selectedCounts, "mode=" + mode + ", dp_size=" + dpSize);
        } finally {
            registry.close();
        }
    }
}
