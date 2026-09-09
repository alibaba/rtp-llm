package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.resource.DecodeResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.config.ConfigService;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class DecodeCostSelectionTest {
    private ConfigService config;
    private EndpointRegistry registry;
    private CostBasedDecodeStrategy strategy;
    private BalanceContext context;

    @BeforeEach
    void setUp() {
        config = new ConfigService();
        assertInstanceOf(RoutingConfig.MinCostDecodeSelectorConfig.class,
                config.loadBalanceConfig().getRouter().getRoles().getDecode().getSelector());
        registry = new EndpointRegistry(config, () -> null, mock(BatchSchedulerReporter.class));
        var factory = mock(ResourceMeasureFactory.class);
        var measure = mock(DecodeResourceMeasure.class);
        when(factory.getMeasure(any())).thenReturn(measure);
        when(measure.isResourceAvailable(any())).thenReturn(true);
        strategy = new CostBasedDecodeStrategy(new EngineWorkerStatus(registry), factory);
        Request request = new Request();
        request.setRequestId(100L);
        request.setSeqLen(1);
        context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(config.loadBalanceConfig());
    }

    @Test
    void defaultsToRatioRatherThanAbsoluteKvUsage() {
        worker("127.0.0.1", 1000, 200);
        worker("127.0.0.2", 10000, 1000);
        assertSelected("127.0.0.2");
    }

    @Test
    void runningSizeIncludesQueuedReservations() {
        DecodeEndpoint first = worker("127.0.0.1", 10000, 100);
        worker("127.0.0.2", 10000, 1000);
        first.reserve(1L, 1, 1);
        expression("running_size");
        assertSelected("127.0.0.2");
    }

    @Test
    void kvCostIncludesLocalPredictedReservations() {
        DecodeEndpoint first = worker("127.0.0.1", 10000, 100);
        worker("127.0.0.2", 10000, 1000);
        first.reserve(1L, 1, 2000);
        assertSelected("127.0.0.2");
    }

    @Test
    void customFormulaUsesConfiguredMaximumAndNonlinearFunctions() {
        DecodeEndpoint first = worker("127.0.0.1", 10000, 100);
        worker("127.0.0.2", 10000, 1000);
        first.reserve(1L, 1, 1);
        context.getConfig().getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(2L);
        expression("0.7 * running_size / max_running_size + pow(kvcache_used_ratio, 2)");
        assertSelected("127.0.0.2");
    }

    @Test
    void nonFiniteWorkersAreExcludedAndAllNonFiniteFailsWithoutReservation() {
        DecodeEndpoint first = worker("127.0.0.1", 10000, 0);
        DecodeEndpoint second = worker("127.0.0.2", 10000, 1000);
        expression("1 / kvcache_used");
        assertSelected("127.0.0.2");
        assertEquals(0, first.getTotalLoad());
        assertEquals(0, second.getTotalLoad());
        expression("sqrt(-1)");
        assertThrows(IllegalStateException.class, this::select);
        assertEquals(0, first.getTotalLoad());
        assertEquals(0, second.getTotalLoad());
    }

    @Test
    void formulaCannotSelectAWorkerWithoutPromptKvCapacity() {
        worker("127.0.0.1", 1000, 1000);
        worker("127.0.0.2", 1000, 100);
        expression("-kvcache_used");
        assertSelected("127.0.0.2");
    }

    @Test
    void equalNegativeCostsCanSelectBothWorkers() {
        worker("127.0.0.1", 10000, 100);
        worker("127.0.0.2", 10000, 1000);
        expression("-2");
        Set<String> selected = new HashSet<>();
        for (int i = 0; i < 200; i++) {
            ServerStatus result = select();
            assertTrue(result.isSuccess());
            selected.add(result.getServerIp());
            release(result);
        }
        assertEquals(Set.of("127.0.0.1", "127.0.0.2"), selected);
    }

    private DecodeEndpoint worker(String ip, long capacity, long used) {
        WorkerStatus status = new WorkerStatus();
        status.setIp(ip);
        status.setPort(8080);
        status.setGrpcPort(9090);
        status.setAlive(true);
        status.getTotalKvCacheTokens().set(capacity);
        status.getAvailableKvCacheTokens().set(capacity - used);
        DecodeEndpoint endpoint = (DecodeEndpoint) registry.ensureEndpoint(RoleType.DECODE, ip + ":8080", status);
        endpoint.onWorkerStatusUpdate(status, new WorkerStatusResponse());
        return endpoint;
    }

    private void expression(String expression) {
        context.getConfig().getRouter().getRoles().getDecode().getCostEstimator().setExpression(expression);
    }

    private ServerStatus select() {
        return strategy.select(context, RoleType.DECODE, null);
    }

    private void assertSelected(String ip) {
        ServerStatus result = select();
        assertTrue(result.isSuccess());
        assertEquals(ip, result.getServerIp());
        release(result);
    }

    private void release(ServerStatus result) {
        strategy.rollBack(registry.get(RoleType.DECODE, result.getServerIp() + ":8080"), 100L);
    }
}
