package org.flexlb.balance.dp;

import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.balance.strategy.LoadBalanceStrategyFactory;
import org.flexlb.balance.strategy.LoadBalancer;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class DefaultDispatchPlannerTest {

    private EngineWorkerStatus engineWorkerStatus;
    private ResourceMeasureFactory resourceMeasureFactory;
    private LoadBalancer decodeSelector;
    private DefaultDispatchPlanner planner;

    @BeforeEach
    void setUp() {
        engineWorkerStatus = mock(EngineWorkerStatus.class);
        resourceMeasureFactory = mock(ResourceMeasureFactory.class);
        decodeSelector = mock(LoadBalancer.class);

        ResourceMeasure measure = mock(ResourceMeasure.class);
        when(resourceMeasureFactory.getMeasure(any())).thenReturn(measure);
        when(measure.isResourceAvailable(any())).thenReturn(true);

        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.WEIGHTED_CACHE, decodeSelector);

        planner = new DefaultDispatchPlanner(engineWorkerStatus, resourceMeasureFactory,
                mock(CacheAwareService.class), mock(PrefillTimePredictor.class));
    }

    @Test
    void selectDecodeWorker_returns_successful_status() {
        when(decodeSelector.select(any(), eq(RoleType.DECODE), anyString())).thenReturn(okDecode());

        BalanceContext ctx = makeCtx(1);
        ServerStatus result = planner.selectDecodeWorker(ctx, "g1");

        assertNotNull(result);
        assertTrue(result.isSuccess());
        assertEquals(RoleType.DECODE, result.getRole());
        assertEquals("10.0.0.99", result.getServerIp());
    }

    @Test
    void selectDecodeWorker_returns_failed_when_no_worker() {
        when(decodeSelector.select(any(), eq(RoleType.DECODE), anyString())).thenReturn(failedDecode());

        BalanceContext ctx = makeCtx(1);
        ServerStatus result = planner.selectDecodeWorker(ctx, "g1");

        assertNotNull(result);
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), result.getCode());
    }

    @Test
    void selectDecodeWorker_returns_null_when_selector_returns_null() {
        when(decodeSelector.select(any(), eq(RoleType.DECODE), anyString())).thenReturn(null);

        BalanceContext ctx = makeCtx(1);
        ServerStatus result = planner.selectDecodeWorker(ctx, "g1");

        assertNull(result);
    }

    @Test
    void selectPrefillWorker_returns_best_candidate() {
        WorkerStatus w = workerStatus("10.0.0.1", 4);
        when(engineWorkerStatus.selectModelWorkerStatus(eq(RoleType.PREFILL), any())).thenReturn(map(w));
        when(engineWorkerStatus.selectModelWorkerStatus(eq(RoleType.DECODE), any()))
                .thenReturn(Map.of("10.0.0.99:8081", decodeWorkerStatus()));

        WorkerStatus result = planner.selectPrefillWorker("m1", new FlexlbConfig(),
                makeCtx(1), k -> 0);

        assertNotNull(result);
        assertEquals("10.0.0.1", result.getIp());
    }

    @Test
    void selectPrefillWorker_returns_null_when_no_candidates() {
        when(engineWorkerStatus.selectModelWorkerStatus(eq(RoleType.PREFILL), any()))
                .thenReturn(Map.of());

        WorkerStatus result = planner.selectPrefillWorker("m1", new FlexlbConfig(),
                makeCtx(1), k -> 0);

        assertNull(result);
    }

    // ============== helpers ==============

    private static BalanceContext makeCtx(long requestId) {
        BalanceContext ctx = new BalanceContext();
        Request r = new Request();
        r.setRequestId(requestId);
        ctx.setRequest(r);
        ctx.setConfig(new FlexlbConfig());
        return ctx;
    }

    private static WorkerStatus workerStatus(String ip, long dpSize) {
        WorkerStatus w = new WorkerStatus();
        w.setIp(ip);
        w.setPort(8080);
        w.setDpSize(dpSize);
        w.setAlive(true);
        w.setGroup("g1");
        return w;
    }

    private static WorkerStatus decodeWorkerStatus() {
        WorkerStatus w = new WorkerStatus();
        w.setIp("10.0.0.99");
        w.setPort(8081);
        w.setAlive(true);
        w.setGroup("g1");
        return w;
    }

    private static Map<String, WorkerStatus> map(WorkerStatus... ws) {
        Map<String, WorkerStatus> m = new HashMap<>();
        for (WorkerStatus w : ws) {
            m.put(w.getIpPort(), w);
        }
        return m;
    }

    private static ServerStatus okDecode() {
        ServerStatus s = new ServerStatus();
        s.setSuccess(true);
        s.setRole(RoleType.DECODE);
        s.setServerIp("10.0.0.99");
        s.setHttpPort(8081);
        s.setGrpcPort(9081);
        s.setGroup("g1");
        return s;
    }

    private static ServerStatus failedDecode() {
        return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
    }
}
