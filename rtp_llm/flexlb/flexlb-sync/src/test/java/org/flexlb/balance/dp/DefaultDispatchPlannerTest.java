package org.flexlb.balance.dp;

import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.balance.strategy.LoadBalanceStrategyFactory;
import org.flexlb.balance.strategy.LoadBalancer;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class DefaultDispatchPlannerTest {

    private EngineWorkerStatus engineWorkerStatus;
    private ResourceMeasureFactory resourceMeasureFactory;
    private GroupSelector groupSelector;
    private LoadBalancer decodeSelector;
    private DefaultDispatchPlanner planner;

    @BeforeEach
    void setUp() {
        engineWorkerStatus = mock(EngineWorkerStatus.class);
        resourceMeasureFactory = mock(ResourceMeasureFactory.class);
        groupSelector = mock(GroupSelector.class);
        decodeSelector = mock(LoadBalancer.class);

        ResourceMeasure measure = mock(ResourceMeasure.class);
        when(resourceMeasureFactory.getMeasure(any())).thenReturn(measure);
        when(measure.isResourceAvailable(any())).thenReturn(true);

        // Register the decode selector under WEIGHTED_CACHE — that's what
        // FlexlbConfig.getStrategyForRoleType(DECODE) returns by default.
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.WEIGHTED_CACHE, decodeSelector);

        planner = new DefaultDispatchPlanner(engineWorkerStatus, resourceMeasureFactory, groupSelector);
    }

    @Test
    void happy_path_one_batch_with_all_requests_paired() {
        WorkerStatus w = workerStatus("10.0.0.1", 4);
        when(engineWorkerStatus.selectModelWorkerStatus(eq(RoleType.PREFILL), any())).thenReturn(map(w));
        when(groupSelector.select(any(), any())).thenReturn(w);
        when(decodeSelector.select(any(), eq(RoleType.DECODE), anyString())).thenAnswer(inv -> okDecode());

        DispatchPlan result = planner.plan(drained(4), context(4));

        assertEquals(1, result.batches().size());
        assertEquals(4, result.batches().get(0).size());
        assertTrue(result.failures().isEmpty());
        assertEquals("10.0.0.1", result.batches().get(0).prefillIp());
    }

    @Test
    void no_dp_enabled_candidates_fails_all_with_NO_PREFILL_WORKER() {
        // Only single-rank workers — must be filtered out by the dp_size>1 gate.
        WorkerStatus single = workerStatus("10.0.0.5", 1);
        when(engineWorkerStatus.selectModelWorkerStatus(eq(RoleType.PREFILL), any())).thenReturn(map(single));

        DispatchPlan result = planner.plan(drained(3), context(4));
        assertTrue(result.batches().isEmpty());
        assertEquals(3, result.failures().size());
        assertTrue(result.failures().stream()
                .allMatch(f -> f.reason() == StrategyErrorType.NO_PREFILL_WORKER));
    }

    @Test
    void group_selector_returning_null_fails_all() {
        WorkerStatus w = workerStatus("10.0.0.1", 4);
        when(engineWorkerStatus.selectModelWorkerStatus(eq(RoleType.PREFILL), any())).thenReturn(map(w));
        when(groupSelector.select(any(), any())).thenReturn(null);

        DispatchPlan result = planner.plan(drained(2), context(4));
        assertTrue(result.batches().isEmpty());
        assertEquals(2, result.failures().size());
    }

    @Test
    void decode_selection_failing_for_one_request_drops_only_that_one() {
        WorkerStatus w = workerStatus("10.0.0.1", 4);
        when(engineWorkerStatus.selectModelWorkerStatus(eq(RoleType.PREFILL), any())).thenReturn(map(w));
        when(groupSelector.select(any(), any())).thenReturn(w);

        AtomicInteger n = new AtomicInteger();
        when(decodeSelector.select(any(), eq(RoleType.DECODE), anyString())).thenAnswer(inv -> {
            int call = n.getAndIncrement();
            return call == 1 ? failedDecode() : okDecode();  // 2nd call fails
        });

        DispatchPlan result = planner.plan(drained(4), context(4));
        assertEquals(1, result.batches().size());
        assertEquals(3, result.batches().get(0).size(), "victim is dropped from the batch");
        assertEquals(1, result.failures().size());
        assertEquals(StrategyErrorType.NO_DECODE_WORKER, result.failures().get(0).reason());
    }

    @Test
    void decode_failing_for_all_requests_yields_zero_batches() {
        WorkerStatus w = workerStatus("10.0.0.1", 4);
        when(engineWorkerStatus.selectModelWorkerStatus(eq(RoleType.PREFILL), any())).thenReturn(map(w));
        when(groupSelector.select(any(), any())).thenReturn(w);
        when(decodeSelector.select(any(), eq(RoleType.DECODE), anyString())).thenAnswer(inv -> failedDecode());

        DispatchPlan result = planner.plan(drained(2), context(4));
        assertTrue(result.batches().isEmpty(), "no decode pairing ⇒ no prefill work either");
        assertEquals(2, result.failures().size());
    }

    @Test
    void empty_drain_returns_empty_result() {
        DispatchPlan result = planner.plan(List.of(), context(4));
        assertTrue(result.batches().isEmpty());
        assertTrue(result.failures().isEmpty());
    }

    // ============== helpers ==============

    private static List<QueuedRequest> drained(int n) {
        List<QueuedRequest> out = new java.util.ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            BalanceContext ctx = new BalanceContext();
            Request r = new Request();
            r.setRequestId(i + 1);
            ctx.setRequest(r);
            out.add(QueuedRequest.of(ctx, new CompletableFuture<>()));
        }
        return out;
    }

    private static DispatchContext context(int dpSize) {
        return new DispatchContext("m1", dpSize, new FlexlbConfig());
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
