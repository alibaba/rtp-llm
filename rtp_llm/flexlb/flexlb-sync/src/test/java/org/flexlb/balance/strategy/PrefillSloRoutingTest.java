package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.balance.scheduler.QueueExecutionSettings;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class PrefillSloRoutingTest {
    FlexlbConfig config;
    EndpointRegistry registry;
    CacheAwareService cache;
    CostBasedPrefillStrategy strategy;
    PrefillEndpoint worker;
    AtomicLong ids = new AtomicLong();
    List<RequestRoute> held = new CopyOnWriteArrayList<>();

    void setup(String dispatch) { setup(dispatch, 3000); }

    void setup(String dispatch, long sloMs) {
        config=ConfigService.parse("""
                {"ttftSloMs":%d,"dispatcher":{"type":"%s"},
                 "router":{"roles":{"prefill":{"executionTimeEstimator":{"type":"FORMULA","expression":"sum(computeTokens) / 70"}}}}}
                """.formatted(sloMs, dispatch));
        initialize();
    }

    void initialize() {
        registry=StrategyTestSupport.endpointRegistry(SchedulingTestConfig.configService(config));
        cache=mock(CacheAwareService.class);
        when(cache.findMatchingEngines(any(),any(),any())).thenReturn(Map.of());
        worker=publish("10.0.0.1");
        strategy=new CostBasedPrefillStrategy(registry,cache,mock(EngineHealthReporter.class),mock(CacheMetricsReporter.class),SchedulingTestConfig.configService(config));
    }
    PrefillEndpoint publish(String ip) {
        return (PrefillEndpoint)StrategyTestSupport.publishEndpoint(registry,RoleType.PREFILL,ip+":8080",
                StrategyTestSupport.workerStatus(RoleType.PREFILL,null,ip,8080,8081,true,10_000_000,10_000_000));
    }
    RequestRoute select(long tokens, long age, String session) { return select(tokens, age, session, 50); }
    RequestRoute select(long tokens, long age, String session, int priority) {
        long id=ids.incrementAndGet(), now=System.currentTimeMillis(), created=now-age;
        var request=Request.builder().requestId(id).sessionId(session).seqLen(tokens).apiKey("tenant").model("model")
                .cacheKeyBlockSize(Math.max(1,tokens/10)).blockCacheKeys(List.of(1L,2L)).build();
        var ctx=new RequestContext(config); ctx.data().setRequest(request);
        ctx.data().setSchedulingMetadata(SchedulingMetadata.explicit(priority,config.resolveExpiresAtMs(created)));
        ctx.register(id,created,config.resolveExpiresAtMs(created),config.queueScheduler().getQueueTimeoutMs());
        var route=new RequestRoute(ctx);
        var result=strategy.select(request,ctx.data().getSchedulingMetadata(),RoleType.PREFILL,null,route);
        if(result.status()==PlacementResult.Status.SUCCESS) {held.add(route); return route;}
        route.close(); return null;
    }
    @AfterEach void close() { held.forEach(RequestRoute::close); if(registry!=null) registry.close(); }

    @Test
    void uncalibratedLearningWorkerObtainsAnIsolatedExecutionSample() {
        config = ConfigService.parse("{\"ttftSloMs\":30000}");
        initialize();
        assertNotNull(select(70_000, 0, null));
        assertNull(select(70_000, 0, null), "a fresh worker must finish its first compute sample before accumulating a queue");
        var other = publish("10.0.0.2");
        assertSame(other, select(70_000, 0, null).prefillEp(), "other idle workers can calibrate concurrently");
        var capacity = (PrefillComputeCapacity) org.springframework.test.util.ReflectionTestUtils.getField(worker, "computeCapacity");
        capacity.observe(70_000, 1000);
        assertNotNull(select(70_000, 0, null), "known capacity removes the startup gate");
    }

    @ParameterizedTest @ValueSource(strings={"NON_BATCH","BATCH"})
    void observedComputeWorkBoundsOptimisticPredictionsWithoutCountingPlansTwice(String mode) {
        setup(mode);
        var capacity = (PrefillComputeCapacity) org.springframework.test.util.ReflectionTestUtils.getField(worker, "computeCapacity");
        capacity.observe(70_000, 2000); // observed 35K TPS; configured predictor still assumes 70K
        assertNotNull(select(70_000, 0, null));
        assertNotNull(select(14_000, 0, null)); // 2.4s total, including unpublished plans
        assertNull(select(14_000, 0, null)); // 2.8s exceeds the 2.7s budget despite a 1.4s prediction
        assertNull(select(700, 1000, null)); // measured work also consumes the aged request's remaining budget
    }

    @Test
    void committedEngineWorkRetainsItsOriginalSloAgainstLaterBatchMembers() {
        setup("NON_BATCH", 30000);
        var first = select(70_000, 25000, null);
        assertNotNull(first);
        org.flexlb.balance.scheduler.SchedulerTestSupport.bindEndpointOwner(worker, first);
        try (var pin = worker.tryPinGeneration()) {
            var reservation = worker.reserveUnqueuedRoute(pin, first, 1000).reservation();
            try (var preparation = org.flexlb.balance.endpoint.EndpointTestSupport.preparation(reservation);
                 var handoff = worker.commit(new org.flexlb.balance.endpoint.PrefillState.CommitPlan(
                         org.flexlb.balance.endpoint.PrefillState.CommitKind.DIRECT,
                         List.of(first), List.of(reservation), null, new long[0], null, System.currentTimeMillis()))) { }
        }
        assertEquals(0, worker.ownership.tokenWorkload().plannedExecutionMs());
        assertNull(select(140_000, 0, null), "Engine batching cannot extend the old request's remaining 2s budget");
        assertNotNull(select(700, 0, null));
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"NON_BATCH,100", "BATCH,100", "NON_BATCH,1000", "BATCH,1000", "NON_BATCH,10000", "BATCH,10000"})
    void admissionScalesWithUserTargetInsteadOfFixedRequestCount(String mode, long slo) {
        setup(mode,slo);
        int accepted=0;
        while(accepted<1200 && select(700,0,null)!=null) accepted++;
        // Each request reserves 10ms. Allow time spent executing this real routing test,
        // but never allow more plans than the user target can accommodate.
        assertTrue(accepted>=(slo*7/10)/10,"too little admitted work: "+accepted);
        assertTrue(accepted<=slo*9/100,"time budget was overspent: "+accepted);
    }

    @ParameterizedTest @ValueSource(strings={"NON_BATCH","BATCH"})
    void admitsManyShortRequestsWithoutManualInflightLimit(String mode) {
        setup(mode);
        assertEquals(0,worker.requestLimit);
        assertEquals(0,QueueExecutionSettings.capture(config).maxOutstandingRequests());
        for(int i=0;i<64;i++) assertNotNull(select(700,0,null),"ten-ms work must not stop at legacy inflight=2");
        assertEquals(640,worker.ownership.tokenWorkload().plannedExecutionMs());
    }
    @ParameterizedTest @ValueSource(strings={"NON_BATCH","BATCH"})
    void oneLongPlanBlocksMoreWorkWhileEquivalentCountOfShortPlansFits(String mode) {
        setup(mode);
        assertNotNull(select(182_000,0,null)); // 2600ms
        assertNull(select(14_000,0,null)); // total 2800ms exceeds 2700ms admission budget
        assertNotNull(select(700,0,null));
        held.forEach(RequestRoute::close);
        assertEquals(0,worker.ownership.tokenWorkload().plannedExecutionMs());
        assertNotNull(select(14_000,0,null));
    }
    @ParameterizedTest @ValueSource(strings={"NON_BATCH","BATCH"})
    void spentGlobalQueueTimeCannotBeResetByReselection(String mode) {
        setup(mode);
        assertNotNull(select(700,0,null)); // prevent idle-only burst drain exception
        assertNull(select(70_000,2000,null));
        assertNotNull(select(70_000,0,null));
    }
    @ParameterizedTest @ValueSource(strings={"NON_BATCH","BATCH"})
    void protectsOlderUnpublishedPlanFromNewBatchMember(String mode) {
        setup(mode);
        assertNotNull(select(700,2500,null));
        assertNull(select(35000,0,null)); // new request fits its own target, but not the old one's
        assertNotNull(select(700,0,null));
    }
    @ParameterizedTest @ValueSource(strings={"NON_BATCH","BATCH"})
    void idleWorkerCanDrainLateBurstButCannotExtendItsDeadline(String mode) {
        setup(mode);
        assertNotNull(select(700,4000,null));
        held.forEach(RequestRoute::close);
        assertNull(select(70_000,5500,null));
    }
    @ParameterizedTest @ValueSource(strings={"NON_BATCH","BATCH"})
    void concurrentSelectionsCannotDoubleSpendTimeBudget(String mode) throws Exception {
        setup(mode);
        try(var pool=Executors.newFixedThreadPool(8)) {
            List<Future<RequestRoute>> results=new ArrayList<>();
            for(int i=0;i<80;i++) results.add(pool.submit(()->select(7000,0,null)));
            int count=0; for(var result:results) if(result.get()!=null) count++;
            assertTrue(count>=24 && count<=27,"unexpected admitted count="+count);
            assertTrue(worker.ownership.tokenWorkload().plannedExecutionMs()<=2700);
        }
    }
    @ParameterizedTest @ValueSource(strings={"NON_BATCH","BATCH"})
    void sessionUsesOriginalWorkerUntilTimeBudgetFullThenOnlyIdleTarget(String mode) {
        setup(mode);
        var first=select(700,0,"session"); assertNotNull(first); first.rememberSessionWorker(); first.close();
        var target=publish("10.0.0.2");
        // Keep cache affinity on the original worker for the large query.
        when(cache.findMatchingEngines(any(),any(),any())).thenReturn(Map.of(worker.ipPort(),9));
        var longPlan=select(1_890_000,0,"session"); assertNotNull(longPlan);
        assertSame(worker,longPlan.prefillEp());
        when(cache.findMatchingEngines(any(),any(),any())).thenReturn(Map.of());
        assertNull(select(700,0,"session"));
        var rescued=select(700,1000,"session"); assertNotNull(rescued);
        assertSame(target,rescued.prefillEp()); assertEquals(0,rescued.hitCache());
        assertEquals(700,target.ownership.tokenWorkload().plannedTokens());
    }
    @ParameterizedTest @ValueSource(strings={"NON_BATCH","BATCH"})
    void longQueryCanUseIdleWorkerWhenCacheLeaderCannotMeetSlo(String mode) {
        setup(mode, 30000);
        var idleWorker = publish("10.0.0.2");
        when(cache.findMatchingEngines(any(), any(), any()))
                .thenReturn(Map.of(worker.ipPort(), 9));
        // Keep the cache leader busy for the entire 27s execution budget.
        assertNotNull(select(18_900_000, 0, null));
        // CSV-sized long input: a cold execution needs 1.77s at 70K tokens/s.
        // It has already waited 5s but still fits easily within the 30s target.
        assertNull(select(124086, 0, null), "cache affinity still waits before escaping");
        assertNull(select(124086, 26000, null), "cold execution must fit the remaining SLO budget");
        assertNull(select(2_000_000, 5000, null), "an idle worker cannot make oversized work feasible");
        var rescued = select(124086, 5000, null);
        assertNotNull(rescued, "an idle worker must not reject feasible work at the legacy 1s rescue cap");
        assertSame(idleWorker, rescued.prefillEp());
        assertEquals(0, rescued.hitCache());
        assertEquals(124086, idleWorker.ownership.tokenWorkload().plannedTokens());
        assertNull(select(124086, 5000, null), "rescue must not build another queue on the target");
    }

    @ParameterizedTest @ValueSource(strings={"NON_BATCH","BATCH"})
    void coldWorkWaitsBeforeItCanFillTheQueueAheadOfHotWork(String mode) {
        setup(mode,30000);
        when(cache.findMatchingEngines(any(),any(),any())).thenReturn(Map.of(worker.ipPort(),9));
        for(int i=0;i<3;i++) assertNotNull(select(700000,0,null));
        when(cache.findMatchingEngines(any(),any(),any())).thenReturn(Map.of());
        assertNull(select(70000,0,null));
        when(cache.findMatchingEngines(any(),any(),any())).thenReturn(Map.of(worker.ipPort(),9));
        assertNotNull(select(700000,0,null));
    }
    @ParameterizedTest @ValueSource(strings={"NON_BATCH","BATCH"})
    void lowPressureColdArrivalAndHigherPriorityColdWorkRemainAdmissible(String mode) {
        setup(mode,30000);
        when(cache.findMatchingEngines(any(),any(),any())).thenReturn(Map.of(worker.ipPort(),9));
        assertNotNull(select(70000,0,null));
        when(cache.findMatchingEngines(any(),any(),any())).thenReturn(Map.of());
        assertNotNull(select(7000,0,null));
        when(cache.findMatchingEngines(any(),any(),any())).thenReturn(Map.of(worker.ipPort(),9));
        assertNotNull(select(700000,0,null));
        when(cache.findMatchingEngines(any(),any(),any())).thenReturn(Map.of());
        assertNotNull(select(70000,0,null,90));
        held.forEach(RequestRoute::close);
        assertEquals(0,worker.ownership.cacheWorkAtOrAbove(0).inputTokens());
        assertNotNull(select(70000,0,null));
    }
}
