package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;
import java.util.OptionalLong;

/**
 * RejectOutliers leave-one-out contract tests. The rejectOutliers pipeline
 * lives behind private CandidateSet internals (no upstream select-level
 * prefill test scaffolding exists yet), so these tests build a CandidateSet
 * via reflection and invoke the private filter directly. The candidate
 * values are public records, so only the set plumbing needs reflection.
 */
class CostBasedPrefillStrategyTest {

    private final CostBasedPrefillStrategy strategy =
            new CostBasedPrefillStrategy(
                    Mockito.mock(EngineWorkerStatus.class),
                    Mockito.mock(CacheAwareService.class),
                    Mockito.mock(ResourceMeasureFactory.class),
                    Mockito.mock(EngineHealthReporter.class));

    @AfterEach
    void tearDown() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
    }

    @Test
    void imbalanceFilterUsesOthersAverageExcludingSelf() throws Exception {
        // Two modeled engines, pendingCount 0 on both keeps the hotspot axis
        // inert (zero others-average exemption) so the test isolates the
        // drain-imbalance axis. With the old self-inclusive average the
        // threshold was 3.0 * (350+100)/2 = 675 — drain 350 can mathematically
        // never exceed it, so the old formula provably never filtered this
        // fleet. The leave-one-out baseline (others avg = 100, threshold =
        // 300) must reject the busy engine: 350 > 300.
        CostBasedPrefillStrategy.CandidateSet feasible = new CostBasedPrefillStrategy.CandidateSet();
        addCandidate(feasible, "10.0.0.1:8080", modeled(350L, 0L));
        addCandidate(feasible, "10.0.0.2:8080", modeled(100L, 0L));

        Map<String, Integer> rejections = new HashMap<>();
        Object result = rejectOutliers(feasible, new FlexlbConfig(), rejections);

        CostBasedPrefillStrategy.CandidateSet survivors = survivors(result);
        Assertions.assertEquals(1, survivors.size(),
                "drain=350ms vs others-avg=100ms must be outlier-rejected");
        Assertions.assertEquals("10.0.0.2:8080", survivors.endpointAddress(0));
        Assertions.assertEquals(1, rejections.get("IMBALANCE_FILTERED"));
        survivors.close();
    }

    @Test
    void hotspotFilterUsesOthersAverageExcludingSelf() throws Exception {
        // Drain projections absent on both engines keep the imbalance axis
        // inert; pendingCount [4, 1] mirrors the decode hotspot case. Old
        // self-inclusive threshold: 3.0 * (4+1)/2 = 7.5 — 4 can never exceed
        // it. Leave-one-out others avg = 1, threshold = 3, 4 > 3 — filtered.
        CostBasedPrefillStrategy.CandidateSet feasible = new CostBasedPrefillStrategy.CandidateSet();
        addCandidate(feasible, "10.0.0.1:8080", modeled(null, 4L));
        addCandidate(feasible, "10.0.0.2:8080", modeled(null, 1L));

        Map<String, Integer> rejections = new HashMap<>();
        Object result = rejectOutliers(feasible, new FlexlbConfig(), rejections);

        CostBasedPrefillStrategy.CandidateSet survivors = survivors(result);
        Assertions.assertEquals(1, survivors.size(),
                "pending=4 vs others-avg=1 must be outlier-rejected");
        Assertions.assertEquals("10.0.0.2:8080", survivors.endpointAddress(0));
        Assertions.assertEquals(1, rejections.get("HOTSPOT_FILTERED"));
        survivors.close();
    }

    @Test
    void singleCandidateSkipsOutlierRejection() throws Exception {
        // A lone engine with extreme drain and pending count: there are no
        // "other" engines to be an outlier against, so the relative outlier
        // checks must be skipped — rejecting the only candidate could only
        // yield NO_AVAILABLE_WORKER with nothing to gain (the least-loaded
        // fallback would rescue it anyway, so it must stay in the set).
        CostBasedPrefillStrategy.CandidateSet feasible = new CostBasedPrefillStrategy.CandidateSet();
        addCandidate(feasible, "10.0.0.1:8080", modeled(100_000L, 100_000L));

        Map<String, Integer> rejections = new HashMap<>();
        Object result = rejectOutliers(feasible, new FlexlbConfig(), rejections);

        CostBasedPrefillStrategy.CandidateSet survivors = survivors(result);
        Assertions.assertEquals(1, survivors.size(),
                "a lone engine has no 'others' to be an outlier against and must survive");
        Assertions.assertEquals("10.0.0.1:8080", survivors.endpointAddress(0));
        Assertions.assertTrue(rejections.isEmpty());
        survivors.close();
    }

    /** Modeled projection with the given drain (empty = absent) and pending count. */
    private static RouteProjection.Candidate modeled(
            Long projectedDrainMs, long pendingCount) {
        RouteProjection.Result projection = new RouteProjection.Result(
                RouteProjection.Result.State.MODELED,
                OptionalLong.of(5_000L),
                projectedDrainMs == null
                        ? OptionalLong.empty() : OptionalLong.of(projectedDrainMs),
                5_000L,
                RouteProjection.Result.InitialHeadDisposition.NONE,
                "");
        return new RouteProjection.Candidate(
                projection, 0L, 0L, OptionalLong.of(pendingCount));
    }

    private static void addCandidate(
            CostBasedPrefillStrategy.CandidateSet set,
            String endpointAddress,
            RouteProjection.Candidate candidate) throws Exception {
        WorkerEndpoint.GenerationPin pin =
                Mockito.mock(WorkerEndpoint.GenerationPin.class);
        Method add = CostBasedPrefillStrategy.CandidateSet.class.getDeclaredMethod(
                "addCandidate",
                String.class,
                WorkerEndpoint.GenerationPin.class,
                RouteProjection.Candidate.class);
        add.setAccessible(true);
        add.invoke(set, endpointAddress, pin, candidate);
    }

    private Object rejectOutliers(
            CostBasedPrefillStrategy.CandidateSet feasible,
            FlexlbConfig config,
            Map<String, Integer> rejections) throws Exception {
        Method reject = CostBasedPrefillStrategy.class.getDeclaredMethod(
                "rejectOutliers",
                CostBasedPrefillStrategy.CandidateSet.class,
                FlexlbConfig.class,
                Map.class);
        reject.setAccessible(true);
        return reject.invoke(strategy, feasible, config, rejections);
    }

    private static CostBasedPrefillStrategy.CandidateSet survivors(Object filterResult) throws Exception {
        Field candidates = filterResult.getClass().getDeclaredField("candidates");
        candidates.setAccessible(true);
        return (CostBasedPrefillStrategy.CandidateSet) candidates.get(filterResult);
    }
}
