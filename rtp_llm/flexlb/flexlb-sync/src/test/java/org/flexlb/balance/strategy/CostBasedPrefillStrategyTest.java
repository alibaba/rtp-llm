package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.Map;
import java.util.OptionalLong;

/**
 * RejectOutliers upstream-contract tests. The rejectOutliers pipeline
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
    void singleCandidateSkipsOutlierRejection() throws Exception {
        // A lone engine with extreme drain and pending count: with the
        // upstream self-inclusive average the average IS the engine's own
        // value, so own > multiplier * avg can never hold — the engine must
        // stay in the survivor set.
        CostBasedPrefillStrategy.CandidateSet feasible = new CostBasedPrefillStrategy.CandidateSet();
        addCandidate(feasible, "10.0.0.1:8080", modeled(100_000L, 100_000L));

        Map<String, Integer> rejections = new HashMap<>();
        Object result = rejectOutliers(feasible, new FlexlbConfig(), rejections);

        CostBasedPrefillStrategy.CandidateSet survivors = survivors(result);
        Assertions.assertEquals(1, survivors.size(),
                "a lone engine's self-inclusive average equals its own value, so it must survive");
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

    /**
     * Invokes the private rejectOutliers filter. Upstream's schedule refactor
     * widened the signature with a pool-wide blocker map
     * (Map&lt;RoleType, Integer&gt;); a fresh empty EnumMap mirrors the production
     * call site in evaluateCandidates.
     */
    private Object rejectOutliers(
            CostBasedPrefillStrategy.CandidateSet feasible,
            FlexlbConfig config,
            Map<String, Integer> rejections) throws Exception {
        Method reject = CostBasedPrefillStrategy.class.getDeclaredMethod(
                "rejectOutliers",
                CostBasedPrefillStrategy.CandidateSet.class,
                FlexlbConfig.class,
                Map.class,
                Map.class);
        reject.setAccessible(true);
        return reject.invoke(strategy, feasible, config, rejections,
                new EnumMap<>(RoleType.class));
    }

    private static CostBasedPrefillStrategy.CandidateSet survivors(Object filterResult) throws Exception {
        Field candidates = filterResult.getClass().getDeclaredField("candidates");
        candidates.setAccessible(true);
        return (CostBasedPrefillStrategy.CandidateSet) candidates.get(filterResult);
    }
}
