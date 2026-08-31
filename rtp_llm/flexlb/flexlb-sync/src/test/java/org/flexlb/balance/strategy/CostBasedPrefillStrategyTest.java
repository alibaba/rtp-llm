package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.util.List;

/**
 * RejectOutliers upstream-contract tests. The rejectOutliers pipeline lives
 * behind private {@code Candidate} internals (no upstream select-level
 * prefill test scaffolding exists yet), so these tests build the private
 * candidate list via reflection and invoke the private static filter
 * directly. The filter only reads {@code pendingCount} and
 * {@code committedWaitMs}, so the pin/endpoint constructor slots can stay
 * {@code null}.
 */
class CostBasedPrefillStrategyTest {

    @Test
    void singleCandidateSkipsOutlierRejection() throws Exception {
        // A lone engine with extreme drain and pending count. Two guards
        // keep it selectable: the size &lt;= 1 early return, and past that
        // the self-inclusive average — the average IS the engine's own
        // value, so own &gt; multiplier * avg can never hold. A lone engine
        // must stay in the survivor set either way.
        List<Object> feasible = List.of(candidate(100_000L, 100_000L));

        Object survivors = rejectOutliers(feasible, new FlexlbConfig());

        Assertions.assertEquals(1, ((List<?>) survivors).size(),
                "a lone engine's self-inclusive average equals its own value, "
                        + "so it must survive");
    }

    /**
     * Builds one private {@code Candidate} via reflection. Constructor slot
     * order: pin, endpoint, scoreMs, prefillMs, cacheHit,
     * routingCacheMatchTokens, committedWaitMs, pendingCount,
     * lastSelectedTime.
     */
    private static Object candidate(
            long committedWaitMs, long pendingCount) throws Exception {
        Class<?> candidateClass = Class.forName(
                "org.flexlb.balance.strategy.CostBasedPrefillStrategy$Candidate");
        Constructor<?> constructor = candidateClass.getDeclaredConstructor(
                WorkerEndpoint.GenerationPin.class,
                PrefillEndpoint.class,
                long.class, long.class, long.class, long.class,
                long.class, long.class, long.class);
        constructor.setAccessible(true);
        return constructor.newInstance(
                null, null, 0L, 0L, 0L, 0L,
                committedWaitMs, pendingCount, 0L);
    }

    /**
     * Invokes the private static rejectOutliers filter. A default
     * {@code FlexlbConfig} chains to
     * EstimatedTtftSelectorConfig/RandomWithinToleranceConfig whose
     * outlierRejection defaults to 3.0 multipliers, so the filter logic
     * (not just the early returns) executes — mirroring the production
     * call site in evaluateCandidates.
     */
    private static Object rejectOutliers(
            List<Object> candidates, FlexlbConfig config) throws Exception {
        Method reject = CostBasedPrefillStrategy.class.getDeclaredMethod(
                "rejectOutliers", List.class, FlexlbConfig.class);
        reject.setAccessible(true);
        return reject.invoke(null, candidates, config);
    }
}
