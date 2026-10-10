package org.flexlb.balance.eviction;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.scheduler.RequestRequirements;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.dao.master.WorkerStatus;
import static org.flexlb.balance.scheduler.SchedulingTestConfig.decodeRequirements;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRequestView;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.enums.DecodeTaskPhase;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import java.math.BigInteger;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Exact-value contracts for {@link EvictionPlanner#planDecode}: the frozen
 * decode admission recompute of {@code current − victims + incoming} for
 * slot and KV capacity.
 *
 * <p>Requirements under test:
 * <ul>
 *   <li>Slot deficit = max(0, engineLoad + 1 − concurrencyLimit). No deficit → no eviction.</li>
 *   <li>Both prompt supply and complete-output usage must fit the same KV budget as admission.</li>
 *   <li>Equal-priority entries are NEVER victims (strict &lt;).</li>
 *   <li>Greedy KV selection takes the largest-bucket release first; covers the
 *       deficit exactly or fails entirely—never a partial set.</li>
 *   <li>Cost = h × f(priority) × g(phase) [× lengthWaste for KV], with exact numeric values.</li>
 * </ul>
 */
@DisplayName("EvictionPlanner.planDecode recompute contracts")
class EvictionPlannerDecodeContractTest {

    @Test
    void negativeHarmPropagatesDiagnosticFormattingFailure() {
        var failure = new IllegalStateException("diagnostic formatting failed");
        BigInteger harm = new BigInteger("-1") {
            @Override
            public String toString() {
                throw failure;
            }
        };
        assertSame(failure, assertThrows(IllegalStateException.class,
                () -> PriorityHarmProfile.builder().add(50, harm)));
    }

    @Test
    void scalarDiagnosticCostUsesExactHarmAndSaturatesAtLongBoundary() {
        int[] priorities = {1, 30, 39, 40, 50, 60, 70, 100};
        long[] factors = {1L, 1L, 1L, 1_024L, 1_048_576L, 1_073_741_824L,
                1_099_511_627_776L, 1_099_511_627_776L};
        BigInteger maximum = BigInteger.valueOf(Long.MAX_VALUE);
        for (int i = 0; i < priorities.length; i++) {
            BigInteger factor = BigInteger.valueOf(factors[i]);
            BigInteger boundary = maximum.divide(factor);
            for (BigInteger harm : List.of(BigInteger.ZERO, BigInteger.ONE, boundary,
                    boundary.add(BigInteger.ONE), maximum, maximum.multiply(maximum))) {
                var profile = PriorityHarmProfile.builder().add(priorities[i], harm).build();
                assertEquals(harm.multiply(factor).min(maximum).longValueExact(), profile.totalCost());
            }
        }
        var lower = PriorityHarmProfile.builder().add(30, BigInteger.valueOf(5)).build();
        var higher = PriorityHarmProfile.builder().add(40, BigInteger.valueOf(3)).build();
        assertEquals(3_077L, lower.plus(higher).totalCost());
        assertEquals(5L, lower.totalCost());
        assertEquals(0L, PriorityHarmProfile.empty().totalCost());
        var saturated = PriorityHarmProfile.builder().add(30, maximum).build();
        assertEquals(Long.MAX_VALUE, saturated.plus(higher).totalCost());
        assertTrue(saturated.compareTo(higher) < 0, "scalar saturation must not decide exact priority ordering");
    }

    private static final long H_SLOT = 4L;
    private static final long H_KV = 8L;
    private static org.flexlb.config.PreemptionConfig engineOwned() {
        PreemptionConfig p = new PreemptionConfig();
        p.setAllowedVictimStages(EnumSet.of(VictimStage.DECODE_ENGINE_OWNED));
        return p;
    }

    private static DecodeRequestView accepted(long id, int priority, long kvTokens) {
        return new DecodeRequestView(
                id, priority, kvTokens, kvTokens,
                DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                true, 0L, false);
    }

    private static EndpointFixture endpoint(
            long realKvAvailable, long realKvTotal,
            int engineLoad, long concurrencyLimit,
            List<DecodeRequestView> accepted) {
        var usage = new DecodeResources.CapacityUsage(engineLoad, realKvTotal, realKvAvailable,
                0L, Math.max(0L, realKvTotal - realKvAvailable));
        WorkerStatus status = WorkerStatus.createDiscovered(
                org.flexlb.dao.route.RoleType.DECODE, "default", "decode-a", 8080, 8081, "test");
        var routing = new DecodeResources.DecodeRoutingView("decode-a", status.getGenerationId(),
                status.topologySnapshot(), status.committedWorkerStatus(), 0L, engineLoad, engineLoad,
                usage, usage, 0L);
        var requests = accepted.stream().collect(java.util.stream.Collectors.toMap(
                DecodeRequestView::requestId, java.util.function.Function.identity()));
        return new EndpointFixture(new DecodeResources.AdmissionCapacity(concurrencyLimit, 100L),
                new DecodeResources.ResourceSnapshot(routing, requests, 0, 0));
    }

    private record EndpointFixture(DecodeResources.AdmissionCapacity capacity,
                                   DecodeResources.ResourceSnapshot snapshot) { }

    private static DecodeEvictionProposal planDecode(RequestRequirements request,
            DecodeResources.ResourceSnapshot snapshot, PreemptionConfig policy, Map<String, String> failures) {
        return EvictionPlanner.planDecode(request, snapshot, policy, failures).proposal();
    }

    private static DecodeEvictionProposal plan(
            int priority, long hardKvTokens, EndpointFixture ep,
            Map<String, String> failures) {
        return planDecode(
                decodeRequirements(priority, hardKvTokens, hardKvTokens, ep.capacity()), ep.snapshot(), engineOwned(), failures);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void victimStagePolicyGatesEngineVictimsOnly(boolean engineOwnedAllowed) {
        var engine = endpoint(1_000L, 1_000L, 1, 1, List.of(accepted(1L, 10, 100L)));
        var enginePolicy = new PreemptionConfig();
        enginePolicy.setAllowedVictimStages(engineOwnedAllowed
                ? EnumSet.of(VictimStage.DECODE_ENGINE_OWNED) : EnumSet.noneOf(VictimStage.class));
        var proposal = planDecode(decodeRequirements(70, 0L, 0L, engine.capacity()), engine.snapshot(),
                enginePolicy, new HashMap<>());
        if (engineOwnedAllowed) {
            assertEquals(List.of(1L), victimIds(proposal));
        } else {
            assertNull(proposal);
        }
        var local = new DecodeRequestView(2L, 10, 100L, 100L,
                DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED, true, 1L, false);
        var policy = new PreemptionConfig();
        policy.setAllowedVictimStages(EnumSet.of(VictimStage.DECODE_RESERVED));
        var localEndpoint = endpoint(1_000L, 1_000L, 1, 1, List.of(local));
        var localProposal = planDecode(decodeRequirements(70, 0L, 0L, localEndpoint.capacity()), localEndpoint.snapshot(),
                policy, new HashMap<>());
        assertNotNull(localProposal);
        assertEquals(List.of(2L), victimIds(localProposal));
    }

    private static List<Long> victimIds(DecodeEvictionProposal p) {
        return p.victims().stream().map(DecodeRequestView::requestId).toList();
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void combinedDeficitSelectsBothDimensionsWithoutRepeatingVictims(boolean kvFirst) {
        List<DecodeRequestView> candidates = kvFirst
                ? List.of(accepted(1L, 30, 2_000L), accepted(2L, 30, 100L),
                        accepted(3L, 30, 100L), accepted(4L, 30, 100L), accepted(5L, 30, 2_000L))
                : List.of(accepted(1L, 30, 100L), accepted(2L, 30, 100L), accepted(3L, 30, 2_000L));
        EndpointFixture ep = endpoint(0L, 10_000L,
                kvFirst ? 102 : 3, kvFirst ? 100 : 2, candidates);
        long demand = kvFirst ? 4_000L : 1_200L;
        DecodeEvictionProposal proposal = plan(70, demand, ep, new HashMap<>());

        assertEquals(DecodeEvictionProposal.CASE_SLOT_AND_KV, proposal.evictionCase());
        assertEquals(kvFirst ? List.of(1L, 5L, 2L) : List.of(1L, 2L, 3L), victimIds(proposal));
        long cost = kvFirst ? 320L : 256L;
        assertEquals(cost, proposal.priorityHarmProfile().totalCost());
        assertEquals(PriorityHarmProfile.builder().add(30, BigInteger.valueOf(cost)).build(),
                proposal.priorityHarmProfile());
        DecodeResources.CapacityRelease release = DecodeResources.CapacityRelease.NONE;
        for (DecodeRequestView victim : proposal.victims()) {
            release = release.plus(victim.placementRelease());
        }
        assertEquals(kvFirst ? 4_100L : 2_200L, proposal.freedKvTokens());
        assertTrue(ep.capacity().evaluate(ep.snapshot().routing().placementUsage(), demand, demand, release).fits());
    }

    // ─── No deficit ─────────────────────────────────────────────────────

    @Nested
    @DisplayName("No deficit")
    class NoDef {

        @Test
        void sufficientCapacityReportsNoDeficit() {
            // slotDeficit = max(0, 2+1-4) = 0; kvDeficit = 200 < 500 → 0.
            Map<String, String> f = new HashMap<>();
            var ep = endpoint(500L, 1000L, 2, 4L, List.of());
            var decision = EvictionPlanner.planDecode(
                    decodeRequirements(70, 200L, 200L, ep.capacity()), ep.snapshot(), engineOwned(), f);
            assertTrue(decision.deficit().fits());
            assertNull(decision.evictionCase());
            assertNull(decision.proposal());
            assertEquals("decode_capacity_sufficient", f.get("decode-a"));
        }
    }

    // ─── Slot deficit ───────────────────────────────────────────────────

    @Nested
    @DisplayName("Slot deficit recompute")
    class SlotDef {

        @Test
        void oneSlotDeficitEvictsExactlyOneLowerPriorityAccepted() {
            // slotDeficit = max(0, 1+1-1) = 1; kvDeficit = 0 (realKvTotal=0).
            Map<String, String> f = new HashMap<>();
            DecodeEvictionProposal p = plan(70, 0L,
                    endpoint(1000L, 0L, 1, 1L, List.of(accepted(1L, 30, 128L))), f);
            assertEquals(List.of(1L), victimIds(p));
            assertEquals(DecodeEvictionProposal.CASE_SLOT, p.evictionCase());
            // cost = H_SLOT * f(30) * g(ACCEPTED) = 4 * 1 * 16 = 64
            assertEquals(64L, p.priorityHarmProfile().totalCost());
            assertEquals(128L, p.freedKvTokens());
        }

        @Test
        void equalPriorityCandidateNeverYields() {
            Map<String, String> f = new HashMap<>();
            var ep = endpoint(1000L, 0L, 1, 1L, List.of(accepted(1L, 70, 128L)));
            var decision = EvictionPlanner.planDecode(
                    decodeRequirements(70, 0L, 0L, ep.capacity()), ep.snapshot(), engineOwned(), f);
            assertFalse(decision.deficit().fits());
            assertEquals(DecodeEvictionProposal.CASE_SLOT, decision.evictionCase());
            assertNull(decision.proposal());
            assertEquals("insufficient_lower_priority_candidates", f.get("decode-a"));
        }

        @Test
        void slotEvictionCanAlsoSatisfyKvDeficit() {
            var ep = endpoint(100L, 1000L, 1, 1L, List.of(accepted(1L, 30, 512L)));
            var decision = EvictionPlanner.planDecode(
                    decodeRequirements(70, 300L, 300L, ep.capacity()), ep.snapshot(), engineOwned(), new HashMap<>());
            assertEquals(DecodeEvictionProposal.CASE_SLOT_AND_KV, decision.evictionCase());
            assertEquals(DecodeEvictionProposal.CASE_SLOT, decision.proposal().evictionCase());
            assertEquals(List.of(1L), victimIds(decision.proposal()));
        }

        @Test
        void noPriorityCandidateNeverYields() {
            Map<String, String> f = new HashMap<>();
            assertNull(plan(70, 0L,
                    endpoint(1000L, 0L, 1, 1L, List.of(accepted(1L, 0, 128L))), f));
            assertEquals("insufficient_lower_priority_candidates", f.get("decode-a"));
        }
    }

    // ─── KV deficit ─────────────────────────────────────────────────────

    @Nested
    @DisplayName("KV deficit recompute")
    class KvDef {

        @Test
        void greedyKvSelectionTakesTheLargestReleaseFirst() {
            // limit=0→slotDeficit=0. kvDeficit = 300 - 100 = 200.
            // Two victims: kv2048(bucket2) and kv512(bucket1). Largest bucket first;
            // 2048 alone covers 200 → sole victim.
            Map<String, String> f = new HashMap<>();
            DecodeEvictionProposal p = plan(70, 300L,
                    endpoint(100L, 1000L, 0, 0L,
                            List.of(accepted(1L, 30, 2048L), accepted(2L, 30, 512L))), f);
            assertEquals(List.of(1L), victimIds(p));
            assertEquals(DecodeEvictionProposal.CASE_KV, p.evictionCase());
            // cost = H_KV * f(30) * g(ACCEPTED) * lengthWasteCost(2048)
            //      = 8 * 1 * 16 * round(sqrt(ceil(2048/1024))) = 8*16*round(sqrt(2))
            //      = 8 * 16 * 1 = 128  (sqrt(2)=1.41→round=1)
            assertEquals(128L, p.priorityHarmProfile().totalCost());
            assertEquals(2048L, p.freedKvTokens());
        }

        @Test
        void kvGreedyExactlyMeetsDeficitWithTwoVictims() {
            // kvDeficit = 700 - 100 = 600. Victims: kv512 + kv512 = 1024 >= 600.
            // But greedy takes one by one. First 512 < 600, so adds second.
            Map<String, String> f = new HashMap<>();
            DecodeEvictionProposal p = plan(70, 700L,
                    endpoint(100L, 1000L, 0, 0L,
                            List.of(accepted(1L, 30, 512L), accepted(2L, 30, 512L))), f);
            assertEquals(2, p.victims().size());
            assertEquals(1024L, p.freedKvTokens());
        }

        @Test
        void insufficientReleasableKvIsInfeasible() {
            // kvDeficit = 100000 - 100 = 99900. Only 128 releasable.
            Map<String, String> f = new HashMap<>();
            assertNull(plan(70, 100_000L,
                    endpoint(100L, 1000L, 0, 0L, List.of(accepted(1L, 30, 128L))), f));
            assertEquals("insufficient_releasable_kv", f.get("decode-a"));
        }
    }
}
