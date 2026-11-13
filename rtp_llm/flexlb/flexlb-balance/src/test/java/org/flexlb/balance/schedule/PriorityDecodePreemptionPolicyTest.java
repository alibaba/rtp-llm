package org.flexlb.balance.schedule;

import org.flexlb.balance.schedule.DecodePreemptionPolicy.DecodePreemptionPlan;
import org.flexlb.balance.schedule.PriorityDecodePreemptionPolicy.PriorityHarmProfile;
import org.flexlb.config.scheduler.PreemptionConfig;
import org.flexlb.config.scheduler.PreemptionStage;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.DecodeResources;
import org.flexlb.worker.endpoint.DecodeResources.DecodeRequestView;
import org.flexlb.worker.endpoint.Request;
import org.flexlb.worker.status.DecodeTaskPhase;
import org.flexlb.worker.status.WorkerStatus;
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
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Exact-value contracts for {@link PriorityDecodePreemptionPolicy#planDecode}: the frozen
 * decode admission recompute of {@code current − requests selected for preemption + incoming} for
 * slot and KV capacity.
 *
 * <p>Requirements under test:
 * <ul>
 *   <li>Slot deficit = max(0, engineLoad + 1 − concurrencyLimit). No deficit → no preemption.</li>
 *   <li>Both prompt supply and complete-output usage must fit the same KV budget as admission.</li>
 *   <li>Equal-priority entries are NEVER requests selected for preemption (strict &lt;).</li>
 *   <li>Greedy KV selection takes the largest-bucket release first; covers the
 *       deficit exactly or fails entirely—never a partial set.</li>
 *   <li>Cost = h × f(priority) × g(phase) [× lengthWaste for KV], with exact numeric values.</li>
 * </ul>
 */
@DisplayName("PriorityDecodePreemptionPolicy.planDecode recompute contracts")
class PriorityDecodePreemptionPolicyTest {
    private static final long H_SLOT = 4L;
    private static final long H_KV = 8L;

    @Test
    void negativeHarmPropagatesDiagnosticFormattingFailure() {
        var failure = new IllegalStateException("diagnostic formatting failed");
        BigInteger harm =
                new BigInteger("-1") {
                    @Override
                    public String toString() {
                        throw failure;
                    }
                };
        assertSame(
                failure,
                assertThrows(
                        IllegalStateException.class,
                        () -> PriorityHarmProfile.builder().add(50, harm)));
    }

    @Test
    void scalarDiagnosticCostUsesExactHarmAndSaturatesAtLongBoundary() {
        int[] priorities = {1, 30, 39, 40, 50, 60, 70, 100};
        long[] factors = {
            1L, 1L, 1L, 1_024L, 1_048_576L, 1_073_741_824L, 1_099_511_627_776L, 1_099_511_627_776L
        };
        BigInteger maximum = BigInteger.valueOf(Long.MAX_VALUE);
        for (int i = 0; i < priorities.length; i++) {
            BigInteger factor = BigInteger.valueOf(factors[i]);
            BigInteger boundary = maximum.divide(factor);
            for (BigInteger harm :
                    List.of(
                            BigInteger.ZERO,
                            BigInteger.ONE,
                            boundary,
                            boundary.add(BigInteger.ONE),
                            maximum,
                            maximum.multiply(maximum))) {
                var profile = PriorityHarmProfile.builder().add(priorities[i], harm).build();
                assertEquals(
                        harm.multiply(factor).min(maximum).longValueExact(), profile.totalCost());
            }
        }
        var lower = PriorityHarmProfile.builder().add(30, BigInteger.valueOf(5)).build();
        var higher = PriorityHarmProfile.builder().add(40, BigInteger.valueOf(3)).build();
        assertEquals(3_077L, lower.plus(higher).totalCost());
        assertEquals(5L, lower.totalCost());
        assertEquals(0L, PriorityHarmProfile.empty().totalCost());
        var saturated = PriorityHarmProfile.builder().add(30, maximum).build();
        assertEquals(Long.MAX_VALUE, saturated.plus(higher).totalCost());
        assertTrue(
                saturated.compareTo(higher) < 0,
                "scalar saturation must not decide exact priority ordering");
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void combinedDeficitSelectsBothDimensionsWithoutRepeatingVictims(boolean kvFirst) {
        List<DecodeRequestView> candidates =
                kvFirst
                        ? List.of(
                                accepted(1L, 30, 2_000L),
                                accepted(2L, 30, 100L),
                                accepted(3L, 30, 100L),
                                accepted(4L, 30, 100L),
                                accepted(5L, 30, 2_000L))
                        : List.of(
                                accepted(1L, 30, 100L),
                                accepted(2L, 30, 100L),
                                accepted(3L, 30, 2_000L));
        EndpointFixture ep =
                endpoint(0L, 10_000L, kvFirst ? 102 : 3, kvFirst ? 100 : 2, candidates);
        long demand = kvFirst ? 4_000L : 1_200L;
        DecodePreemptionPlan proposal = plan(70, demand, ep, new HashMap<>());

        assertEquals(DecodePreemptionPlan.CASE_SLOT_AND_KV, proposal.preemptionReason());
        assertEquals(
                kvFirst ? List.of(1L, 5L, 2L) : List.of(1L, 2L, 3L), requestIdsToPreempt(proposal));
        long cost = kvFirst ? 320L : 256L;
        assertEquals(cost, proposal.diagnosticCost());
        DecodeResources.Capacity release = DecodeResources.Capacity.NONE;
        for (DecodeRequestView requestToPreempt : proposal.requestsToPreempt()) {
            release = release.plus(requestToPreempt.placementRelease());
        }
        assertEquals(kvFirst ? 4_100L : 2_200L, proposal.freedKvTokens());
        assertTrue(
                ep.capacity()
                        .evaluate(ep.snapshot().selection().usage(), demand, demand, release)
                        .isZero());
    }

    private static DecodeRequestView accepted(long id, int priority, long kvTokens) {
        return new DecodeRequestView(
                String.valueOf(id),
                priority,
                kvTokens,
                kvTokens,
                DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                new DecodeEndpoint.RequestRegistration(1L, String.valueOf(id)));
    }

    private static EndpointFixture endpoint(
            long realKvAvailable,
            long realKvTotal,
            int engineLoad,
            long concurrencyLimit,
            List<DecodeRequestView> accepted) {
        var usage =
                new DecodeResources.CapacityUsage(
                        engineLoad,
                        realKvTotal,
                        realKvAvailable,
                        0L,
                        Math.max(0L, realKvTotal - realKvAvailable));
        WorkerStatus status =
                WorkerStatus.createDiscovered(
                        org.flexlb.config.service.RoleType.DECODE,
                        "default",
                        "decode-a",
                        8080,
                        8081,
                        "test");
        var routing =
                new DecodeEndpoint.SelectionSnapshot(
                        "decode-a",
                        status.getGenerationId(),
                        status.topologySnapshot(),
                        status.committedWorkerStatus(),
                        0L,
                        usage);
        var requests =
                accepted.stream()
                        .collect(
                                java.util.stream.Collectors.toMap(
                                        DecodeRequestView::requestId,
                                        java.util.function.Function.identity()));
        return new EndpointFixture(
                new DecodeResources.AdmissionCapacity(concurrencyLimit, 100L),
                new DecodeResources.ResourceSnapshot(routing, requests));
    }

    private static DecodePreemptionPlan plan(
            int priority, long hardKvTokens, EndpointFixture ep, Map<String, String> failures) {
        return planDecode(
                Request.builder().requestId("99").seqLen(hardKvTokens).maxNewTokens(0).build(),
                priority,
                ep.capacity(),
                ep.snapshot(),
                engineOwned(),
                failures);
    }

    private static DecodePreemptionPlan planDecode(
            Request request,
            int priority,
            DecodeResources.AdmissionCapacity capacity,
            DecodeResources.ResourceSnapshot snapshot,
            PreemptionConfig policy,
            Map<String, String> failures) {
        return new PriorityDecodePreemptionPolicy()
                .planDecode(request, priority, capacity, snapshot, policy, failures)
                .proposal();
    }

    private static org.flexlb.config.scheduler.PreemptionConfig engineOwned() {
        PreemptionConfig p = new PreemptionConfig();
        p.setAllowedPreemptionStages(EnumSet.of(PreemptionStage.DECODE_ENGINE_OWNED));
        return p;
    }

    private static List<Long> requestIdsToPreempt(DecodePreemptionPlan p) {
        return p.requestsToPreempt().stream()
                .map(view -> Long.parseLong(view.requestId()))
                .toList();
    }

    private record EndpointFixture(
            DecodeResources.AdmissionCapacity capacity,
            DecodeResources.ResourceSnapshot snapshot) {}

    // ─── No deficit ─────────────────────────────────────────────────────

    /**
     * Checks that sufficient capacity requires no preemption.
     */
    @Nested
    @DisplayName("No deficit")
    class NoDef {

        @Test
        void sufficientCapacityReportsNoDeficit() {
            // slotDeficit = max(0, 2+1-4) = 0; kvDeficit = 200 < 500 → 0.
            Map<String, String> f = new HashMap<>();
            var ep = endpoint(500L, 1000L, 2, 4L, List.of());
            var decision =
                    new PriorityDecodePreemptionPolicy()
                            .planDecode(
                                    Request.builder()
                                            .requestId("99")
                                            .seqLen(200L)
                                            .maxNewTokens(0)
                                            .build(),
                                    70,
                                    ep.capacity(),
                                    ep.snapshot(),
                                    engineOwned(),
                                    f);
            assertTrue(decision.deficit().isZero());
            assertNull(decision.preemptionReason());
            assertNull(decision.proposal());
            assertEquals("decode_capacity_sufficient", f.get("decode-a"));
        }
    }

    // ─── Slot deficit ───────────────────────────────────────────────────

    /**
     * Checks priority selection when concurrent request capacity is insufficient.
     */
    @Nested
    @DisplayName("Slot deficit recompute")
    class SlotDef {

        @Test
        void oneSlotDeficitEvictsExactlyOneLowerPriorityAccepted() {
            // slotDeficit = max(0, 1+1-1) = 1; kvDeficit = 0 (realKvTotal=0).
            Map<String, String> f = new HashMap<>();
            DecodePreemptionPlan p =
                    plan(70, 0L, endpoint(1000L, 0L, 1, 1L, List.of(accepted(1L, 30, 128L))), f);
            assertEquals(List.of(1L), requestIdsToPreempt(p));
            assertEquals(DecodePreemptionPlan.CASE_SLOT, p.preemptionReason());
            // cost = H_SLOT * f(30) * g(ACCEPTED) = 4 * 1 * 16 = 64
            assertEquals(64L, p.diagnosticCost());
            assertEquals(128L, p.freedKvTokens());
        }

        @Test
        void equalPriorityCandidateNeverYields() {
            Map<String, String> f = new HashMap<>();
            var ep = endpoint(1000L, 0L, 1, 1L, List.of(accepted(1L, 70, 128L)));
            var decision =
                    new PriorityDecodePreemptionPolicy()
                            .planDecode(
                                    Request.builder()
                                            .requestId("99")
                                            .seqLen(0L)
                                            .maxNewTokens(0)
                                            .build(),
                                    70,
                                    ep.capacity(),
                                    ep.snapshot(),
                                    engineOwned(),
                                    f);
            assertFalse(decision.deficit().isZero());
            assertEquals(DecodePreemptionPlan.CASE_SLOT, decision.preemptionReason());
            assertNull(decision.proposal());
            assertEquals("insufficient_lower_priority_candidates", f.get("decode-a"));
        }

        @Test
        void slotEvictionCanAlsoSatisfyKvDeficit() {
            var ep = endpoint(100L, 1000L, 1, 1L, List.of(accepted(1L, 30, 512L)));
            var decision =
                    new PriorityDecodePreemptionPolicy()
                            .planDecode(
                                    Request.builder()
                                            .requestId("99")
                                            .seqLen(300L)
                                            .maxNewTokens(0)
                                            .build(),
                                    70,
                                    ep.capacity(),
                                    ep.snapshot(),
                                    engineOwned(),
                                    new HashMap<>());
            assertEquals(DecodePreemptionPlan.CASE_SLOT_AND_KV, decision.preemptionReason());
            assertEquals(DecodePreemptionPlan.CASE_SLOT, decision.proposal().preemptionReason());
            assertEquals(List.of(1L), requestIdsToPreempt(decision.proposal()));
        }

        @Test
        void noPriorityCandidateNeverYields() {
            Map<String, String> f = new HashMap<>();
            assertNull(plan(70, 0L, endpoint(1000L, 0L, 1, 1L, List.of(accepted(1L, 0, 128L))), f));
            assertEquals("insufficient_lower_priority_candidates", f.get("decode-a"));
        }
    }

    // ─── KV deficit ─────────────────────────────────────────────────────

    /**
     * Checks which requests release enough KV for the incoming request.
     */
    @Nested
    @DisplayName("KV deficit recompute")
    class KvDef {

        @Test
        void greedyKvSelectionTakesTheLargestReleaseFirst() {
            // limit=0→slotDeficit=0. kvDeficit = 300 - 100 = 200.
            // Two requests selected for preemption: kv2048(bucket2) and kv512(bucket1). Largest
            // bucket first;
            // 2048 alone covers 200 → sole request selected for preemption.
            Map<String, String> f = new HashMap<>();
            DecodePreemptionPlan p =
                    plan(
                            70,
                            300L,
                            endpoint(
                                    100L,
                                    1000L,
                                    0,
                                    0L,
                                    List.of(accepted(1L, 30, 2048L), accepted(2L, 30, 512L))),
                            f);
            assertEquals(List.of(1L), requestIdsToPreempt(p));
            assertEquals(DecodePreemptionPlan.CASE_KV, p.preemptionReason());
            // cost = H_KV * f(30) * g(ACCEPTED) * lengthWasteCost(2048)
            //      = 8 * 1 * 16 * round(sqrt(ceil(2048/1024))) = 8*16*round(sqrt(2))
            //      = 8 * 16 * 1 = 128  (sqrt(2)=1.41→round=1)
            assertEquals(128L, p.diagnosticCost());
            assertEquals(2048L, p.freedKvTokens());
        }

        @Test
        void kvGreedyExactlyMeetsDeficitWithTwoVictims() {
            // kvDeficit = 700 - 100 = 600. requests selected for preemption: kv512 + kv512 = 1024
            // >= 600.
            // But greedy takes one by one. First 512 < 600, so adds second.
            Map<String, String> f = new HashMap<>();
            DecodePreemptionPlan p =
                    plan(
                            70,
                            700L,
                            endpoint(
                                    100L,
                                    1000L,
                                    0,
                                    0L,
                                    List.of(accepted(1L, 30, 512L), accepted(2L, 30, 512L))),
                            f);
            assertEquals(2, p.requestsToPreempt().size());
            assertEquals(1024L, p.freedKvTokens());
        }

        @Test
        void insufficientReleasableKvIsInfeasible() {
            // kvDeficit = 100000 - 100 = 99900. Only 128 releasable.
            Map<String, String> f = new HashMap<>();
            assertNull(
                    plan(
                            70,
                            100_000L,
                            endpoint(100L, 1000L, 0, 0L, List.of(accepted(1L, 30, 128L))),
                            f));
            assertEquals("insufficient_releasable_kv", f.get("decode-a"));
        }
    }
}
