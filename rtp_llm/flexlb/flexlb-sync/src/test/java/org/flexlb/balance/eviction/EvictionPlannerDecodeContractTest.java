package org.flexlb.balance.eviction;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.eviction.EngineCancelChannel.CancelOutcome;
import org.flexlb.balance.eviction.model.PriorityRequestEnvelope;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.enums.DecodeTaskPhase;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * Exact-value contracts for {@link EvictionPlanner#planDecode}: the frozen
 * decode admission recompute of {@code current − victims + incoming} for
 * slot and KV capacity.
 *
 * <p>Requirements under test:
 * <ul>
 *   <li>Slot deficit = max(0, engineLoad + 1 − concurrencyLimit). No deficit → no eviction.</li>
 *   <li>KV deficit = hardKvTokens − realKvAvailable (only when realKvTotal &gt; 0).</li>
 *   <li>Equal-priority entries are NEVER victims (strict &lt;).</li>
 *   <li>Greedy KV selection takes the largest-bucket release first; covers the
 *       deficit exactly or fails entirely—never a partial set.</li>
 *   <li>Cost = h × f(priority) × g(phase) [× lengthWaste for KV], with exact numeric values.</li>
 * </ul>
 */
@DisplayName("EvictionPlanner.planDecode recompute contracts")
class EvictionPlannerDecodeContractTest {

    private static final long H_SLOT = 4L;
    private static final long H_KV = 8L;
    private static final EngineCancelChannel SUPPORTING_CHANNEL =
            new EngineCancelChannel() {
                @Override
                public boolean isSupported(DecodeEndpoint endpoint) {
                    return true;
                }

                @Override
                public CompletableFuture<CancelOutcome> cancel(
                        CancelTarget target, long a, long b) {
                    return CompletableFuture.completedFuture(CancelOutcome.accepted());
                }
            };

    private static PreemptionConfig engineOwned() {
        PreemptionConfig p = new PreemptionConfig();
        p.setAllowedVictimStages(EnumSet.of(VictimStage.DECODE_ENGINE_OWNED));
        return p;
    }

    private static PriorityRequestEnvelope envelope(int priority, long hardKvTokens) {
        return new PriorityRequestEnvelope(999L, priority, 0L, 0L, 0L, hardKvTokens, 0L);
    }

    private static DecodeRequestSnapshot accepted(long id, int priority, long kvTokens) {
        return new DecodeRequestSnapshot(
                id, priority, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, kvTokens,
                0L, true, false, 0L);
    }

    private static DecodeEndpointSnapshot endpoint(
            long realKvAvailable, long realKvTotal,
            int engineLoad, long concurrencyLimit,
            List<DecodeRequestSnapshot> accepted) {
        return new DecodeEndpointSnapshot(
                null, "decode-a", realKvAvailable, realKvTotal,
                engineLoad, engineLoad, concurrencyLimit,
                0L, 0L, List.of(), accepted, List.of());
    }

    private static DecodeEvictionProposal plan(
            PriorityRequestEnvelope env, DecodeEndpointSnapshot ep,
            Map<String, String> failures) {
        return EvictionPlanner.planDecode(
                env, List.of(ep), engineOwned(), SUPPORTING_CHANNEL, failures);
    }

    private static List<Long> victimIds(DecodeEvictionProposal p) {
        return p.victims().stream().map(DecodeRequestSnapshot::requestId).toList();
    }

    // ─── No deficit ─────────────────────────────────────────────────────

    @Nested
    @DisplayName("No deficit")
    class NoDef {

        @Test
        void sufficientCapacityReportsNoDeficit() {
            // slotDeficit = max(0, 2+1-4) = 0; kvDeficit = 200 < 500 → 0.
            Map<String, String> f = new HashMap<>();
            assertNull(plan(envelope(70, 200L),
                    endpoint(500L, 1000L, 2, 4L, List.of()), f));
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
            DecodeEvictionProposal p = plan(envelope(70, 0L),
                    endpoint(1000L, 0L, 1, 1L, List.of(accepted(1L, 30, 128L))), f);
            assertEquals(List.of(1L), victimIds(p));
            assertEquals(DecodeEvictionProposal.CASE_SLOT, p.evictionCase());
            // cost = H_SLOT * f(30) * g(ACCEPTED) = 4 * 1 * 16 = 64
            assertEquals(64L, p.totalCost());
            assertEquals(128L, p.freedKvTokens());
        }

        @Test
        void equalPriorityCandidateNeverYields() {
            Map<String, String> f = new HashMap<>();
            assertNull(plan(envelope(70, 0L),
                    endpoint(1000L, 0L, 1, 1L, List.of(accepted(1L, 70, 128L))), f));
            assertEquals("insufficient_lower_priority_candidates", f.get("decode-a"));
        }

        @Test
        void noPriorityCandidateNeverYields() {
            Map<String, String> f = new HashMap<>();
            assertNull(plan(envelope(70, 0L),
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
            DecodeEvictionProposal p = plan(envelope(70, 300L),
                    endpoint(100L, 1000L, 0, 0L,
                            List.of(accepted(1L, 30, 2048L), accepted(2L, 30, 512L))), f);
            assertEquals(List.of(1L), victimIds(p));
            assertEquals(DecodeEvictionProposal.CASE_KV, p.evictionCase());
            // cost = H_KV * f(30) * g(ACCEPTED) * lengthWasteCost(2048)
            //      = 8 * 1 * 16 * round(sqrt(ceil(2048/1024))) = 8*16*round(sqrt(2))
            //      = 8 * 16 * 1 = 128  (sqrt(2)=1.41→round=1)
            assertEquals(128L, p.totalCost());
            assertEquals(2048L, p.freedKvTokens());
        }

        @Test
        void kvGreedyExactlyMeetsDeficitWithTwoVictims() {
            // kvDeficit = 700 - 100 = 600. Victims: kv512 + kv512 = 1024 >= 600.
            // But greedy takes one by one. First 512 < 600, so adds second.
            Map<String, String> f = new HashMap<>();
            DecodeEvictionProposal p = plan(envelope(70, 700L),
                    endpoint(100L, 1000L, 0, 0L,
                            List.of(accepted(1L, 30, 512L), accepted(2L, 30, 512L))), f);
            assertEquals(2, p.victims().size());
            assertEquals(1024L, p.freedKvTokens());
        }

        @Test
        void insufficientReleasableKvIsInfeasible() {
            // kvDeficit = 100000 - 100 = 99900. Only 128 releasable.
            Map<String, String> f = new HashMap<>();
            assertNull(plan(envelope(70, 100_000L),
                    endpoint(100L, 1000L, 0, 0L, List.of(accepted(1L, 30, 128L))), f));
            assertEquals("insufficient_releasable_kv", f.get("decode-a"));
        }
    }
}
