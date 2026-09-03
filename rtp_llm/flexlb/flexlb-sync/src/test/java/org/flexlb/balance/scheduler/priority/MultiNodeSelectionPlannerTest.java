package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.PriorityOrderingConfig;
import org.flexlb.config.QueueSchedulerConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.enums.DecodeTaskPhase;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Task34 类别一：复杂多节点选择正确性 — 3~8 个 endpoint 快照矩阵下的纯
 * planner 选择与比较序验证（不同队列深度 / KV 余量 / totalLoad / 可行性）。
 *
 * <p>覆盖：跨 endpoint 队列驱逐按 priority harm→victimCount→tie-break 选对、
 * victim 集合选对（低优在前、同优 newest-first）、decode slot/KV/combined
 * 三场景比较序与去重、不可行 endpoint 绝不入选、全不可行→明确失败、
 * 仅一可行→必选、并列→endpointId 确定性 tie-break。
 */
class MultiNodeSelectionPlannerTest {

    private FlexlbConfig config;
    private Map<String, String> failures;
    private EngineCancelChannel cancelChannel;

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
        QueueSchedulerConfig scheduler = new QueueSchedulerConfig();
        PriorityOrderingConfig ordering = new PriorityOrderingConfig();
        PreemptionConfig preemption = new PreemptionConfig();
        preemption.setAllowedVictimStages(EnumSet.of(
                VictimStage.DECODE_RESERVED, VictimStage.DECODE_ENGINE_OWNED));
        ordering.setPreemption(preemption);
        scheduler.setOrdering(ordering);
        config.setScheduler(scheduler);
        failures = new HashMap<>();
        cancelChannel = new EngineCancelChannel() {
            @Override
            public boolean isSupported(DecodeEndpoint endpoint) {
                return true;
            }

            @Override
            public CompletableFuture<CancelOutcome> cancel(CancelTarget target,
                                                           String requestId,
                                                           long timeoutMs) {
                return CompletableFuture.completedFuture(CancelOutcome.unsupported());
            }
        };
    }

    // ==================== prefill：跨 endpoint 矩阵选择 ====================

    @Test
    void prefill_matrix_skips_every_infeasible_endpoint_and_picks_cheapest_feasible() {
        long now = System.currentTimeMillis();
        // ep1 未满、ep2 无界、ep3 只有同优先级候选 —— 不可入选；
        // ep4 只能驱逐 P40，ep5 驱逐一个 P30，ep6 驱逐九个 P30。
        // ep5 精确伤害最小，且 ep6 不会再因固定 victim cap 被判不可行。
        PrefillQueueSnapshot ep1NotFull = queue("ep1", 5, snap("11", 30, now));
        PrefillQueueSnapshot ep2Unbounded = queue("ep2", 0, snap("21", 30, now));
        PrefillQueueSnapshot ep3EqualPriority = queue("ep3", 1, snap("31", 70, now));
        PrefillQueueSnapshot ep4EvictsP40 = queue("ep4", 1, snap("41", 40, now));
        PrefillQueueSnapshot ep5EvictsP30 = queue("ep5", 1, snap("51", 30, now));
        // ep6：容量 1、9 项 → deficit = 9，仍是可行方案。
        QueuedRequestSnapshot[] ep6Items = new QueuedRequestSnapshot[9];
        for (int i = 0; i < ep6Items.length; i++) {
            ep6Items[i] = snap(String.valueOf(61 + i), 30, now + i);
        }
        PrefillQueueSnapshot ep6ManyVictims = queue("ep6", 1, ep6Items);

        PrefillEvictionProposal proposal = EvictionPlanner.planPrefillQueue(
                envelope("9", 70),
                List.of(ep1NotFull, ep2Unbounded, ep3EqualPriority,
                        ep4EvictsP40, ep5EvictsP30, ep6ManyVictims),
                failures);

        assertNotNull(proposal);
        assertEquals("ep5", proposal.endpointId());
        assertEquals(List.of("51"), proposal.victims().stream()
                .map(QueuedRequestSnapshot::requestId).toList());
        assertEquals(PriorityCostFunction.f(30), proposal.rawCost());
        // 不可行 endpoint 全部带明确原因；ep6 可行但伤害高于 ep5。
        assertEquals("queue_not_full", failures.get("ep1"));
        assertEquals("queue_unbounded", failures.get("ep2"));
        assertEquals("insufficient_lower_priority_candidates", failures.get("ep3"));
        assertNull(failures.get("ep6"));
    }

    @Test
    void prefill_equal_net_cost_prefers_fewer_victims_then_endpoint_id() {
        long now = System.currentTimeMillis();
        // epA：2 个 P30 victim（rawCost 2）；epB：1 个 P30（rawCost 1）——
        // netCost 不同时直接选 netCost 小者。
        PrefillQueueSnapshot epA = queue("epA", 2,
                snap("1", 30, now),
                snap("2", 30, now),
                snap("3", 70, now));
        PrefillQueueSnapshot epB = queue("epB", 1, snap("4", 30, now));

        PrefillEvictionProposal proposal = EvictionPlanner.planPrefillQueue(
                envelope("9", 70), List.of(epA, epB), failures);

        assertNotNull(proposal);
        assertEquals("epB", proposal.endpointId());
        assertEquals(1, proposal.victims().size());
    }

    @Test
    void prefill_victim_set_orders_low_priority_first_then_newer_arrival() {
        long now = System.currentTimeMillis();
        // 容量 3、已有 5 项，deficit = 3：应选 [P30 较新, P30 较旧, P40]，
        // P50 绝不进 victim 集合。
        PrefillQueueSnapshot queue = queue("ep1", 3,
                snap("1", 50, now),
                snap("2", 40, now),
                snap("3", 30, now),
                snap("4", 30, now + 500),
                snap("5", 50, now));

        PrefillEvictionProposal proposal = EvictionPlanner.planPrefillQueue(
                envelope("9", 70), List.of(queue), failures);

        assertNotNull(proposal);
        assertEquals(List.of("4", "3", "2"), proposal.victims().stream()
                .map(QueuedRequestSnapshot::requestId).toList());
        assertEquals(2 * PriorityCostFunction.f(30) + PriorityCostFunction.f(40),
                proposal.rawCost());
    }

    @Test
    void prefill_identical_endpoints_tie_break_is_deterministic_across_permutations() {
        long now = System.currentTimeMillis();
        PrefillQueueSnapshot epA = queue("epA", 1, snap("1", 30, now));
        PrefillQueueSnapshot epB = queue("epB", 1, snap("2", 30, now));
        // 除 victim requestId / endpointId 外完全等价 → PlanCost tie-break 后
        // 仍按 endpointId asc 确定性收敛，且与输入顺序无关。
        List<List<PrefillQueueSnapshot>> permutations = List.of(
                List.of(epA, epB), List.of(epB, epA));

        for (List<PrefillQueueSnapshot> permutation : permutations) {
            PrefillEvictionProposal proposal = EvictionPlanner.planPrefillQueue(
                    envelope("9", 70), permutation, new HashMap<>());
            assertNotNull(proposal);
            assertEquals("epA", proposal.endpointId(),
                    "tie-break must not depend on input order: " + permutation);
        }
    }

    @Test
    void prefill_only_feasible_endpoint_is_always_selected() {
        long now = System.currentTimeMillis();
        PrefillQueueSnapshot infeasible1 = queue("ep1", 3, snap("1", 30, now));
        PrefillQueueSnapshot infeasible2 = queue("ep2", 1, snap("2", 70, now));
        // 唯一可行者虽然要驱逐一个 P60（很贵）也必须被选中
        PrefillQueueSnapshot onlyFeasible = queue("ep3", 1, snap("3", 60, now));

        PrefillEvictionProposal proposal = EvictionPlanner.planPrefillQueue(
                envelope("9", 70), List.of(infeasible1, infeasible2, onlyFeasible),
                failures);

        assertNotNull(proposal);
        assertEquals("ep3", proposal.endpointId());
        assertEquals(PriorityCostFunction.f(60), proposal.rawCost());
    }

    @Test
    void prefill_all_infeasible_fails_explicitly_with_reason_per_endpoint() {
        long now = System.currentTimeMillis();
        PrefillQueueSnapshot notFull = queue("ep1", 9, snap("1", 30, now));
        PrefillQueueSnapshot equalPriority = queue("ep2", 1, snap("2", 50, now));
        PrefillQueueSnapshot higherPriority = queue("ep3", 1, snap("3", 70, now));

        PrefillEvictionProposal proposal = EvictionPlanner.planPrefillQueue(
                envelope("9", 50), List.of(notFull, equalPriority, higherPriority),
                failures);

        assertNull(proposal);
        assertEquals(3, failures.size());
        assertEquals("queue_not_full", failures.get("ep1"));
        assertEquals("insufficient_lower_priority_candidates", failures.get("ep2"));
        assertEquals("insufficient_lower_priority_candidates", failures.get("ep3"));
    }

    // ==================== decode：slot / KV / combined 跨 endpoint ====================

    @Test
    void decode_matrix_picks_cheapest_across_slot_kv_and_infeasible_endpoints() {
        // d1：slot-full 只能驱逐 P40 → 4x1024；d2：kv-full 驱逐 P30 → 8x1x1x1=8；
        // d3：slot-full Engine-owned P30 → 4×stage(4)=16；d4：容量充足不可行 → 必选 d3。
        DecodeEndpointSnapshot d1 = endpoint("d1", 1, 100_000, 200_000, 2, 2, List.of(
                reserved("11", 40, 128),
                reserved("12", 70, 128)));
        DecodeEndpointSnapshot d2 = endpoint("d2", 2, 100, 10_000, 2, 0, List.of(
                reserved("21", 30, 2_048)));
        DecodeEndpointSnapshot d3 = endpoint("d3", 3, 100_000, 200_000, 2, 2, List.of(
                reserved("31", 30, 128),
                reserved("32", 70, 128)));
        DecodeEndpointSnapshot d4 = endpoint("d4", 4, 100_000, 200_000, 0, 2, List.of());

        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(50, 128), List.of(d1, d2, d3, d4),
                config, cancelChannel, failures);

        assertNotNull(proposal);
        assertEquals("d3", proposal.endpointId());
        assertEquals(DecodeEvictionProposal.CASE_SLOT, proposal.evictionCase());
        assertEquals(List.of("31"), proposal.victims().stream()
                .map(DecodeRequestSnapshot::requestId).toList());
        assertEquals(16, proposal.totalCost());
        assertEquals("decode_capacity_sufficient", failures.get("d4"));
    }

    @Test
    void decode_single_case_plan_beats_combined_plan_on_another_endpoint() {
        // d1 同时 slot+KV 双缺口（combined，victim 两个 P30）；
        // d2 仅 slot 缺口（单 case，victim 一个 P30）→ priorityCost 更小者胜。
        DecodeEndpointSnapshot d1 = endpoint("d1", 1, 0, 100_000, 2, 1, List.of(
                reserved("11", 30, 100),
                reserved("12", 30, 1_000)));
        DecodeEndpointSnapshot d2 = endpoint("d2", 2, 100_000, 200_000, 1, 1, List.of(
                reserved("21", 30, 128)));

        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(50, 1_000), List.of(d1, d2),
                config, cancelChannel, failures);

        assertNotNull(proposal);
        assertEquals("d2", proposal.endpointId());
        assertEquals(DecodeEvictionProposal.CASE_SLOT, proposal.evictionCase());
        assertEquals(1, proposal.victims().size());
    }

    @Test
    void decode_identical_endpoints_tie_break_is_deterministic_across_permutations() {
        DecodeEndpointSnapshot dA = endpoint("dA", 1, 100_000, 200_000, 1, 1, List.of(
                reserved("1", 30, 128)));
        DecodeEndpointSnapshot dB = endpoint("dB", 2, 100_000, 200_000, 1, 1, List.of(
                reserved("1", 30, 128)));

        for (List<DecodeEndpointSnapshot> permutation
                : List.of(List.of(dA, dB), List.of(dB, dA))) {
            DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                    incoming(50, 128), permutation,
                    config, cancelChannel, new HashMap<>());
            assertNotNull(proposal);
            assertEquals("dA", proposal.endpointId(),
                    "tie-break must not depend on input order");
        }
    }

    @Test
    void decode_all_infeasible_fails_explicitly_and_only_feasible_wins() {
        // 全不可行 → null + 每个 endpoint 都有原因
        DecodeEndpointSnapshot equalPriority = endpoint("d1", 1, 100_000, 200_000, 1, 1,
                List.of(reserved("1", 50, 128)));
        DecodeEndpointSnapshot sufficient = endpoint("d2", 2, 100_000, 200_000, 0, 4, List.of());
        assertNull(EvictionPlanner.planDecode(
                incoming(50, 128), List.of(equalPriority, sufficient),
                config, cancelChannel, failures));
        assertEquals("insufficient_lower_priority_candidates", failures.get("d1"));
        assertEquals("decode_capacity_sufficient", failures.get("d2"));

        // 加入唯一可行者（哪怕驱逐 P40 很贵）→ 必选
        DecodeEndpointSnapshot onlyFeasible = endpoint("d3", 3, 100_000, 200_000, 1, 1,
                List.of(reserved("3", 40, 128)));
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(50, 128), List.of(equalPriority, sufficient, onlyFeasible),
                config, cancelChannel, new HashMap<>());
        assertNotNull(proposal);
        assertEquals("d3", proposal.endpointId());
    }

    // ==================== 比较序公理化验证 ====================

    @Test
    void plan_cost_order_applies_each_tier_in_sequence() {
        PriorityHarmProfile twoP30 = profile(30, 2);
        PlanCost base = new PlanCost(twoP30, 30, 10, 2, "7");

        // exact priority harm 最优先：任意数量 P30 都优于一个 P40。
        assertTrue(PlanCost.ORDER.compare(base,
                new PlanCost(profile(40, 1), 40, 1, 1, "1")) < 0);
        // 再 victimCount asc
        assertTrue(PlanCost.ORDER.compare(base,
                new PlanCost(twoP30, 30, 10, 3, "1")) < 0);
        // 最后 deterministicTieBreak asc
        assertTrue(PlanCost.ORDER.compare(base,
                new PlanCost(twoP30, 30, 10, 2, "8")) < 0);
        assertEquals(0, PlanCost.ORDER.compare(base,
                new PlanCost(twoP30, 30, 999, 2, "7")));
    }

    @Test
    void prefill_proposal_order_sorts_a_shuffled_matrix_deterministically() {
        PrefillEvictionProposal cheapest = proposal("epC", 1, 1, 30);
        PrefillEvictionProposal sameHarmA = proposal("epA", 2, 2, 30);
        PrefillEvictionProposal sameHarmB = proposal("epB", 2, 2, 30);
        PrefillEvictionProposal expensive = proposal("epD", 1, 1_024, 40);

        List<PrefillEvictionProposal> shuffled = new ArrayList<>(
                List.of(expensive, sameHarmB, cheapest, sameHarmA));
        Collections.shuffle(shuffled, new Random(42));
        shuffled.sort(PrefillEvictionProposal.ORDER);

        assertEquals(List.of("epC", "epA", "epB", "epD"),
                shuffled.stream().map(PrefillEvictionProposal::endpointId).toList());
    }

    @Test
    void decode_proposal_order_uses_plan_cost_then_endpoint_id() {
        DecodeEvictionProposal cheap = decodeProposal("dB", 4,
                new PlanCost(profile(30, 1), 30, 1, 1, "1"));
        DecodeEvictionProposal sameCost = decodeProposal("dA", 4,
                new PlanCost(profile(30, 1), 30, 1, 1, "1"));
        DecodeEvictionProposal expensive = decodeProposal("dC", 8,
                new PlanCost(profile(40, 1), 40, 1_024, 1, "1"));

        List<DecodeEvictionProposal> sorted = new ArrayList<>(List.of(expensive, cheap, sameCost));
        sorted.sort(DecodeEvictionProposal.ORDER);

        assertEquals(List.of("dA", "dB", "dC"),
                sorted.stream().map(DecodeEvictionProposal::endpointId).toList());
    }

    // ==================== helpers ====================

    private static PrefillQueueSnapshot queue(String endpointId, int capacity,
                                              QueuedRequestSnapshot... items) {
        return new PrefillQueueSnapshot(endpointId, 1L, capacity, List.of(items));
    }

    private static QueuedRequestSnapshot snap(String requestId, int priority,
                                              long arrivalMs) {
        return new QueuedRequestSnapshot(requestId, priority, arrivalMs,
                128, 0, QueuedRequestSnapshot.PREFILL_QUEUED);
    }

    private static PriorityRequestEnvelope envelope(String requestId, int priority) {
        long now = System.currentTimeMillis();
        return new PriorityRequestEnvelope(requestId, priority, 128, 8,
                now, 128, 136);
    }

    private static DecodeEndpointSnapshot endpoint(String id, long version,
                                                   long realKvAvailable, long realKvTotal,
                                                   int totalLoad, long concurrencyLimit,
                                                   List<DecodeRequestSnapshot> reserved) {
        long hardKv = reserved.stream().mapToLong(DecodeRequestSnapshot::kvTokens).sum();
        long expectedKv = reserved.stream().mapToLong(DecodeRequestSnapshot::expectedKvTokens).sum();
        return new DecodeEndpointSnapshot(null, id, version, realKvAvailable, realKvTotal,
                totalLoad, totalLoad, concurrencyLimit, hardKv, expectedKv,
                reserved, List.of(), List.of());
    }

    private static DecodeRequestSnapshot reserved(String requestId, int priority,
                                                  long kvTokens) {
        return new DecodeRequestSnapshot(requestId, priority,
                DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN,
                kvTokens, kvTokens + 8, true, false);
    }

    private static PriorityRequestEnvelope incoming(int priority, long seqLen) {
        return new PriorityRequestEnvelope("999", priority, seqLen, 8,
                System.currentTimeMillis(), seqLen, seqLen + 8);
    }

    private static PrefillEvictionProposal proposal(String endpointId, int victimCount,
                                                    long rawCost, int minVictimPriority) {
        long now = 1_000_000L;
        List<QueuedRequestSnapshot> victims = new ArrayList<>();
        for (int i = 0; i < victimCount; i++) {
            victims.add(snap(String.valueOf(100 + i), minVictimPriority, now));
        }
        return new PrefillEvictionProposal(endpointId, 1L, victims, rawCost,
                new PlanCost(profile(minVictimPriority, victimCount), minVictimPriority,
                        rawCost, victimCount, "100"));
    }

    private static PriorityHarmProfile profile(int priority, long harm) {
        return PriorityHarmProfile.builder().add(priority, harm).build();
    }

    private static DecodeEvictionProposal decodeProposal(String endpointId, long totalCost,
                                                         PlanCost cost) {
        return new DecodeEvictionProposal(endpointId, 1L,
                List.of(reserved("1", cost.minVictimPriority(), 128)),
                DecodeEvictionProposal.CASE_SLOT, totalCost, 128, cost);
    }
}
