package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.Test;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

class PrefillCostOrderTest {
    private static PrefillCostOrder.Candidate<String> c(String name, int priority, long seq, long input, long compute) {
        return new PrefillCostOrder.Candidate<>(name, priority, seq, input, compute);
    }
    @Test
    void priorityThenFreshBenefitThenUnknownFifoUsesAllFreeSlots() {
        var ready = List.of(c("lowHot", 10, 1, 100, 0), c("cold", 50, 2, 100, 100),
                c("hot", 50, 3, 100, 10), c("earlyUnknown", 50, 0, 0, 0),
                c("lateUnknown", 50, 5, 0, 0), c("highUnknown", 90, 4, 0, 0));
        assertEquals(List.of("highUnknown", "hot", "cold", "earlyUnknown", "lateUnknown", "lowHot"),
                PrefillCostOrder.first(ready, true, 6));
        assertEquals(List.of("lowHot", "hot", "cold", "earlyUnknown", "highUnknown", "lateUnknown"),
                PrefillCostOrder.first(ready, false, 6));
    }

    @Test
    void closeRatiosAtLongMaxValueKeepTheirExactOrder() {
        assertEquals(List.of("better", "worse"), PrefillCostOrder.first(List.of(
                c("worse", 50, 1, Long.MAX_VALUE, Long.MAX_VALUE - 1),
                c("better", 50, 2, Long.MAX_VALUE, Long.MAX_VALUE - 2)), false, 2));
    }

    @Test
    void boundedSelectionMatchesStableFullOrderingAcrossBacklogs() {
        var random = new Random(20261114L);
        for (int round = 0; round < 300; round++) {
            var ready = new ArrayList<PrefillCostOrder.Candidate<Integer>>();
            int size = 1 + random.nextInt(400);
            for (int i = 0; i < size; i++) {
                long input = round % 7 == 0 ? Long.MAX_VALUE - random.nextInt(100) : 1000L * (1 + random.nextInt(10));
                Long compute = round % 3 == 0 ? input : round % 3 == 1 ? 0L : input / (1 + random.nextInt(10));
                if (round % 11 == 0 && i == 0) { compute = null; }
                var candidate = new PrefillCostOrder.Candidate<>(i, random.nextInt(3), random.nextInt(20), input, compute);
                ready.add(candidate);
            }
            for (boolean priority : new boolean[]{false, true}) {
                for (int limit : new int[]{0, 1, 8, 512}) {
                    assertEquals(referenceFirst(ready, priority, limit),
                            PrefillCostOrder.first(ready, priority, limit),
                            "round=" + round + ", priority=" + priority + ", limit=" + limit);
                }
            }
        }
    }

    private static List<Integer> referenceFirst(List<PrefillCostOrder.Candidate<Integer>> ready,
            boolean priority, int limit) {
        if (limit <= 0 || ready.isEmpty()) { return List.of(); }
        Comparator<PrefillCostOrder.Candidate<Integer>> byPriority =
                Comparator.comparingInt(candidate -> priority ? -candidate.priority() : 0);
        Comparator<PrefillCostOrder.Candidate<Integer>> byRatio = (left, right) ->
                BigInteger.valueOf(left.computeTokens()).multiply(BigInteger.valueOf(right.inputTokens()))
                        .compareTo(BigInteger.valueOf(right.computeTokens()).multiply(BigInteger.valueOf(left.inputTokens())));
        var known = ready.stream().filter(c -> c.computeTokens() != null).sorted(byPriority.thenComparing(byRatio)
                .thenComparingLong(PrefillCostOrder.Candidate::computeTokens)
                .thenComparingLong(PrefillCostOrder.Candidate::inputTokens)
                .thenComparingLong(PrefillCostOrder.Candidate::sequence));
        var unknown = ready.stream().filter(c -> c.computeTokens() == null)
                .sorted(Comparator.comparingLong(PrefillCostOrder.Candidate::sequence));
        return java.util.stream.Stream.concat(known, unknown).sorted(byPriority)
                .limit(limit).map(PrefillCostOrder.Candidate::value).toList();
    }
}
