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
    @Test void equalComputeCountsCachedTokensAsBenefit() {
        assertEquals(List.of("hot"), PrefillCostOrder.first(List.of(
                c("shortCold",50,1,1024,1024),c("hot",50,2,9216,1024)),List.of(),false,8));
    }
    @Test void evenTinyColdDoesNotDisplaceAnExistingWarmConversation() {
        assertEquals(List.of("hot"), PrefillCostOrder.first(List.of(
                c("tinyCold",50,1,64,64),c("hot",50,2,9216,1024)),List.of(),false,8));
    }
    @Test void equalBenefitUsesShorterComputeFirstAndRetainsParallelism() {
        assertEquals(List.of("small","large"), PrefillCostOrder.first(List.of(
                c("large",50,1,18000,2000),c("small",50,2,9000,1000)),List.of(),false,8));
        assertEquals(List.of("large"), PrefillCostOrder.first(List.of(c("large",50,2,18000,2000)),
                List.of(c("small",50,1,9000,1000)),false,8));
    }
    @Test void shortColdRunsWhenOnlyColdWorkIsReady() {
        assertEquals(List.of("short","long"), PrefillCostOrder.first(List.of(
                c("long",50,1,9000,9000),c("short",50,2,1000,1000)),List.of(),false,8));
    }
    @Test void runningHigherBenefitHoldsSpareSlots() {
        assertEquals(List.of(), PrefillCostOrder.first(List.of(c("cold",50,2,100,100)),
                List.of(c("hot",50,1,9000,1000)),false,8));
    }
    @Test void businessPriorityWinsBeforeBenefit() {
        assertEquals(List.of("high"), PrefillCostOrder.first(List.of(c("high",90,2,100,100),
                c("low",10,3,9000,1000)),List.of(c("runningLow",10,1,9000,1000)),true,8));
    }
    @Test void missingEstimateFallsBackToPriorityAndFifo() {
        var unknown = new PrefillCostOrder.Candidate<String>("unknown",50,1,0,null);
        assertEquals(List.of("unknown","known"), PrefillCostOrder.first(
                List.of(c("known",50,2,100,10),unknown),List.of(),false,8));
    }
    @Test void ratioComparisonDoesNotOverflowOrLoseCloseDifferences() {
        assertEquals(List.of("better"), PrefillCostOrder.first(List.of(
                c("worse",50,1,Long.MAX_VALUE,Long.MAX_VALUE-1),
                c("better",50,2,Long.MAX_VALUE,Long.MAX_VALUE-2)),List.of(),false,8));
    }
    @Test void runningHigherBusinessPriorityPrecedesBetterBenefit() {
        assertEquals(List.of(), PrefillCostOrder.first(List.of(c("warmLow", 10, 2, 9000, 1000)),
                List.of(c("coldHigh", 90, 1, 100, 100)), true, 8));
        assertEquals(List.of("warmLow"), PrefillCostOrder.first(List.of(c("warmLow", 10, 2, 9000, 1000)),
                List.of(c("coldHigh", 90, 1, 100, 100)), false, 8));
    }
    @Test void zeroComputeAndEqualBenefitHaveDeterministicOrder() {
        assertEquals(List.of("first", "second"), PrefillCostOrder.first(List.of(
                c("second", 50, 2, 1000, 0), c("first", 50, 1, 1000, 0),
                c("cold", 50, 0, 10, 10)), List.of(), false, 8));
    }

    @Test
    void boundedSelectionMatchesStableFullOrderingAcrossBacklogs() {
        var random = new Random(20261114L);
        for (int round = 0; round < 300; round++) {
            var ready = new ArrayList<PrefillCostOrder.Candidate<Integer>>();
            var running = new ArrayList<PrefillCostOrder.Candidate<Integer>>();
            int size = 1 + random.nextInt(400);
            for (int i = 0; i < size; i++) {
                long input = round % 7 == 0 ? Long.MAX_VALUE - random.nextInt(100) : 1000L * (1 + random.nextInt(10));
                Long compute = round % 3 == 0 ? input : round % 3 == 1 ? 0L : input / (1 + random.nextInt(10));
                if (round % 11 == 0 && i == 0) { compute = null; }
                var candidate = new PrefillCostOrder.Candidate<>(i, random.nextInt(3), random.nextInt(20), input, compute);
                ready.add(candidate);
                if (round % 5 == 0 && i < 2) { running.add(candidate); }
            }
            for (boolean priority : new boolean[]{false, true}) {
                for (int limit : new int[]{0, 1, 8, 512}) {
                    assertEquals(referenceFirst(ready, running, priority, limit),
                            PrefillCostOrder.first(ready, running, priority, limit),
                            "round=" + round + ", priority=" + priority + ", limit=" + limit);
                }
            }
        }
    }

    private static List<Integer> referenceFirst(List<PrefillCostOrder.Candidate<Integer>> ready,
            List<PrefillCostOrder.Candidate<Integer>> running, boolean priority, int limit) {
        if (limit <= 0 || ready.isEmpty()) { return List.of(); }
        Comparator<PrefillCostOrder.Candidate<Integer>> byPriority =
                Comparator.comparingInt(candidate -> priority ? -candidate.priority() : 0);
        var all = new ArrayList<>(ready);
        all.addAll(running);
        if (all.stream().anyMatch(candidate -> candidate.inputTokens() <= 0 || candidate.computeTokens() == null
                || candidate.computeTokens() < 0 || candidate.computeTokens() > candidate.inputTokens())) {
            return ready.stream().sorted(byPriority.thenComparingLong(PrefillCostOrder.Candidate::sequence))
                    .limit(limit).map(PrefillCostOrder.Candidate::value).toList();
        }
        Comparator<PrefillCostOrder.Candidate<Integer>> byRatio = (left, right) ->
                BigInteger.valueOf(left.computeTokens()).multiply(BigInteger.valueOf(right.inputTokens()))
                        .compareTo(BigInteger.valueOf(right.computeTokens()).multiply(BigInteger.valueOf(left.inputTokens())));
        var sorted = ready.stream().sorted(byPriority.thenComparing(byRatio)
                .thenComparingLong(PrefillCostOrder.Candidate::computeTokens)
                .thenComparingLong(PrefillCostOrder.Candidate::inputTokens)
                .thenComparingLong(PrefillCostOrder.Candidate::sequence)).toList();
        var best = sorted.getFirst();
        for (var active : running) {
            if (priority && active.priority() < best.priority()) { continue; }
            if (priority && active.priority() > best.priority() || byRatio.compare(active, best) < 0) { return List.of(); }
        }
        return sorted.stream().filter(candidate -> (!priority || candidate.priority() == best.priority())
                && byRatio.compare(candidate, best) == 0).limit(limit).map(PrefillCostOrder.Candidate::value).toList();
    }

}
