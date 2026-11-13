package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.Test;

import java.util.List;

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

}
