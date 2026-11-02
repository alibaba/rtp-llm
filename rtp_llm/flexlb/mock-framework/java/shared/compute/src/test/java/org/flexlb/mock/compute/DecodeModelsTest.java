package org.flexlb.mock.compute;

import static org.junit.jupiter.api.Assertions.*;
import org.flexlb.mock.api.model.*;
import org.junit.jupiter.api.Test;

class DecodeModelsTest {
    private DecodeStepModel step(int k, double alpha, long seed) { return new DecodeStepModel(new DecodeSpec(k, alpha, seed, null)); }
    private StoppingModel stopping(double p, long seed) { return new StoppingModel(new StoppingSpec(p, seed)); }

    @Test void verificationPlanIsIndependentOfAcceptanceAndOutputTruncation() {
        var rejected = step(4, 0, 1); var accepted = step(4, 1, 1);
        var plan = rejected.plan(0, 10);
        assertEquals(plan, accepted.plan(0, 10));
        assertEquals(5, plan.verifyTokens()); assertEquals(5, plan.additionalKvTokens());
        assertEquals(1, rejected.candidates(plan, "r", 0)); assertEquals(5, accepted.candidates(plan, "r", 0));
        var tail = accepted.plan(0, 4);
        assertEquals(5, tail.verifyTokens()); assertEquals(5, tail.additionalKvTokens());
        assertEquals(5, accepted.candidates(tail, "r", 0));
    }
    @Test void cappedGeometricHasTheExpectedAcceptedOutputDistribution() {
        var model = step(3, .8, 42); var plan = model.plan(0, 100); double sum = 0; int[] counts = new int[5];
        for (int i = 0; i < 20000; i++) { int n = model.candidates(plan, "r" + i, 0); sum += n; counts[n]++; }
        assertEquals(1 + .8 + .64 + .512, sum / 20000, .035);
        assertEquals(.2, counts[1] / 20000.0, .02); assertEquals(.512, counts[4] / 20000.0, .02);
    }
    @Test void eosAndLengthClampOnlyAcceptedOutput() {
        assertEquals(new ComputeModel.Result(4, true), stopping(0, 1).complete("r", 0, 5, 0, 4, false));
        assertEquals(new ComputeModel.Result(3, true), stopping(1, 1).complete("r", 0, 5, 3, 10, false));
        assertEquals(new ComputeModel.Result(5, false), stopping(1, 1).complete("r", 0, 5, 0, 10, true));
        assertEquals(new ComputeModel.Result(1, true), stopping(0, 1).complete("r", 3, 5, 0, 4, true));
        assertEquals(new ComputeModel.Result(2, true), stopping(1, 1).complete("r", 0, 5, 9, 2, false));
        var rejected = step(4, 0, 1); var plan = rejected.plan(0, 10);
        assertEquals(new ComputeModel.Result(1, false), stopping(1, 1).complete("r", 0,
            rejected.candidates(plan, "r", 0), 2, 10, false));
    }
    @Test void perTokenEosHasGeometricMeanAndIsStableAcrossRoundWidths() {
        var stop = stopping(.1, 42); double sum = 0;
        for (int i = 0; i < 20000; i++) sum += stop.complete("r" + i, 0, 100, 1, 100, false).committedTokens();
        assertEquals(10, sum / 20000, .3);
        for (int i = 0; i < 100; i++) {
            String key = "r" + i;
            int expected = stop.complete(key, 0, 100, 3, 100, false).committedTokens();
            for (int width : new int[]{1, 2, 5, 13}) {
                int generated = 0;
                while (true) {
                    var result = stop.complete(key, generated, width, 3, 100, false);
                    generated += result.committedTokens(); if (result.finished()) break;
                }
                assertEquals(expected, generated);
            }
        }
    }
    @Test void fractionalProgressIsAnAggregateModelAndTheTailCanProduceFiveButCommitFour() {
        var model = new DecodeStepModel(new DecodeSpec(null, null, null, 2.6)); int total = 0;
        for (int round = 0; round < 100; round++) {
            var plan = model.plan(round, 1000); int count = model.candidates(plan, "r", round);
            assertEquals(1, plan.verifyTokens()); assertEquals(0, plan.proposedTokens());
            assertEquals(count - 1, plan.additionalKvTokens()); total += count;
        }
        assertEquals(260, total);
        var mean = new DecodeStepModel(new DecodeSpec(null, null, null, 5.0)); var plan = mean.plan(0, 4);
        assertEquals(5, mean.candidates(plan, "r", 0)); assertEquals(3, plan.additionalKvTokens());
        assertEquals(new ComputeModel.Result(4, true), stopping(0, 1).complete("r", 0, 5, 0, 4, false));
    }
    @Test void randomAcceptanceIsKeyedAndInvalidParametersAreRejected() {
        var model = step(4, .6, 42); var plan = model.plan(7, 80); int expected = model.candidates(plan, "r", 7);
        for (int i = 0; i < 100; i++) model.candidates(model.plan(i, 40), "other" + i, i);
        assertEquals(expected, model.candidates(plan, "r", 7));
        assertThrows(IllegalArgumentException.class, () -> new DecodeSpec(3, .8, null, null));
        assertThrows(IllegalArgumentException.class, () -> new DecodeSpec(3, null, 1L, null));
        assertThrows(IllegalArgumentException.class, () -> new DecodeSpec(3, 1.1, 1L, null));
        assertThrows(IllegalArgumentException.class, () -> new DecodeSpec(1, 1.0, null, 2.6));
        assertThrows(IllegalArgumentException.class, () -> new DecodeSpec(null, null, null, Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> new StoppingSpec(.1, null));
        assertThrows(IllegalArgumentException.class, () -> new StoppingSpec(Double.NaN, 1L));
        assertThrows(IllegalArgumentException.class, () -> model.plan(0, 0));
        assertThrows(IllegalArgumentException.class, () -> stopping(.1, 1).complete("", 0, 1, 0, 10, false));
    }
}
