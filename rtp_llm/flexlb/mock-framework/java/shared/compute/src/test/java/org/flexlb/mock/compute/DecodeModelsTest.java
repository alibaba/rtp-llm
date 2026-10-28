package org.flexlb.mock.compute;

import static org.junit.jupiter.api.Assertions.*;
import org.flexlb.mock.api.model.*;
import org.junit.jupiter.api.Test;

class DecodeModelsTest {
    private DecodeStepModel step(int k, double alpha, long seed) { return new DecodeStepModel(new DecodeSpec(k, alpha, seed)); }
    private StoppingModel stopping(double p, long seed) { return new StoppingModel(new StoppingSpec(p, seed)); }

    @Test void ordinaryAllRejectedAllAcceptedAndBudgetTail() {
        assertEquals(new DecodeStepModel.Step(0, 0, 1, 1), step(0, 0, 1).sample("r", 0, 8));
        assertEquals(new DecodeStepModel.Step(3, 0, 1, 4), step(3, 0, 1).sample("r", 0, 8));
        assertEquals(new DecodeStepModel.Step(3, 3, 4, 4), step(3, 1, 1).sample("r", 0, 8));
        assertEquals(new DecodeStepModel.Step(1, 1, 2, 2), step(3, 1, 1).sample("r", 0, 2));
    }
    @Test void cappedGeometricHasTheExpectedPrefixDistribution() {
        var model = step(3, .8, 42); double sum = 0; int[] counts = new int[4]; int samples = 20000;
        for (int i = 0; i < samples; i++) { var plan = model.sample("r" + i, 0, 100); sum += plan.committedTokens(); counts[plan.acceptedTokens()]++; }
        assertEquals(1 + .8 + .64 + .512, sum / samples, .035);
        assertEquals(.2, counts[0] / (double) samples, .02);
        assertEquals(.512, counts[3] / (double) samples, .02);
    }
    @Test void stoppingRespectsLimitsIgnoreEosAndTheGeometricMean() {
        assertEquals(20, stopping(0, 1).length("r", 3, 20, false));
        assertEquals(3, stopping(1, 1).length("r", 3, 20, false));
        assertEquals(20, stopping(1, 1).length("r", 3, 20, true));
        double sum = 0; var model = stopping(.1, 42);
        for (int i = 0; i < 20000; i++) sum += model.length("r" + i, 1, 100, false);
        assertEquals(10, sum / 20000, .3);
    }
    @Test void randomResultsAreKeyedAndSpeculationDoesNotResampleTheStopLength() {
        var stop = stopping(.07, 31); var slow = step(0, 0, 42); var fast = step(5, .8, 42);
        int expected = stop.length("r", 1, 100, false);
        for (var model : new DecodeStepModel[]{slow, fast}) {
            int generated = 0, round = 0;
            while (generated < expected) generated += Math.min(expected - generated, model.sample("r", round++, 100 - generated).committedTokens());
            assertEquals(expected, generated);
        }
        var model = step(4, .6, 42); var original = model.sample("r", 7, 80);
        for (int i = 0; i < 100; i++) model.sample("other" + i, i, 40);
        assertEquals(original, model.sample("r", 7, 80));
        assertEquals(expected, stop.length("r", 1, 100, false));
    }
    @Test void invalidProbabilitiesMissingSeedsAndLimitsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> new DecodeSpec(3, .8, null));
        assertThrows(IllegalArgumentException.class, () -> new DecodeSpec(3, null, 1L));
        assertThrows(IllegalArgumentException.class, () -> new DecodeSpec(3, 1.1, 1L));
        assertThrows(IllegalArgumentException.class, () -> new StoppingSpec(.1, null));
        assertThrows(IllegalArgumentException.class, () -> new StoppingSpec(Double.NaN, 1L));
        assertThrows(IllegalArgumentException.class, () -> step(1, .5, 1).sample("r", 0, 0));
        assertThrows(IllegalArgumentException.class, () -> stopping(.1, 1).length("r", 11, 10, false));
        assertThrows(IllegalArgumentException.class, () -> stopping(.1, 1).length("", 0, 10, false));
    }
}
