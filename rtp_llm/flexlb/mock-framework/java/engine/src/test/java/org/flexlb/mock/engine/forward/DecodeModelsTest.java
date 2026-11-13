package org.flexlb.mock.engine.forward;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

/**
 * Checks deterministic Decode progress and EOS decisions.
 */
class DecodeModelsTest {
    @Test void eosAndLengthClampOnlyAcceptedOutput() {
        assertEquals(new ComputeModel.Result(4, true, 5), stopping(0, 1).complete("r", 0, 5, 0, 4, false));
        assertEquals(new ComputeModel.Result(3, true, 5), stopping(1, 1).complete("r", 0, 5, 3, 10, false));
        assertEquals(new ComputeModel.Result(5, false, 5), stopping(1, 1).complete("r", 0, 5, 0, 10, true));
        assertEquals(new ComputeModel.Result(1, true, 5), stopping(0, 1).complete("r", 3, 5, 0, 4, true));
        assertEquals(new ComputeModel.Result(2, true, 5), stopping(1, 1).complete("r", 0, 5, 9, 2, false));

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
        var model = new DecodeStepModel(new DecodeStepModel.Config(2.6));
        int total = 0;
        for (int round = 0; round < 100; round++) total += model.tokens(round);
        assertEquals(260, total);
        assertEquals(1, new DecodeStepModel(new DecodeStepModel.Config(null)).tokens(0));
        assertEquals(5, new DecodeStepModel(new DecodeStepModel.Config(5.0)).tokens(0));
    }

    @Test void invalidParametersAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> new DecodeStepModel.Config(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> new DecodeStepModel.Config(.9));
        assertThrows(IllegalArgumentException.class, () -> new StoppingModel.Config(.1, null));
        assertThrows(IllegalArgumentException.class, () -> new StoppingModel.Config(Double.NaN, 1L));
        assertThrows(IllegalArgumentException.class, () -> new DecodeStepModel(new DecodeStepModel.Config(null)).tokens(-1));
        assertThrows(IllegalArgumentException.class, () -> stopping(.1, 1).complete("", 0, 1, 0, 10, false));
    }

    private StoppingModel stopping(double p, long seed) { return new StoppingModel(new StoppingModel.Config(p, seed)); }
}
