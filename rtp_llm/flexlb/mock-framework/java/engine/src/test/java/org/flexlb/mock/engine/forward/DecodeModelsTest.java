package org.flexlb.mock.engine.forward;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

/**
 * Checks deterministic Decode progress and Gaussian output lengths.
 */
class DecodeModelsTest {
    @Test void gaussianOutputUsesRelativeSpreadAndStableRequestSeeds() throws Exception {
        var compute = ComputeModel.fromJson(new ObjectMapper().readTree("""
            {"forward":{"expression":"1"},"output":{"mean_tokens":1000,"stddev_percent":10,"seed":42}}
            """));
        double sum = 0, squares = 0;
        for (int i = 0; i < 10000; i++) {
            String key = "r" + i;
            int length = compute.decodeOutputLimit(key, 0, 10000, false);
            assertEquals(Math.min(length, 950), compute.decodeOutputLimit(key, 0, 950, false));
            sum += length;
            squares += (double) length * length;
        }
        double mean = sum / 10000;
        assertEquals(1000, mean, 5);
        assertEquals(100, Math.sqrt(squares / 10000 - mean * mean), 5);
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
        assertThrows(IllegalArgumentException.class, () -> new DecodeStepModel(new DecodeStepModel.Config(null)).tokens(-1));
    }
}
