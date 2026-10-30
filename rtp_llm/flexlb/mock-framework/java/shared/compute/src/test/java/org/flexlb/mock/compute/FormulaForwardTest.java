package org.flexlb.mock.compute;

import static org.junit.jupiter.api.Assertions.*;
import java.time.Duration;
import java.util.*;
import org.flexlb.balance.prediction.PrefillTimeFormula;
import org.junit.jupiter.api.Test;

class FormulaForwardTest {
    static ForwardBatch.Request request(String id, long input, long cache) {
        return new ForwardBatch.Request(id, input, cache, 32, 4, 1, 0, List.of());
    }
    static ForwardBatch batch(ForwardBatch.Request... requests) {
        return new ForwardBatch("batch-0", List.of(new ForwardBatch.Partition(0, List.of(requests))));
    }

    @Test void agreesWithProductionPrefillBindingsWithoutCopyingFormulaImplementation() {
        String expression = "param(baseMs, 1.25) + 2*batchSize + 0.5*totalComputeTokens"
            + " + 0.01*sum(computeTokens^2) + 0.25*maxInputTokens";
        var production = PrefillTimeFormula.parse(expression);
        double[] outer = {2, 0, 0, 0, 0, 30, 6, 24, 20, 16};
        var items = List.of(new double[]{2, 10, 2, 8, 1}, new double[]{2, 20, 4, 16, 1});
        var result = new FormulaForward(expression).estimate(batch(request("a", 10, 2), request("b", 20, 4)));
        assertEquals(production.evaluateAsDouble(outer, items), result.toNanos() / 1e6);
    }

    @Test void preservesPerRequestDistributionAndFractionalNanosecondPrecision() {
        var input = batch(request("a", 10, 2), request("b", 20, 4));
        assertEquals(Duration.ofMillis(576), new FormulaForward("totalComputeTokens^2").estimate(input));
        assertEquals(Duration.ofMillis(320), new FormulaForward("sum(computeTokens^2)").estimate(input));
        assertEquals(Duration.ofNanos(125), new FormulaForward("0.000125").estimate(input));
        assertEquals(Duration.ZERO, new FormulaForward("0").estimate(input));
    }

    @Test void supportsDecodeStepAndContextWithoutMovingSchedulingIntoSharedLayer() {
        var input = batch(request("a", 10, 2), request("b", 20, 4));
        assertEquals(Duration.ofMillis(38), new FormulaForward("sum(contextTokens)").estimate(input));
        assertEquals(Duration.ofMillis(2), new FormulaForward("totalStepTokens").estimate(input));
        assertEquals(Duration.ofMillis(3), new FormulaForward("param(baseMs, 1) + batchSize").estimate(input));
    }

    @Test void retainsImageShapeAndUsesRawPixelsOnlyWhenFormulaAsksForThem() {
        var image = new ForwardBatch.Image("image", 640, 480, 3);
        var request = new ForwardBatch.Request("a", 10, 2, 32, 0, 8, 9, List.of(image, image));
        var input = batch(request);
        assertEquals(2, input.dps().getFirst().requests().getFirst().images().size());
        assertEquals(Duration.ofMillis(614402), new FormulaForward("totalImagePixels + totalImageCount").estimate(input));
    }

    @Test void refusesAmbiguousScopesInvalidResultsAndUnknownVariables() {
        for (String expression : List.of("typo + 1", "sum(batchSize)", "sum(totalComputeTokens)", "1 +"))
            assertThrows(IllegalArgumentException.class, () -> new FormulaForward(expression), expression);
        for (String expression : List.of("inputTokens", "1/0", "sqrt(-1)", "-1", "1e300"))
            assertThrows(IllegalArgumentException.class, () -> new FormulaForward(expression).estimate(batch(request("a", 1, 0))), expression);
    }

    @Test void retainsMultipleDpsButDefaultFormulaDoesNotGuessHowToCombineThem() {
        var input = new ForwardBatch("batch", List.of(new ForwardBatch.Partition(0, List.of(request("a", 1, 0))),
            new ForwardBatch.Partition(1, List.of(request("b", 2, 0)))));
        assertEquals(2, input.dps().size());
        assertThrows(IllegalArgumentException.class, () -> new FormulaForward("totalInputTokens").estimate(input));
    }

    @Test void validatesAndFreezesInputWithoutChangingEngineState() {
        var mutable = new ArrayList<>(List.of(request("a", 1, 0)));
        var dp = new ForwardBatch.Partition(0, mutable); mutable.clear();
        assertEquals(1, dp.requests().size());
        assertThrows(UnsupportedOperationException.class, () -> dp.requests().clear());
        assertThrows(IllegalArgumentException.class, () -> request("a", 1, 2));
        assertThrows(IllegalArgumentException.class, () -> new FormulaForward("1").estimate(batch()));
    }

    @Test void formulaConstantsAreScopedValidatedAndDoNotChangeOtherModels() {
        var constants = new HashMap<>(Map.of("cpSize", 4.0));
        var model = new FormulaForward("cpSize + batchSize + totalComputeTokens", constants);
        constants.put("cpSize", 100.0);
        assertEquals(Duration.ofMillis(7), model.estimate(batch(request("a", 2, 0))));
        assertThrows(IllegalArgumentException.class, () -> new FormulaForward("cpSize"));
        assertThrows(IllegalArgumentException.class, () -> new FormulaForward("1", Map.of("batchSize", 2.0)));
        assertThrows(IllegalArgumentException.class, () -> new FormulaForward("1", Map.of("cpSize", Double.NaN)));
        assertThrows(IllegalArgumentException.class, () -> new FormulaForward("sum(cpSize)", Map.of("cpSize", 4.0)));
    }
}
