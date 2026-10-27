package org.flexlb.mock.compute;

import java.time.Duration;
import java.util.*;
import java.util.function.ToDoubleFunction;
import org.flexlb.balance.prediction.ArithmeticFormula;

/** Uses the real Master's formula engine; no copied parser or historical calibration fallback. */
public final class FormulaForward implements ForwardModel {
    private static final Map<String, ToDoubleFunction<ForwardBatch.Request>> REQUEST_VARIABLES = variables();
    private static final Map<String, Integer> INDICES;
    private static final Set<String> BATCH_VARIABLES;
    static {
        Map<String, Integer> indices = new LinkedHashMap<>();
        indices.put("batchSize", indices.size()); indices.put("dpRank", indices.size());
        Set<String> batch = new HashSet<>(indices.keySet());
        for (String name : REQUEST_VARIABLES.keySet()) {
            indices.put(name, indices.size());
            for (String prefix : List.of("total", "max")) {
                String aggregate = prefix + Character.toUpperCase(name.charAt(0)) + name.substring(1);
                indices.put(aggregate, indices.size()); batch.add(aggregate);
            }
        }
        INDICES = Collections.unmodifiableMap(indices); BATCH_VARIABLES = Set.copyOf(batch);
    }

    private final ArithmeticFormula formula;

    public FormulaForward(String expression) {
        if (expression == null || expression.isBlank() || expression.length() > 32768)
            throw new IllegalArgumentException("Compute expression must contain 1..32768 characters");
        formula = ArithmeticFormula.parse(expression, INDICES, BATCH_VARIABLES, true);
    }

    public static Set<String> variableNames() { return INDICES.keySet(); }

    @Override public Duration estimate(ForwardBatch batch) {
        if (batch.dps().size() != 1)
            throw new IllegalArgumentException("FormulaForward currently supports one DP; multiple DPs need an explicit model");
        var dp = batch.dps().getFirst();
        if (dp.requests().isEmpty()) throw new IllegalArgumentException("Forward requires a nonempty batch");
        double[] outer = new double[INDICES.size()];
        Arrays.fill(outer, Double.NaN); // Per-request names outside sum() are deliberately not batch totals.
        outer[INDICES.get("batchSize")] = dp.requests().size();
        outer[INDICES.get("dpRank")] = dp.rank();
        List<double[]> items = new ArrayList<>(dp.requests().size());
        for (var request : dp.requests()) {
            double[] row = new double[INDICES.size()];
            Arrays.fill(row, Double.NaN);
            for (var entry : REQUEST_VARIABLES.entrySet()) row[INDICES.get(entry.getKey())] = entry.getValue().applyAsDouble(request);
            items.add(row);
        }
        for (String name : REQUEST_VARIABLES.keySet()) {
            double total = 0, maximum = Double.NEGATIVE_INFINITY;
            for (var row : items) {
                double value = row[INDICES.get(name)]; total += value; maximum = Math.max(maximum, value);
            }
            String suffix = Character.toUpperCase(name.charAt(0)) + name.substring(1);
            outer[INDICES.get("total" + suffix)] = total; outer[INDICES.get("max" + suffix)] = maximum;
        }
        return milliseconds(formula.evaluateAsDouble(outer, items));
    }

    static Duration milliseconds(double value) {
        double nanos = value * 1e6;
        if (!Double.isFinite(nanos) || nanos < 0 || nanos >= Long.MAX_VALUE / 4.0)
            throw new IllegalArgumentException("Formula/noise must produce finite, nonnegative, representable milliseconds; use request variables inside sum()");
        return Duration.ofNanos(Math.round(nanos));
    }

    private static Map<String, ToDoubleFunction<ForwardBatch.Request>> variables() {
        var values = new LinkedHashMap<String, ToDoubleFunction<ForwardBatch.Request>>();
        values.put("inputTokens", ForwardBatch.Request::inputTokens);
        values.put("hitCacheTokens", ForwardBatch.Request::hitCacheTokens);
        values.put("computeTokens", ForwardBatch.Request::computeTokens);
        values.put("hasHitCache", r -> r.hitCacheTokens() > 0 ? 1 : 0);
        values.put("outputTokens", ForwardBatch.Request::outputTokens);
        values.put("generatedTokens", ForwardBatch.Request::generatedTokens);
        values.put("contextTokens", ForwardBatch.Request::contextTokens);
        values.put("stepTokens", ForwardBatch.Request::stepTokens);
        values.put("priority", ForwardBatch.Request::priority);
        values.put("imageCount", r -> r.images().size());
        values.put("imagePixels", r -> r.images().stream().mapToDouble(ForwardBatch.Image::pixels).sum());
        return Collections.unmodifiableMap(values);
    }
}
