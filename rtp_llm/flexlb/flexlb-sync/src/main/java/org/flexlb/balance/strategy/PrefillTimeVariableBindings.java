package org.flexlb.balance.strategy;

import org.flexlb.balance.scheduler.BatchItem;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.ToDoubleFunction;

final class PrefillTimeVariableBindings {

    private static final String BATCH_SIZE = "batchSize";
    private static final Map<String, ToDoubleFunction<RequestStats>> REQUEST_BINDINGS = requestBindings();

    private PrefillTimeVariableBindings() {
    }

    static boolean supports(String name) {
        return BATCH_SIZE.equals(name) || REQUEST_BINDINGS.containsKey(name);
    }

    static EvaluationVariables singleRequestVariables(long totalTokens, long hitCacheTokens) {
        RequestStats request = requestStats(totalTokens, hitCacheTokens);
        Map<String, Double> topLevelVars = requestVars(request);
        topLevelVars.put(BATCH_SIZE, 1.0);
        Map<String, Double> requestVars = new HashMap<>(topLevelVars);
        requestVars.remove(BATCH_SIZE);
        return new EvaluationVariables(topLevelVars, List.of(requestVars));
    }

    static EvaluationVariables batchVariables(List<BatchItem> items) {
        List<Map<String, Double>> itemVars = new ArrayList<>(items.size());
        for (BatchItem item : items) {
            RequestStats request = requestStats(item.seqLen(), item.hitCache());
            itemVars.add(requestVars(request));
        }
        Map<String, Double> topLevelVars = new HashMap<>();
        topLevelVars.put(BATCH_SIZE, (double) items.size());
        return new EvaluationVariables(topLevelVars, itemVars);
    }

    private static RequestStats requestStats(long totalTokens, long hitCacheTokens) {
        long inputTokens = Math.max(0L, totalTokens);
        long boundedHitCacheTokens = Math.max(0L, Math.min(hitCacheTokens, inputTokens));
        return new RequestStats(inputTokens, boundedHitCacheTokens);
    }

    private static Map<String, Double> requestVars(RequestStats request) {
        return bind(REQUEST_BINDINGS, request);
    }

    private static <T> Map<String, Double> bind(Map<String, ToDoubleFunction<T>> bindings, T source) {
        Map<String, Double> vars = new HashMap<>(bindings.size());
        bindings.forEach((name, value) -> vars.put(name, value.applyAsDouble(source)));
        return vars;
    }

    private static Map<String, ToDoubleFunction<RequestStats>> requestBindings() {
        Map<String, ToDoubleFunction<RequestStats>> bindings = new LinkedHashMap<>();
        bindings.put("inputTokens", request -> request.inputTokens);
        bindings.put("hitCacheTokens", request -> request.hitCacheTokens);
        bindings.put("computeTokens", RequestStats::computeTokens);
        bindings.put("hasHitCache", RequestStats::hasHitCache);
        return Collections.unmodifiableMap(bindings);
    }

    private static final class RequestStats {
        private final long inputTokens;
        private final long hitCacheTokens;

        private RequestStats(long inputTokens, long hitCacheTokens) {
            this.inputTokens = inputTokens;
            this.hitCacheTokens = hitCacheTokens;
        }

        private double computeTokens() {
            return inputTokens - hitCacheTokens;
        }

        private double hasHitCache() {
            return hitCacheTokens > 0 ? 1.0 : 0.0;
        }
    }

    record EvaluationVariables(Map<String, Double> topLevelVars,
                               List<Map<String, Double>> itemVars) {
    }
}
