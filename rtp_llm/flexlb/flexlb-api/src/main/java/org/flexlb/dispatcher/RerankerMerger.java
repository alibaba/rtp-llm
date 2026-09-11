package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.List;

/**
 * Request full unsorted child results, then apply global sorting and top_k once. Partial failures fail
 * the request.
 */
public final class RerankerMerger {

    private RerankerMerger() {}

    public static String validate(JSONObject body) {
        if (!(body.get("query") instanceof String)) {
            return "query must be a string";
        }
        if (body.containsKey("sorted") && !(body.get("sorted") instanceof Boolean)) {
            return "sorted must be a boolean";
        }
        Object topK = body.get("top_k");
        if (topK != null && integralValue(topK) == null) {
            return "top_k must be an integer or null";
        }
        return null;
    }

    public static void prepare(JSONObject chunkBody) {
        // Restore global sorting and top-k only after all scores arrive.
        chunkBody.put("sorted", false);
        chunkBody.remove("top_k");
    }

    public static void merge(JSONObject mergedBody, List<SubBatchResult> subs, List<Integer> failedIndices,
                             JSONObject originalRequest) {
        BatchEndpointSpec spec = BatchEndpointSpec.RERANKER;
        // BatchHandler fails closed for this endpoint. Do not attempt to sort the generic null
        // placeholders: this body is discarded in favor of a 500 response.
        if (!failedIndices.isEmpty()) {
            return;
        }

        JSONArray results = mergedBody.getJSONArray(spec.getResponseArrayField());

        long totalTokens = 0;
        for (SubBatchResult sub : subs) {
            Object tokenValue = sub.body().get("total_tokens");
            if (!(tokenValue instanceof Number tokenNumber)) {
                throw new IllegalStateException("reranker response is missing total_tokens");
            }
            totalTokens = Math.addExact(totalTokens, tokenNumber.longValue());

            JSONArray localResults = sub.body().getJSONArray(spec.getResponseArrayField());
            boolean[] seen = new boolean[sub.chunkSize()];
            for (Object value : localResults) {
                if (!(value instanceof JSONObject item)) {
                    throw new IllegalStateException("reranker result item must be an object");
                }
                Object indexValue = item.get("index");
                BigInteger exactIndex = integralValue(indexValue);
                if (exactIndex == null) {
                    throw new IllegalStateException("reranker result item is missing index");
                }
                final int localIndex;
                try {
                    localIndex = exactIndex.intValueExact();
                } catch (ArithmeticException outOfRange) {
                    throw new IllegalStateException(
                            "reranker result index is outside its chunk", outOfRange);
                }
                if (localIndex < 0 || localIndex >= sub.chunkSize()) {
                    throw new IllegalStateException("reranker result index is outside its chunk");
                }
                if (seen[localIndex]) {
                    throw new IllegalStateException("reranker result indices must be unique within a chunk");
                }
                seen[localIndex] = true;
                item.put("index", Math.addExact(sub.startIndex(), localIndex));
                scoreOf(item); // Validate before sorting, including the sorted=false path.
            }

        }
        mergedBody.put("total_tokens", totalTokens);

        boolean sorted = !originalRequest.containsKey("sorted")
                || originalRequest.getBooleanValue("sorted");
        if (sorted && results.size() > 1) {
            // List.sort is stable, so equal scores retain original document order just like
            // Python's stable list.sort in RerankerRenderer.
            results.sort((left, right) -> compareScores((JSONObject) left, (JSONObject) right));
        }

        if (originalRequest.get("top_k") != null) {
            BigInteger topK = integralValue(originalRequest.get("top_k"));
            truncateLikePython(results, topK);
        }
    }

    private static int compareScores(JSONObject left, JSONObject right) {
        double leftScore = scoreOf(left);
        double rightScore = scoreOf(right);
        // Treat signed zero as a tie, matching Python float equality and preserving input order.
        if (leftScore == rightScore) {
            return 0;
        }
        return Double.compare(rightScore, leftScore);
    }

    private static double scoreOf(JSONObject item) {
        Object value = item.get("relevance_score");
        if (!(value instanceof Number number)) {
            throw new IllegalStateException("reranker result item is missing relevance_score");
        }
        double score = number.doubleValue();
        if (!Double.isFinite(score)) {
            throw new IllegalStateException("reranker relevance_score must be finite");
        }
        return score;
    }

    /** Mirrors Python {@code values[:min(len(values), top_k)]}, including negative top_k. */
    private static void truncateLikePython(JSONArray values, BigInteger topK) {
        BigInteger size = BigInteger.valueOf(values.size());
        BigInteger end = topK.min(size);
        if (end.signum() < 0) {
            end = size.add(end);
            if (end.signum() < 0) {
                end = BigInteger.ZERO;
            }
        }
        values.subList(end.intValueExact(), values.size()).clear();
    }

    /** Returns null for non-integral JSON values. */
    private static BigInteger integralValue(Object value) {
        if (!(value instanceof Number number)) {
            return null;
        }
        try {
            return switch (number) {
                case BigInteger integer -> integer;
                case BigDecimal decimal -> decimal.toBigIntegerExact();
                case Float f -> BigDecimal.valueOf(f.doubleValue()).toBigIntegerExact();
                case Double d -> BigDecimal.valueOf(d).toBigIntegerExact();
                default -> BigInteger.valueOf(number.longValue());
            };
        } catch (ArithmeticException | NumberFormatException ignored) {
            return null;
        }
    }
}
