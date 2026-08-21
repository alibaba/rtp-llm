package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;

/**
 * Endpoint hooks for Voyage-style {@code /v1/reranker} requests.
 *
 * <p>FE applies {@code sorted} and {@code top_k} inside each request. Those operations are not
 * distributive across chunks: concatenating each shard's top-k cannot recover the global top-k.
 * Every child request therefore asks FE for its complete, input-ordered scores
 * ({@code sorted=false}, no {@code top_k}); after the generic merger has verified that every
 * successful child returned one result per document, this class rebases local result indices,
 * performs one stable global sort, applies the caller's top-k once, and sums total tokens.
 */
public final class RerankerMerger implements BatchEndpointSpec.ChunkBodyTransformer,
        BatchEndpointSpec.PostMerger, BatchEndpointSpec.RequestValidator {

    public static final RerankerMerger INSTANCE = new RerankerMerger();

    private RerankerMerger() {}

    @Override
    public String validate(JSONObject body) {
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

    @Override
    public void apply(JSONObject originalBody, List<JSONObject> chunkBodies) {
        for (JSONObject chunkBody : chunkBodies) {
            // FE must return every score in document order. Global request semantics are restored
            // after all chunks arrive; applying either operation here loses information.
            chunkBody.put("sorted", false);
            chunkBody.remove("top_k");
        }
    }

    @Override
    public void apply(JSONObject mergedBody, List<SubBatchResult> subs, List<Integer> failedIndices,
                      BatchEndpointSpec spec, JSONObject originalRequest) {
        // BatchHandler fails closed for this endpoint. Do not attempt to sort the generic null
        // placeholders: this body is discarded in favor of a 500 response.
        if (!failedIndices.isEmpty()) {
            return;
        }

        JSONArray results = mergedBody.getJSONArray(spec.getResponseArrayField());
        if (results == null) {
            throw new IllegalStateException("reranker response is missing results");
        }

        long totalTokens = 0;
        for (SubBatchResult sub : subs) {
            if (!ResponseMerger.wellFormed(sub, spec)) {
                throw new IllegalStateException("reranker sub-batch is not well formed");
            }
            Object tokenValue = sub.body().get("total_tokens");
            if (!(tokenValue instanceof Number tokenNumber)) {
                throw new IllegalStateException("reranker response is missing total_tokens");
            }
            totalTokens = Math.addExact(totalTokens, tokenNumber.longValue());

            JSONArray localResults = sub.body().getJSONArray(spec.getResponseArrayField());
            for (Object value : localResults) {
                if (!(value instanceof JSONObject item)) {
                    throw new IllegalStateException("reranker result item must be an object");
                }
                Object indexValue = item.get("index");
                if (!(indexValue instanceof Number indexNumber)) {
                    throw new IllegalStateException("reranker result item is missing index");
                }
                long localIndex = indexNumber.longValue();
                if (localIndex < 0 || localIndex >= sub.chunkSize()) {
                    throw new IllegalStateException("reranker result index is outside its chunk");
                }
                item.put("index", Math.addExact(sub.startIndex(), Math.toIntExact(localIndex)));
                scoreOf(item); // Validate before sorting, including the sorted=false path.
            }
        }
        mergedBody.put("total_tokens", totalTokens);

        boolean sorted = originalRequest == null
                || !originalRequest.containsKey("sorted")
                || originalRequest.getBooleanValue("sorted");
        if (sorted && results.size() > 1) {
            // List.sort is stable, so equal scores retain original document order just like
            // Python's stable list.sort in RerankerRenderer.
            List<Object> ranked = new ArrayList<>(results);
            ranked.sort((left, right) -> compareScores((JSONObject) left, (JSONObject) right));
            results.clear();
            results.addAll(ranked);
        }

        if (originalRequest != null && originalRequest.get("top_k") != null) {
            BigInteger topK = integralValue(originalRequest.get("top_k"));
            if (topK == null) {
                // validate() runs before fanout; retain a defensive guard for direct unit callers.
                throw new IllegalArgumentException("top_k must be an integer or null");
            }
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
        int keep = end.intValueExact();
        while (values.size() > keep) {
            values.remove(values.size() - 1);
        }
    }

    /** Returns null for non-integral JSON values. */
    private static BigInteger integralValue(Object value) {
        if (value instanceof BigInteger integer) {
            return integer;
        }
        if (value instanceof BigDecimal decimal) {
            try {
                return decimal.toBigIntegerExact();
            } catch (ArithmeticException ignored) {
                return null;
            }
        }
        if (value instanceof Byte || value instanceof Short
                || value instanceof Integer || value instanceof Long) {
            return BigInteger.valueOf(((Number) value).longValue());
        }
        if (value instanceof Float || value instanceof Double) {
            double number = ((Number) value).doubleValue();
            if (!Double.isFinite(number) || number != Math.rint(number)) {
                return null;
            }
            return BigDecimal.valueOf(number).toBigIntegerExact();
        }
        return null;
    }
}
