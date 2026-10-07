package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;

/** Test-side equivalent of FIFOScheduler::fitsPrefillTokenLimits at 5d478bb3c.
 * Physical KV leases remain a separate admission gate; these are execution budgets.
 */
record MockPrefillBatchPolicy(int maxRequests, long maxTokens, long maxKvLen,
                             long maxSeqLen, int cpSize, boolean forceSingle,
                             long uncachedStop, int maxWaitingRequests, int maxInitedKvStreams) {
    static MockPrefillBatchPolicy load(JsonNode node) {
        if (node.isMissingNode() || node.isNull()) return null;
        MockPrefillBatchPolicy policy = new MockPrefillBatchPolicy(
                node.path("max_requests").asInt(0), node.path("max_batch_tokens").asLong(0),
                node.path("max_batch_kv_len").asLong(0), node.path("max_seq_len").asLong(0),
                node.path("cp_size").asInt(1), node.path("force_single").asBoolean(false),
                node.path("max_batch_tokens_without_cache").asLong(0),
                node.path("max_waiting_requests").asInt(0), node.path("max_inited_kv_streams").asInt(0));
        if (policy.maxRequests <= 0 || policy.maxTokens <= 0 || policy.maxKvLen < 0
                || policy.maxSeqLen <= 0 || policy.cpSize <= 0 || policy.uncachedStop < 0
                || policy.maxWaitingRequests < 0 || policy.maxInitedKvStreams < 0) {
            throw new IllegalArgumentException("prefill.fifo requires positive max_requests, "
                    + "max_batch_tokens, max_seq_len, cp_size and nonnegative remaining limits");
        }
        return policy;
    }

    Budget newBudget() { return new Budget(); }

    final class Budget {
        private int count;
        private long fullTokens;
        private long computeTokens;
        private long maxLength;

        private long paddedCompute(long full, long hit) {
            long compute = Math.max(0, full - hit);
            long alignment = cpSize > 1 ? 2L * cpSize : 1;
            return (compute + alignment - 1) / alignment * alignment;
        }

        boolean fits(long full, long hit) {
            if (count >= maxRequests || (forceSingle && count > 0)
                    || (uncachedStop > 0 && computeTokens >= uncachedStop)) return false;
            if (count == 0 && Math.max(0, full - hit) < maxSeqLen) return true;
            if (maxKvLen > 0) {
                return fullTokens < maxKvLen && full < maxKvLen - fullTokens
                        && computeTokens < maxTokens
                        && paddedCompute(full, hit) < maxTokens - computeTokens;
            }
            return fullTokens < maxTokens && full < maxTokens - fullTokens
                    && (Math.max(maxLength, full) == 0
                    || count + 1L <= (maxTokens - 1) / Math.max(maxLength, full));
        }

        void add(long full, long hit) {
            count++;
            fullTokens += full;
            computeTokens += paddedCompute(full, hit);
            maxLength = Math.max(maxLength, full);
        }
    }
}
