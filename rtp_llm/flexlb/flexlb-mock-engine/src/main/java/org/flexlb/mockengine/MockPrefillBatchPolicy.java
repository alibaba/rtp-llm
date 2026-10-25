package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;

/** Test-side FIFO execution budgets, matching FIFOScheduler token admission.
 * Physical KV leases remain a separate admission gate; these are execution budgets.
 */
record MockPrefillBatchPolicy(int maxRequests, long maxTokens, long maxKvLen,
                             long maxSeqLen, int cpSize, boolean forceSingle,
                             long uncachedStop, int maxWaitingRequests, int maxInitedKvStreams,
                             boolean faultLimitsEnabled, boolean cpEnabled) {
    static MockPrefillBatchPolicy load(JsonNode node) {
        if (node.isMissingNode() || node.isNull()) return null;
        MockPrefillBatchPolicy policy = new MockPrefillBatchPolicy(
                node.path("max_requests").asInt(0), node.path("max_batch_tokens").asLong(0),
                node.path("max_batch_kv_len").asLong(0), node.path("max_seq_len").asLong(0),
                node.path("cp_size").asInt(1), node.path("force_single").asBoolean(true),
                node.path("max_batch_tokens_without_cache").asLong(0),
                node.path("max_waiting_requests").asInt(0), node.path("max_inited_kv_streams").asInt(0),
                node.path("fault_limits_enabled").asBoolean(false),
                node.path("cp_enabled").asBoolean(node.path("cp_size").asInt(1) > 1));
        if (policy.maxRequests <= 0 || policy.maxTokens <= 0 || policy.maxKvLen < 0
                || policy.maxSeqLen <= 0 || policy.cpSize <= 0 || policy.uncachedStop < 0
                || policy.maxWaitingRequests < 0 || policy.maxInitedKvStreams < 0
                || (!policy.cpEnabled && policy.cpSize != 1)) {
            throw new IllegalArgumentException("prefill.fifo requires positive max_requests, "
                    + "max_batch_tokens, max_seq_len, cp_size and nonnegative remaining limits");
        }
        return policy;
    }

    Budget newBudget() { return new Budget(); }

    boolean allowsKvInitialization(long initializedStreams, boolean alreadyInitialized) {
        return alreadyInitialized || maxInitedKvStreams == 0 || initializedStreams < maxInitedKvStreams;
    }

    final class Budget {
        private int count;
        private long fullTokens;
        private long computeTokens;
        private long maxLength;
        private long sequences;

        private long paddedCompute(long full, long hit) {
            long compute = Math.max(0, full - hit);
            long alignment = cpSize > 1 ? 2L * cpSize : 1;
            return (compute + alignment - 1) / alignment * alignment;
        }

        boolean fits(long full, long hit, int width) {
            if (full < 0 || hit < 0 || hit > full || width <= 0) {
                throw new IllegalArgumentException("invalid FIFO request shape");
            }
            if (count >= maxRequests || (cpEnabled && forceSingle && count > 0)
                    || (uncachedStop > 0 && computeTokens >= uncachedStop)) return false;
            if (count == 0 && Math.max(0, full - hit) < maxSeqLen) return true;
            // Full logical tokens include reused prefixes. Check multiplication
            // by division so even oversized synthetic inputs cannot overflow.
            if (fullTokens >= maxTokens
                    || (full > 0 && width > (maxTokens - fullTokens - 1) / full)) return false;
            long longest = Math.max(maxLength, full);
            if (longest > 0 && sequences + width > (maxTokens - 1) / longest) return false;
            // Captured mock-only limits are inert unless explicitly requested
            // for a fault experiment; they never replace the real token gates.
            return !faultLimitsEnabled || maxKvLen == 0
                    || (fullTokens < maxKvLen
                        && (full == 0 || width <= (maxKvLen - fullTokens - 1) / full));
        }

        void add(long full, long hit, int width) {
            count++;
            fullTokens = Math.addExact(fullTokens, Math.multiplyExact(full, width));
            computeTokens = Math.addExact(computeTokens, Math.multiplyExact(paddedCompute(full, hit), width));
            maxLength = Math.max(maxLength, full);
            sequences += width;
        }
    }
}
