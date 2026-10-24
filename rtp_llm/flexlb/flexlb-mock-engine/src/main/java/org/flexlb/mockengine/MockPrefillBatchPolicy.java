package org.flexlb.mockengine;

/** FIFO execution budgets, matching FIFOScheduler token admission.
 * Physical KV leases remain a separate admission gate; these are execution budgets.
 */
record MockPrefillBatchPolicy(int maxRequests, long maxTokens,
                             long maxSeqLen, int cpSize, boolean forceSingle,
                             long uncachedStop, int maxInitedKvStreams,
                             boolean cpEnabled) {
    Budget newBudget() { return new Budget(true); }
    Budget newBudget(boolean applyUncachedQuota) { return new Budget(applyUncachedQuota); }

    boolean allowsKvInitialization(long initializedStreams, boolean alreadyInitialized) {
        return alreadyInitialized || maxInitedKvStreams == 0 || initializedStreams < maxInitedKvStreams;
    }

    final class Budget {
        private final boolean applyUncachedQuota;
        private Boolean prefillOnly;
        private int count;
        private long fullTokens;
        private long computeTokens;
        private long maxLength;
        private long sequences;

        Budget(boolean applyUncachedQuota) { this.applyUncachedQuota = applyUncachedQuota; }

        private long paddedCompute(long full, long hit) {
            long compute = Math.max(0, full - hit);
            long alignment = cpSize > 1 ? 2L * cpSize : 1;
            return (compute + alignment - 1) / alignment * alignment;
        }

        boolean fits(long full, long hit, int width) {
            return fits(full, hit, width, false);
        }

        boolean fits(long full, long hit, int width, boolean candidatePrefillOnly) {
            if (full < 0 || hit < 0 || hit > full || width <= 0) {
                throw new IllegalArgumentException("invalid FIFO request shape");
            }
            if (prefillOnly != null && prefillOnly != candidatePrefillOnly) return false;
            if (count >= maxRequests || (cpEnabled && forceSingle && count > 0)
                    || (applyUncachedQuota && uncachedStop > 0 && computeTokens >= uncachedStop)) return false;
            if (count == 0 && Math.max(0, full - hit) < maxSeqLen) return true;
            // Full logical tokens include reused prefixes. Check multiplication
            // by division so even oversized synthetic inputs cannot overflow.
            long available = maxTokens - (candidatePrefillOnly ? 0 : 1);
            if (fullTokens > available
                    || (full > 0 && width > (available - fullTokens) / full)) return false;
            long longest = Math.max(maxLength, full);
            if (longest > 0 && sequences + width > available / longest) return false;
            return true;
        }

        void add(long full, long hit, int width) {
            add(full, hit, width, false);
        }

        void add(long full, long hit, int width, boolean candidatePrefillOnly) {
            prefillOnly = candidatePrefillOnly;
            count++;
            fullTokens = Math.addExact(fullTokens, Math.multiplyExact(full, width));
            computeTokens = Math.addExact(computeTokens, Math.multiplyExact(paddedCompute(full, hit), width));
            maxLength = Math.max(maxLength, full);
            sequences += width;
        }
    }
}
