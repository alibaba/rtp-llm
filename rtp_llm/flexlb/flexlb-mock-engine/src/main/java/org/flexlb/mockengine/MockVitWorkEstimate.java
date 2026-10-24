package org.flexlb.mockengine;

/** MMWorkEstimate: sums for additive dimensions; maximum for attention segment. */
record MockVitWorkEstimate(long inputPatches, long outputTokens, long workspaceBytes,
                           long maxAttentionSegment, long attentionWork) {
    static final MockVitWorkEstimate ZERO = new MockVitWorkEstimate(0, 0, 0, 0, 0);
    MockVitWorkEstimate {
        if (inputPatches < 0 || outputTokens < 0 || workspaceBytes < 0 || maxAttentionSegment < 0 || attentionWork < 0)
            throw new IllegalArgumentException("ViT work dimensions must be nonnegative");
    }
    MockVitWorkEstimate plus(MockVitWorkEstimate other) {
        return new MockVitWorkEstimate(Math.addExact(inputPatches, other.inputPatches),
                Math.addExact(outputTokens, other.outputTokens), Math.addExact(workspaceBytes, other.workspaceBytes),
                Math.max(maxAttentionSegment, other.maxAttentionSegment), Math.addExact(attentionWork, other.attentionWork));
    }
    boolean fits(MockVitWorkEstimate budget) {
        return within(inputPatches, budget.inputPatches) && within(outputTokens, budget.outputTokens)
                && within(workspaceBytes, budget.workspaceBytes) && within(maxAttentionSegment, budget.maxAttentionSegment)
                && within(attentionWork, budget.attentionWork);
    }
    private static boolean within(long value, long limit) { return limit == 0 || value <= limit; }
}
