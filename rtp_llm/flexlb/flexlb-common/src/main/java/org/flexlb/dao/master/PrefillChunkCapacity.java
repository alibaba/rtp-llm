package org.flexlb.dao.master;

import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

/**
 * Consumes a single engine's grant under the worker generation lock.
 * Repeated observations retain spent tokens and slots; cancelled requests never refund an
 * uncertain dispatch. A newer chunk replaces the previous immutable grant.
 */
public final class PrefillChunkCapacity {
    private PrefillChunk definition;

    private long remainingTokens;
    private int remainingRequests;
    private long stepTokenBudget;

    private String epoch;

    private long latestChunkId;

    private final Set<String> previousEpochs = new HashSet<>();

    public void update(PrefillChunk next, long budget) {
        stepTokenBudget = Math.max(0L, budget);
        if (next == null || !next.isValid() || budget <= 0) {
            definition = null;
            return;
        }
        if (!Objects.equals(epoch, next.engineEpoch())) {
            if (previousEpochs.contains(next.engineEpoch())) {
                return;
            }
            if (epoch != null) {
                previousEpochs.add(epoch);
            }
            epoch = next.engineEpoch();
            latestChunkId = 0;
            definition = null;
        }
        if (next.chunkId() <= latestChunkId) {
            // A cleared/expired grant cannot be reopened by a repeated report.
            return;
        }
        latestChunkId = next.chunkId();
        definition = next;
        remainingTokens = next.inputTokenCapacity();
        remainingRequests = next.requestCapacity();
    }

    public long stepTokenBudget() {
        return stepTokenBudget;
    }

    public PrefillChunk definition() {
        return definition;
    }

    public PrefillChunk reserve(long inputTokens, long nowMs, long requestDeadlineMs) {
        if (remainingTokens(inputTokens, nowMs, requestDeadlineMs) == Long.MAX_VALUE) {
            return null;
        }
        remainingTokens -= inputTokens;
        remainingRequests--;
        return definition;
    }

    public long remainingTokens(long inputTokens, long nowMs, long requestDeadlineMs) {
        return definition != null
                        && nowMs < definition.dispatchDeadlineMs()
                        && definition.dispatchDeadlineMs() < requestDeadlineMs
                        && remainingRequests > 0
                        && inputTokens > 0
                        && remainingTokens >= inputTokens
                ? remainingTokens
                : Long.MAX_VALUE;
    }

    public void refundUnpublished(PrefillChunk chunk, long tokens) {
        if (definition == chunk) {
            remainingTokens = Math.min(definition.inputTokenCapacity(), remainingTokens + tokens);
            remainingRequests = Math.min(definition.requestCapacity(), remainingRequests + 1);
        }
    }
}
