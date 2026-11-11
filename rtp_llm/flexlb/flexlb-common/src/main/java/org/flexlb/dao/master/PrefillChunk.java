package org.flexlb.dao.master;

/**
 * An engine's immutable grant of raw input tokens and request slots.
 * The dispatch deadline is Unix epoch milliseconds, not the local freeze time.
 */
public record PrefillChunk(
        String engineEpoch,
        long chunkId,
        long inputTokenCapacity,
        int requestCapacity,
        long dispatchDeadlineMs) {
    public boolean isValid() {
        return engineEpoch != null
                && !engineEpoch.isBlank()
                && chunkId > 0
                && inputTokenCapacity > 0
                && requestCapacity > 0
                && dispatchDeadlineMs > 0;
    }
}
