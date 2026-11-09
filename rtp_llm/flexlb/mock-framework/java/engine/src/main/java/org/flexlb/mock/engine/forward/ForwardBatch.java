package org.flexlb.mock.engine.forward;

import java.util.*;

/** Engine supplies the chosen batch and already-resolved cache hits. No cache or scheduling happens here. */
public record ForwardBatch(String id, List<Request> requests) {
    public ForwardBatch {
        nonblank(id);
        requests = List.copyOf(requests);
    }
    public record Request(String id, long inputTokens, long hitCacheTokens, long outputTokens,
                          long generatedTokens, long stepTokens, int priority, List<Image> images) {
        public Request {
            nonblank(id);
            if (inputTokens < 0 || hitCacheTokens < 0 || hitCacheTokens > inputTokens || outputTokens < 0
                    || generatedTokens < 0 || stepTokens < 0)
                throw new IllegalArgumentException("Invalid request lengths/cache hit");
            Math.addExact(inputTokens, generatedTokens);
            images = List.copyOf(images);
        }
        public long computeTokens() { return inputTokens - hitCacheTokens; }
        public long contextTokens() { return Math.addExact(inputTokens, generatedTokens); }
    }
    /** Original image shape only; no patching, resizing or feature-token calculation. */
    public record Image(String id, int width, int height, int channels) {
        public Image {
            nonblank(id);
            if (width <= 0 || height <= 0 || channels <= 0) throw new IllegalArgumentException("Invalid image shape");
        }
        public long pixels() { return (long) width * height; }
    }
    private static void nonblank(String value) {
        if (value == null || value.isBlank()) throw new IllegalArgumentException("Identity must be nonempty");
    }
}
