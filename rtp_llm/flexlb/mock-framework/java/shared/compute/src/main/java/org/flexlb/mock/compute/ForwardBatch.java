package org.flexlb.mock.compute;

import java.util.*;

/** Engine supplies the chosen batch and already-resolved cache hits. No cache or scheduling happens here. */
public record ForwardBatch(String id, List<Partition> dps) {
    public ForwardBatch {
        nonblank(id); dps = List.copyOf(dps);
        if (dps.isEmpty() || dps.stream().map(Partition::rank).distinct().count() != dps.size())
            throw new IllegalArgumentException("Batch needs distinct DP ranks");
    }
    public record Partition(int rank, List<Request> requests) {
        public Partition {
            if (rank < 0) throw new IllegalArgumentException("DP rank must be nonnegative");
            requests = List.copyOf(requests);
        }
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
