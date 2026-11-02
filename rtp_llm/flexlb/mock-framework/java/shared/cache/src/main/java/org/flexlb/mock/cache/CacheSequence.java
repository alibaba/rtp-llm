package org.flexlb.mock.cache;

import java.util.*;

/** Immutable semantic input and cache-only media identity; no model or RPC dependency. */
public record CacheSequence(int blockSize, List<Integer> tokens, List<Span> spans, long seed) {
    public CacheSequence(int blockSize, List<Integer> tokens, List<Span> spans) { this(blockSize, tokens, spans, 0); }
    public record Span(int offset, List<Integer> rowHashes) {
        public Span { rowHashes = List.copyOf(rowHashes); }
    }
    public CacheSequence {
        if (blockSize < 1) throw new IllegalArgumentException("Invalid cache block size");
        tokens = List.copyOf(tokens); spans = List.copyOf(spans);
        long previous = 0;
        for (var span : spans) {
            long end = (long) span.offset() + span.rowHashes().size();
            if (span.offset() < previous || span.rowHashes().isEmpty() || end > tokens.size())
                throw new IllegalArgumentException("Invalid multimodal cache span");
            previous = end;
        }
    }
    public List<Long> keys() { return keys(0, 0); }

    /** Append only output tokens whose KV has actually been computed. Mock output repeats one token. */
    public List<Long> keys(int computedOutputTokens, int outputToken) {
        if (computedOutputTokens < 0) throw new IllegalArgumentException("Negative computed output length");
        int length = Math.addExact(tokens.size(), computedOutputTokens);
        var keys = new ArrayList<Long>(); long hash = seed;
        for (int start = 0; (long) start + blockSize <= length; start += blockSize) {
            for (int i = start; i < start + blockSize; i++)
                hash = roll(hash, i < tokens.size() ? tokens.get(i) : outputToken);
            for (var span : spans) {
                int begin = Math.max(start, span.offset());
                int end = Math.min(start + blockSize, span.offset() + span.rowHashes().size());
                if (begin >= end) continue;
                for (int value : new int[]{-42, begin - start, span.rowHashes().size(), begin - span.offset(), end - begin})
                    hash = roll(hash, value);
                for (int i = begin; i < end; i++) hash = roll(hash, span.rowHashes().get(i - span.offset()));
            }
            keys.add(hash);
        }
        return List.copyOf(keys);
    }
    private static long roll(long hash, long value) {
        return hash ^ (value + 0x9e3779b97f4a7c15L + (hash << 12) + (hash >> 32));
    }
}
