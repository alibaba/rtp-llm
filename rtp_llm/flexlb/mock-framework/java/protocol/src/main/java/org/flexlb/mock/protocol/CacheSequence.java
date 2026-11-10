package org.flexlb.mock.protocol;

import java.util.*;

/**
 * Immutable semantic input, cache-only media identity and model-owned cache key namespace.
 */
public final class CacheSequence {
    private final int blockSize, hashBlockSize, keyStride;

    private final List<Integer> tokens;
    private final List<Span> spans;

    private final long seed;

    private final List<Long> promptKeys;

    public CacheSequence(int blockSize, List<Integer> tokens, List<Span> spans) { this(blockSize, tokens, spans, 0); }

    public CacheSequence(int blockSize, List<Integer> tokens, List<Span> spans, long seed) {
        this(blockSize,tokens,spans,seed,1);
    }

    public CacheSequence(int blockSize, List<Integer> tokens, List<Span> spans, long seed, int keyStride) {
        if (blockSize < 1 || keyStride < 1) throw new IllegalArgumentException("Invalid cache block size");
        tokens = List.copyOf(tokens); spans = List.copyOf(spans);
        long previous = 0;
        for (var span : spans) {
            long end = (long) span.offset() + span.rowHashes().size();
            if (span.offset() < previous || span.rowHashes().isEmpty() || end > tokens.size())
                throw new IllegalArgumentException("Invalid multimodal cache span");
            previous = end;
        }
        this.hashBlockSize=blockSize; this.keyStride=keyStride;
        this.blockSize = Math.multiplyExact(blockSize,keyStride); this.tokens = tokens; this.spans = spans; this.seed = seed;
        promptKeys = appendKeys(List.of(), 0, 0);
    }

    public int blockSize() { return blockSize; }

    public List<Integer> tokens() { return tokens; }

    public List<Span> spans() { return spans; }

    public long seed() { return seed; }

    public List<Long> keys() { return promptKeys; }

    /** Append only output tokens whose KV has actually been computed. Mock output repeats one token. */
    public List<Long> keys(int computedOutputTokens, int outputToken) {
        if (computedOutputTokens < 0) throw new IllegalArgumentException("Negative computed output length");
        if (computedOutputTokens == 0) return promptKeys;
        return appendKeys(promptKeys, computedOutputTokens, outputToken);
    }

    @Override public boolean equals(Object other) {
        return other instanceof CacheSequence sequence && hashBlockSize == sequence.hashBlockSize && keyStride == sequence.keyStride && seed == sequence.seed
            && tokens.equals(sequence.tokens) && spans.equals(sequence.spans);
    }

    @Override public int hashCode() { return Objects.hash(hashBlockSize, keyStride, tokens, spans, seed); }

    @Override public String toString() {
        return "CacheSequence[blockSize=" + blockSize + ", hashBlockSize=" + hashBlockSize + ", keyStride=" + keyStride + ", tokens=" + tokens + ", spans=" + spans + ", seed=" + seed + "]";
    }

    private List<Long> appendKeys(List<Long> prefix, int computedOutputTokens, int outputToken) {
        int length = Math.addExact(tokens.size(), computedOutputTokens);
        int completeTokens = length - length % blockSize;
        int prefixTokens = prefix.size() * blockSize;
        if (completeTokens == prefixTokens) return prefix;

        var keys = new ArrayList<>(prefix);
        long hash = prefix.isEmpty() ? seed : prefix.getLast();
        int spanIndex = 0;
        // Hash at physical block boundaries and retain only complete logical block keys.
        for (int start = prefixTokens; start < completeTokens; start += hashBlockSize) {
            for (int i = start; i < start + hashBlockSize; i++)
                hash = roll(hash, i < tokens.size() ? tokens.get(i) : outputToken);
            while (spanIndex < spans.size() && spans.get(spanIndex).offset() + spans.get(spanIndex).rowHashes().size() <= start)
                spanIndex++;
            for (int j = spanIndex; j < spans.size() && spans.get(j).offset() < start + hashBlockSize; j++) {
                var span = spans.get(j);
                int begin = Math.max(start, span.offset());
                int end = Math.min(start + hashBlockSize, span.offset() + span.rowHashes().size());
                if (begin >= end) continue;
                for (int value : new int[]{-42, begin - start, span.rowHashes().size(), begin - span.offset(), end - begin})
                    hash = roll(hash, value);
                for (int i = begin; i < end; i++) hash = roll(hash, span.rowHashes().get(i - span.offset()));
            }
            if ((start + hashBlockSize) % blockSize == 0) keys.add(hash);
        }
        return List.copyOf(keys);
    }

    private static long roll(long hash, long value) {
        return hash ^ (value + 0x9e3779b97f4a7c15L + (hash << 12) + (hash >> 32));
    }

    /**
     * Stores the prompt offset and cache identities of the rows for one media input.
     */
    public record Span(int offset, List<Integer> rowHashes) {
        public Span { rowHashes = List.copyOf(rowHashes); }
    }
}
