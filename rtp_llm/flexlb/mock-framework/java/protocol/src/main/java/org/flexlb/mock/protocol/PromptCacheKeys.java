package org.flexlb.mock.protocol;

import java.util.*;
import java.util.function.IntUnaryOperator;
import org.flexlb.engine.grpc.EngineRpcService.MockDecodePromptPB;

/**
 * Immutable prompt length and complete cache block keys.
 */
public final class PromptCacheKeys {
    private final int blockSize, keyStride, inputLength;

    private final List<Long> promptKeys;

    public PromptCacheKeys(int blockSize, List<Integer> tokens, List<Span> spans) { this(blockSize, tokens, spans, 0); }

    public PromptCacheKeys(int blockSize, List<Integer> tokens, List<Span> spans, long seed) {
        this(blockSize,tokens,spans,seed,1);
    }

    public PromptCacheKeys(int blockSize, List<Integer> tokens, List<Span> spans, long seed, int keyStride) {
        if (blockSize < 1 || keyStride < 1) throw new IllegalArgumentException("Invalid cache block size");
        long previous = 0;
        for (var span : spans) {
            long end = (long) span.offset() + span.rowHashes().size();
            if (span.offset() < previous || span.rowHashes().isEmpty() || end > tokens.size())
                throw new IllegalArgumentException("Invalid multimodal cache span");
            previous = end;
        }
        this.blockSize = Math.multiplyExact(blockSize, keyStride);
        this.keyStride = keyStride;
        inputLength = tokens.size();
        promptKeys = computePromptKeys(tokens, spans, seed, blockSize);
    }

    private PromptCacheKeys(PromptCacheKeys physical, int stride) {
        keyStride = stride;
        blockSize = Math.multiplyExact(physical.blockSize, stride);
        inputLength = physical.inputLength;
        var keys = new ArrayList<Long>(physical.promptKeys.size() / stride);
        for (int i = stride - 1; i < physical.promptKeys.size(); i += stride)
            keys.add(physical.promptKeys.get(i));
        promptKeys = List.copyOf(keys);
    }

    private PromptCacheKeys(MockDecodePromptPB prompt, int expectedBlockSize) {
        if (expectedBlockSize < 1 || prompt.getBlockSize() != expectedBlockSize || prompt.getInputLength() < 1
                || prompt.getBlockKeysCount() != prompt.getInputLength() / expectedBlockSize)
            throw new IllegalArgumentException("Invalid Decode prompt cache identity");
        blockSize = expectedBlockSize;
        keyStride = 1;
        inputLength = prompt.getInputLength();
        promptKeys = prompt.getBlockKeysList();
    }

    /** Projects an already hashed physical sequence into Prefill's logical blocks. */
    public PromptCacheKeys withStride(int stride) {
        if (stride < 1 || keyStride != 1)
            throw new IllegalArgumentException("Expected physical prompt keys");
        return stride == 1 ? this : new PromptCacheKeys(this, stride);
    }

    public static PromptCacheKeys fromDecodePrompt(MockDecodePromptPB prompt, int blockSize) {
        return new PromptCacheKeys(prompt, blockSize);
    }

    public MockDecodePromptPB decodePrompt() {
        if (keyStride != 1) throw new IllegalStateException("Expected physical prompt keys");
        return MockDecodePromptPB.newBuilder().setInputLength(inputLength).setBlockSize(blockSize)
            .addAllBlockKeys(promptKeys).build();
    }

    public int blockSize() { return blockSize; }

    public List<Long> keys() { return promptKeys; }

    public int inputLength() { return inputLength; }

    @Override public boolean equals(Object other) {
        return other instanceof PromptCacheKeys sequence && blockSize == sequence.blockSize && keyStride == sequence.keyStride
            && inputLength == sequence.inputLength && promptKeys.equals(sequence.promptKeys);
    }

    @Override public int hashCode() { return Objects.hash(blockSize, keyStride, inputLength, promptKeys); }

    private List<Long> computePromptKeys(List<Integer> tokens, List<Span> spans, long seed, int hashBlockSize) {
        int completeTokens = inputLength - inputLength % blockSize;
        var keys = new ArrayList<Long>(completeTokens / blockSize);
        long hash = seed;
        int spanIndex = 0;
        IntUnaryOperator tokenAt = tokens instanceof com.google.protobuf.Internal.IntList ints ? ints::getInt : tokens::get;
        // Hash at physical block boundaries and retain only complete logical block keys.
        for (int start = 0; start < completeTokens; start += hashBlockSize) {
            for (int i = start; i < start + hashBlockSize; i++)
                hash = roll(hash, tokenAt.applyAsInt(i));
            while (spanIndex < spans.size() && spans.get(spanIndex).offset() + spans.get(spanIndex).rowHashes().size() <= start)
                spanIndex++;
            for (int j = spanIndex; j < spans.size() && spans.get(j).offset() < start + hashBlockSize; j++) {
                var span = spans.get(j);
                for (int value : mediaHeader(span, start, start + hashBlockSize)) hash = roll(hash, value);
                int begin = Math.max(start, span.offset()) - span.offset();
                int end = Math.min(start + hashBlockSize - span.offset(), span.rowHashes().size());
                for (int i = begin; i < end; i++) hash = roll(hash, span.rowHashes().get(i));
            }
            if ((start + hashBlockSize) % blockSize == 0) keys.add(hash);
        }
        return List.copyOf(keys);
    }

    private static long roll(long hash, long value) {
        return hash ^ (value + 0x9e3779b97f4a7c15L + (hash << 12) + (hash >> 32));
    }

    private static int[] mediaHeader(Span span, int start, int end) {
        int begin = Math.max(start, span.offset());
        int limit = Math.min(end, span.offset() + span.rowHashes().size());
        return new int[]{-42, begin - start, span.rowHashes().size(), begin - span.offset(), limit - begin};
    }

    /**
     * Stores the prompt offset and cache identities of the rows for one media input.
     */
    public record Span(int offset, List<Integer> rowHashes) {
        public Span { rowHashes = List.copyOf(rowHashes); }
    }
}
