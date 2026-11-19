package org.flexlb.mock.protocol;

import java.util.*;
import java.util.function.IntUnaryOperator;
import org.flexlb.mock.protocol.grpc.MockEngineProtocol.MockDecodePromptPB;

/**
 * Stores the prompt length and keys for complete cache blocks. Hashes advance at physical block
 * boundaries, and Prefill retains one key per group of physical blocks when CP is enabled.
 */
public final class PromptCacheKeys {
    private final int physicalBlockSize, tokensPerKey, inputLength;

    private final List<Long> promptKeys;

    public PromptCacheKeys(int physicalBlockSize, List<Integer> tokens, List<Span> spans) { this(physicalBlockSize, tokens, spans, 0); }

    public PromptCacheKeys(int physicalBlockSize, List<Integer> tokens, List<Span> spans, long seed) {
        this(physicalBlockSize, tokens, spans, seed, 1);
    }

    public PromptCacheKeys(int physicalBlockSize, List<Integer> tokens, List<Span> spans, long seed, int cpSize) {
        if (physicalBlockSize < 1 || cpSize < 1) throw new IllegalArgumentException("Invalid cache block size");
        long previous = 0;
        for (var span : spans) {
            long end = (long) span.offset() + span.rowHashes().size();
            if (span.offset() < previous || span.rowHashes().isEmpty() || end > tokens.size())
                throw new IllegalArgumentException("Invalid multimodal cache span");
            previous = end;
        }
        this.physicalBlockSize = physicalBlockSize;
        tokensPerKey = Math.multiplyExact(physicalBlockSize, cpSize);
        inputLength = tokens.size();
        promptKeys = computePromptKeys(tokens, spans, seed);
    }

    private PromptCacheKeys(PromptCacheKeys physical, int cpSize) {
        physicalBlockSize = physical.physicalBlockSize;
        tokensPerKey = Math.multiplyExact(physicalBlockSize, cpSize);
        inputLength = physical.inputLength;
        var keys = new ArrayList<Long>(physical.promptKeys.size() / cpSize);
        for (int i = cpSize - 1; i < physical.promptKeys.size(); i += cpSize)
            keys.add(physical.promptKeys.get(i));
        promptKeys = List.copyOf(keys);
    }

    private PromptCacheKeys(MockDecodePromptPB prompt, int expectedPhysicalBlockSize) {
        if (expectedPhysicalBlockSize < 1 || prompt.getBlockSize() != expectedPhysicalBlockSize || prompt.getInputLength() < 1
                || prompt.getBlockKeysCount() != prompt.getInputLength() / expectedPhysicalBlockSize)
            throw new IllegalArgumentException("Invalid Decode prompt cache identity");
        physicalBlockSize = expectedPhysicalBlockSize;
        tokensPerKey = physicalBlockSize;
        inputLength = prompt.getInputLength();
        promptKeys = prompt.getBlockKeysList();
    }

    /**
     * Keeps the last key in each complete group of CP physical blocks.
     */
    public PromptCacheKeys toPrefillKeys(int cpSize) {
        if (cpSize < 1 || tokensPerKey != physicalBlockSize)
            throw new IllegalArgumentException("Expected physical prompt keys");
        return cpSize == 1 ? this : new PromptCacheKeys(this, cpSize);
    }

    public static PromptCacheKeys fromDecodePrompt(MockDecodePromptPB prompt, int physicalBlockSize) {
        return new PromptCacheKeys(prompt, physicalBlockSize);
    }

    public MockDecodePromptPB toDecodePrompt() {
        if (tokensPerKey != physicalBlockSize) throw new IllegalStateException("Expected physical prompt keys");
        return MockDecodePromptPB.newBuilder().setInputLength(inputLength).setBlockSize(physicalBlockSize)
            .addAllBlockKeys(promptKeys).build();
    }

    public int physicalBlockSize() { return physicalBlockSize; }

    public int tokensPerKey() { return tokensPerKey; }

    public List<Long> keys() { return promptKeys; }

    public int inputLength() { return inputLength; }

    @Override public boolean equals(Object other) {
        return other instanceof PromptCacheKeys keys && physicalBlockSize == keys.physicalBlockSize && tokensPerKey == keys.tokensPerKey
            && inputLength == keys.inputLength && promptKeys.equals(keys.promptKeys);
    }

    @Override public int hashCode() { return Objects.hash(physicalBlockSize, tokensPerKey, inputLength, promptKeys); }

    private List<Long> computePromptKeys(List<Integer> tokens, List<Span> spans, long seed) {
        int completeTokens = inputLength - inputLength % tokensPerKey;
        var keys = new ArrayList<Long>(completeTokens / tokensPerKey);
        long hash = seed;
        int spanIndex = 0;
        IntUnaryOperator tokenAt = tokens instanceof com.google.protobuf.Internal.IntList ints ? ints::getInt : tokens::get;
        // Hash at physical block boundaries and retain only complete logical block keys.
        for (int start = 0; start < completeTokens; start += physicalBlockSize) {
            for (int i = start; i < start + physicalBlockSize; i++)
                hash = roll(hash, tokenAt.applyAsInt(i));
            while (spanIndex < spans.size() && spans.get(spanIndex).offset() + spans.get(spanIndex).rowHashes().size() <= start)
                spanIndex++;
            for (int j = spanIndex; j < spans.size() && spans.get(j).offset() < start + physicalBlockSize; j++) {
                var span = spans.get(j);
                for (int value : mediaHeader(span, start, start + physicalBlockSize)) hash = roll(hash, value);
                int begin = Math.max(start, span.offset()) - span.offset();
                int end = Math.min(start + physicalBlockSize - span.offset(), span.rowHashes().size());
                for (int i = begin; i < end; i++) hash = roll(hash, span.rowHashes().get(i));
            }
            if ((start + physicalBlockSize) % tokensPerKey == 0) keys.add(hash);
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
