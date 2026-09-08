package org.flexlb.balance.scheduler;

/**
 * Resource shape of a prefill batch.
 *
 * <p>The Engine executes a context batch as a padded rectangle, so compute
 * admission is based on {@code maxSeqLen * batchSize}. KV admission is based
 * on the sum of the individual sequence lengths.
 */
record BatchShape(int size, long maxSeqLen, long paddedTokens, long kvTokens) {

    static BatchShape empty() {
        return new BatchShape(0, 0, 0, 0);
    }

    BatchShape add(BatchItem item) {
        int nextSize = size + 1;
        long nextMaxSeqLen = Math.max(maxSeqLen, Math.max(0, item.seqLen()));
        return new BatchShape(
                nextSize,
                nextMaxSeqLen,
                saturatedMultiply(nextMaxSeqLen, nextSize),
                saturatedAdd(kvTokens, Math.max(0, item.seqLen())));
    }

    boolean fitsCompute(long capacity) {
        return capacity > 0 && paddedTokens < capacity;
    }

    boolean fitsKv(long capacity) {
        return capacity == Long.MAX_VALUE || (capacity >= 0 && kvTokens <= capacity);
    }

    private static long saturatedAdd(long left, long right) {
        return left > Long.MAX_VALUE - right ? Long.MAX_VALUE : left + right;
    }

    private static long saturatedMultiply(long value, int multiplier) {
        if (value == 0 || multiplier == 0) {
            return 0;
        }
        return value > Long.MAX_VALUE / multiplier ? Long.MAX_VALUE : value * multiplier;
    }
}
