package org.flexlb.balance.prediction;

/**
 * Prefill-time prediction contract.
 *
 * <p>One scheduling operation captures one immutable {@link Evaluator}. Online
 * learning may publish a replacement evaluator concurrently, but it cannot
 * change an evaluation already in progress.
 */
public interface PrefillTimePredictor {

    /** Capture the immutable model used by one scheduling operation. */
    Evaluator evaluator();

    interface Evaluator {
        /** Estimate one request from its input and cache-hit token counts. */
        long estimateMs(long totalTokens, long hitTokens);

        /** Estimate one payload-free batch. */
        double predictBatchMs(PrefillBatchFeatures features);
    }

    enum LearningResult {
        MODEL_UNCHANGED,
        MODEL_UPDATED
    }

    /**
     * Publish learning from one completed batch when applicable.
     *
     * @return whether a replacement evaluator was published
     */
    LearningResult learn(
            PrefillBatchFeatures features, long predictedMs, long actualMs);
}
