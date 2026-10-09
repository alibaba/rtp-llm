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
        /**
         * Identity of the immutable model snapshot behind this evaluator.
         *
         * <p>Independent endpoint predictors may return the same object only
         * when their predictions are guaranteed identical for equal inputs.
         * Projection uses this identity to reuse an equal singleton prediction
         * while it still evaluates every endpoint's queue and cache state.
         */
        default Object snapshotIdentity() {
            return this;
        }

        /** Estimate one request from its input and cache-hit token counts. */
        long estimateMs(long totalTokens, long hitTokens);

        /** Fresh state for one append-only planning operation on this immutable model. */
        default BatchPrediction newBatchPrediction() {
            java.util.List<PrefillBatchFeatures.Item> items = new java.util.ArrayList<>();
            return (seqLen, hitCache) -> {
                items.add(new PrefillBatchFeatures.Item(seqLen, hitCache));
                return predictBatchMs(new PrefillBatchFeatures(items));
            };
        }

        /** Estimate one payload-free batch. */
        double predictBatchMs(PrefillBatchFeatures features);
    }

    /** Not shared or reused after a membership/model change; result retains fractional milliseconds. */
    @FunctionalInterface
    interface BatchPrediction {
        double append(long seqLen, long hitCache);
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
