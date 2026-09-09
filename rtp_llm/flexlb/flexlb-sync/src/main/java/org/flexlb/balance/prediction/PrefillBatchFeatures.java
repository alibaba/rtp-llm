package org.flexlb.balance.prediction;

import java.util.ArrayList;
import java.util.List;
import java.util.function.ToLongFunction;

/** Immutable, payload-free features retained for prediction and learning. */
public record PrefillBatchFeatures(List<Item> items) {

    public PrefillBatchFeatures {
        items = List.copyOf(items);
    }

    /**
     * Materialize predictor features without depending on a scheduling item
     * type.
     */
    public static <T> PrefillBatchFeatures from(
            List<T> source,
            ToLongFunction<? super T> seqLen,
            ToLongFunction<? super T> hitCache) {
        List<Item> features = new ArrayList<>(source.size());
        for (T item : source) {
            features.add(new Item(
                    seqLen.applyAsLong(item),
                    hitCache.applyAsLong(item)));
        }
        return new PrefillBatchFeatures(features);
    }

    public int batchSize() {
        return items.size();
    }

    public record Item(long seqLen, long hitCache) {
        public Item {
            if (seqLen < 0L) {
                throw new IllegalArgumentException(
                        "seqLen must be non-negative");
            }
            if (hitCache < 0L || hitCache > seqLen) {
                throw new IllegalArgumentException(
                        "hitCache must be in [0, seqLen]");
            }
        }
    }
}
