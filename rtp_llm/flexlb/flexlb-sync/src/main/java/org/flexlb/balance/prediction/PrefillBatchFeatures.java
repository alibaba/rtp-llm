package org.flexlb.balance.prediction;

import java.util.ArrayList;
import java.util.List;
import java.util.function.ToLongFunction;

import static com.google.common.base.Preconditions.checkArgument;

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

    public record Item(long seqLen, long hitCache) implements ArithmeticFormula.Variables {
        public Item {
            checkArgument(seqLen >= 0L, "seqLen must be non-negative");
            checkArgument(hitCache >= 0L && hitCache <= seqLen, "hitCache must be in [0, seqLen]");
        }

        @Override
        public double variable(int index) {
            return switch (index) {
                case PrefillTimeFormula.IDX_INPUT_TOKENS -> seqLen;
                case PrefillTimeFormula.IDX_HIT_CACHE_TOKENS -> hitCache;
                case PrefillTimeFormula.IDX_COMPUTE_TOKENS -> seqLen - hitCache;
                case PrefillTimeFormula.IDX_HAS_HIT_CACHE -> hitCache > 0L ? 1.0 : 0.0;
                default -> 0.0;
            };
        }
    }
}
