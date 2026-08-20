package org.flexlb.balance.strategy;

import org.flexlb.balance.scheduler.BatchItem;

import java.util.List;

/** Lightweight, payload-free input features retained for predictor learning. */
public record PrefillBatchFeatures(List<Item> items) {

    public PrefillBatchFeatures {
        items = items == null ? List.of() : List.copyOf(items);
    }

    public static PrefillBatchFeatures from(List<BatchItem> batchItems) {
        if (batchItems == null || batchItems.isEmpty()) {
            return new PrefillBatchFeatures(List.of());
        }
        return new PrefillBatchFeatures(batchItems.stream()
                .map(item -> new Item(item.seqLen(), item.hitCache()))
                .toList());
    }

    public int batchSize() {
        return items.size();
    }

    public record Item(long seqLen, long hitCache) {}
}
