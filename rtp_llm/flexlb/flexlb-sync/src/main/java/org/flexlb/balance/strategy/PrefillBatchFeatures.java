package org.flexlb.balance.strategy;

import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;

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

    /**
     * Rebuild payload-free {@link BatchItem} views for predictors compiled
     * against the legacy learning callback.
     */
    public List<BatchItem> toBatchItems() {
        return items.stream().map(item -> {
            Request request = new Request();
            request.setSeqLen(item.seqLen());
            BalanceContext context = new BalanceContext();
            context.setRequest(request);

            ServerStatus prefill = new ServerStatus();
            DebugInfo debugInfo = new DebugInfo();
            debugInfo.setHitCacheLen(item.hitCache());
            prefill.setDebugInfo(debugInfo);
            return new BatchItem(context, null, null, prefill,
                    null, null, null, 0);
        }).toList();
    }

    public record Item(long seqLen, long hitCache) {}
}
