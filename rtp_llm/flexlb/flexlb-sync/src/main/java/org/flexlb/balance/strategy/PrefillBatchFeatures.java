package org.flexlb.balance.strategy;

import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

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

    /**
     * Build shape-aware predictor input for one aggregate placement request.
     *
     * <p>Aggregate placements intentionally omit item-local cache keys, so their conservative
     * cache-hit value is zero. Missing, negative, overflowing, or sum-mismatched wire data returns
     * {@code null}; callers then preserve the rolling-upgrade-compatible aggregate estimate.
     */
    public static PrefillBatchFeatures fromAggregateDemand(BalanceContext context) {
        if (context == null || !context.isAggregateDemand() || context.getRequest() == null) {
            return null;
        }
        List<Long> seqLens = context.getBatchSeqLens();
        if (seqLens == null || seqLens.isEmpty() || context.getRequest().getSeqLen() < 0L) {
            return null;
        }

        List<Item> features = new ArrayList<>(seqLens.size());
        long total = 0L;
        for (Long boxedSeqLen : seqLens) {
            if (boxedSeqLen == null || boxedSeqLen < 0L
                    || boxedSeqLen > Long.MAX_VALUE - total) {
                return null;
            }
            total += boxedSeqLen;
            features.add(new Item(boxedSeqLen, 0L));
        }
        if (total != context.getRequest().getSeqLen()) {
            return null;
        }
        return new PrefillBatchFeatures(features);
    }

    /**
     * Build request-granular members for an aggregate placement reservation.
     *
     * <p>During a rolling upgrade either aligned list may be absent. Invalid, incomplete, or
     * duplicate metadata therefore keeps the legacy aggregate reservation (an empty member list)
     * instead of inventing identities that could reconcile the wrong Engine task.
     */
    public static List<BatchItem> aggregateReservationItems(BalanceContext context) {
        PrefillBatchFeatures features = fromAggregateDemand(context);
        if (features == null) {
            return List.of();
        }
        List<Long> requestIds = context.getBatchRequestIds();
        if (requestIds == null || requestIds.size() != features.items().size()
                || requestIds.isEmpty()
                || requestIds.get(0) == null
                || requestIds.get(0) != context.getRequestId()) {
            return List.of();
        }

        Set<Long> uniqueIds = new HashSet<>(requestIds.size());
        List<BatchItem> reservations = new ArrayList<>(requestIds.size());
        for (int i = 0; i < requestIds.size(); i++) {
            Long requestId = requestIds.get(i);
            if (requestId == null || !uniqueIds.add(requestId)) {
                return List.of();
            }
            reservations.add(BatchItem.placementReservation(
                    requestId, features.items().get(i).seqLen()));
        }
        return List.copyOf(reservations);
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
