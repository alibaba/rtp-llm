package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.strategy.CacheTokenMatch;
import org.flexlb.config.RoutingConfig.QueuedRequestPolicy;
import org.flexlb.dao.master.CacheStatus;

import java.util.List;
import java.util.function.Predicate;

/** Chooses one eligible request from a snapshot of the source's dispatch queue. */
final class QueuedMigrationCandidates {
    /** One position per source. The migration scan advances it after each choice. */
    static final class Cursor {
        private long sequence;
        private int index = -1;
        private QueuedRequestPolicy policy;
        private boolean fromTail;

        void reset() { index = -1; }

        private int previousIndex(List<ScheduledRequest> items, QueuedRequestPolicy policy, boolean fromTail) {
            if (index < 0 || this.policy != policy || this.fromTail != fromTail) { return -1; }
            if (index < items.size() && items.get(index).enqueueSeq() == sequence) { return index; }
            // Dispatch and new arrivals can shift the previous request within the queue.
            for (int i = items.size() - 1; i >= 0; i--) {
                if (items.get(i).enqueueSeq() == sequence) { return i; }
            }
            return -1;
        }
    }

    private record Scored(int index, long hits, long input) {
        private int compareRate(Scored other) {
            return PrefillCostOrder.compareTokenRatios(hits, input, other.hits, other.input);
        }

        private int compareTo(Scored other, boolean highestFirst) {
            int byRate = compareRate(other);
            if (byRate == 0) { return Integer.compare(other.index, index); }
            return highestFirst ? -byRate : byRate;
        }
    }

    static ScheduledRequest candidate(PrefillEndpoint source, QueuedRequestPolicy policy,
                                      boolean costOrdered, Predicate<ScheduledRequest> eligible, Cursor cursor) {
        if (policy == QueuedRequestPolicy.DISABLED) {
            cursor.reset();
            return null;
        }
        var items = source.captureQueueSnapshot().items();
        boolean fromTail = costOrdered && policy == QueuedRequestPolicy.LOWEST_HIT_RATE_FIRST;
        int previous = cursor.previousIndex(items, policy, fromTail);
        int next = fromTail || items.size() <= 1 ? nextFromTail(items, previous, eligible)
                : nextByHitRate(items, previous, eligible, source.getStatus().getCacheStatus(), policy);
        cursor.index = next;
        if (next < 0) { return null; }
        cursor.sequence = items.get(next).enqueueSeq();
        cursor.policy = policy;
        cursor.fromTail = fromTail;
        return items.get(next);
    }

    private static int nextFromTail(List<ScheduledRequest> items, int previous, Predicate<ScheduledRequest> eligible) {
        int index = previous;
        for (int seen = 0; seen < items.size(); seen++) {
            index = index <= 0 ? items.size() - 1 : index - 1;
            if (eligible.test(items.get(index))) { return index; }
        }
        return -1;
    }

    private static int nextByHitRate(List<ScheduledRequest> items, int previous, Predicate<ScheduledRequest> eligible,
                                     CacheStatus cache, QueuedRequestPolicy policy) {
        boolean highestFirst = policy == QueuedRequestPolicy.HIGHEST_HIT_RATE_FIRST;
        Scored last = previous < 0 ? null : score(items.get(previous), previous, cache);
        Scored best = null;
        Scored successor = null;
        for (int i = items.size() - 1; i >= 0; i--) {
            if (!eligible.test(items.get(i))) { continue; }
            Scored current = i == previous ? last : score(items.get(i), i, cache);
            if (best == null || current.compareTo(best, highestFirst) < 0) { best = current; }
            if (last != null && current.compareTo(last, highestFirst) > 0
                    && (successor == null || current.compareTo(successor, highestFirst) < 0)) {
                successor = current;
            }
        }
        // Strict minimum only rotates among ties, even if the previous candidate could not move.
        if (successor != null && (policy != QueuedRequestPolicy.MIN_SOURCE_CACHE_HIT_RATE
                || successor.compareRate(best) == 0)) {
            return successor.index();
        }
        return best == null ? -1 : best.index();
    }

    private static Scored score(ScheduledRequest item, int index, CacheStatus cache) {
        long input = item.seqLen();
        if (input <= 0) { return new Scored(index, 0, 1); }
        var request = item.ctx().getRequest();
        var keys = request.getBlockCacheKeys();
        long block = request.getCacheKeyBlockSize();
        if (block <= 0 && cache != null) { block = cache.getBlockSize(); }
        // A missing report or missing key metadata leaves the placement estimate as the best evidence.
        if (cache == null || cache.getCachedKeys() == null || keys == null || block <= 0) {
            return new Scored(index, Math.clamp(item.hitCache(), 0, input), input);
        }
        int matchedBlocks = 0;
        for (Long key : keys) {
            if (key == null || !cache.getCachedKeys().contains(key)) { break; }
            matchedBlocks++;
        }
        return new Scored(index, CacheTokenMatch.fromBlocks(input, block, matchedBlocks).effectiveHitTokens(), input);
    }
}
