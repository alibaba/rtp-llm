package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/** Cacheable work still owned by a Prefill worker. All operations use the global queue lock. */
final class PendingPrefillPrefixes {
    // In lexicographic order a longest common prefix is found on either side of the search key.
    private static final Comparator<List<Long>> KEY_ORDER = (a, b) -> {
        int n = commonBlocks(a, b);
        return n == Math.min(a.size(), b.size()) ? Integer.compare(a.size(), b.size())
                : Long.compare(a.get(n), b.get(n));
    };
    private record Scope(String model, RoleType role, String group, long blockSize, List<String> affinityKeys) { }
    // The sequence distinguishes independent producers with the same block keys.
    private record Prefix(List<Long> keys, long sequence) { }
    private static final Comparator<Prefix> PREFIX_ORDER = Comparator.comparing(Prefix::keys, KEY_ORDER)
            .thenComparingLong(Prefix::sequence);
    private record Producer(Scope scope, Prefix prefix, int priority, Set<GlobalQueueEntry> followers) { }
    private record Wait(GlobalQueueEntry entry, Set<GlobalQueueEntry> subscribers, long expectedTokens, long untilMs) { }
    private record CacheObservation(PrefillEndpoint source, long completedAtNanos, long expectedTokens, long untilMs) { }

    // Search the two neighbours in each eligible priority bucket. Lower-priority
    // producers cannot hide a match or make a higher-priority request wait.
    private final Map<Scope, NavigableMap<Integer, NavigableMap<Prefix, Producer>>> pools = new HashMap<>();
    private final Map<RequestExecution, Producer> producers = new IdentityHashMap<>();
    private final Map<GlobalQueueEntry, Wait> waiting = new IdentityHashMap<>();
    private final Map<GlobalQueueEntry, CacheObservation> observations = new IdentityHashMap<>();
    // Cache pollers only read key membership without the queue lock. Sets are owned by the queue.
    private final Map<PrefillEndpoint, Set<GlobalQueueEntry>> cacheWaiters = new ConcurrentHashMap<>();
    private final TreeSet<Wait> deadlines = new TreeSet<>(Comparator.comparingLong(Wait::untilMs)
            .thenComparingLong(wait -> wait.entry().sequence));
    private final Consumer<GlobalQueueEntry> ready;

    PendingPrefillPrefixes(Consumer<GlobalQueueEntry> ready) {
        this.ready = ready;
    }

    private static Scope scope(Request req, ServerStatus prefill) {
        return new Scope(req.getModel(), prefill.getRole(), prefill.getGroup(), req.getCacheKeyBlockSize(),
                req.getCacheAffinityKeys() == null ? List.of() : List.copyOf(req.getCacheAffinityKeys()));
    }

    static List<Long> cacheableBlocks(BalanceContext context) {
        if (context == null || context.getConfig() == null || !context.getConfig().isPrefillCostOrder()
                || context.getRequest() == null) { return List.of(); }
        var affinity = context.getConfig().getRouter().getRoles().getPrefill().getCacheAffinity();
        if (affinity == null || affinity.getMaxExtraTtftMs() <= 0L) { return List.of(); }
        Request req = context.getRequest();
        var keys = req.getBlockCacheKeys();
        if (keys == null || req.getCacheKeyBlockSize() <= 0 || req.getSeqLen() <= 0) { return List.of(); }
        int count = (int) Math.min(keys.size(), req.getSeqLen() / req.getCacheKeyBlockSize());
        var blocks = keys.subList(0, count);
        for (Long key : blocks) { if (key == null) { return List.of(); } }
        return List.copyOf(blocks);
    }

    void published(GlobalQueueEntry entry, RequestExecution item, long residentPrefixTokens) {
        long cacheableTokens = (long) entry.cacheableBlocks.size() * entry.context.getRequest().getCacheKeyBlockSize();
        if (item.prefillEnded || residentPrefixTokens >= cacheableTokens) { return; }
        Scope scope = scope(entry.context.getRequest(), item.prefill());
        var priorities = pools.computeIfAbsent(scope, ignored -> new TreeMap<>());
        var pool = priorities.computeIfAbsent(entry.priority, ignored -> new TreeMap<>(PREFIX_ORDER));
        Prefix prefix = new Prefix(entry.cacheableBlocks, entry.sequence);
        Producer producer = new Producer(scope, prefix, entry.priority, new HashSet<>());
        pool.put(prefix, producer);
        producers.put(item, producer);
    }

    boolean awaitBetterPrefix(GlobalQueueEntry entry, ServerStatus selected, long resident, long nowMs) {
        // A finished producer and its newly resident blocks arrive on independent RPCs.
        var observation = observations.get(entry);
        if (observation != null) {
            if (resident < observation.expectedTokens() && nowMs < observation.untilMs()) {
                if (observation.source().awaitingCachePollAfter(observation.completedAtNanos())) {
                    var subscribers = cacheWaiters.computeIfAbsent(observation.source(), ignored -> new HashSet<>());
                    park(entry, subscribers, observation.expectedTokens(), observation.untilMs());
                    // Publish the subscription before rechecking the watermark to avoid a lost poll.
                    if (observation.source().awaitingCachePollAfter(observation.completedAtNanos())) { return true; }
                }
                // The index may have advanced during this placement lookup. Retry once against the new view.
                wake(entry);
                return true;
            }
            removeWait(entry);
        }
        Request req = entry.context.getRequest();
        var priorities = pools.get(scope(req, selected));
        if (priorities == null) { return false; }
        RoutingConfig.CacheAffinityConfig affinity = entry.context.getConfig().getRouter().getRoles().getPrefill()
                .getCacheAffinity();
        long elapsed = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - entry.queuedAtNanos);
        if (affinity == null || elapsed >= affinity.getMaxExtraTtftMs()) { return false; }
        long blockSize = req.getCacheKeyBlockSize();
        if (resident >= (long) entry.cacheableBlocks.size() * blockSize) { return false; }
        Prefix target = new Prefix(entry.cacheableBlocks, entry.sequence);
        Producer best = null;
        int bestBlocks = 0;
        for (var pool : priorities.tailMap(entry.priority, true).descendingMap().values()) {
            var left = pool.floorEntry(target);
            var right = pool.ceilingEntry(target);
            int leftBlocks = left == null ? 0 : commonBlocks(left.getKey().keys(), target.keys());
            int rightBlocks = right == null ? 0 : commonBlocks(right.getKey().keys(), target.keys());
            int blocks = Math.max(leftBlocks, rightBlocks);
            if (blocks > bestBlocks) {
                bestBlocks = blocks;
                best = leftBlocks >= rightBlocks ? left.getValue() : right.getValue();
                if (blocks == target.keys().size()) { break; }
            }
        }
        long potential = bestBlocks * blockSize;
        if (best == null || potential <= resident || potential * RoutingConfig.PERCENTAGE_SCALE
                < affinity.getMinPrefixHitPercent() * req.getSeqLen()) { return false; }
        long until = nowMs + Math.min(Long.MAX_VALUE - nowMs, affinity.getMaxExtraTtftMs() - elapsed);
        park(entry, best.followers(), potential, until);
        return true;
    }

    private void park(GlobalQueueEntry entry, Set<GlobalQueueEntry> subscribers, long expectedTokens, long untilMs) {
        Wait wait = new Wait(entry, subscribers, expectedTokens, untilMs);
        subscribers.add(entry);
        waiting.put(entry, wait);
        deadlines.add(wait);
    }

    private static int commonBlocks(List<Long> a, List<Long> b) {
        int limit = Math.min(a.size(), b.size());
        int n = 0;
        while (n < limit && a.get(n).equals(b.get(n))) { n++; }
        return n;
    }

    boolean isWaiting(GlobalQueueEntry entry) { return waiting.containsKey(entry); }

    void removeWait(GlobalQueueEntry entry) {
        Wait wait = waiting.remove(entry);
        if (wait != null) {
            wait.subscribers().remove(entry);
            deadlines.remove(wait);
        }
        var observation = observations.remove(entry);
        if (observation != null && wait != null && wait.subscribers().isEmpty()) {
            cacheWaiters.remove(observation.source(), wait.subscribers());
        }
    }

    boolean finished(RequestExecution item, boolean completed) {
        item.prefillEnded = true;
        Producer producer = producers.remove(item);
        if (producer == null) { return false; }
        var priorities = pools.get(producer.scope());
        var pool = priorities.get(producer.priority());
        pool.remove(producer.prefix());
        if (pool.isEmpty()) { priorities.remove(producer.priority()); }
        if (priorities.isEmpty()) { pools.remove(producer.scope()); }
        if (producer.followers().isEmpty()) { return false; }
        long completedAtNanos = System.nanoTime();
        PrefillEndpoint source = item.prefillEp();
        boolean verifyIndex = completed && source != null && source.awaitingCachePollAfter(completedAtNanos);
        for (var entry : new ArrayList<>(producer.followers())) {
            Wait wait = waiting.get(entry);
            wake(entry);
            if (verifyIndex) {
                observations.put(entry, new CacheObservation(source, completedAtNanos, wait.expectedTokens(), wait.untilMs()));
            }
        }
        return true;
    }

    boolean observingCache(PrefillEndpoint source) {
        return cacheWaiters.containsKey(source);
    }

    boolean cachePollCompleted(PrefillEndpoint source) {
        var subscribers = cacheWaiters.get(source);
        if (subscribers == null) { return false; }
        boolean woke = false;
        for (var entry : new ArrayList<>(subscribers)) {
            if (!source.awaitingCachePollAfter(observations.get(entry).completedAtNanos())) {
                wake(entry);
                woke = true;
            }
        }
        return woke;
    }

    private void wake(GlobalQueueEntry entry) {
        removeWait(entry);
        ready.accept(entry);
    }

    long resumeExpired(long nowMs) {
        while (!deadlines.isEmpty() && deadlines.first().untilMs() <= nowMs) { wake(deadlines.first().entry()); }
        return deadlines.isEmpty() ? Long.MAX_VALUE : deadlines.first().untilMs() - nowMs;
    }

    void clear() {
        pools.clear();
        producers.clear();
        waiting.clear();
        deadlines.clear();
        observations.clear();
        cacheWaiters.clear();
    }
}
