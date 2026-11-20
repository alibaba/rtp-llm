package org.flexlb.balance.strategy;

import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;

import java.lang.ref.WeakReference;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/** A dispatch hint may delay a duplicate fill but must never increase an estimated hit length. */
final class PendingPrefixFills {
    static final long MAX_WAIT_MS = 3_000L;
    private final Cache<Key, Fill> pending = CacheBuilder.newBuilder().maximumSize(16_384)
            .expireAfterWrite(MAX_WAIT_MS, TimeUnit.MILLISECONDS).build();
    // A partial index update can move the first miss into a fill already being computed.
    // Keep its terminal block as a second lookup; limit probes on arbitrarily long cold prompts.
    private static final int TERMINAL_PROBES = 64;
    private final Cache<Long, Fill> terminals = CacheBuilder.newBuilder().maximumSize(16_384)
            .expireAfterWrite(MAX_WAIT_MS, TimeUnit.MILLISECONDS).build();
    private final Consumer<PrefillEndpoint> reconsider;

    PendingPrefixFills(Consumer<PrefillEndpoint> reconsider) {
        this.reconsider = reconsider;
    }

    record Key(RoleType role, String group, long blockSize, int offset, long root, long first, long second) { }
    record Suffix(Key key, int lastIndex, long lastBlock) { }

    static Suffix suffix(Request request, RoleType role, String group, long hitTokens) {
        long size = request.getCacheKeyBlockSize();
        var keys = request.getBlockCacheKeys();
        if (size <= 0 || keys == null || hitTokens < 0 || hitTokens % size != 0) { return null; }
        int complete = (int) Math.min(keys.size(), request.getSeqLen() / size);
        long index = hitTokens / size;
        if (complete - index < 2) { return null; }
        int offset = (int) index;
        var key = new Key(role, group, size, offset, keys.getFirst(), keys.get(offset), keys.get(offset + 1));
        return new Suffix(key, complete - 1, keys.get(complete - 1));
    }

    // Accessed under the strategy's selection lock. Retain no prompt or RPC payload.
    private static final class Fill {
        final Suffix suffix;
        final WeakReference<RequestContext> owner;
        final PrefillEndpoint endpoint;
        final long requestId;
        final int priority;
        final long untilMs;
        boolean armed;

        Fill(Suffix suffix, RequestContext ctx, PrefillEndpoint endpoint, int priority, long untilMs) {
            this.suffix = suffix;
            this.owner = new WeakReference<>(ctx);
            this.endpoint = endpoint;
            this.requestId = ctx.data().getRequestId();
            this.priority = priority;
            this.untilMs = untilMs;
        }

        boolean confirmed(RequestContext ctx) {
            if (endpoint.isGenerationRetiringOrRetired()) { return false; }
            var state = ctx.snapshot();
            if (state.pendingEnd() != null || state.cancellation() != null && state.cancellation().reason() != null
                    || state.delivery() != null && state.delivery().abandonment() != null
                    || state.outcome() != null && state.outcome().errorType() != null) { return false; }
            var response = ctx.data().getFuture();
            if (response == null) { return false; }
            try {
                var reply = response.getNow(null);
                if (reply == null || !reply.isSuccess() || reply.getServerStatus() == null) { return false; }
                var topology = endpoint.getStatus().topologySnapshot();
                return reply.getServerStatus().stream().anyMatch(status -> status.isSuccess() && status.getRequestId() == requestId
                        && status.getRole() == suffix.key().role()
                        && topology.ip().equals(status.getServerIp()) && topology.port() == status.getHttpPort());
            } catch (CancellationException | CompletionException failed) {
                return false;
            }
        }
    }

    PrefillEndpoint waitingFor(Suffix suffix, Request request, int priority, Set<String> leaders,
                               long latestWakeMs, long nowMs) {
        if (suffix == null) { return null; }
        Fill fill = findProducer(suffix, request, leaders, latestWakeMs, nowMs);
        if (fill == null) { return null; }
        // A confirmed producer has already left Master placement. With one cache leader both requests
        // must use that Worker. Preserve a higher priority's option to use an alternative immediately.
        var tasks = fill.endpoint.getStatus().committedEngineObservation().runningTaskList();
        var task = tasks.get(Long.toString(fill.requestId));
        if (task != null && (task.errorCode() != 0 || task.endTimeMs() > 0
                || task.priorityPreemptionProgress() != PriorityPreemptionProgress.NONE)) { return null; }
        if (priority > fill.priority && leaders.size() > 1 && (task == null || task.phase() != TaskPhase.RUNNING)) {
            return null;
        }
        if (!fill.armed) {
            fill.armed = true;
            CompletableFuture.delayedExecutor(fill.untilMs - nowMs, TimeUnit.MILLISECONDS)
                    .execute(() -> reconsider.accept(fill.endpoint));
        }
        return fill.endpoint;
    }

    private Fill findProducer(Suffix suffix, Request request, Set<String> leaders, long latestWakeMs, long nowMs) {
        var exact = pending.getIfPresent(suffix.key());
        if (overlaps(exact, suffix, request, leaders, latestWakeMs, nowMs)) { return exact; }
        var keys = request.getBlockCacheKeys();
        int first = Math.max(suffix.key().offset() + 1, suffix.lastIndex() - TERMINAL_PROBES + 1);
        for (int i = suffix.lastIndex(); i >= first; i--) {
            var producer = terminals.getIfPresent(keys.get(i));
            if (producer != null && producer.suffix.lastIndex() == i
                    && overlaps(producer, suffix, request, leaders, latestWakeMs, nowMs)) { return producer; }
        }
        return null;
    }

    private static boolean overlaps(Fill fill, Suffix suffix, Request request, Set<String> leaders,
                                    long latestWakeMs, long nowMs) {
        if (fill == null || fill.requestId == request.getRequestId() || nowMs >= fill.untilMs
                || fill.untilMs > latestWakeMs || !leaders.contains(fill.endpoint.ipPort())) { return false; }
        var source = fill.suffix.key();
        var target = suffix.key();
        if (source.role() != target.role() || source.blockSize() != target.blockSize()
                || !Objects.equals(source.group(), target.group())) { return false; }
        var owner = fill.owner.get();
        if (owner == null) { return false; }
        // Queue ordering can dispatch a longer turn first. Check the complete shorter prefix,
        // including interior blocks: a terminal-key collision alone establishes no dependency.
        int end = Math.min(fill.suffix.lastIndex(), suffix.lastIndex());
        if (end <= Math.max(source.offset(), target.offset())) { return false; }
        var offered = request.getBlockCacheKeys();
        var warming = owner.data().getRequest().getBlockCacheKeys();
        return warming.get(end).equals(offered.get(end))
                && warming.subList(0, end + 1).equals(offered.subList(0, end + 1))
                && fill.confirmed(owner);
    }

    void remember(Suffix suffix, RequestContext ctx, PrefillEndpoint endpoint, int priority,
                  long nowMs, long expiresAtMs) {
        if (suffix == null) { return; }
        long duration = Math.min(MAX_WAIT_MS, (expiresAtMs - nowMs) / 10);
        if (duration > 0) {
            var fill = new Fill(suffix, ctx, endpoint, priority, nowMs + duration);
            pending.put(suffix.key(), fill);
            terminals.put(suffix.lastBlock(), fill);
        }
    }
}
