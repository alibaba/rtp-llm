package org.flexlb.balance.strategy;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import org.flexlb.balance.endpoint.EndpointRegistry.PrefillRoutingEntry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.RoutingConfig.HotspotReleaseConfig;
import org.flexlb.dao.route.RoleType;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.function.LongSupplier;
import java.util.stream.Collectors;

import static com.google.common.math.LongMath.saturatedAdd;

/** Samples outstanding load within the exact role and group used by a placement. */
final class PrefillHotspots {
    private static final long SAMPLE_NANOS = TimeUnit.MILLISECONDS.toNanos(100);
    private static final long LOAD_WINDOW_NANOS = TimeUnit.SECONDS.toNanos(1);
    private static final long SOURCE_WINDOW_NANOS = TimeUnit.SECONDS.toNanos(1);
    private static final long STALE_NANOS = TimeUnit.SECONDS.toNanos(1);

    private record Scope(RoleType role, String group) { }

    private final Cache<Scope, Pool> pools;
    private final LongSupplier nanoTime;

    PrefillHotspots(LongSupplier nanoTime) {
        this.nanoTime = nanoTime;
        pools = Caffeine.newBuilder().ticker(nanoTime::getAsLong)
                .expireAfterAccess(Math.max(STALE_NANOS, SOURCE_WINDOW_NANOS), TimeUnit.NANOSECONDS).build();
    }

    Snapshot sample(RoleType role, String group, List<PrefillRoutingEntry> candidates,
                    HotspotReleaseConfig config) {
        Scope scope = new Scope(role, group);
        if (candidates.isEmpty()) {
            pools.invalidate(scope);
            return Snapshot.EMPTY;
        }
        Pool pool = pools.get(scope, ignored -> new Pool());
        return pool.sample(candidates, config, nanoTime.getAsLong());
    }

    private record Load(long requests, long compute) { }

    /** Strict majorities of same-pool comparisons over recent distinct samples. */
    private record LoadPosition(boolean highRequests, boolean highCompute, boolean lowRequests, boolean lowCompute) { }

    /** Integer loads strictly below the mean are exactly those below its ceiling. */
    private record Signal(long outlierLimit, long meanCeiling) {
        private static Signal measure(long[] values, long floor, double factor) {
            int count = values.length;
            long whole = 0;
            long remainders = 0;
            // Split before adding: the sum of long token counts can overflow.
            for (long value : values) {
                whole += value / count;
                remainders += value % count;
            }
            long meanCeiling = whole + (remainders + count - 1L) / count;
            Arrays.sort(values);
            int mid = count / 2;
            double median = count % 2 == 1 ? values[mid] : 0.5 * values[mid - 1] + 0.5 * values[mid];
            // Requiring a strictly lower peer also excludes uniform pools, even at long precision limits.
            long limit = Math.max(values[0], Math.max(floor, (long) (factor * median)));
            return new Signal(limit, meanCeiling);
        }

        private boolean allows(long source, long target) {
            return source > outlierLimit && target < meanCeiling && target < source;
        }
    }

    private enum HotspotRole { HOT, REDIRECT_SOURCE }

    /** Read-only view shared by every decision within one pool sampling interval. */
    static final class Snapshot {
        private static final Signal NONE = new Signal(Long.MAX_VALUE, 0L);
        static final Snapshot EMPTY = new Snapshot(Map.of(), Map.of(), Map.of(), NONE, NONE);

        private final Map<PrefillEndpoint, Load> loads;
        private final Map<PrefillEndpoint, LoadPosition> positions;
        private final Map<PrefillEndpoint, HotspotRole> hotspots;
        private final Signal requests;
        private final Signal compute;

        private Snapshot(Map<PrefillEndpoint, Load> loads, Map<PrefillEndpoint, LoadPosition> positions,
                         Map<PrefillEndpoint, HotspotRole> hotspots,
                         Signal requests, Signal compute) {
            this.loads = Map.copyOf(loads);
            this.positions = Map.copyOf(positions);
            this.hotspots = Map.copyOf(hotspots);
            this.requests = requests;
            this.compute = compute;
        }

        boolean isHot(PrefillEndpoint endpoint) {
            return hotspots.containsKey(endpoint);
        }

        boolean isRedirectSource(PrefillEndpoint endpoint) {
            return hotspots.get(endpoint) == HotspotRole.REDIRECT_SOURCE;
        }

        boolean mayRedirect(PrefillEndpoint source, PrefillEndpoint target, PrefillTokenWorkload.Snapshot current) {
            return mayRedirect(source, target, loads.get(source), current);
        }

        boolean mayMoveQueuedRequest(PrefillEndpoint source, PrefillEndpoint target) {
            return mayRedirect(source, target, new Load(source.observedRequestCount(), source.pendingTokenWork()),
                    target.tokenWorkloadSnapshot());
        }

        boolean hasQueuedMigrationReceiver(PrefillEndpoint source, List<PrefillRoutingEntry> members) {
            if (!isRedirectSource(source)) { return false; }
            Load from = new Load(source.observedRequestCount(), source.pendingTokenWork());
            if (from.requests() <= requests.outlierLimit() && from.compute() <= compute.outlierLimit()) { return false; }
            for (var entry : members) {
                var target = entry.endpoint();
                if (!isHot(target) && !target.engineQueueAdmissionBlocked()
                        && mayRedirect(source, target, from, target.tokenWorkloadSnapshot())) { return true; }
            }
            return false;
        }

        private boolean mayRedirect(PrefillEndpoint source, PrefillEndpoint target, Load from,
                                    PrefillTokenWorkload.Snapshot current) {
            if (!isRedirectSource(source) || isHot(target)) {
                return false;
            }
            LoadPosition to = positions.get(target);
            if (to == null) { return false; }
            LoadPosition sourcePosition = positions.get(source);
            long targetRequests = saturatedAdd(
                    current.observedRequests(), current.plannedRequests());
            long targetCompute = saturatedAdd(
                    current.pendingTokens(), current.plannedTokens());
            if (targetRequests > requests.outlierLimit() || targetCompute > compute.outlierLimit()) {
                return false;
            }
            return (sourcePosition.highRequests() && to.lowRequests() && requests.allows(from.requests(), targetRequests))
                    || (sourcePosition.highCompute() && to.lowCompute() && compute.allows(from.compute(), targetCompute));
        }
    }

    private static final class Pool {
        private record Sample(long at, Snapshot snapshot) { }
        private record Reading(long at, Map<PrefillEndpoint, Load> loads, Signal requests, Signal compute) { }

        private Map<PrefillEndpoint, Long> aboveSince = Map.of();
        private final ArrayDeque<Reading> history = new ArrayDeque<>();
        private volatile Sample last;
        private Set<PrefillEndpoint> windowSources = Set.of();
        private long windowStart;

        Snapshot sample(List<PrefillRoutingEntry> entries, HotspotReleaseConfig config, long now) {
            Sample sampled = last;
            if (sampled != null && now - sampled.at() < SAMPLE_NANOS) {
                return sampled.snapshot();
            }
            return refresh(entries, config, now);
        }

        private synchronized Snapshot refresh(List<PrefillRoutingEntry> entries,
                                              HotspotReleaseConfig config, long now) {
            if (last != null && now - last.at() < SAMPLE_NANOS) {
                return last.snapshot();
            }
            long[] requests = new long[entries.size()];
            long[] compute = new long[entries.size()];
            Map<PrefillEndpoint, Load> loads = new HashMap<>();
            for (int i = 0; i < entries.size(); i++) {
                PrefillEndpoint endpoint = entries.get(i).endpoint();
                requests[i] = endpoint.observedRequestCount();
                compute[i] = endpoint.pendingTokenWork();
                loads.put(endpoint, new Load(requests[i], compute[i]));
            }
            Signal requestSignal = Signal.measure(requests, config.getSetpointFloor(), config.getOutlierFactor());
            Signal computeSignal = Signal.measure(compute, config.getComputeTokenFloor(), config.getOutlierFactor());
            while (!history.isEmpty() && now - history.getFirst().at() >= LOAD_WINDOW_NANOS) {
                history.removeFirst();
            }
            history.addLast(new Reading(now, loads, requestSignal, computeSignal));
            long duration = TimeUnit.MILLISECONDS.toNanos(config.getSustainedPressureMs());
            boolean fresh = last != null && now - last.at() < STALE_NANOS;
            Map<PrefillEndpoint, Long> continuing = new HashMap<>();
            int topK = config.getSourceTopK();
            HotspotRole defaultRole = topK == 0 ? HotspotRole.REDIRECT_SOURCE : HotspotRole.HOT;
            Map<PrefillEndpoint, HotspotRole> hotspots = new HashMap<>();
            Map<PrefillEndpoint, LoadPosition> positions = new HashMap<>();
            for (var entry : loads.entrySet()) {
                PrefillEndpoint endpoint = entry.getKey();
                LoadPosition position = windowPosition(endpoint);
                positions.put(endpoint, position);
                Load load = entry.getValue();
                if (load.requests() <= requestSignal.outlierLimit() && load.compute() <= computeSignal.outlierLimit()) {
                    continue;
                }
                long since = fresh ? aboveSince.getOrDefault(endpoint, now) : now;
                continuing.put(endpoint, since);
                boolean highNowAndInWindow = (position.highRequests() && load.requests() > requestSignal.outlierLimit())
                        || (position.highCompute() && load.compute() > computeSignal.outlierLimit());
                if (highNowAndInWindow && now - since >= duration) {
                    hotspots.put(endpoint, defaultRole);
                }
            }
            aboveSince = continuing;
            if (topK > 0) {
                selectWindowSources(entries, loads, hotspots.keySet(), topK, now);
                for (PrefillEndpoint source : windowSources) {
                    hotspots.replace(source, HotspotRole.REDIRECT_SOURCE);
                }
            }
            Snapshot snapshot = hotspots.isEmpty() ? Snapshot.EMPTY
                    : new Snapshot(loads, positions, hotspots, requestSignal, computeSignal);
            last = new Sample(now, snapshot);
            return snapshot;
        }

        private LoadPosition windowPosition(PrefillEndpoint endpoint) {
            int samples = 0;
            int highRequests = 0;
            int highCompute = 0;
            int lowRequests = 0;
            int lowCompute = 0;
            for (var it = history.descendingIterator(); it.hasNext();) {
                Reading reading = it.next();
                Load load = reading.loads().get(endpoint);
                if (load == null) { break; } // A new or rejoined endpoint starts its own history.
                samples++;
                if (load.requests() > reading.requests().outlierLimit()) { highRequests++; }
                if (load.compute() > reading.compute().outlierLimit()) { highCompute++; }
                if (load.requests() < reading.requests().meanCeiling()) { lowRequests++; }
                if (load.compute() < reading.compute().meanCeiling()) { lowCompute++; }
            }
            int required = Math.max(2, samples / 2 + 1);
            return new LoadPosition(highRequests >= required, highCompute >= required,
                    lowRequests >= required, lowCompute >= required);
        }

        private record Source(PrefillEndpoint endpoint, String address, long tokens, long requests) { }

        private void selectWindowSources(List<PrefillRoutingEntry> entries, Map<PrefillEndpoint, Load> loads,
                                         Set<PrefillEndpoint> hotspots, int topK, long now) {
            if (!windowSources.isEmpty() && now - windowStart < SOURCE_WINDOW_NANOS) { return; }
            windowStart = now;
            if (hotspots.size() <= topK) {
                windowSources = Set.copyOf(hotspots);
            } else {
                windowSources = entries.stream().filter(e -> hotspots.contains(e.endpoint()))
                        .map(e -> {
                            Load load = loads.get(e.endpoint());
                            long tokens = e.endpoint().queuedComputeTokens().orElse(load.compute());
                            return new Source(e.endpoint(), e.address(), tokens, load.requests());
                        })
                        .sorted(Comparator.comparingLong(Source::tokens).reversed()
                                .thenComparing(Comparator.comparingLong(Source::requests).reversed())
                                .thenComparing(Source::address))
                        .limit(topK).map(Source::endpoint).collect(Collectors.toUnmodifiableSet());
            }
        }
    }
}
