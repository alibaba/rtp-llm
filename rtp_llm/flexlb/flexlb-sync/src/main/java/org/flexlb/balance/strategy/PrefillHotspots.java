package org.flexlb.balance.strategy;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import org.flexlb.balance.endpoint.EndpointRegistry.PrefillRoutingEntry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.RoutingConfig.HotspotReleaseConfig;
import org.flexlb.dao.route.RoleType;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.function.LongSupplier;

import static com.google.common.math.LongMath.saturatedAdd;

/** Samples outstanding load within the exact role and group used by a placement. */
final class PrefillHotspots {
    private static final long SAMPLE_NANOS = TimeUnit.MILLISECONDS.toNanos(100);
    private static final long STALE_NANOS = TimeUnit.SECONDS.toNanos(1);

    private record Scope(RoleType role, String group) { }

    private final Cache<Scope, Pool> pools;
    private final LongSupplier nanoTime;

    PrefillHotspots(LongSupplier nanoTime) {
        this.nanoTime = nanoTime;
        pools = Caffeine.newBuilder().ticker(nanoTime::getAsLong)
                .expireAfterAccess(STALE_NANOS, TimeUnit.NANOSECONDS).build();
    }

    Snapshot sample(RoleType role, String group, List<PrefillRoutingEntry> candidates,
                    HotspotReleaseConfig config) {
        if (candidates.isEmpty()) {
            return Snapshot.EMPTY;
        }
        Pool pool = pools.get(new Scope(role, group), ignored -> new Pool());
        return pool.sample(candidates, config, nanoTime.getAsLong());
    }

    private record Load(long requests, long compute) { }

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

    /** Read-only view shared by every decision within one pool sampling interval. */
    static final class Snapshot {
        private static final Signal NONE = new Signal(Long.MAX_VALUE, 0L);
        static final Snapshot EMPTY = new Snapshot(Map.of(), Set.of(), NONE, NONE);

        private final Map<PrefillEndpoint, Load> loads;
        private final Set<PrefillEndpoint> hotspots;
        private final Signal requests;
        private final Signal compute;

        private Snapshot(Map<PrefillEndpoint, Load> loads, Set<PrefillEndpoint> hotspots,
                         Signal requests, Signal compute) {
            this.loads = Map.copyOf(loads);
            this.hotspots = Set.copyOf(hotspots);
            this.requests = requests;
            this.compute = compute;
        }

        boolean isHot(PrefillEndpoint endpoint) {
            return hotspots.contains(endpoint);
        }

        boolean mayRedirect(PrefillEndpoint source, PrefillEndpoint target, PrefillTokenWorkload.Snapshot current) {
            if (!isHot(source) || isHot(target)) {
                return false;
            }
            Load from = loads.get(source);
            if (!loads.containsKey(target)) { return false; }
            long targetRequests = saturatedAdd(
                    current.observedRequests(), current.plannedRequests());
            long targetCompute = saturatedAdd(
                    current.pendingTokens(), current.plannedTokens());
            return requests.allows(from.requests(), targetRequests) || compute.allows(from.compute(), targetCompute);
        }
    }

    private static final class Pool {
        private record Sample(long at, Snapshot snapshot) { }

        private Map<PrefillEndpoint, Long> aboveSince = Map.of();
        private volatile Sample last;

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
            long duration = TimeUnit.MILLISECONDS.toNanos(config.getSustainedPressureMs());
            boolean fresh = last != null && now - last.at() < STALE_NANOS;
            Map<PrefillEndpoint, Long> continuing = new HashMap<>();
            Set<PrefillEndpoint> hotspots = new HashSet<>();
            for (var entry : loads.entrySet()) {
                Load load = entry.getValue();
                if (load.requests() <= requestSignal.outlierLimit() && load.compute() <= computeSignal.outlierLimit()) {
                    continue;
                }
                PrefillEndpoint endpoint = entry.getKey();
                long since = fresh ? aboveSince.getOrDefault(endpoint, now) : now;
                continuing.put(endpoint, since);
                if (now - since >= duration) {
                    hotspots.add(endpoint);
                }
            }
            aboveSince = continuing;
            Snapshot snapshot = hotspots.isEmpty() ? Snapshot.EMPTY
                    : new Snapshot(loads, hotspots, requestSignal, computeSignal);
            last = new Sample(now, snapshot);
            return snapshot;
        }
    }
}
