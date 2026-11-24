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

    Set<PrefillEndpoint> sample(RoleType role, String group, List<PrefillRoutingEntry> candidates,
                               HotspotReleaseConfig config) {
        if (candidates.isEmpty()) {
            return Set.of();
        }
        Pool pool = pools.get(new Scope(role, group), ignored -> new Pool());
        return pool.sample(candidates, config, nanoTime.getAsLong());
    }

    private static final class Pool {
        private record Sample(long at, Set<PrefillEndpoint> hotspots) { }

        private Map<PrefillEndpoint, Long> aboveSince = Map.of();
        private volatile Sample last;

        Set<PrefillEndpoint> sample(List<PrefillRoutingEntry> entries, HotspotReleaseConfig config, long now) {
            Sample sampled = last;
            if (sampled != null && now - sampled.at() < SAMPLE_NANOS) {
                return sampled.hotspots();
            }
            return refresh(entries, config, now);
        }

        private synchronized Set<PrefillEndpoint> refresh(List<PrefillRoutingEntry> entries,
                                                          HotspotReleaseConfig config, long now) {
            if (last != null && now - last.at() < SAMPLE_NANOS) {
                return last.hotspots();
            }
            long[] requests = new long[entries.size()];
            long[] compute = new long[entries.size()];
            boolean spare = false;
            for (int i = 0; i < entries.size(); i++) {
                PrefillEndpoint endpoint = entries.get(i).endpoint();
                requests[i] = endpoint.observedRequestCount();
                compute[i] = endpoint.pendingTokenWork();
                if (requests[i] == 0 && compute[i] == 0 && endpoint.getInflightBatchCount() == 0) {
                    spare = true;
                }
            }
            Map<PrefillEndpoint, Long> continuing = new HashMap<>();
            Set<PrefillEndpoint> hotspots = new HashSet<>();
            if (spare) {
                long requestLimit = threshold(requests, config.getSetpointFloor(), config.getOutlierFactor());
                long computeLimit = threshold(compute, config.getComputeTokenFloor(), config.getOutlierFactor());
                long duration = TimeUnit.MILLISECONDS.toNanos(config.getSustainedPressureMs());
                boolean fresh = last != null && now - last.at() < STALE_NANOS;
                for (int i = 0; i < entries.size(); i++) {
                    if (requests[i] <= requestLimit && compute[i] <= computeLimit) {
                        continue;
                    }
                    PrefillEndpoint endpoint = entries.get(i).endpoint();
                    long since = fresh ? aboveSince.getOrDefault(endpoint, now) : now;
                    continuing.put(endpoint, since);
                    if (now - since >= duration) {
                        hotspots.add(endpoint);
                    }
                }
            }
            aboveSince = continuing;
            last = new Sample(now, Set.copyOf(hotspots));
            return last.hotspots();
        }
    }

    private static long threshold(long[] values, long floor, double factor) {
        long[] sorted = values.clone();
        Arrays.sort(sorted);
        int mid = sorted.length / 2;
        double median = sorted.length % 2 == 1 ? sorted[mid] : 0.5 * sorted[mid - 1] + 0.5 * sorted[mid];
        return Math.max(floor, (long) (factor * median));
    }
}
