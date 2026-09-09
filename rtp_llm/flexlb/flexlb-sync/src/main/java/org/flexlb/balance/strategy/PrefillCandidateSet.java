package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.projection.RouteProjection;

import java.util.Arrays;

/** Reusable columns for modeled Prefill candidates from one fleet snapshot. */
final class PrefillCandidateSet {

    private static final int INITIAL_CAPACITY = 16;
    private String[] endpointAddresses = new String[0];
    private PrefillEndpoint[] endpoints = new PrefillEndpoint[0];
    private long[] projectedTtftMs = new long[0];
    private long[] incomingPrefillMs = new long[0];
    private long[] cacheHitTokens = new long[0];
    private long[] routingCacheMatchTokens = new long[0];
    private long[] ownershipVersions = new long[0];
    private int size;
    long minimumTtftRank;
    boolean hasKnownTtft;
    long minimumCacheHit;
    long maximumCacheHit;
    long maximumRoutingCacheMatchTokens;

    void reset(int expectedCapacity) {
        Arrays.fill(endpointAddresses, 0, size, null);
        Arrays.fill(endpoints, 0, size, null);
        ensureCapacity(expectedCapacity);
        size = 0;
        minimumTtftRank = Long.MAX_VALUE;
        hasKnownTtft = false;
        minimumCacheHit = Long.MAX_VALUE;
        maximumCacheHit = 0L;
        maximumRoutingCacheMatchTokens = 0L;
    }

    void addCandidate(String address, PrefillEndpoint endpoint,
                      RouteProjection.CandidateView candidate,
                      long ownershipVersion) {
        if (!candidate.selectable() && !candidate.engineWorkUnmodeled()) {
            throw new IllegalArgumentException("candidate requires available capacity");
        }
        ensureCapacity(size + 1);
        endpointAddresses[size] = address;
        endpoints[size] = endpoint;
        projectedTtftMs[size] = candidate.projectedTtftMsValue();
        incomingPrefillMs[size] = candidate.incomingPrefillMs();
        cacheHitTokens[size] = candidate.cacheHitTokens();
        routingCacheMatchTokens[size] = candidate.routingCacheMatchTokens();
        ownershipVersions[size] = ownershipVersion;
        hasKnownTtft |= projectedTtftMs[size] >= 0L;
        minimumTtftRank = Math.min(minimumTtftRank, ttftRank(size));
        minimumCacheHit = Math.min(minimumCacheHit, cacheHitTokens[size]);
        maximumCacheHit = Math.max(maximumCacheHit, cacheHitTokens[size]);
        maximumRoutingCacheMatchTokens = Math.max(
                maximumRoutingCacheMatchTokens, routingCacheMatchTokens[size]);
        size++;
    }

    PrefillEndpoint endpoint(int index) { return endpoints[index]; }
    String endpointAddress(int index) { return endpointAddresses[index]; }
    long cacheHit(int index) { return cacheHitTokens[index]; }
    long ttftRank(int index) { return projectedTtftMs[index] < 0L ? Long.MAX_VALUE : projectedTtftMs[index]; }
    long projectedTtftMs(int index) { return projectedTtftMs[index]; }
    long prefillMs(int index) { return incomingPrefillMs[index]; }
    long routingCacheMatchTokens(int index) { return routingCacheMatchTokens[index]; }
    long ownershipVersion(int index) { return ownershipVersions[index]; }
    int size() { return size; }

    private void ensureCapacity(int expectedCapacity) {
        if (endpoints.length >= expectedCapacity) {
            return;
        }
        int capacity = Math.max(expectedCapacity,
                Math.max(INITIAL_CAPACITY, endpoints.length << 1));
        endpointAddresses = Arrays.copyOf(endpointAddresses, capacity);
        endpoints = Arrays.copyOf(endpoints, capacity);
        projectedTtftMs = Arrays.copyOf(projectedTtftMs, capacity);
        incomingPrefillMs = Arrays.copyOf(incomingPrefillMs, capacity);
        cacheHitTokens = Arrays.copyOf(cacheHitTokens, capacity);
        routingCacheMatchTokens = Arrays.copyOf(routingCacheMatchTokens, capacity);
        ownershipVersions = Arrays.copyOf(ownershipVersions, capacity);
    }

}
