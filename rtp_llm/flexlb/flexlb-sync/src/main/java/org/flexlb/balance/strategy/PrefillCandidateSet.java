package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.projection.RouteProjection;

import java.util.Arrays;

import static com.google.common.math.LongMath.saturatedAdd;

/** Reusable columns for modeled Prefill candidates from one fleet snapshot. */
final class PrefillCandidateSet {

    private static final int INITIAL_CAPACITY = 16;
    private String[] endpointAddresses = new String[0];
    private PrefillEndpoint[] endpoints = new PrefillEndpoint[0];
    private PrefillTokenWorkload.Snapshot[] workloads = new PrefillTokenWorkload.Snapshot[0];
    private long[] projectedTtftMs = new long[0];
    private long[] incomingPrefillMs = new long[0];
    private long[] cacheHitTokens = new long[0];
    private long[] routingCacheMatchTokens = new long[0];
    private long[] ownershipVersions = new long[0];
    private String[] admissionBlocks = new String[0];
    private PrefillHotspots.Snapshot hotspots = PrefillHotspots.Snapshot.EMPTY;
    private int size;
    long minimumProjectedTtftMs;
    long maximumCacheHit;
    long maximumRoutingCacheMatchTokens;

    void reset(int expectedCapacity) {
        Arrays.fill(endpointAddresses, 0, size, null);
        Arrays.fill(endpoints, 0, size, null);
        Arrays.fill(admissionBlocks, 0, size, null);
        Arrays.fill(workloads, 0, size, null);
        ensureCapacity(expectedCapacity);
        size = 0;
        hotspots = PrefillHotspots.Snapshot.EMPTY;
        minimumProjectedTtftMs = Long.MAX_VALUE;
        maximumCacheHit = 0L;
        maximumRoutingCacheMatchTokens = 0L;
    }

    void addCandidate(String address, PrefillEndpoint endpoint,
                      RouteProjection.CandidateView candidate,
                      long ownershipVersion) {
        addCandidate(address, endpoint, candidate, null, ownershipVersion, endpoint.tokenWorkloadSnapshot());
    }

    void addCandidate(String address, PrefillEndpoint endpoint,
                      RouteProjection.CandidateView candidate,
                      String admissionBlock,
                      long ownershipVersion, PrefillTokenWorkload.Snapshot workload) {
        if (!candidate.hasPrediction()) {
            throw new IllegalArgumentException("candidate requires a work prediction");
        }
        int index = addEndpoint(address, endpoint, candidate.cacheHitTokens(), candidate.routingCacheMatchTokens());
        workloads[index] = workload;
        projectedTtftMs[index] = saturatedAdd(candidate.projectedTtftMsValue(), workload.plannedDurationMs());
        incomingPrefillMs[index] = candidate.incomingPrefillMs();
        ownershipVersions[index] = ownershipVersion;
        admissionBlocks[index] = admissionBlock;
        minimumProjectedTtftMs = Math.min(minimumProjectedTtftMs, projectedTtftMs[index]);
    }

    /** The source owns the queued request already. Only its prefix is needed to price the move. */
    void addQueuedSource(String address, PrefillEndpoint source, long hit, long routingHit) {
        int index = addEndpoint(address, source, hit, routingHit);
        projectedTtftMs[index] = Long.MAX_VALUE;
        incomingPrefillMs[index] = 0L;
    }

    private int addEndpoint(String address, PrefillEndpoint endpoint, long hit, long routingHit) {
        ensureCapacity(size + 1);
        int index = size++;
        endpointAddresses[index] = address;
        endpoints[index] = endpoint;
        cacheHitTokens[index] = hit;
        routingCacheMatchTokens[index] = routingHit;
        maximumCacheHit = Math.max(maximumCacheHit, hit);
        maximumRoutingCacheMatchTokens = Math.max(maximumRoutingCacheMatchTokens, routingHit);
        return index;
    }

    void setHotspots(PrefillHotspots.Snapshot hotspots) { this.hotspots = hotspots; }
    boolean isHotspot(int index) { return hotspots.isRedirectSource(endpoints[index]); }
    int indexOf(PrefillEndpoint endpoint) {
        for (int i = 0; i < size; i++) {
            if (endpoints[i] == endpoint) { return i; }
        }
        return -1;
    }
    boolean mayRedirect(int source, int target) {
        return hotspots.mayRedirect(endpoints[source], endpoints[target], workloads[target]);
    }
    PrefillTokenWorkload.Snapshot workload(int index) { return workloads[index]; }
    PrefillEndpoint endpoint(int index) { return endpoints[index]; }
    String endpointAddress(int index) { return endpointAddresses[index]; }
    long cacheHit(int index) { return cacheHitTokens[index]; }
    /** A full prefix can save the same modeled work as a partial prefix because the last block is recomputed. */
    int compareCacheMatch(int left, int right) {
        int effective = Long.compare(cacheHitTokens[left], cacheHitTokens[right]);
        return effective != 0 ? effective : Long.compare(routingCacheMatchTokens[left], routingCacheMatchTokens[right]);
    }
    long projectedTtftMs(int index) { return projectedTtftMs[index]; }
    /** Estimated Prefill execution time for the incoming request on this candidate. */
    long prefillMs(int index) { return incomingPrefillMs[index]; }
    long routingCacheMatchTokens(int index) { return routingCacheMatchTokens[index]; }
    long ownershipVersion(int index) { return ownershipVersions[index]; }
    String admissionBlock(int index) { return admissionBlocks[index]; }
    int size() { return size; }

    private void ensureCapacity(int expectedCapacity) {
        if (endpoints.length >= expectedCapacity) {
            return;
        }
        int capacity = Math.max(expectedCapacity,
                Math.max(INITIAL_CAPACITY, endpoints.length << 1));
        endpointAddresses = Arrays.copyOf(endpointAddresses, capacity);
        endpoints = Arrays.copyOf(endpoints, capacity);
        workloads = Arrays.copyOf(workloads, capacity);
        projectedTtftMs = Arrays.copyOf(projectedTtftMs, capacity);
        incomingPrefillMs = Arrays.copyOf(incomingPrefillMs, capacity);
        cacheHitTokens = Arrays.copyOf(cacheHitTokens, capacity);
        routingCacheMatchTokens = Arrays.copyOf(routingCacheMatchTokens, capacity);
        ownershipVersions = Arrays.copyOf(ownershipVersions, capacity);
        admissionBlocks = Arrays.copyOf(admissionBlocks, capacity);
    }

}
