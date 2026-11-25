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
        ensureCapacity(size + 1);
        endpointAddresses[size] = address;
        endpoints[size] = endpoint;
        workloads[size] = workload;
        projectedTtftMs[size] = saturatedAdd(
                candidate.projectedTtftMsValue(), workload.plannedDurationMs());
        incomingPrefillMs[size] = candidate.incomingPrefillMs();
        cacheHitTokens[size] = candidate.cacheHitTokens();
        routingCacheMatchTokens[size] = candidate.routingCacheMatchTokens();
        ownershipVersions[size] = ownershipVersion;
        admissionBlocks[size] = admissionBlock;
        minimumProjectedTtftMs = Math.min(minimumProjectedTtftMs, projectedTtftMs[size]);
        maximumCacheHit = Math.max(maximumCacheHit, cacheHitTokens[size]);
        maximumRoutingCacheMatchTokens = Math.max(
                maximumRoutingCacheMatchTokens, routingCacheMatchTokens[size]);
        size++;
    }

    void setHotspots(PrefillHotspots.Snapshot hotspots) { this.hotspots = hotspots; }
    boolean isHotspot(int index) { return hotspots.isHot(endpoints[index]); }
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
