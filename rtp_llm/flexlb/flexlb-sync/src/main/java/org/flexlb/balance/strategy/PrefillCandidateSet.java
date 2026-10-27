package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.projection.RequestTimeEstimation;

import java.util.AbstractList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import static com.google.common.base.Preconditions.checkArgument;

/** Thread-confined fleet capture; ranking compacts survivors into the same columns. */
final class PrefillCandidateSet {

    private static final int INITIAL_CAPACITY = 16;
    String[] endpointAddresses = new String[0];
    int[] tieIndexes = new int[0];
    private PrefillEndpoint[] endpoints = new PrefillEndpoint[0];
    private long[] projectedTtftMs = new long[0];
    private long[] incomingPrefillMs = new long[0];
    private long[] cacheHitTokens = new long[0];
    private long[] routingCacheMatchTokens = new long[0];
    private long[] ownershipVersions = new long[0];
    private String[] admissionBlocks = new String[0];
    private int size;
    int registeredCount;
    // Borrowed only during the synchronous cache lookup, before ranking compacts these columns.
    final List<String> addresses =
            new AbstractList<>() {
                @Override
                public String get(int index) {
                    return endpointAddresses[Objects.checkIndex(index, registeredCount)];
                }

                @Override
                public int size() {
                    return registeredCount;
                }
            };
    boolean inUse;
    long minimumTtftRank = Long.MAX_VALUE;
    boolean hasKnownTtft;
    long maximumCacheHit;
    long maximumRoutingCacheMatchTokens;

    void capture(List<EndpointRegistry.PrefillRoutingEntry> directory, String group) {
        clear();
        for (var entry : directory) {
            if (group == null
                    || group.equals(entry.endpoint().getStatus().topologySnapshot().group())) {
                ensureCapacity(registeredCount + 1);
                endpointAddresses[registeredCount] = entry.address();
                endpoints[registeredCount++] = entry.endpoint();
            }
        }
    }

    void clear() {
        int retained = Math.max(size, registeredCount);
        Arrays.fill(endpointAddresses, 0, retained, null);
        Arrays.fill(endpoints, 0, retained, null);
        Arrays.fill(admissionBlocks, 0, size, null);
        size = 0;
        registeredCount = 0;
        minimumTtftRank = Long.MAX_VALUE;
        hasKnownTtft = false;
        maximumCacheHit = 0L;
        maximumRoutingCacheMatchTokens = 0L;
    }

    void addCandidate(
            String address,
            PrefillEndpoint endpoint,
            RequestTimeEstimation.CandidateView candidate,
            long ownershipVersion) {
        checkArgument(
                candidate.selectable() || candidate.engineWorkUnmodeled(),
                "candidate requires available capacity");
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
        maximumCacheHit = Math.max(maximumCacheHit, cacheHitTokens[size]);
        maximumRoutingCacheMatchTokens =
                Math.max(maximumRoutingCacheMatchTokens, routingCacheMatchTokens[size]);
        size++;
    }

    void admissionBlock(String detail) {
        admissionBlocks[size - 1] = detail;
    }

    String admissionBlock(int index) {
        return admissionBlocks[index];
    }

    PrefillEndpoint endpoint(int index) {
        return endpoints[index];
    }

    String endpointAddress(int index) {
        return endpointAddresses[index];
    }

    long cacheHit(int index) {
        return cacheHitTokens[index];
    }

    long ttftRank(int index) {
        return projectedTtftMs[index] < 0L ? Long.MAX_VALUE : projectedTtftMs[index];
    }

    long projectedTtftMs(int index) {
        return projectedTtftMs[index];
    }

    long prefillMs(int index) {
        return incomingPrefillMs[index];
    }

    long routingCacheMatchTokens(int index) {
        return routingCacheMatchTokens[index];
    }

    long ownershipVersion(int index) {
        return ownershipVersions[index];
    }

    int size() {
        return size;
    }

    int capacity() {
        return endpoints.length;
    }

    private void ensureCapacity(int expectedCapacity) {
        if (endpoints.length >= expectedCapacity) {
            return;
        }
        int capacity =
                Math.max(expectedCapacity, Math.max(INITIAL_CAPACITY, endpoints.length << 1));
        endpointAddresses = Arrays.copyOf(endpointAddresses, capacity);
        tieIndexes = Arrays.copyOf(tieIndexes, capacity);
        endpoints = Arrays.copyOf(endpoints, capacity);
        admissionBlocks = Arrays.copyOf(admissionBlocks, capacity);
        projectedTtftMs = Arrays.copyOf(projectedTtftMs, capacity);
        incomingPrefillMs = Arrays.copyOf(incomingPrefillMs, capacity);
        cacheHitTokens = Arrays.copyOf(cacheHitTokens, capacity);
        routingCacheMatchTokens = Arrays.copyOf(routingCacheMatchTokens, capacity);
        ownershipVersions = Arrays.copyOf(ownershipVersions, capacity);
    }
}
