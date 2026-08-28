package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.projection.RouteProjection;

import java.util.Arrays;
import java.util.BitSet;

/** Planner-local primitive columns and summaries for full-fleet Prefill selection. */
final class PrefillCandidateSet {

    private static final int INITIAL_CAPACITY = 16;

    private String[] endpointAddresses = new String[0];
    private PrefillEndpoint[] endpoints = new PrefillEndpoint[0];
    private RouteProjection.Candidate.State[] states =
            new RouteProjection.Candidate.State[0];
    private long[] projectedTtftMs = new long[0];
    private long[] projectedDrainMs = new long[0];
    private long[] incomingPrefillMs = new long[0];
    private long[] cacheHitTokens = new long[0];
    private long[] routingCacheMatchTokens = new long[0];
    private long[] pendingCounts = new long[0];
    private long[] ownershipVersions = new long[0];
    /** Max-heap of candidate indices used to retain the shortest K TTFTs. */
    private int[] shortestTtftHeap = new int[0];
    private int size;
    long projectedDrainTotalMs;
    int knownDrainCount;
    long pendingRequestTotal;
    long maximumPendingCount;
    long maximumProjectedDrainMs;
    long minimumProjectedTtftMs;
    long minimumCacheHit;
    long maximumCacheHit;
    long maximumRoutingCacheMatchTokens;

    void reset(int expectedCapacity) {
        clearReferences(0, size);
        ensureCapacity(expectedCapacity);
        size = 0;
        resetSummary();
    }

    void addCandidate(
            String endpointAddress,
            PrefillEndpoint endpoint,
            RouteProjection.CandidateView candidate,
            long ownershipVersion) {
        ensureCapacity(size + 1);
        endpointAddresses[size] = endpointAddress;
        endpoints[size] = endpoint;
        states[size] = candidate.state();
        projectedTtftMs[size] = candidate.projectedTtftMsValue();
        projectedDrainMs[size] = candidate.projectedDrainMsValue();
        incomingPrefillMs[size] = candidate.incomingPrefillMs();
        cacheHitTokens[size] = candidate.cacheHitTokens();
        routingCacheMatchTokens[size] = candidate.routingCacheMatchTokens();
        pendingCounts[size] = candidate.pendingCountValue();
        ownershipVersions[size] = ownershipVersion;
        size++;
        summarize(size - 1);
    }

    boolean selectable(int index) {
        return states[index] == RouteProjection.Candidate.State.MODELED
                && projectedTtftMs[index] != RouteProjection.Candidate.UNKNOWN;
    }

    boolean engineWorkUnmodeled(int index) {
        return states[index]
                == RouteProjection.Candidate.State.UNMODELED_ENGINE_WORK;
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

    long projectedTtftMs(int index) {
        return projectedTtftMs[index];
    }

    long prefillMs(int index) {
        return incomingPrefillMs[index];
    }

    boolean hasProjectedDrain(int index) {
        return projectedDrainMs[index] != RouteProjection.Candidate.UNKNOWN;
    }

    long projectedDrainMs(int index) {
        return projectedDrainMs[index];
    }

    long pendingCount(int index) {
        return pendingCounts[index];
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

    /**
     * Select the configured shortest-TTFT pool from the complete evaluated
     * fleet without allocating boxed indices or sorting every candidate.
     */
    BitSet shortestTtftPool(int configuredCount) {
        int count = Math.min(Math.max(1, configuredCount), size);
        BitSet pool = new BitSet(size);
        if (count == size) {
            pool.set(0, size);
            return pool;
        }
        ensureHeapCapacity(count);
        int heapSize = 0;
        for (int candidate = 0; candidate < size; candidate++) {
            if (heapSize < count) {
                shortestTtftHeap[heapSize] = candidate;
                siftWorseUp(heapSize++);
                continue;
            }
            if (isBetterTtft(candidate, shortestTtftHeap[0])) {
                shortestTtftHeap[0] = candidate;
                siftWorseDown(0, heapSize);
            }
        }
        for (int index = 0; index < heapSize; index++) {
            pool.set(shortestTtftHeap[index]);
        }
        return pool;
    }

    void beginRetainedSummary() {
        resetSummary();
    }

    void retainAndSummarize(int source, int target) {
        retainAt(source, target);
        summarize(target);
    }

    void finishRetainedSummary(int retainedSize) {
        clearReferences(retainedSize, size);
        size = retainedSize;
    }

    private void clearReferences(int from, int to) {
        Arrays.fill(endpointAddresses, from, to, null);
        Arrays.fill(endpoints, from, to, null);
        Arrays.fill(states, from, to, null);
    }

    private void retainAt(int source, int target) {
        if (source == target) {
            return;
        }
        endpointAddresses[target] = endpointAddresses[source];
        endpoints[target] = endpoints[source];
        states[target] = states[source];
        projectedTtftMs[target] = projectedTtftMs[source];
        projectedDrainMs[target] = projectedDrainMs[source];
        incomingPrefillMs[target] = incomingPrefillMs[source];
        cacheHitTokens[target] = cacheHitTokens[source];
        routingCacheMatchTokens[target] = routingCacheMatchTokens[source];
        pendingCounts[target] = pendingCounts[source];
        ownershipVersions[target] = ownershipVersions[source];
    }

    private void resetSummary() {
        projectedDrainTotalMs = 0L;
        knownDrainCount = 0;
        pendingRequestTotal = 0L;
        maximumPendingCount = 0L;
        maximumProjectedDrainMs = 0L;
        minimumProjectedTtftMs = Long.MAX_VALUE;
        minimumCacheHit = Long.MAX_VALUE;
        maximumCacheHit = 0L;
        maximumRoutingCacheMatchTokens = 0L;
    }

    private void summarize(int index) {
        long drainMs = projectedDrainMs[index];
        if (drainMs != RouteProjection.Candidate.UNKNOWN) {
            projectedDrainTotalMs = saturatingAdd(
                    projectedDrainTotalMs, drainMs);
            maximumProjectedDrainMs = Math.max(
                    maximumProjectedDrainMs, drainMs);
            knownDrainCount++;
        }
        long pending = pendingCounts[index];
        pendingRequestTotal = saturatingAdd(pendingRequestTotal, pending);
        maximumPendingCount = Math.max(maximumPendingCount, pending);
        if (selectable(index)) {
            minimumProjectedTtftMs = Math.min(
                    minimumProjectedTtftMs, projectedTtftMs[index]);
        }
        minimumCacheHit = Math.min(minimumCacheHit, cacheHitTokens[index]);
        maximumCacheHit = Math.max(maximumCacheHit, cacheHitTokens[index]);
        maximumRoutingCacheMatchTokens = Math.max(
                maximumRoutingCacheMatchTokens,
                routingCacheMatchTokens[index]);
    }

    private void ensureCapacity(int expectedCapacity) {
        if (states.length >= expectedCapacity) {
            return;
        }
        int capacity = Math.max(
                expectedCapacity,
                Math.max(INITIAL_CAPACITY, states.length << 1));
        endpointAddresses = Arrays.copyOf(endpointAddresses, capacity);
        endpoints = Arrays.copyOf(endpoints, capacity);
        states = Arrays.copyOf(states, capacity);
        projectedTtftMs = Arrays.copyOf(projectedTtftMs, capacity);
        projectedDrainMs = Arrays.copyOf(projectedDrainMs, capacity);
        incomingPrefillMs = Arrays.copyOf(incomingPrefillMs, capacity);
        cacheHitTokens = Arrays.copyOf(cacheHitTokens, capacity);
        routingCacheMatchTokens = Arrays.copyOf(
                routingCacheMatchTokens, capacity);
        pendingCounts = Arrays.copyOf(pendingCounts, capacity);
        ownershipVersions = Arrays.copyOf(ownershipVersions, capacity);
    }

    private void ensureHeapCapacity(int expectedCapacity) {
        if (shortestTtftHeap.length < expectedCapacity) {
            shortestTtftHeap = new int[expectedCapacity];
        }
    }

    private void siftWorseUp(int child) {
        while (child > 0) {
            int parent = (child - 1) >>> 1;
            if (!isWorseTtft(
                    shortestTtftHeap[child], shortestTtftHeap[parent])) {
                return;
            }
            swapHeap(child, parent);
            child = parent;
        }
    }

    private void siftWorseDown(int parent, int heapSize) {
        while (true) {
            int left = (parent << 1) + 1;
            if (left >= heapSize) {
                return;
            }
            int worseChild = left;
            int right = left + 1;
            if (right < heapSize && isWorseTtft(
                    shortestTtftHeap[right], shortestTtftHeap[left])) {
                worseChild = right;
            }
            if (!isWorseTtft(
                    shortestTtftHeap[worseChild],
                    shortestTtftHeap[parent])) {
                return;
            }
            swapHeap(parent, worseChild);
            parent = worseChild;
        }
    }

    private boolean isBetterTtft(int left, int right) {
        long leftTtft = projectedTtftMs[left];
        long rightTtft = projectedTtftMs[right];
        return leftTtft < rightTtft
                || leftTtft == rightTtft && left < right;
    }

    private boolean isWorseTtft(int left, int right) {
        return isBetterTtft(right, left);
    }

    private void swapHeap(int left, int right) {
        int value = shortestTtftHeap[left];
        shortestTtftHeap[left] = shortestTtftHeap[right];
        shortestTtftHeap[right] = value;
    }

    private static long saturatingAdd(long left, long right) {
        return left > Long.MAX_VALUE - right
                ? Long.MAX_VALUE : left + right;
    }

    /** Per-planner reusable columns; a planner never re-enters selection. */
    static final class Scratch {
        final PrefillCandidateSet modeled = new PrefillCandidateSet();
        final PrefillCandidateSet unmodeled = new PrefillCandidateSet();

        void reset(int expectedCapacity) {
            modeled.reset(expectedCapacity);
            unmodeled.reset(0);
        }
    }
}
