package org.flexlb.balance.strategy;

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
    PrefillSelectionStrategy.CandidateInput[] capturedSources =
            new PrefillSelectionStrategy.CandidateInput[0];
    String affinityReason;
    long[] projectedTtftMs = new long[0];
    long[] incomingPrefillMs = new long[0];
    long[] cacheHitTokens = new long[0];
    long[] routingCacheMatchTokens = new long[0];
    long[] ownershipVersions = new long[0];
    String[] admissionBlocks = new String[0];
    int size;
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

    void capture(List<? extends PrefillSelectionStrategy.CandidateInput> directory, String group) {
        clear();
        for (var source : directory) {
            if (group == null || group.equals(source.group())) {
                ensureCapacity(registeredCount + 1);
                endpointAddresses[registeredCount] = source.address();
                capturedSources[registeredCount++] = source;
            }
        }
    }

    void clear() {
        int retained = Math.max(size, registeredCount);
        Arrays.fill(endpointAddresses, 0, retained, null);
        Arrays.fill(capturedSources, 0, registeredCount, null);
        affinityReason = null;
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
            PrefillSelectionStrategy.CandidateInput input,
            RequestTimeEstimation.CandidateView candidate,
            long ownershipVersion) {
        checkArgument(
                candidate.selectable() || candidate.engineWorkUnmodeled(),
                "candidate requires available capacity");
        ensureCapacity(size + 1);
        endpointAddresses[size] = address;
        capturedSources[size] = input;
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

    long ttftRank(int index) {
        return projectedTtftMs[index] < 0L ? Long.MAX_VALUE : projectedTtftMs[index];
    }

    private void ensureCapacity(int expectedCapacity) {
        if (capturedSources.length >= expectedCapacity) {
            return;
        }
        int capacity =
                Math.max(expectedCapacity, Math.max(INITIAL_CAPACITY, capturedSources.length << 1));
        endpointAddresses = Arrays.copyOf(endpointAddresses, capacity);
        capturedSources = Arrays.copyOf(capturedSources, capacity);
        tieIndexes = Arrays.copyOf(tieIndexes, capacity);
        admissionBlocks = Arrays.copyOf(admissionBlocks, capacity);
        projectedTtftMs = Arrays.copyOf(projectedTtftMs, capacity);
        incomingPrefillMs = Arrays.copyOf(incomingPrefillMs, capacity);
        cacheHitTokens = Arrays.copyOf(cacheHitTokens, capacity);
        routingCacheMatchTokens = Arrays.copyOf(routingCacheMatchTokens, capacity);
        ownershipVersions = Arrays.copyOf(ownershipVersions, capacity);
    }
}
