package org.flexlb.balance.eviction;

import org.flexlb.balance.endpoint.DecodeResources.DecodeRequestView;

import java.util.Comparator;
import java.util.List;

import static com.google.common.base.Preconditions.checkArgument;

/** Immutable eviction proposal for one Decode snapshot. Victims are strictly lower priority;
 * Master-local withdrawal and Engine Cancel are separate plans. Engine victims require release proof. */
public record DecodeEvictionProposal(
        String endpointId,
        List<DecodeRequestView> victims,
        String evictionCase,
        long freedKvTokens,
        PriorityHarmProfile priorityHarmProfile,
        long deterministicTieBreak) {

    public DecodeEvictionProposal {
        victims = List.copyOf(victims);
        boolean hasLocal = victims.stream().anyMatch(victim -> victim.phase().isMasterQueued());
        boolean hasCancel = victims.stream().anyMatch(victim -> victim.phase().requiresEngineCancel());
        checkArgument(!hasLocal || !hasCancel, "decode proposal cannot mix Master-local and Engine-Cancel victims");
    }

    public boolean requiresEngineCancel() {
        return !victims.isEmpty() && victims.get(0).phase().requiresEngineCancel();
    }

    /** Concurrency slots exhausted (design doc 11). */
    public static final String CASE_SLOT = "decode_slot_full";
    /** Real KV available below the incoming hard demand (design doc 12). */
    public static final String CASE_KV = "decode_kv_full";
    /** Both deficits at once (design doc 13). */
    public static final String CASE_SLOT_AND_KV = "decode_slot_and_kv_full";

    /**
     * Prefer less exact priority harm, then fewer victims and the smallest request id.
     * Scalar diagnostic cost never participates in priority-safety ordering.
     */
    public static final Comparator<DecodeEvictionProposal> ORDER = Comparator
            .comparing(DecodeEvictionProposal::priorityHarmProfile)
            .thenComparingInt(proposal -> proposal.victims().size())
            .thenComparingLong(DecodeEvictionProposal::deterministicTieBreak)
            .thenComparing(DecodeEvictionProposal::endpointId);
}
