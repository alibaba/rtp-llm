package org.flexlb.balance.eviction;

import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.endpoint.DecodeResources.CapacityDeficit;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRequestView;
import org.flexlb.balance.endpoint.DecodeResources.ResourceSnapshot;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.dao.loadbalance.Request;

import java.util.List;
import java.util.Map;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Pure victim selection for a single Decode endpoint snapshot. Inputs are borrowed read-only;
 * only the supplied diagnostic map may be written. A proposal is a plan, never a resource claim.
 * Implementations must choose lower-priority, unclaimed victims from this snapshot and enabled
 * stages, with one ownership kind. The executor revalidates identity before resource mutation.
 */
@FunctionalInterface
public interface DecodePreemptionPolicy {
    DecodeEvictionDecision planDecode(
            Request request,
            int incomingPriority,
            AdmissionCapacity capacity,
            ResourceSnapshot snapshot,
            PreemptionConfig preemption,
            Map<String, String> failures);

    /**
     * One snapshot's capacity decision and its optional single-ownership eviction.
     */
    public record DecodeEvictionDecision(CapacityDeficit deficit, DecodeEvictionProposal proposal) {
        public String evictionCase() {
            if (deficit.requests() > 0L && deficit.needsKv()) {
                return DecodeEvictionProposal.CASE_SLOT_AND_KV;
            }
            if (deficit.requests() > 0L) {
                return DecodeEvictionProposal.CASE_SLOT;
            }
            return deficit.needsKv() ? DecodeEvictionProposal.CASE_KV : null;
        }
    }

    /**
     * Immutable eviction proposal for one Decode snapshot. Victims are strictly lower priority;
     * Master-local withdrawal and Engine Cancel are separate plans. Engine victims require release proof.
     */
    public record DecodeEvictionProposal(
            String endpointId,
            List<DecodeRequestView> victims,
            String evictionCase,
            long freedKvTokens,
            long diagnosticCost) {
        /** Concurrency slots exhausted (design doc 11). */
        public static final String CASE_SLOT = "decode_slot_full";

        /** Real KV available below the incoming hard demand (design doc 12). */
        public static final String CASE_KV = "decode_kv_full";

        /** Both deficits at once (design doc 13). */
        public static final String CASE_SLOT_AND_KV = "decode_slot_and_kv_full";

        public DecodeEvictionProposal {
            victims = List.copyOf(victims);
            boolean hasLocal = victims.stream().anyMatch(victim -> victim.phase().isMasterQueued());
            boolean hasCancel =
                    victims.stream().anyMatch(victim -> victim.phase().requiresEngineCancel());
            checkArgument(
                    !hasLocal || !hasCancel,
                    "decode proposal cannot mix Master-local and Engine-Cancel victims");
        }

        public boolean requiresEngineCancel() {
            return !victims.isEmpty() && victims.get(0).phase().requiresEngineCancel();
        }
    }
}
