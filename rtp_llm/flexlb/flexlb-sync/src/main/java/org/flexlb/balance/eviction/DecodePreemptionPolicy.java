package org.flexlb.balance.eviction;

import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.endpoint.DecodeResources.CapacityDeficit;
import org.flexlb.balance.endpoint.DecodeResources.ResourceSnapshot;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.dao.loadbalance.Request;

import java.util.Map;

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

    /** One snapshot's capacity decision and its optional single-ownership eviction. */
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
}
