package org.flexlb.balance.preemption;

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
 * Selects lower-priority requests to preempt from one Decode endpoint snapshot, without changing resources.
 * A plan contains either locally queued requests or requests requiring Engine Cancel; the executor
 * validates their identity again before applying the plan.
 */
@FunctionalInterface
public interface DecodePreemptionPolicy {
    DecodePreemptionDecision planDecode(
            Request request,
            int incomingPriority,
            AdmissionCapacity capacity,
            ResourceSnapshot snapshot,
            PreemptionConfig preemption,
            Map<String, String> failures);

    /**
     * One snapshot's capacity decision and its optional single-ownership preemption.
     */
    public record DecodePreemptionDecision(CapacityDeficit deficit, DecodePreemptionPlan proposal) {
        public String preemptionReason() {
            if (deficit.requests() > 0L && deficit.needsKv()) {
                return DecodePreemptionPlan.CASE_SLOT_AND_KV;
            }
            if (deficit.requests() > 0L) {
                return DecodePreemptionPlan.CASE_SLOT;
            }
            return deficit.needsKv() ? DecodePreemptionPlan.CASE_KV : null;
        }
    }

    /**
     * Holds the lower-priority requests selected for preemption and the capacity shortage they address.
     * Local route withdrawal and Engine Cancel use separate plans; cancelling an engine request
     * does not itself establish that its resources have been released.
     */
    public record DecodePreemptionPlan(
            String endpointId,
            List<DecodeRequestView> requestsToPreempt,
            String preemptionReason,
            long freedKvTokens,
            long diagnosticCost) {
        /** Concurrency slots exhausted (design doc 11). */
        public static final String CASE_SLOT = "decode_slot_full";

        /** Real KV available below the incoming hard demand (design doc 12). */
        public static final String CASE_KV = "decode_kv_full";

        /** Both deficits at once (design doc 13). */
        public static final String CASE_SLOT_AND_KV = "decode_slot_and_kv_full";

        public DecodePreemptionPlan {
            requestsToPreempt = List.copyOf(requestsToPreempt);
            boolean hasLocal =
                    requestsToPreempt.stream()
                            .anyMatch(
                                    requestToPreempt -> requestToPreempt.phase().isMasterQueued());
            boolean hasCancel =
                    requestsToPreempt.stream()
                            .anyMatch(
                                    requestToPreempt ->
                                            requestToPreempt.phase().requiresEngineCancel());
            checkArgument(
                    !hasLocal || !hasCancel,
                    "decode proposal cannot mix Master-local and Engine-Cancel victims");
        }

        public boolean requiresEngineCancel() {
            return !requestsToPreempt.isEmpty()
                    && requestsToPreempt.get(0).phase().requiresEngineCancel();
        }
    }
}
