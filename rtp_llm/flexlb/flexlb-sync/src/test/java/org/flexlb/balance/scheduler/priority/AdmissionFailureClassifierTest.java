package org.flexlb.balance.scheduler.priority;

import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.enums.DecodeTaskPhase;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Causal attribution must use known priorities on the dimensions in deficit. */
class AdmissionFailureClassifierTest {

    @Test
    void confirmedOccupantWithoutPriorityProvenanceIsAdmissionUnavailable() {
        DecodeRequestSnapshot unattributed = request(
                "1", 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 128, false, false);
        DecodeEndpointSnapshot endpoint = endpoint(
                1, 1, 1_000, 2_000, List.of(), List.of(unattributed), List.of());

        AdmissionFailure failure = AdmissionFailureClassifier.classifyDecode(
                incoming(50, 128), List.of(endpoint));

        assertFailure(failure, StrategyErrorType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED);
    }

    @Test
    void queuedReservationDoesNotExplainEngineSlotDeficit() {
        DecodeRequestSnapshot higherQueued = request(
                "1", 70, DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED,
                0, true, true);
        DecodeEndpointSnapshot endpoint = endpoint(
                1, 1, 1_000, 2_000,
                List.of(higherQueued), List.of(), List.of());

        AdmissionFailure failure = AdmissionFailureClassifier.classifyDecode(
                incoming(50, 128), List.of(endpoint));

        assertFailure(failure, StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED);
    }

    @Test
    void validPriorityValueWithoutProvenanceIsNotTrusted() {
        DecodeRequestSnapshot untrustedP70 = request(
                "1", 70, DecodeTaskPhase.RUNNING, 0, false, false);
        DecodeEndpointSnapshot endpoint = endpoint(
                1, 1, 1_000, 2_000,
                List.of(), List.of(), List.of(untrustedP70));

        AdmissionFailure failure = AdmissionFailureClassifier.classifyDecode(
                incoming(50, 128), List.of(endpoint));

        assertFailure(failure, StrategyErrorType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED);
    }

    @Test
    void higherPriorityOnSlotDoesNotExplainKvOnlyDeficit() {
        DecodeRequestSnapshot higherAccepted = request(
                "1", 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                0, true, false);
        DecodeRequestSnapshot sameQueued = request(
                "2", 50, DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED,
                256, true, true);
        DecodeEndpointSnapshot endpoint = endpoint(
                1, 0, 0, 2_000,
                List.of(sameQueued), List.of(higherAccepted), List.of());

        AdmissionFailure failure = AdmissionFailureClassifier.classifyDecode(
                incoming(50, 128), List.of(endpoint));

        assertFailure(failure, StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.SAME_PRIORITY_AHEAD);
    }

    @Test
    void queuedReservationWithoutPriorityProvenanceCanExplainKvDeficit() {
        DecodeRequestSnapshot unattributedQueued = request(
                "1", 50, DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED,
                512, false, true);
        DecodeEndpointSnapshot endpoint = endpoint(
                1, 0, 0, 2_000,
                List.of(unattributedQueued), List.of(), List.of());

        AdmissionFailure failure = AdmissionFailureClassifier.classifyDecode(
                incoming(50, 128), List.of(endpoint));

        assertFailure(failure, StrategyErrorType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED);
    }

    @Test
    void unattributedOccupantOnResidualDimensionOverridesKnownPriorityLabel() {
        DecodeRequestSnapshot unattributed = request(
                "1", 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 0, false, false);
        DecodeRequestSnapshot higher = request(
                "2", 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 0, true, false);
        DecodeEndpointSnapshot endpoint = endpoint(
                "decode-1", 2, 2, 1_000, 2_000,
                List.of(), List.of(unattributed, higher), List.of());

        AdmissionFailure failure = AdmissionFailureClassifier.classifyDecode(
                incoming(50, 128), List.of(endpoint));

        assertFailure(failure, StrategyErrorType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED);
    }

    @Test
    void knownHigherSlotCapacityWinsWhenItFullyCoversResidual() {
        DecodeRequestSnapshot unattributed = request(
                "1", 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 0, false, false);
        DecodeRequestSnapshot higher = request(
                "2", 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 0, true, false);
        DecodeEndpointSnapshot endpoint = endpoint(
                "decode-1", 2, 2, 2, 1_000, 2_000,
                List.of(), List.of(unattributed, higher), List.of());

        AdmissionFailure failure = AdmissionFailureClassifier.classifyDecode(
                incoming(50, 128), List.of(endpoint));

        assertFailure(failure, StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.HIGHER_PRIORITY_AHEAD);
    }

    @Test
    void knownHigherAndSameSlotsTogetherCoverResidualDespiteUnattributedOccupant() {
        DecodeRequestSnapshot unattributed = request(
                "1", 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 0, false, false);
        DecodeRequestSnapshot higher = request(
                "2", 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 0, true, false);
        DecodeRequestSnapshot same = request(
                "3", 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 0, true, false);
        DecodeEndpointSnapshot endpoint = endpoint(
                "decode-1", 3, 3, 2, 1_000, 2_000,
                List.of(), List.of(unattributed, higher, same), List.of());

        AdmissionFailure failure = AdmissionFailureClassifier.classifyDecode(
                incoming(50, 128), List.of(endpoint));

        assertFailure(failure, StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.HIGHER_PRIORITY_AHEAD);
    }

    @Test
    void knownSameKvCapacityWinsWhenItFullyCoversResidual() {
        DecodeRequestSnapshot unattributed = request(
                "1", 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 256, false, false);
        DecodeRequestSnapshot same = request(
                "2", 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 256, true, false);
        DecodeEndpointSnapshot endpoint = endpoint(
                "decode-1", 0, 0, 0, 2_000,
                List.of(), List.of(unattributed, same), List.of());

        AdmissionFailure failure = AdmissionFailureClassifier.classifyDecode(
                incoming(50, 128), List.of(endpoint));

        assertFailure(failure, StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.SAME_PRIORITY_AHEAD);
    }

    @Test
    void unattributedKvIsCausalWhenKnownProtectedKvCannotCoverResidual() {
        DecodeRequestSnapshot unattributed = request(
                "1", 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 128, false, false);
        DecodeRequestSnapshot higher = request(
                "2", 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 64, true, false);
        DecodeEndpointSnapshot endpoint = endpoint(
                "decode-1", 0, 0, 0, 2_000,
                List.of(), List.of(unattributed, higher), List.of());

        AdmissionFailure failure = AdmissionFailureClassifier.classifyDecode(
                incoming(50, 128), List.of(endpoint));

        assertFailure(failure, StrategyErrorType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED);
    }

    @Test
    void knownHigherAndSameKvTogetherCoverResidualDespiteUnattributedOccupant() {
        DecodeRequestSnapshot unattributed = request(
                "1", 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 128, false, false);
        DecodeRequestSnapshot higher = request(
                "2", 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 64, true, false);
        DecodeRequestSnapshot same = request(
                "3", 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 64, true, false);
        DecodeEndpointSnapshot endpoint = endpoint(
                "decode-1", 0, 0, 0, 2_000,
                List.of(), List.of(unattributed, higher, same), List.of());

        AdmissionFailure failure = AdmissionFailureClassifier.classifyDecode(
                incoming(50, 128), List.of(endpoint));

        assertFailure(failure, StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.HIGHER_PRIORITY_AHEAD);
    }

    @Test
    void differingEndpointCausesFallBackToResourceExhausted() {
        DecodeEndpointSnapshot higher = endpoint(
                "decode-higher", 1, 1, 1_000, 2_000, List.of(),
                List.of(request("1", 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                        0, true, false)), List.of());
        DecodeEndpointSnapshot same = endpoint(
                "decode-same", 1, 1, 1_000, 2_000, List.of(),
                List.of(request("2", 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                        0, true, false)), List.of());

        AdmissionFailure failure = AdmissionFailureClassifier.classifyDecode(
                incoming(50, 128), List.of(higher, same));

        assertFailure(failure, StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED);
    }

    @Test
    void oneEndpointWithUnattributedBlockerMakesClusterAttributionUnavailable() {
        DecodeEndpointSnapshot unattributed = endpoint(
                "decode-unattributed", 1, 1, 1_000, 2_000, List.of(),
                List.of(request("1", 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                        0, false, false)), List.of());
        DecodeEndpointSnapshot higher = endpoint(
                "decode-higher", 1, 1, 1_000, 2_000, List.of(),
                List.of(request("2", 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                        0, true, false)), List.of());

        AdmissionFailure failure = AdmissionFailureClassifier.classifyDecode(
                incoming(50, 128), List.of(unattributed, higher));

        assertFailure(failure, StrategyErrorType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED);
    }

    @Test
    void unattributedEndpointDominatesEndpointWithSnapshotCapacity() {
        DecodeEndpointSnapshot unattributed = endpoint(
                "decode-unattributed", 1, 1, 1_000, 2_000, List.of(),
                List.of(request("1", 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                        0, false, false)), List.of());
        DecodeEndpointSnapshot snapshotHasCapacity = endpoint(
                "decode-capacity", 0, 0, 1_000, 2_000,
                List.of(), List.of(), List.of());

        AdmissionFailure failure = AdmissionFailureClassifier.classifyDecode(
                incoming(50, 128), List.of(unattributed, snapshotHasCapacity));

        assertFailure(failure, StrategyErrorType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED);
    }

    @Test
    void unanimousEndpointCauseRemainsTyped() {
        DecodeEndpointSnapshot first = endpoint(
                "decode-1", 1, 1, 1_000, 2_000, List.of(),
                List.of(request("1", 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                        0, true, false)), List.of());
        DecodeEndpointSnapshot second = endpoint(
                "decode-2", 1, 1, 1_000, 2_000, List.of(),
                List.of(request("2", 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                        0, true, false)), List.of());

        AdmissionFailure failure = AdmissionFailureClassifier.classifyDecode(
                incoming(50, 128), List.of(first, second));

        assertFailure(failure, StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.HIGHER_PRIORITY_AHEAD);
    }

    @Test
    void hardKvLargerThanEveryKnownEndpointIsResourceExhausted() {
        DecodeEndpointSnapshot first = endpoint(
                "decode-1", 0, 0, 100, 100, List.of(), List.of(), List.of());
        DecodeEndpointSnapshot second = endpoint(
                "decode-2", 0, 0, 200, 200, List.of(), List.of(), List.of());

        AdmissionFailure failure = AdmissionFailureClassifier.classifyDecode(
                incoming(50, 256), List.of(first, second));

        assertFailure(failure, StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED);
    }

    @Test
    void physicallyImpossibleEndpointDoesNotManufactureUnattributedCause() {
        DecodeEndpointSnapshot tooSmallWithUnattributedOccupant = endpoint(
                "decode-too-small", 1, 1, 0, 100, List.of(),
                List.of(request("1", 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                        0, false, false)), List.of());
        DecodeEndpointSnapshot higher = endpoint(
                "decode-higher", 1, 1, 1_000, 2_000, List.of(),
                List.of(request("2", 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                        0, true, false)), List.of());

        AdmissionFailure failure = AdmissionFailureClassifier.classifyDecode(
                incoming(50, 128), List.of(tooSmallWithUnattributedOccupant, higher));

        // The first endpoint cannot fit the request even when empty, so its
        // occupant provenance cannot cause the cluster rejection. The two
        // remaining endpoint causes differ and conservatively fold to 8431.
        assertFailure(failure, StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED);
    }

    @Test
    void prefillWithoutSnapshotDeficitIsResourceExhausted() {
        PrefillQueueSnapshot queue = new PrefillQueueSnapshot(
                "prefill-1", 1, 4,
                List.of(new QueuedRequestSnapshot("1", 70, 0,
                        128, 0, QueuedRequestSnapshot.PREFILL_QUEUED)));

        AdmissionFailure failure = AdmissionFailureClassifier.classifyPrefill(
                incoming(50, 128), queue);

        assertFailure(failure, StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED);
    }

    @Test
    void prefillResidualBlockedByLegacyOccupantIsAdmissionUnavailable() {
        PrefillQueueSnapshot queue = new PrefillQueueSnapshot(
                "prefill-1", 1, 1,
                List.of(new QueuedRequestSnapshot("1", 0, 0,
                        128, 0, QueuedRequestSnapshot.PREFILL_QUEUED)));

        AdmissionFailure failure = AdmissionFailureClassifier.classifyPrefill(
                incoming(50, 128), queue);

        assertFailure(failure, StrategyErrorType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED);
    }

    @Test
    void prefillProvenHigherBlockerWinsWhenItCoversResidual() {
        PrefillQueueSnapshot queue = new PrefillQueueSnapshot(
                "prefill-1", 1, 2,
                List.of(
                        queued("1", 70),
                        new QueuedRequestSnapshot("2", 0, 0,
                                128, 0, QueuedRequestSnapshot.PREFILL_QUEUED)));

        AdmissionFailure failure = AdmissionFailureClassifier.classifyPrefill(
                incoming(50, 128), queue);

        assertFailure(failure, StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.HIGHER_PRIORITY_AHEAD);
    }

    @Test
    void prefillUnattributedOccupantIsNeededBeyondKnownProtectedCapacity() {
        PrefillQueueSnapshot queue = new PrefillQueueSnapshot(
                "prefill-1", 1, 1,
                List.of(
                        queued("1", 70),
                        new QueuedRequestSnapshot("2", 0, 0,
                                128, 0, QueuedRequestSnapshot.PREFILL_QUEUED)));

        AdmissionFailure failure = AdmissionFailureClassifier.classifyPrefill(
                incoming(50, 128), queue);

        assertFailure(failure, StrategyErrorType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED);
    }

    @Test
    void prefillLegacyOccupantDoesNotMatterWhenLowerPriorityCapacityIsEnough() {
        PrefillQueueSnapshot queue = new PrefillQueueSnapshot(
                "prefill-1", 1, 2,
                List.of(
                        new QueuedRequestSnapshot("1", 0, 0,
                                128, 0, QueuedRequestSnapshot.PREFILL_QUEUED),
                        queued("2", 30)));

        AdmissionFailure failure = AdmissionFailureClassifier.classifyPrefill(
                incoming(50, 128), queue);

        // Evicting the one lower-priority item would cover the one-slot
        // deficit, so the unprioritized item is not causally relevant.
        assertFailure(failure, StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED);
    }

    @Test
    void queuedTimeoutUsesHigherPriorityPrefix() {
        AdmissionFailure failure = AdmissionFailureClassifier.classifyQueuedTimeout(
                50, List.of(
                        queued("1", 50),
                        queued("2", 70)));

        assertFailure(failure, StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.HIGHER_PRIORITY_AHEAD);
    }

    @Test
    void queuedTimeoutUsesSamePriorityFifoPrefix() {
        AdmissionFailure failure = AdmissionFailureClassifier.classifyQueuedTimeout(
                50, List.of(queued("1", 30), queued("2", 50)));

        assertFailure(failure, StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.SAME_PRIORITY_AHEAD);
    }

    @Test
    void queuedTimeoutWithoutProtectedPrefixIsResourceExhausted() {
        AdmissionFailure failure = AdmissionFailureClassifier.classifyQueuedTimeout(
                50, List.of(queued("1", 30)));

        assertFailure(failure, StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED);
    }

    @Test
    void queuedTimeoutWithUnattributedItemAheadIsAdmissionUnavailable() {
        AdmissionFailure failure = AdmissionFailureClassifier.classifyQueuedTimeout(
                50, List.of(queued("1", 0)));

        assertFailure(failure, StrategyErrorType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED);
    }

    @Test
    void queuedTimeoutUsesProvenHigherBeforeUnattributedFallback() {
        AdmissionFailure failure = AdmissionFailureClassifier.classifyQueuedTimeout(
                50, List.of(queued("1", 0), queued("2", 70)));

        assertFailure(failure, StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.HIGHER_PRIORITY_AHEAD);
    }

    private static QueuedRequestSnapshot queued(String requestId, int priority) {
        return new QueuedRequestSnapshot(requestId, priority, Long.parseLong(requestId),
                128, 0, QueuedRequestSnapshot.PREFILL_QUEUED);
    }

    private static DecodeEndpointSnapshot endpoint(
            int totalLoad,
            int engineLoad,
            long realKvAvailable,
            long realKvTotal,
            List<DecodeRequestSnapshot> reserved,
            List<DecodeRequestSnapshot> accepted,
            List<DecodeRequestSnapshot> running) {
        return endpoint("decode-1", totalLoad, engineLoad, realKvAvailable,
                realKvTotal, reserved, accepted, running);
    }

    private static DecodeEndpointSnapshot endpoint(
            String endpointId,
            int totalLoad,
            int engineLoad,
            long realKvAvailable,
            long realKvTotal,
            List<DecodeRequestSnapshot> reserved,
            List<DecodeRequestSnapshot> accepted,
            List<DecodeRequestSnapshot> running) {
        return endpoint(endpointId, totalLoad, engineLoad, 1,
                realKvAvailable, realKvTotal, reserved, accepted, running);
    }

    private static DecodeEndpointSnapshot endpoint(
            String endpointId,
            int totalLoad,
            int engineLoad,
            long concurrencyLimit,
            long realKvAvailable,
            long realKvTotal,
            List<DecodeRequestSnapshot> reserved,
            List<DecodeRequestSnapshot> accepted,
            List<DecodeRequestSnapshot> running) {
        long hardKv = reserved.stream().mapToLong(DecodeRequestSnapshot::kvTokens).sum();
        return new DecodeEndpointSnapshot(null, endpointId, 1,
                realKvAvailable, realKvTotal, totalLoad, engineLoad, concurrencyLimit,
                hardKv, hardKv, reserved, accepted, running);
    }

    private static DecodeRequestSnapshot request(String requestId,
                                                  int priority,
                                                  DecodeTaskPhase phase,
                                                  long kvTokens,
                                                  boolean priorityKnown,
                                                  boolean queued) {
        return new DecodeRequestSnapshot(requestId, priority, phase,
                kvTokens, kvTokens, priorityKnown, queued);
    }

    private static PriorityRequestEnvelope incoming(int priority, long hardKvTokens) {
        return new PriorityRequestEnvelope("999", priority, hardKvTokens, 0,
                0, hardKvTokens, hardKvTokens);
    }

    private static void assertFailure(AdmissionFailure actual,
                                      StrategyErrorType errorType,
                                      AdmissionRejectReason reason) {
        assertEquals(errorType, actual.errorType());
        assertEquals(reason, actual.reason());
    }
}
