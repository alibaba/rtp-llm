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
    void unknownConfirmedPriorityIsNotGuessedAsDefaultFifty() {
        DecodeRequestSnapshot unknown = request(
                1, 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 128, false, false);
        DecodeEndpointSnapshot endpoint = endpoint(
                1, 1, 1_000, 2_000, List.of(), List.of(unknown), List.of());

        AdmissionFailure failure = AdmissionFailureClassifier.classifyDecode(
                incoming(50, 128), List.of(endpoint));

        assertFailure(failure, StrategyErrorType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED);
    }

    @Test
    void higherPriorityQueuedReservationExplainsSlotDeficit() {
        DecodeRequestSnapshot higherQueued = request(
                1, 70, DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED,
                512, true, true);
        DecodeRequestSnapshot sameAccepted = request(
                2, 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                0, true, false);
        DecodeEndpointSnapshot endpoint = endpoint(
                1, 1, 1_000, 2_000,
                List.of(higherQueued), List.of(sameAccepted), List.of());

        AdmissionFailure failure = AdmissionFailureClassifier.classifyDecode(
                incoming(50, 128), List.of(endpoint));

        assertFailure(failure, StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.HIGHER_PRIORITY_AHEAD);
    }

    @Test
    void higherPriorityOnSlotDoesNotExplainKvOnlyDeficit() {
        DecodeRequestSnapshot higherAccepted = request(
                1, 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                0, true, false);
        DecodeRequestSnapshot sameQueued = request(
                2, 50, DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED,
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
    void unknownPriorityOnReservedSlotForcesUnknownAttribution() {
        DecodeRequestSnapshot unknownQueued = request(
                1, 50, DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED,
                512, false, true);
        DecodeEndpointSnapshot endpoint = endpoint(
                1, 1, 1_000, 2_000,
                List.of(unknownQueued), List.of(), List.of());

        AdmissionFailure failure = AdmissionFailureClassifier.classifyDecode(
                incoming(50, 128), List.of(endpoint));

        assertFailure(failure, StrategyErrorType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED);
    }

    @Test
    void unknownPriorityOnResidualDimensionOverridesKnownPriorityLabels() {
        DecodeRequestSnapshot unknown = request(
                1, 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 0, false, false);
        DecodeRequestSnapshot higher = request(
                2, 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 0, true, false);
        DecodeEndpointSnapshot endpoint = endpoint(
                "decode-1", 2, 2, 1_000, 2_000,
                List.of(), List.of(unknown, higher), List.of());

        AdmissionFailure failure = AdmissionFailureClassifier.classifyDecode(
                incoming(50, 128), List.of(endpoint));

        assertFailure(failure, StrategyErrorType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED);
    }

    @Test
    void differingEndpointCausesDegradeToUnknown() {
        DecodeEndpointSnapshot higher = endpoint(
                "decode-higher", 1, 1, 1_000, 2_000, List.of(),
                List.of(request(1, 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                        0, true, false)), List.of());
        DecodeEndpointSnapshot same = endpoint(
                "decode-same", 1, 1, 1_000, 2_000, List.of(),
                List.of(request(2, 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                        0, true, false)), List.of());

        AdmissionFailure failure = AdmissionFailureClassifier.classifyDecode(
                incoming(50, 128), List.of(higher, same));

        assertFailure(failure, StrategyErrorType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED);
    }

    @Test
    void unanimousEndpointCauseRemainsTyped() {
        DecodeEndpointSnapshot first = endpoint(
                "decode-1", 1, 1, 1_000, 2_000, List.of(),
                List.of(request(1, 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                        0, true, false)), List.of());
        DecodeEndpointSnapshot second = endpoint(
                "decode-2", 1, 1, 1_000, 2_000, List.of(),
                List.of(request(2, 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
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
    void prefillWithoutSnapshotDeficitIsUnknown() {
        PrefillQueueSnapshot queue = new PrefillQueueSnapshot(
                "prefill-1", 1, 4,
                List.of(new QueuedRequestSnapshot(1, 70, 0, 0,
                        128, 0, QueuedRequestSnapshot.PREFILL_QUEUED)));

        AdmissionFailure failure = AdmissionFailureClassifier.classifyPrefill(
                incoming(50, 128), queue);

        assertFailure(failure, StrategyErrorType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED);
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
        long hardKv = reserved.stream().mapToLong(DecodeRequestSnapshot::kvTokens).sum();
        return new DecodeEndpointSnapshot(null, endpointId, 1,
                realKvAvailable, realKvTotal, totalLoad, engineLoad, 1,
                hardKv, hardKv, reserved, accepted, running);
    }

    private static DecodeRequestSnapshot request(long requestId,
                                                  int priority,
                                                  DecodeTaskPhase phase,
                                                  long kvTokens,
                                                  boolean priorityKnown,
                                                  boolean queued) {
        return new DecodeRequestSnapshot(requestId, priority, phase,
                kvTokens, kvTokens, 0, priorityKnown, queued);
    }

    private static PriorityRequestEnvelope incoming(int priority, long hardKvTokens) {
        return new PriorityRequestEnvelope(999, priority, hardKvTokens, 0,
                0, 0, 0, hardKvTokens, hardKvTokens);
    }

    private static void assertFailure(AdmissionFailure actual,
                                      StrategyErrorType errorType,
                                      AdmissionRejectReason reason) {
        assertEquals(errorType, actual.errorType());
        assertEquals(reason, actual.reason());
    }
}
