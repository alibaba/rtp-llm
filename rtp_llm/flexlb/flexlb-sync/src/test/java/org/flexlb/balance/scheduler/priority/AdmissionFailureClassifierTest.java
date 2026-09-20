package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Causal attribution must use known priorities on the dimensions in deficit. */
class AdmissionFailureClassifierTest {

    @Test
    void kvOccupancyWithoutTaskDetailsIsAdmissionUnavailable() {
        Response failure = classifyDecode(incoming(50, 128),
                List.of(endpoint(0, 0, 64, 2_000, List.of(), List.of(), List.of())));
        assertFailure(failure, StrategyErrorType.ADMISSION_UNAVAILABLE, AdmissionRejectReason.UNSPECIFIED);
    }

    @Test
    void availabilityWatermarkIsNotAnUnexplainedSelectionFailure() {
        DecodeEndpointSnapshot endpoint = endpoint(0, 0, 1_000, 2_000, List.of(), List.of(), List.of());
        Response failure = classifyDecode(incoming(50, 128),
                List.of(endpoint));
        assertFailure(failure, StrategyErrorType.RESOURCE_EXHAUSTED, AdmissionRejectReason.RESOURCE_EXHAUSTED);
    }

    @Test
    void confirmedOccupantWithoutPriorityProvenanceIsAdmissionUnavailable() {
        DecodeRequestSnapshot unattributed = request(
                1, 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 128, false, false);
        DecodeEndpointSnapshot endpoint = endpoint(
                1, 1, 1_000, 2_000, List.of(), List.of(unattributed), List.of());

        Response failure = classifyDecode(
                incoming(50, 128), List.of(endpoint));

        assertFailure(failure, StrategyErrorType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED);
    }

    @Test
    void queuedReservationDoesNotExplainEngineSlotDeficit() {
        DecodeRequestSnapshot higherQueued = request(
                1, 70, DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED,
                0, true, true);
        DecodeEndpointSnapshot endpoint = endpoint(
                1, 1, 1_000, 2_000,
                List.of(higherQueued), List.of(), List.of());

        Response failure = classifyDecode(
                incoming(50, 128), List.of(endpoint));

        // The one engine slot is occupied by an unreported request, not the queued reservation.
        assertFailure(failure, StrategyErrorType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED);
    }

    @Test
    void validPriorityValueWithoutProvenanceIsNotTrusted() {
        DecodeRequestSnapshot untrustedP70 = request(
                1, 70, DecodeTaskPhase.RUNNING, 0, false, false);
        DecodeEndpointSnapshot endpoint = endpoint(
                1, 1, 1_000, 2_000,
                List.of(), List.of(), List.of(untrustedP70));

        Response failure = classifyDecode(
                incoming(50, 128), List.of(endpoint));

        assertFailure(failure, StrategyErrorType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED);
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

        Response failure = classifyDecode(
                incoming(50, 128), List.of(endpoint));

        assertFailure(failure, StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.SAME_PRIORITY_AHEAD);
    }

    @Test
    void queuedReservationWithoutPriorityProvenanceCanExplainKvDeficit() {
        DecodeRequestSnapshot unattributedQueued = request(
                1, 50, DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED,
                512, false, true);
        DecodeEndpointSnapshot endpoint = endpoint(
                1, 0, 0, 2_000,
                List.of(unattributedQueued), List.of(), List.of());

        Response failure = classifyDecode(
                incoming(50, 128), List.of(endpoint));

        assertFailure(failure, StrategyErrorType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED);
    }

    @Test
    void unattributedOccupantOnResidualDimensionOverridesKnownPriorityLabel() {
        DecodeRequestSnapshot unattributed = request(
                1, 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 0, false, false);
        DecodeRequestSnapshot higher = request(
                2, 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 0, true, false);
        DecodeEndpointSnapshot endpoint = endpoint(
                "decode-1", 2, 2, 1_000, 2_000,
                List.of(), List.of(unattributed, higher), List.of());

        Response failure = classifyDecode(
                incoming(50, 128), List.of(endpoint));

        assertFailure(failure, StrategyErrorType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED);
    }

    @Test
    void knownHigherSlotCapacityWinsWhenItFullyCoversResidual() {
        DecodeRequestSnapshot unattributed = request(
                1, 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 0, false, false);
        DecodeRequestSnapshot higher = request(
                2, 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 0, true, false);
        DecodeEndpointSnapshot endpoint = endpoint(
                "decode-1", 2, 2, 2, 1_000, 2_000,
                List.of(), List.of(unattributed, higher), List.of());

        Response failure = classifyDecode(
                incoming(50, 128), List.of(endpoint));

        assertFailure(failure, StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.HIGHER_PRIORITY_AHEAD);
    }

    @Test
    void knownHigherAndSameSlotsTogetherCoverResidualDespiteUnattributedOccupant() {
        DecodeRequestSnapshot unattributed = request(
                1, 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 0, false, false);
        DecodeRequestSnapshot higher = request(
                2, 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 0, true, false);
        DecodeRequestSnapshot same = request(
                3, 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 0, true, false);
        DecodeEndpointSnapshot endpoint = endpoint(
                "decode-1", 3, 3, 2, 1_000, 2_000,
                List.of(), List.of(unattributed, higher, same), List.of());

        Response failure = classifyDecode(
                incoming(50, 128), List.of(endpoint));

        assertFailure(failure, StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.HIGHER_PRIORITY_AHEAD);
    }

    @Test
    void knownSameKvCapacityWinsWhenItFullyCoversResidual() {
        DecodeRequestSnapshot unattributed = request(
                1, 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 256, false, false);
        DecodeRequestSnapshot same = request(
                2, 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 256, true, false);
        DecodeEndpointSnapshot endpoint = endpoint(
                "decode-1", 0, 0, 0, 2_000,
                List.of(), List.of(unattributed, same), List.of());

        Response failure = classifyDecode(
                incoming(50, 128), List.of(endpoint));

        assertFailure(failure, StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.SAME_PRIORITY_AHEAD);
    }

    @Test
    void unattributedKvIsCausalWhenKnownProtectedKvCannotCoverResidual() {
        DecodeRequestSnapshot unattributed = request(
                1, 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 128, false, false);
        DecodeRequestSnapshot higher = request(
                2, 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 64, true, false);
        DecodeEndpointSnapshot endpoint = endpoint(
                "decode-1", 0, 0, 0, 2_000,
                List.of(), List.of(unattributed, higher), List.of());

        Response failure = classifyDecode(
                incoming(50, 128), List.of(endpoint));

        assertFailure(failure, StrategyErrorType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED);
    }

    @Test
    void knownHigherAndSameKvTogetherCoverResidualDespiteUnattributedOccupant() {
        DecodeRequestSnapshot unattributed = request(
                1, 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 128, false, false);
        DecodeRequestSnapshot higher = request(
                2, 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 64, true, false);
        DecodeRequestSnapshot same = request(
                3, 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING, 64, true, false);
        DecodeEndpointSnapshot endpoint = endpoint(
                "decode-1", 0, 0, 0, 2_000,
                List.of(), List.of(unattributed, higher, same), List.of());

        Response failure = classifyDecode(
                incoming(50, 128), List.of(endpoint));

        assertFailure(failure, StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.HIGHER_PRIORITY_AHEAD);
    }

    @Test
    void differingEndpointCausesFallBackToResourceExhausted() {
        DecodeEndpointSnapshot higher = endpoint(
                "decode-higher", 1, 1, 1_000, 2_000, List.of(),
                List.of(request(1, 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                        0, true, false)), List.of());
        DecodeEndpointSnapshot same = endpoint(
                "decode-same", 1, 1, 1_000, 2_000, List.of(),
                List.of(request(2, 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                        0, true, false)), List.of());

        Response failure = classifyDecode(
                incoming(50, 128), List.of(higher, same));

        assertFailure(failure, StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED);
    }

    @Test
    void oneEndpointWithUnattributedBlockerMakesClusterAttributionUnavailable() {
        DecodeEndpointSnapshot unattributed = endpoint(
                "decode-unattributed", 1, 1, 1_000, 2_000, List.of(),
                List.of(request(1, 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                        0, false, false)), List.of());
        DecodeEndpointSnapshot higher = endpoint(
                "decode-higher", 1, 1, 1_000, 2_000, List.of(),
                List.of(request(2, 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                        0, true, false)), List.of());

        Response failure = classifyDecode(
                incoming(50, 128), List.of(unattributed, higher));

        assertFailure(failure, StrategyErrorType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED);
    }

    @Test
    void unattributedEndpointDominatesEndpointWithSnapshotCapacity() {
        DecodeEndpointSnapshot unattributed = endpoint(
                "decode-unattributed", 1, 1, 1_000, 2_000, List.of(),
                List.of(request(1, 50, DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                        0, false, false)), List.of());
        DecodeEndpointSnapshot snapshotHasCapacity = endpoint(
                "decode-capacity", 0, 0, 1_000, 2_000,
                List.of(), List.of(), List.of());

        Response failure = classifyDecode(
                incoming(50, 128), List.of(unattributed, snapshotHasCapacity));

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

        Response failure = classifyDecode(
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

        Response failure = classifyDecode(
                incoming(50, 256), List.of(first, second));

        assertFailure(failure, StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED);
    }

    @Test
    void physicallyImpossibleEndpointDoesNotManufactureUnattributedCause() {
        DecodeEndpointSnapshot tooSmallWithUnattributedOccupant = endpoint(
                "decode-too-small", 1, 1, 0, 100, List.of(),
                List.of(request(1, 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                        0, false, false)), List.of());
        DecodeEndpointSnapshot higher = endpoint(
                "decode-higher", 1, 1, 1_000, 2_000, List.of(),
                List.of(request(2, 70, DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                        0, true, false)), List.of());

        Response failure = classifyDecode(
                incoming(50, 128), List.of(tooSmallWithUnattributedOccupant, higher));

        // The first endpoint cannot fit the request even when empty, so its
        // occupant provenance cannot cause the cluster rejection. The two
        // remaining endpoint causes differ and conservatively fold to 8431.
        assertFailure(failure, StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED);
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

    private static DecodeRequestSnapshot request(long requestId,
                                                  int priority,
                                                  DecodeTaskPhase phase,
                                                  long kvTokens,
                                                  boolean priorityKnown,
                                                  boolean queued) {
        return new DecodeRequestSnapshot(requestId, priority, phase,
                kvTokens, kvTokens, priorityKnown, queued);
    }

    private static Response classifyDecode(PriorityRequestEnvelope incoming, List<DecodeEndpointSnapshot> endpoints) {
        var summaries = new ArrayList<DecodeEndpoint.AdmissionSnapshot>();
        for (DecodeEndpointSnapshot fixture : endpoints) {
            var status = new WorkerStatus();
            var endpoint = Mockito.spy(new DecodeEndpoint(status));
            var confirmed = new HashMap<String, TaskInfo>();
            for (var layer : List.of(fixture.accepted(), fixture.running())) {
                for (var occupant : layer) {
                    if (occupant.priorityKnown()) {
                        endpoint.reserve(occupant.requestId(), occupant.kvTokens(), occupant.kvTokens(), occupant.priority());
                    }
                    var task = new TaskInfo();
                    task.setRequestId(occupant.requestId());
                    task.setInputLength(occupant.kvTokens());
                    task.setPhase(occupant.phase() == DecodeTaskPhase.RUNNING
                            ? TaskPhase.RUNNING : TaskPhase.KV_ALLOCATED);
                    confirmed.put(String.valueOf(occupant.requestId()), task);
                }
            }
            var report = new WorkerStatusResponse();
            report.setRunningTaskInfo(confirmed);
            endpoint.onWorkerStatusUpdate(status, report);
            for (var occupant : fixture.reserved()) {
                endpoint.reserve(occupant.requestId(), occupant.kvTokens(), occupant.kvTokens(),
                        occupant.priorityKnown() ? occupant.priority() : 0);
                if (occupant.queued()) {
                    endpoint.markQueuedPhase(occupant.requestId());
                }
            }
            // Aggregate engine reports may exceed tracked requests; exercise that gap independently.
            Mockito.doReturn(fixture.engineLoad()).when(endpoint).getEngineLoad();
            Mockito.doReturn(fixture.realKvTotal()).when(endpoint).realKvTotal();
            Mockito.doReturn(fixture.realKvAvailable()).when(endpoint).realKvAvailable();
            summaries.add(endpoint.admissionSnapshot());
        }
        return AdmissionFailureClassifier.classifyDecode(incoming.priority(), incoming.hardKvTokens(),
                endpoints.getFirst().concurrencyLimit(), summaries);
    }

    private static PriorityRequestEnvelope incoming(int priority, long hardKvTokens) {
        return new PriorityRequestEnvelope(999, priority, hardKvTokens, 0,
                0, hardKvTokens, hardKvTokens);
    }

    private static void assertFailure(Response actual,
                                      StrategyErrorType errorType,
                                      AdmissionRejectReason reason) {
        assertEquals(errorType.getErrorCode(), actual.getCode());
        assertEquals(reason, actual.getAdmissionRejectReason());
    }
}
