package org.flexlb.dao.master;

import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class AppliedObservationTest {
    private WorkerStatus discovered() {
        return WorkerStatus.createDiscovered(RoleType.PREFILL, "g", "127.0.0.1", 8080, 9090, "s");
    }

    private WorkerStatus.StatusObservation observation(WorkerStatus status) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setStatusVersion(1L);
        response.setAlive(true);
        return status.freezeStatusResponse(response);
    }

    @Test
    void receivedButUnappliedResponseCannotRefreshApplicationHealth() {
        WorkerStatus status = discovered();
        var observation = observation(status);
        status.lock.lock();
        try {
            status.recordSuccessfulPoll(true);
            assertEquals(0, status.appliedObservation().observedAtUs());
            assertThrows(IllegalArgumentException.class, () -> status.recordAppliedObservation(observation));
            status.publishPreparedStatus(status.prepareNewStatus(observation));
            status.recordAppliedObservation(observation);
            assertEquals(observation.observedAtUs(), status.appliedObservation().observedAtUs());
        } finally {
            status.lock.unlock();
        }
    }

    @Test
    void unchangedVersionHeartbeatRefreshesObservationWithoutReplayingState() {
        WorkerStatus status = discovered();
        status.lock.lock();
        try {
            var initial = observation(status);
            status.publishPreparedStatus(status.prepareNewStatus(initial));
            status.recordAppliedObservation(initial);
            var committed = status.committedWorkerStatus();
            var heartbeat = observation(status);
            status.recordAppliedObservation(heartbeat);
            assertSame(committed, status.committedWorkerStatus());
            assertEquals(heartbeat.observedAtUs(), status.appliedObservation().observedAtUs());
            assertThrows(IllegalArgumentException.class,
                    () -> status.recordAppliedObservation(observation(discovered())));
        } finally {
            status.lock.unlock();
        }
    }
}
