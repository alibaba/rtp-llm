package org.flexlb.worker.endpoint;

import org.flexlb.config.service.RoleType;
import org.flexlb.worker.status.WorkerStatus;

/**
 * Races late admissions with registry detachment to verify every stateful endpoint closes admission before draining requests.
 */
class EndpointCloseLinearizationTest {

    private static WorkerStatus status(RoleType role, int port) {
        return EndpointTestSupport.workerStatus(role, "127.0.0.1", port, port + 1);
    }

    private static java.util.concurrent.CompletionStage<Void> detachUnderGenerationLock(
            EndpointRegistry registry, RoleType role, String address, WorkerStatus expectedStatus) {
        expectedStatus.lock.lock();
        try {
            return registry.closeWorker(role, address, expectedStatus);
        } finally {
            expectedStatus.lock.unlock();
        }
    }

    private record AdmissionAttempt(
            boolean queueOfferAccepted,
            boolean prefillAdmissionAvailable,
            boolean decodeAdmissionAvailable) {}
}
