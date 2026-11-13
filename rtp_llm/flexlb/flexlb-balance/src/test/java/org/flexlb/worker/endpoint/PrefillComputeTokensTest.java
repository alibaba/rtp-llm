package org.flexlb.worker.endpoint;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.status.WorkerStatusResponse;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Checks atomic compute-token reservations and releasing exact records.
 */
class PrefillComputeTokensTest {
    private PrefillEndpoint endpoint;

    @BeforeEach
    void setUp() {
        var config = new FlexlbConfig();
        config.getRouter().getRoles().getPrefill().setMaxComputeTokens(100L);
        var status = EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8081);
        endpoint = new PrefillEndpoint(status, config, new CapacityNotifier());
        var response = new WorkerStatusResponse();
        response.setAlive(true);
        EndpointTestSupport.applyStatus(endpoint, response).run();
    }

    @AfterEach
    void close() {
        EndpointTestSupport.closeAndWait(endpoint);
    }

    @Test
    void chargesTokensUntilExactReservationIsReleased() {
        var first = endpoint.tryRegisterSelection("first", endpoint.selectionSnapshot(), 70L);
        assertEquals(WorkerEndpoint.RegistrationStatus.REGISTERED, first.status());
        var second = endpoint.tryRegisterSelection("second", endpoint.selectionSnapshot(), 30L);
        assertEquals(WorkerEndpoint.RegistrationStatus.REGISTERED, second.status());
        assertEquals(100L, endpoint.selectionSnapshot().computeTokens());
        assertEquals(
                WorkerEndpoint.RegistrationStatus.CAPACITY_FULL,
                endpoint.tryRegisterSelection("third", endpoint.selectionSnapshot(), 1L).status());
        assertTrue(endpoint.rollbackSelection(first.value()));
        assertFalse(endpoint.rollbackSelection(first.value()));
        assertEquals(30L, endpoint.selectionSnapshot().computeTokens());
        assertEquals(
                WorkerEndpoint.RegistrationStatus.REGISTERED,
                endpoint.tryRegisterSelection("third", endpoint.selectionSnapshot(), 70L).status());
        assertEquals(100L, endpoint.selectionSnapshot().computeTokens());
    }

    @Test
    void concurrentSelectionsCannotExceedWorkerCapacity() throws Exception {
        var snapshot = endpoint.selectionSnapshot();
        var start = new CountDownLatch(1);
        try (var workers = Executors.newFixedThreadPool(2)) {
            var first =
                    workers.submit(
                            () -> {
                                assertTrue(start.await(2, TimeUnit.SECONDS));
                                return endpoint.tryRegisterSelection("first", snapshot, 60L);
                            });
            var second =
                    workers.submit(
                            () -> {
                                assertTrue(start.await(2, TimeUnit.SECONDS));
                                return endpoint.tryRegisterSelection("second", snapshot, 60L);
                            });
            start.countDown();
            var statuses =
                    List.of(
                            first.get(2, TimeUnit.SECONDS).status(),
                            second.get(2, TimeUnit.SECONDS).status());
            assertEquals(
                    1L,
                    statuses.stream()
                            .filter(s -> s == WorkerEndpoint.RegistrationStatus.REGISTERED)
                            .count());
            assertEquals(
                    1L,
                    statuses.stream()
                            .filter(s -> s == WorkerEndpoint.RegistrationStatus.VERSION_CONFLICT)
                            .count());
            assertEquals(60L, endpoint.selectionSnapshot().computeTokens());
            assertEquals(
                    WorkerEndpoint.RegistrationStatus.CAPACITY_FULL,
                    endpoint.tryRegisterSelection("retry", endpoint.selectionSnapshot(), 60L)
                            .status());
        }
    }

    @Test
    void rejectsSnapshotFromAnotherWorkerGeneration() {
        var actual = endpoint.selectionSnapshot();
        var stale =
                new PrefillEndpoint.SelectionSnapshot(
                        actual.generationId() + 1, actual.resourceVersion(), 0L, 100L);
        assertEquals(
                WorkerEndpoint.RegistrationStatus.VERSION_CONFLICT,
                endpoint.tryRegisterSelection("request", stale, 10L).status());
        assertEquals(0L, endpoint.selectionSnapshot().computeTokens());
    }
}
