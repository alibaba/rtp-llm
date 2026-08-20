package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DecodeClusterSnapshotTest {

    @Test
    void concurrentMutationAfterFullViewIsRejectedByAdmissionVersion() throws Exception {
        WorkerStatus status = new WorkerStatus();
        status.setIp("10.0.0.1");
        status.setPort(8080);
        status.getAvailableKvCacheTokens().set(10_000);
        status.getTotalKvCacheTokens().set(10_000);

        CountDownLatch viewCaptured = new CountDownLatch(1);
        CountDownLatch mutationFinished = new CountDownLatch(1);
        DecodeEndpoint endpoint = new DecodeEndpoint(status) {
            @Override
            public LayeredAdmissionView layeredAdmissionView() {
                LayeredAdmissionView view = super.layeredAdmissionView();
                viewCaptured.countDown();
                try {
                    if (!mutationFinished.await(1, TimeUnit.SECONDS)) {
                        throw new AssertionError("concurrent mutation did not finish");
                    }
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    throw new AssertionError(interrupted);
                }
                return view;
            }
        };
        endpoint.reserve(1L, 128, 136, 30);

        EndpointRegistry registry = new EndpointRegistry(null, null, null);
        registry.getDecodeEndpoints().put(status.getIpPort(), endpoint);
        FlexlbConfig config = new FlexlbConfig();
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(1L);

        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<DecodeClusterSnapshot> capture = executor.submit(
                    () -> DecodeClusterSnapshot.capture(registry, config));
            assertTrue(viewCaptured.await(1, TimeUnit.SECONDS));
            endpoint.reserve(2L, 64, 72, 20);
            mutationFinished.countDown();

            DecodeEndpointSnapshot snapshot = capture.get(1, TimeUnit.SECONDS)
                    .decodes().get(status.getIpPort());
            assertEquals(List.of(1L), snapshot.reserved().stream()
                    .map(DecodeRequestSnapshot::requestId).toList());
            assertTrue(endpoint.admissionVersion() > snapshot.admissionVersion());

            DecodeEndpoint.ReleaseReserveResult result =
                    endpoint.tryReleaseVictimsAndReserveIncoming(
                            List.of(1L), 3L, 128, 136, 70,
                            snapshot.admissionVersion());
            assertEquals(DecodeEndpoint.ReleaseReserveResult.VERSION_MISMATCH, result);
            assertTrue(endpoint.reservedView().containsKey(1L));
            assertTrue(endpoint.reservedView().containsKey(2L));
            assertFalse(endpoint.reservedView().containsKey(3L));
        } finally {
            mutationFinished.countDown();
            executor.shutdownNow();
            registry.close();
        }
    }
}
