package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class EndpointRetirementLinearizationTest {

    @Test
    void detachClosesEveryStatefulAdmissionGateBeforeFinalDrain()
            throws Exception {
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        EndpointTestSupport.TestRequestRuntime requestRuntime =
                EndpointTestSupport.requestRuntime();
        EndpointRegistry registry = new EndpointRegistry(
                configService,
                requestRuntime.events(),
                mock(BatchSchedulerReporter.class),
                EndpointTestSupport.routeStrategy(requestRuntime),
                new PlacementAvailability());
        String prefillAddress = "127.0.0.1:8100";
        String decodeAddress = "127.0.0.1:8200";
        WorkerStatus prefillStatus = status(RoleType.PREFILL, 8100);
        WorkerStatus decodeStatus = status(RoleType.DECODE, 8200);
        PrefillEndpoint capturedPrefill = (PrefillEndpoint)
                EndpointTestSupport.publishEndpoint(registry,
                        RoleType.PREFILL, prefillAddress, prefillStatus);
        DecodeEndpoint capturedDecode = (DecodeEndpoint)
                EndpointTestSupport.publishEndpoint(registry,
                        RoleType.DECODE, decodeAddress, decodeStatus);

        CountDownLatch referencesCaptured = new CountDownLatch(1);
        CountDownLatch resumeAdmission = new CountDownLatch(1);
        ExecutorService executor = Executors.newSingleThreadExecutor();
        EndpointRegistry.DetachedGeneration detachedPrefill = null;
        EndpointRegistry.DetachedGeneration detachedDecode = null;
        try {
            Future<AdmissionAttempt> delayedAdmission = executor.submit(() -> {
                // These references intentionally outlive their registry mappings.
                PrefillEndpoint prefill = capturedPrefill;
                DecodeEndpoint decode = capturedDecode;
                referencesCaptured.countDown();
                assertTrue(resumeAdmission.await(2, TimeUnit.SECONDS));

                boolean queueOfferAccepted = EndpointTestSupport.offer(
                        prefill, mock(ScheduledRequest.class));
                boolean directPinAvailable;
                try (WorkerEndpoint.GenerationPin pin =
                             prefill.tryPinGeneration()) {
                    directPinAvailable = pin != null;
                }
                boolean decodePinAvailable;
                try (WorkerEndpoint.GenerationPin pin =
                             decode.tryPinGeneration()) {
                    decodePinAvailable = pin != null;
                }
                return new AdmissionAttempt(
                        queueOfferAccepted,
                        directPinAvailable,
                        decodePinAvailable);
            });

            assertTrue(referencesCaptured.await(2, TimeUnit.SECONDS));
            detachedPrefill = detachUnderGenerationLock(
                    registry,
                    RoleType.PREFILL, prefillAddress, prefillStatus);
            detachedDecode = detachUnderGenerationLock(
                    registry,
                    RoleType.DECODE, decodeAddress, decodeStatus);

            assertTrue(detachedPrefill.ownsEndpoint(capturedPrefill));
            assertTrue(detachedDecode.ownsEndpoint(capturedDecode));
            assertFalse(prefillStatus.isActiveGeneration());
            assertFalse(decodeStatus.isActiveGeneration());
            assertNull(registry.get(RoleType.PREFILL, prefillAddress));
            assertNull(registry.get(RoleType.DECODE, decodeAddress));

            // Resume while close/drain has deliberately not run yet. Detach itself
            // must already have made every new generation-local owner impossible.
            resumeAdmission.countDown();
            AdmissionAttempt attempt = delayedAdmission.get(2, TimeUnit.SECONDS);
            assertFalse(attempt.queueOfferAccepted());
            assertFalse(attempt.directPinAvailable());
            assertFalse(attempt.decodePinAvailable());
            assertTrue(capturedDecode.layeredAdmissionView()
                    .reserved().isEmpty());
        } finally {
            resumeAdmission.countDown();
            if (detachedPrefill != null) {
                detachedPrefill.retireAndAwait();
            }
            if (detachedDecode != null) {
                detachedDecode.retireAndAwait();
            }
            registry.close();
            executor.shutdownNow();
        }
    }

    private static WorkerStatus status(RoleType role, int port) {
        return EndpointTestSupport.workerStatus(
                role, "127.0.0.1", port, port + 1);
    }

    private static EndpointRegistry.DetachedGeneration
            detachUnderGenerationLock(
                    EndpointRegistry registry,
                    RoleType role,
                    String address,
                    WorkerStatus expectedStatus) {
        expectedStatus.lock.lock();
        try {
            return registry.detachAndBeginRetirement(
                    role, address, expectedStatus);
        } finally {
            expectedStatus.lock.unlock();
        }
    }

    private record AdmissionAttempt(
            boolean queueOfferAccepted,
            boolean directPinAvailable,
            boolean decodePinAvailable) {
    }
}
