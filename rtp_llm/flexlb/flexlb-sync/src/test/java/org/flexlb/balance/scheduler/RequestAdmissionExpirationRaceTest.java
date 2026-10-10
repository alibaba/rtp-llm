package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.time.Duration;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** A selected failure must not consume the cleanup deadline while admission still owns resources. */
class RequestAdmissionExpirationRaceTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void cleanupFailureStillReleasesAdmissionAndAllowsExpiry(boolean abort) throws Exception {
        var config = SchedulingTestConfig.batchConfig();
        config.getRequestLifecycle().getRequest().setTimeoutMs(300L);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var registry = new RequestRegistry(service, mock(BatchSchedulerReporter.class),
                mock(RequestSchedulerReporter.class));
        try {
            var context = RequestLifecycleTestSupport.context(config, 302L);
            var future = registry.register(context);
            var slot = registry.requestSlot("302");
            var prefill = mock(PrefillEndpoint.class);
            var item = new ScheduledRequest(context, future, new Response(), null, null,
                    prefill, null, null, slot.createdAtMs());
            var cleanupFailure = new IllegalStateException("Prefill cleanup failed");
            doThrow(cleanupFailure).when(prefill).settleFailedRequest(item);
            try (var admission = registry.claimAdmissionHandle("302", future)) {
                assertNotNull(admission);
                assertTrue(registry.commitItemForPublication(item, () -> true));
                registry.failDeliveryPreparation(item, new IllegalStateException("preparation failed"));
                assertSame(cleanupFailure, assertThrows(IllegalStateException.class, () -> {
                    if (abort) {
                        admission.terminate(Response.error(StrategyErrorType.DISPATCH_FAILED));
                    } else {
                        admission.close();
                    }
                }));
            }
            assertFalse(future.get(2L, TimeUnit.SECONDS).isSuccess());
            RequestLifecycleTestSupport.awaitCondition(() -> registry.trackedRequestCount() == 0);
            verify(prefill).expireCommittedItem(item);
            assertEquals(RequestState.Phase.FAILED, slot.snapshot().state());
            assertTimeoutPreemptively(Duration.ofSeconds(2),
                    () -> assertTrue(registry.closeAdmissionAndAwaitMutations()));
        } finally {
            if (registry.closeAdmissionAndAwaitMutations()) {
                registry.closeOutstandingAndTerminalize();
            }
            registry.closeExpiration();
            registry.closePublisher();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void pendingFailureAndLateDecodeStatusCannotStrandAnExpiredAdmission(boolean clientCancellation) throws Exception {
        long requestId = 301L;
        var config = SchedulingTestConfig.newConfig();
        SchedulingTestConfig.useNonBatchDispatcher(config);
        config.getRequestLifecycle().getRequest().setTimeoutMs(300L);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var registry = new RequestRegistry(service, mock(BatchSchedulerReporter.class),
                mock(RequestSchedulerReporter.class));
        try {
            var context = RequestLifecycleTestSupport.context(config, requestId);
            var prefill = mock(PrefillEndpoint.class);
            var decode = mock(DecodeEndpoint.class);
            var reservation = new DecodeEndpoint.ReservationHandle(1L, Long.toString(requestId), 1L);
            var prefillStatus = new ServerStatus();
            prefillStatus.setRole(RoleType.PREFILL);
            prefillStatus.setServerIp("127.0.0.1");
            prefillStatus.setGrpcPort(8081);
            var future = registry.register(context);
            RequestSlot slot = registry.requestSlot(Long.toString(requestId));
            var item = new ScheduledRequest(context, future, new Response(), prefillStatus, null,
                    prefill, decode, reservation, slot.createdAtMs());

            try (var admission = registry.claimAdmissionHandle(Long.toString(requestId), future)) {
                assertNotNull(admission);
                assertTrue(registry.commitItemForPublication(item, () -> true));
                if (clientCancellation) {
                    registry.cancelRequest(Long.toString(requestId), 0L, CancelReason.CLIENT_CANCELLED);
                }
                registry.failDeliveryPreparation(item, new IllegalStateException("preparation failed"));
                if (clientCancellation) {
                    assertFalse(future.isDone());
                } else {
                    assertFalse(future.get(2L, TimeUnit.SECONDS).isSuccess(),
                            "failure publication must not wait for admission cleanup");
                    assertEquals(RequestState.Phase.FAILED, registry.getRequestState(Long.toString(requestId), 0L).state());
                }
                assertFalse(registry.removeExactTerminalRecord(slot, Long.MAX_VALUE));
                registry.expireInactiveRequest(slot, slot.createdAtMs() + 300L);
                synchronized (slot) {
                    assertTrue(slot.inactivityDeadlineAtMs().isEmpty(),
                            "the fired deadline stays disarmed until the admission is completed");
                }
                registry.processDecodeStatus(decode,
                        DecodeEndpoint.WorkerStatusFact.accepted(reservation));
                synchronized (slot) {
                    org.springframework.test.util.ReflectionTestUtils.<RequestSlot.EngineObservation>invokeMethod(slot, "applyDecodeStatusLocked", decode, DecodeEndpoint.WorkerStatusFact.active(reservation), System.currentTimeMillis() + TimeUnit.HOURS.toMillis(1L));
                }
            }

            // Expiration remains a cleanup fact even when the result was already delivered;
            // later Decode activity cannot undo it before the admission owner exits.
            assertFalse(future.get(2L, TimeUnit.SECONDS).isSuccess());
            assertEquals(clientCancellation ? RequestState.Phase.CANCELLED : RequestState.Phase.FAILED,
                    registry.getRequestState(Long.toString(requestId), 0L).state());
            assertEquals(0, registry.trackedRequestCount());
            verify(decode, times(1)).release(reservation, DecodeEndpoint.ReleaseReason.EXPIRED);
            verify(prefill, times(1)).expireCommittedItem(item);
        } finally {
            if (registry.closeAdmissionAndAwaitMutations()) {
                registry.closeOutstandingAndTerminalize();
                registry.closeExpiration();
                registry.closePublisher();
            }
        }
    }
}
