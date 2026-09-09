package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** A deferred delivery failure must not consume the only inactivity-expiration path. */
class RequestAdmissionExpirationRaceTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void pendingFailureAndLateDecodeStatusCannotStrandAnExpiredAdmission(boolean clientCancellation) throws Exception {
        long requestId = 301L;
        var config = new org.flexlb.config.FlexlbConfig();
        SchedulingTestConfig.useNonBatchDispatcher(config);
        config.queueScheduler().getLifecycle().setStaleInflightTimeoutMs(300L);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var registry = new RequestRegistry(service, mock(BatchSchedulerReporter.class),
                mock(RequestSchedulerReporter.class));
        try {
            var context = RequestLifecycleTestSupport.context(config, requestId);
            var prefill = mock(PrefillEndpoint.class);
            var decode = mock(DecodeEndpoint.class);
            var reservation = new DecodeEndpoint.ReservationHandle(1L, requestId, 1L);
            var prefillStatus = new ServerStatus();
            prefillStatus.setRole(RoleType.PREFILL);
            prefillStatus.setServerIp("127.0.0.1");
            prefillStatus.setGrpcPort(8081);
            var future = registry.register(context, 0);
            RequestSlot slot = registry.requestSlot(requestId);
            var item = new ScheduledRequest(context, future, new Response(), prefillStatus, null,
                    prefill, decode, reservation, slot.createdAtMs());

            try (var admission = registry.claimAdmissionMutation(requestId, future)) {
                assertNotNull(admission);
                assertTrue(registry.commitItemForPublication(item, () -> true));
                if (clientCancellation) {
                    registry.cancelRequest(requestId, 0L, CancelReason.CLIENT_CANCELLED);
                }
                registry.failPrepared(item, new IllegalStateException("preparation failed"));
                assertFalse(future.isDone(), "the admission still owns its deferred delivery failure");

                // The automatic timer covers a new timeout; the explicit clock
                // also covers expiry after a previously recorded client cancellation.
                if (clientCancellation) {
                    registry.cancelForRequestInactivity(slot, slot.createdAtMs() + 300L);
                } else {
                    RequestLifecycleTestSupport.awaitCondition(() -> registry.getRequestState(requestId, 0L)
                            .state() == RequestState.Phase.CANCEL_REQUESTED);
                }
                synchronized (slot) {
                    assertTrue(slot.inactivityDeadlineAtMs().isEmpty(),
                            "the fired deadline stays disarmed until the admission is completed");
                }
                assertFalse(future.isDone());
                new EndpointEventProjector(registry).onDecodeStatus(decode,
                        java.util.List.of(DecodeEndpoint.WorkerStatusFact.accepted(reservation)));
                synchronized (slot) {
                    slot.observeWorkerStatus(System.currentTimeMillis() + TimeUnit.HOURS.toMillis(1L));
                }
            }

            // Replaying DELIVERY_FAILURE after Decode ACTIVE no longer yields a terminal;
            // admission completion must still resume the already elected timeout cleanup.
            assertFalse(future.get(2L, TimeUnit.SECONDS).isSuccess());
            assertEquals(clientCancellation ? RequestState.Phase.CANCELLED : RequestState.Phase.TIMED_OUT,
                    registry.getRequestState(requestId, 0L).state());
            assertEquals(0, registry.liveRequestCount());
            verify(decode, times(1)).expireReservationExact(reservation);
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
