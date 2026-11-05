package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.EncoderStrategy;
import org.flexlb.balance.strategy.RandomStrategy;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.NaviSchedPrefillSchedulerConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class ExternalPrefillAdmissionTest {
    @Test
    void delegatedRouteRetainsDecodeAndCompletesFromDecodeEvidence() throws Exception {
        try (Fixture fixture = new Fixture()) {
            var result = fixture.commit();
            assertEquals(PlacementResult.Status.SUCCESS, result.status());
            fixture.registry.publishExternalPrefillRoute(result.value());
            var response = fixture.context.getFuture().get(2, TimeUnit.SECONDS);
            assertTrue(response.isSuccess());
            assertFalse(response.isEnqueuedByMaster());
            assertEquals(
                    List.of(RoleType.DECODE),
                    response.getServerStatus().stream().map(ServerStatus::getRole).toList());
            var slot = fixture.registry.requestSlot("901");
            assertTrue(slot.activeItem().isExternalPrefill());
            assertFalse(slot.activeItem().requiresRouteReservation());
            assertNull(slot.activeItem().prefill());
            assertNull(slot.activeItem().prefillEp());
            assertTrue(slot.decisionDeadlineAtMs().isEmpty());
            assertEquals(RequestState.Phase.ACKNOWLEDGED, slot.snapshot().state());
            assertEquals(1, fixture.registry.trackedRequestCount());
            verify(fixture.permit).dispatch();
            verify(fixture.permit, never()).release();
            verify(fixture.decode, never()).release(any(), any());
            fixture.registry.processDecodeStatus(
                    fixture.decode, DecodeEndpoint.WorkerStatusFact.accepted(fixture.reservation));
            fixture.registry.processDecodeStatus(
                    fixture.decode,
                    DecodeEndpoint.WorkerStatusFact.terminal(fixture.reservation, 0));
            assertEquals(
                    RequestState.Phase.COMPLETED,
                    fixture.registry.getRequestState("901", 0).state());
            assertEquals(0, fixture.registry.trackedRequestCount());
            verify(fixture.decode)
                    .release(
                            fixture.reservation, DecodeEndpoint.ReleaseReason.COUNTERPART_FINISHED);
            verifyNoInteractions(fixture.prefill);
        }
    }

    @Test
    void cancellationAfterDelegationKeepsUnknownDeliveryUntilInactivityExpiry() throws Exception {
        try (Fixture fixture = new Fixture()) {
            var committed = fixture.commit();
            fixture.registry.publishExternalPrefillRoute(committed.value());
            assertTrue(fixture.context.getFuture().get(2, TimeUnit.SECONDS).isSuccess());
            assertEquals(
                    RequestState.Phase.CANCEL_REQUESTED,
                    fixture.registry
                            .cancelRequest("901", 0, CancelReason.CLIENT_CANCELLED)
                            .state());
            verify(fixture.decode, never()).release(any(), any());
            var slot = fixture.registry.requestSlot("901");
            fixture.registry.expireInactiveRequest(slot, Long.MAX_VALUE);
            assertEquals(0, fixture.registry.trackedRequestCount());
            verify(fixture.decode)
                    .release(fixture.reservation, DecodeEndpoint.ReleaseReason.EXPIRED);
            verifyNoInteractions(fixture.prefill);
        }
    }

    @Test
    void cancellationBeforeCommitAcquiresNoDecodeResources() {
        try (Fixture fixture = new Fixture()) {
            fixture.registry.cancelRequest("901", 0, CancelReason.CLIENT_CANCELLED);
            assertEquals(PlacementResult.Status.CLOSED, fixture.commit().status());
            verify(fixture.decode, never())
                    .reserve(any(), anyString(), anyLong(), anyLong(), anyInt(), any());
            verify(fixture.decode, never()).acquireDispatchPermit(any(), any());
            verify(fixture.pin).close();
            assertEquals(0, fixture.registry.trackedRequestCount());
        }
    }

    @Test
    void unavailableDecodeCapacityRollsBackReservationWithoutPrefillWork() {
        try (Fixture fixture = new Fixture()) {
            when(fixture.decode.acquireDispatchPermit(any(), any()))
                    .thenReturn(
                            new DecodeEndpoint.EngineDispatchPermitAcquisition(
                                    DecodeEndpoint.EngineDispatchPermitAcquireStatus.CAPACITY_FULL,
                                    null));
            assertEquals(PlacementResult.Status.BLOCKED, fixture.commit().status());
            verify(fixture.decode)
                    .release(fixture.reservation, DecodeEndpoint.ReleaseReason.LOCAL_ROLLBACK);
            verify(fixture.pin).close();
            verifyNoInteractions(fixture.prefill, fixture.permit);
        }
    }

    @Test
    void externalAdmissionCannotEnterNativePrefillDelivery() {
        try (Fixture fixture = new Fixture();
                var route = fixture.router.selectForExternalPrefill(fixture.context).value();
                var admission = route.prepareAdmission(fixture.context)) {
            assertThrows(
                    IllegalStateException.class,
                    () -> admission.tryCommitDirectRoute(fixture.context, fixture.registry));
            assertThrows(
                    IllegalStateException.class,
                    () ->
                            admission.tryEnqueue(
                                    fixture.context,
                                    fixture.context.getFuture(),
                                    fixture.registry));
            verify(fixture.decode, never()).acquireDispatchPermit(any(), any());
            verifyNoInteractions(fixture.prefill);
        }
    }

    @Test
    void selectionSnapshotSurvivesContextChangesBeforeDelegatedCommit() {
        try (Fixture fixture = new Fixture();
                var route = fixture.router.selectForExternalPrefill(fixture.context).value()) {
            var snapshot = route.decodeRequest();
            fixture.context.getRequest().setSeqLen(4096L);
            fixture.context.getRequest().setMaxNewTokens(1024);
            fixture.context.setSchedulingMetadata(SchedulingMetadata.explicit(2, Long.MAX_VALUE));
            fixture.context
                    .getConfig()
                    .getRouter()
                    .getRoles()
                    .getDecode()
                    .getAvailability()
                    .setMaxEngineRequests(99L);
            try (var admission = route.prepareAdmission(fixture.context)) {
                var committed =
                        admission.tryCommitExternalPrefillRoute(fixture.context, fixture.registry);
                assertEquals(PlacementResult.Status.SUCCESS, committed.status());
                fixture.registry.publishExternalPrefillRoute(committed.value());
                var binding = fixture.registry.requestSlot("901").activeItem().decodeBinding();
                assertEquals(32, binding.inputKvTokens());
                assertEquals(48, binding.inputAndMaxOutputKvTokens());
                assertEquals(50, binding.priority());
                assertSame(snapshot.capacity(), binding.capacity());
                verify(fixture.decode)
                        .reserve(
                                eq(fixture.pin),
                                eq("901"),
                                eq(32L),
                                eq(48L),
                                eq(50),
                                eq(snapshot.capacity()));
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void schedulerPublishesDelegationWithoutPrefillSelectionOrQueueing(boolean direct)
            throws Exception {
        try (Fixture fixture = new Fixture(false)) {
            var navi = new NaviSchedPrefillSchedulerConfig();
            navi.setMasterBizName("master_biz");
            fixture.context.getConfig().setPrefillScheduler(navi);
            fixture.context.getRequest().setSkipPrefill(true);
            if (direct) {
                fixture.context
                        .getConfig()
                        .setScheduler(org.flexlb.config.SchedulerConfig.direct());
                SchedulingTestConfig.useNonBatchDispatcher(fixture.context.getConfig());
            }
            var configService = mock(ConfigService.class);
            when(configService.loadBalanceConfig()).thenReturn(fixture.context.getConfig());
            var endpoints = mock(org.flexlb.balance.endpoint.EndpointRegistry.class);
            when(endpoints.snapshotPrefillEndpoints()).thenReturn(java.util.Map.of());
            var scheduler =
                    new RequestScheduler(
                            configService,
                            fixture.router,
                            endpoints,
                            mock(BatchSchedulerReporter.class),
                            mock(org.flexlb.balance.eviction.EvictionManager.class),
                            fixture.registry,
                            new PlacementAvailability());
            var response = scheduler.submit(fixture.context).get(2, TimeUnit.SECONDS);
            assertTrue(response.isSuccess());
            assertFalse(response.isEnqueuedByMaster());
            assertEquals(
                    new org.flexlb.dao.loadbalance.Response.PrefillDelegation("master_biz"),
                    response.getPrefillDelegation());
            assertEquals(0, scheduler.getQueuedRequestCount());
            assertEquals(0, scheduler.getBlockedRequestCount());
            assertEquals(1, scheduler.getTrackedRequestCount());
            verify(fixture.permit).dispatch();
            verifyNoInteractions(fixture.prefill);
            scheduler.closePlacement();
        }
    }

    @Test
    void enabledDelegationRejectsRequestsThatStillExpectFlexlbPrefill() throws Exception {
        try (Fixture fixture = new Fixture(false)) {
            var scheduler = fixture.delegatedScheduler();
            fixture.context.getRequest().setSkipPrefill(false);
            try {
                var response = scheduler.submit(fixture.context).get(2, TimeUnit.SECONDS);
                assertFalse(response.isSuccess());
                assertEquals(
                        org.flexlb.dao.loadbalance.StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                        response.getCode());
                assertEquals(0, scheduler.getTrackedRequestCount());
                verifyNoInteractions(fixture.prefill);
                verify(fixture.selector, never()).select(any(), any(), any());
            } finally {
                scheduler.closePlacement();
            }
        }
    }

    @Test
    void disabledDelegationRejectsSkipPrefillBeforeAcquiringResources() throws Exception {
        try (Fixture fixture = new Fixture(false)) {
            var configService = mock(ConfigService.class);
            when(configService.loadBalanceConfig()).thenReturn(fixture.context.getConfig());
            var scheduler =
                    new RequestScheduler(
                            configService,
                            fixture.router,
                            mock(org.flexlb.balance.endpoint.EndpointRegistry.class),
                            mock(BatchSchedulerReporter.class),
                            mock(org.flexlb.balance.eviction.EvictionManager.class),
                            fixture.registry,
                            fixture.availability);
            fixture.context.getRequest().setSkipPrefill(true);
            try {
                var response = scheduler.submit(fixture.context).get(2, TimeUnit.SECONDS);
                assertFalse(response.isSuccess());
                assertEquals(
                        org.flexlb.dao.loadbalance.StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                        response.getCode());
                assertEquals(0, scheduler.getTrackedRequestCount());
                verifyNoInteractions(fixture.prefill);
                verify(fixture.selector, never()).select(any(), any(), any());
            } finally {
                scheduler.closePlacement();
            }
        }
    }

    @Test
    void delegatedPrefillNeverStartsAQueueThatCanAcceptWork() {
        var config = SchedulingTestConfig.batchConfig();
        config.setPrefillScheduler(new NaviSchedPrefillSchedulerConfig());
        var batcher =
                new WorkerBatcher(
                        "prefill",
                        mock(org.flexlb.balance.endpoint.PrefillEndpoint.class),
                        config,
                        mock(org.flexlb.balance.delivery.DeliveryStrategy.class),
                        mock(EndpointEventProjector.class));
        try {
            batcher.start();
            var failure =
                    assertThrows(
                            IllegalStateException.class,
                            () -> batcher.offer(mock(ScheduledRequest.class)));
            assertTrue(failure.getMessage().contains("NaviSched"));
        } finally {
            assertNull(batcher.stopAndAwait());
        }
    }

    @Test
    void delegatedQueueWaitsForDecodeAvailabilityInsteadOfRejectingTheRequest() throws Exception {
        try (Fixture fixture = new Fixture(false)) {
            when(fixture.selector.select(eq(fixture.context), any(), isNull()))
                    .thenReturn(PlacementResult.blocked(RoleType.DECODE));
            var scheduler = fixture.delegatedScheduler();
            try {
                var future = scheduler.submit(fixture.context);
                RequestLifecycleTestSupport.awaitCondition(
                        () -> scheduler.getBlockedRequestCount() == 1);
                assertFalse(future.isDone());
                when(fixture.selector.select(eq(fixture.context), any(), isNull()))
                        .thenReturn(PlacementResult.success(fixture.selected));
                fixture.availability.capacityChanged(RoleType.DECODE, "g1", "decode:8001");
                var response = future.get(2, TimeUnit.SECONDS);
                assertTrue(response.isSuccess());
                assertFalse(response.isEnqueuedByMaster());
                assertEquals("master_biz", response.getPrefillDelegation().masterBizName());
                verifyNoInteractions(fixture.prefill);
            } finally {
                scheduler.closePlacement();
            }
        }
    }

    @Test
    void cancellingDelegatedDecodeWaitDoesNotAcquirePrefillOrDecodeResources() throws Exception {
        try (Fixture fixture = new Fixture(false)) {
            when(fixture.selector.select(eq(fixture.context), any(), isNull()))
                    .thenReturn(PlacementResult.blocked(RoleType.DECODE));
            var scheduler = fixture.delegatedScheduler();
            try {
                var future = scheduler.submit(fixture.context);
                RequestLifecycleTestSupport.awaitCondition(
                        () -> scheduler.getBlockedRequestCount() == 1);
                scheduler.cancelRequest("901", 0, CancelReason.CLIENT_CANCELLED);
                assertFalse(future.get(2, TimeUnit.SECONDS).isSuccess());
                RequestLifecycleTestSupport.awaitCondition(
                        () ->
                                scheduler.getQueuedRequestCount() == 0
                                        && scheduler.getTrackedRequestCount() == 0);
                verify(fixture.decode, never()).acquireDispatchPermit(any(), any());
                verifyNoInteractions(fixture.prefill);
            } finally {
                scheduler.closePlacement();
            }
        }
    }

    @Test
    void delegationWithNoDecodeResourceFinishesOnlyTheFlexlbStage() throws Exception {
        try (Fixture fixture = new Fixture(false, true)) {
            var scheduler = fixture.delegatedScheduler();
            try {
                var response = scheduler.submit(fixture.context).get(2, TimeUnit.SECONDS);
                assertTrue(response.isSuccess());
                assertTrue(response.getServerStatus().isEmpty());
                assertFalse(response.isEnqueuedByMaster());
                assertEquals("master_biz", response.getPrefillDelegation().masterBizName());
                RequestLifecycleTestSupport.awaitCondition(
                        () -> scheduler.getTrackedRequestCount() == 0);
                assertEquals(
                        RequestState.Phase.COMPLETED, scheduler.getRequestState("901", 0).state());
                verify(fixture.decode, never()).acquireDispatchPermit(any(), any());
                verifyNoInteractions(fixture.prefill);
            } finally {
                scheduler.closePlacement();
            }
        }
    }

    private static final class Fixture implements AutoCloseable {
        final CostBasedPrefillStrategy prefill = mock(CostBasedPrefillStrategy.class);
        final DecodeEndpoint decode = mock(DecodeEndpoint.class);
        final WorkerEndpoint.GenerationPin pin = mock(WorkerEndpoint.GenerationPin.class);
        final DecodeEndpoint.ReservationHandle reservation =
                new DecodeEndpoint.ReservationHandle(1, "901", 1);
        final DecodeEndpoint.EngineDispatchPermit permit =
                mock(DecodeEndpoint.EngineDispatchPermit.class);
        final BalanceContext context;
        final DefaultRouter router;
        final DecodeSelector selector = mock(DecodeSelector.class);
        final SelectedRole selected = mock(SelectedRole.class);
        final PlacementAvailability availability = new PlacementAvailability();
        final RequestRegistry registry;

        Fixture() {
            this(true);
        }

        Fixture(boolean registerRequest) {
            this(registerRequest, false);
        }

        Fixture(boolean registerRequest, boolean fusionOnly) {
            context = new BalanceContext(SchedulingTestConfig.batchConfig());
            SchedulingTestConfig.usePriorityQueue(context.getConfig());
            Request request = new Request();
            request.setRequestId("901");
            request.setSeqLen(32L);
            request.setMaxNewTokens(16);
            context.setRequest(request);
            context.setSchedulingMetadata(
                    SchedulingMetadata.explicit(
                            50, System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(1)));
            ConfigService config = mock(ConfigService.class);
            when(config.loadBalanceConfig()).thenReturn(context.getConfig());
            registry =
                    new RequestRegistry(
                            config,
                            mock(BatchSchedulerReporter.class),
                            mock(RequestSchedulerReporter.class));
            if (registerRequest) {
                context.setFuture(registry.register(context));
            }
            var model = mock(ModelMetaConfig.class);
            when(model.requiredRoles())
                    .thenReturn(
                            fusionOnly
                                    ? List.of(RoleType.PDFUSION)
                                    : List.of(RoleType.PREFILL, RoleType.DECODE));
            var status = new ServerStatus();
            status.setRequestId("901");
            status.setRole(RoleType.DECODE);
            status.setServerIp("decode");
            status.setHttpPort(8001);
            status.setGroup("g1");
            when(selected.serverStatus()).thenReturn(status);
            when(selected.takeGenerationPin()).thenReturn(pin);
            when(pin.endpoint()).thenReturn(decode);
            when(decode.ipPort()).thenReturn(status.getLogicalIpPort());
            when(selector.select(eq(context), any(), isNull()))
                    .thenReturn(PlacementResult.success(selected));
            when(decode.reserve(eq(pin), eq("901"), eq(32L), eq(48L), eq(50)))
                    .thenReturn(reservation);
            when(decode.reserve(eq(pin), eq("901"), eq(32L), eq(48L), eq(50), any()))
                    .thenReturn(reservation);
            when(decode.acquireDispatchPermit(eq(reservation), any()))
                    .thenReturn(
                            new DecodeEndpoint.EngineDispatchPermitAcquisition(
                                    DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED,
                                    permit));
            when(permit.dispatch())
                    .thenReturn(DecodeEndpoint.EngineDispatchPermitTransferStatus.TRANSFERRED);
            router =
                    new DefaultRouter(
                            prefill,
                            selector,
                            mock(RandomStrategy.class),
                            mock(EncoderStrategy.class),
                            config,
                            model);
        }

        RequestScheduler delegatedScheduler() {
            var navi = new NaviSchedPrefillSchedulerConfig();
            navi.setMasterBizName("master_biz");
            context.getConfig().setPrefillScheduler(navi);
            context.getRequest().setSkipPrefill(true);
            var configService = mock(ConfigService.class);
            when(configService.loadBalanceConfig()).thenReturn(context.getConfig());
            var endpoints = mock(org.flexlb.balance.endpoint.EndpointRegistry.class);
            when(endpoints.snapshotPrefillEndpoints()).thenReturn(java.util.Map.of());
            return new RequestScheduler(
                    configService,
                    router,
                    endpoints,
                    mock(BatchSchedulerReporter.class),
                    mock(org.flexlb.balance.eviction.EvictionManager.class),
                    registry,
                    availability);
        }

        PlacementResult<RequestSlot.DeliveryClaim, PlacementKey> commit() {
            try (var route = router.selectForExternalPrefill(context).value();
                    var admission = route.prepareAdmission(context)) {
                return admission.tryCommitExternalPrefillRoute(context, registry);
            }
        }

        @Override
        public void close() {
            for (var slot : registry.snapshotSlots()) {
                registry.expireInactiveRequest(slot, Long.MAX_VALUE);
            }
            registry.closeAdmissionAndAwaitMutations();
            registry.closeOutstandingAndTerminalize();
            registry.closeExpiration();
            registry.closePublisher();
        }
    }
}
