package org.flexlb.balance.scheduler;

import org.flexlb.balance.RoutingResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.preemption.DecodePreemption;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.VitWorkerSelector;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.master.WorkerStatusTestSupport;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.TheoryCacheHitReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * DIRECT uses real admission ledgers and lifecycle while bypassing waiting queues.
 */
class DirectAdmissionContractTest {

    @Test
    void invalidVisibilityPredictionRollsBackBeforeRequestAcceptsDelivery() throws Exception {
        try (Fixture fixture = new Fixture()) {
            var context = spy(fixture.context(9114L));
            fixture.config.getRequestLifecycle().getDecision().setLifetime(Double.NaN);

            var response =
                    SchedulerTestServices.entry(fixture.scheduler)
                            .submit(context)
                            .get(2, TimeUnit.SECONDS);

            assertFalse(response.isSuccess());
            SchedulerTestSupport.awaitRequestWork(fixture.runtime);
            verify(context, never())
                    .recordDirectDeliveryStarted(
                            any(), any(), org.mockito.ArgumentMatchers.anyLong());
            verify(fixture.decode).release(any(), eq(DecodeResources.ReleaseReason.LOCAL_ROLLBACK));
            verify(fixture.decode).release(any(), any());
            assertNull(EndpointTestSupport.decodeReservation(fixture.decode, 9114L));
            fixture.assertNoPrefillOwnership();
            assertEquals(0, fixture.requests.repository.liveRequestCount());
        }
    }

    @Test
    void directRouteAndDeliveryBecomeVisibleInTheSameSnapshot() throws Exception {
        try (Fixture fixture = new Fixture()) {
            var context = spy(fixture.context(9113L));
            var future =
                    RequestProtocolTestSupport.submitForManualPlacement(fixture.scheduler, context);
            doAnswer(
                            call -> {
                                assertNull(context.snapshot().route());
                                assertNull(context.snapshot().delivery());
                                return call.callRealMethod();
                            })
                    .when(context)
                    .recordDeliveryStarted(any(), org.mockito.ArgumentMatchers.anyLong());

            var result =
                    SchedulerTestSupport.selectAndPublish(
                            fixture.scheduler, context, () -> fixture.router.select(context, null));

            assertEquals(RoutingResult.Status.SUCCESS, result.status());
            assertTrue(future.get(2, TimeUnit.SECONDS).isSuccess());
            assertSame(result.value(), context.snapshot().route());
            assertSame(result.value(), context.snapshot().delivery().identity().route());
            assertNull(context.snapshot().routingContext());
            verify(context, never()).publishRoute(any(), any());
            verify(context).recordDeliveryStarted(any(), org.mockito.ArgumentMatchers.anyLong());
        }
    }

    @Test
    void failedSelectionEndsTheAttemptBeforeCallerRetries() throws Exception {
        try (Fixture fixture = new Fixture()) {
            var context = fixture.context(9111L);
            var future =
                    RequestProtocolTestSupport.submitForManualPlacement(fixture.scheduler, context);
            var failure = new IllegalStateException("selection failed");
            assertSame(
                    failure,
                    assertThrows(
                            IllegalStateException.class,
                            () ->
                                    SchedulerTestSupport.selectAndPublish(
                                            fixture.scheduler,
                                            context,
                                            () -> {
                                                throw failure;
                                            })));
            assertNull(context.snapshot().routingContext());
            assertNull(context.snapshot().route());
            assertFalse(future.isDone());
            var retry =
                    SchedulerTestSupport.selectAndPublish(
                            fixture.scheduler, context, RoutingResult::retry);
            assertEquals(RoutingResult.Status.RETRY, retry.status());
            assertNull(context.snapshot().routingContext());
            fixture.scheduler.cancel(context, 0L, CancelReason.CLIENT_CANCELLED);
            assertFalse(future.get(2, TimeUnit.SECONDS).isSuccess());
            SchedulerTestSupport.awaitRequestWork(fixture.runtime);
            assertEquals(0, fixture.requests.repository.liveRequestCount());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void cancellationBeforeSelectionReturnsReleasesBothReservations(boolean queued)
            throws Exception {
        try (Fixture fixture = new Fixture(16, 1, queued)) {
            var context = fixture.context(9112L);
            var future =
                    RequestProtocolTestSupport.submitForManualPlacement(fixture.scheduler, context);
            var result =
                    SchedulerTestSupport.selectAndPublish(
                            fixture.scheduler,
                            context,
                            () -> {
                                var selected = fixture.router.select(context, null);
                                assertEquals(RoutingResult.Status.SUCCESS, selected.status());
                                assertNotNull(
                                        EndpointTestSupport.decodeReservation(
                                                fixture.decode, 9112L));
                                fixture.scheduler.cancel(
                                        context, 0L, CancelReason.CLIENT_CANCELLED);
                                return selected;
                            });
            assertEquals(RoutingResult.Status.CLOSED, result.status());
            assertFalse(future.get(2, TimeUnit.SECONDS).isSuccess());
            SchedulerTestSupport.awaitRequestWork(fixture.runtime);
            assertNull(context.snapshot().routingContext());
            assertNull(EndpointTestSupport.decodeReservation(fixture.decode, 9112L));
            fixture.assertNoPrefillOwnership();
            assertEquals(0, fixture.requests.repository.liveRequestCount());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void lostPrefillRegistrationReportsPublicationFailureAndRollsBackDecode(boolean queued)
            throws Exception {
        try (Fixture fixture = new Fixture(16, 1, queued)) {
            var context = fixture.context(9100L);
            var future =
                    RequestProtocolTestSupport.submitForManualPlacement(fixture.scheduler, context);
            var routing = SchedulerTestSupport.claim(fixture.scheduler, "9100", future);
            assertNotNull(routing);
            var selected = fixture.router.select(context, null);
            assertEquals(RoutingResult.Status.SUCCESS, selected.status());
            var route = selected.value();
            assertNotNull(EndpointTestSupport.decodeReservation(fixture.decode, 9100L));
            assertTrue(fixture.prefill.releaseRequest(route));
            assertTrue(fixture.prefill.acceptsNewRequests());
            var publication =
                    SchedulerTestSupport.selectAndPublish(
                            fixture.scheduler, routing, () -> RoutingResult.success(route));

            assertEquals(
                    queued ? RoutingResult.Status.BLOCKED : RoutingResult.Status.REJECTED,
                    publication.status());
            assertEquals(
                    StrategyErrorType.DISPATCH_FAILED.getErrorCode(),
                    publication.failure().getCode());
            assertEquals(
                    AdmissionRejectReason.UNSPECIFIED,
                    publication.failure().getAdmissionRejectReason());
            String message = publication.failure().getErrorMessage();
            assertTrue(
                    message.contains(
                            queued
                                    ? "Prefill queue rejected publication"
                                    : "Selected Prefill registration was not valid for commit"));
            assertFalse(message.contains("capacity exhausted"));
            assertFalse(message.contains("priority attribution"));
            assertNull(context.snapshot().route());
            assertNull(context.snapshot().routingContext());
            assertNull(EndpointTestSupport.decodeReservation(fixture.decode, 9100L));
            assertEquals(0, fixture.decode.resourceSnapshot().activeDispatchPermits());
            assertEquals(0, fixture.decode.routingView().engineCapacityUsed());
            assertEquals(0, fixture.prefill.queueDepth());
            assertEquals(
                    0L,
                    fixture.prefill
                            .admissionSummary(0, fixture.prefill.requestLimit)
                            .occupiedRequests());
            assertEquals(0, fixture.prefill.stats().batchCount());
            assertFalse(future.isDone(), "the scheduler still owns the blocked/rejected decision");
            fixture.scheduler.cancel(context, 0L, CancelReason.CLIENT_CANCELLED);
            assertFalse(future.get(2, TimeUnit.SECONDS).isSuccess());
            SchedulerTestSupport.awaitRequestWork(fixture.runtime);
            fixture.assertNoPrefillOwnership();
            assertEquals(0, fixture.requests.repository.liveRequestCount());
        }
    }

    @Test
    void failedUnpublishedRollbackRemainsReachableUntilResourcesAreReleased() throws Exception {
        try (Fixture fixture = new Fixture()) {
            var context = fixture.context(9099L);
            var future =
                    RequestProtocolTestSupport.submitForManualPlacement(fixture.scheduler, context);
            var operation = SchedulerTestSupport.claim(fixture.scheduler, "9099", future);
            assertNotNull(operation);
            var route = fixture.router.select(context, null).value();
            assertNotNull(route);
            var failure = new IllegalStateException("temporary cleanup failure");
            org.mockito.Mockito.doThrow(failure)
                    .doCallRealMethod()
                    .when(fixture.decode)
                    .release(
                            route.decodeReservation(),
                            DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
            assertSame(failure, AbstractRequestScheduler.finishUnpublishedRoute(route, operation));
            assertFalse(future.get(2, TimeUnit.SECONDS).isSuccess());
            SchedulerTestSupport.awaitRequestWork(fixture.runtime);
            assertNull(EndpointTestSupport.decodeReservation(fixture.decode, 9099L));
            fixture.assertNoPrefillOwnership();
            assertEquals(0, fixture.requests.repository.liveRequestCount());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void vitRetirementDuringAdmissionReplansTheWholeRoute(boolean queued) throws Exception {
        try (Fixture fixture = new Fixture(16, 1, queued, true)) {
            var first = new WorkerEndpoint(Fixture.worker(RoleType.VIT, "vit-first"));
            var replacement = new WorkerEndpoint(Fixture.worker(RoleType.VIT, "vit-replacement"));
            var current = new AtomicReference<>(first);
            when(fixture.endpoints.endpointAddressSnapshot(RoleType.VIT))
                    .thenAnswer(call -> List.of(current.get().ipPort()));
            doAnswer(call -> current.get())
                    .when(fixture.endpoints)
                    .selectEndpoint(eq(RoleType.VIT), any());
            doAnswer(
                            call -> {
                                if (current.get() == first) {
                                    first.close();
                                    current.set(replacement);
                                }
                                return call.callRealMethod();
                            })
                    .when(fixture.requests)
                    .currentRoutingContext(any());
            try {
                var context = fixture.context(9016L);
                var response =
                        SchedulerTestServices.entry(fixture.scheduler)
                                .submit(context)
                                .get(3L, TimeUnit.SECONDS);
                SchedulerTestSupport.awaitRequestWork(fixture.runtime);
                assertTrue(response.isSuccess(), response.getErrorMessage());
                assertEquals(
                        "vit-replacement",
                        response.getServerStatus().stream()
                                .filter(status -> status.getRole() == RoleType.VIT)
                                .findFirst()
                                .orElseThrow()
                                .getServerIp());
                assertEquals(1L, fixture.prefill.admissionSummary(0, 0).occupiedRequests());
                assertEquals(1, fixture.decode.routingView().engineCapacityUsed());
                assertEquals(
                        48L,
                        EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()));
                verify(fixture.decode)
                        .release(any(), eq(DecodeResources.ReleaseReason.LOCAL_ROLLBACK));
                verify(fixture.requests, times(2)).currentRoutingContext(any());
            } finally {
                first.close();
                replacement.close();
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void retirementAfterSelectionRetriesOnceAfterReleasingDecode(boolean replacementAlsoRetires)
            throws Exception {
        try (Fixture fixture = new Fixture()) {
            var attempts = new java.util.concurrent.atomic.AtomicInteger();
            var replacement = new AtomicReference<PrefillEndpoint>();
            doAnswer(
                            call -> {
                                var route = call.getArgument(0, RouteResult.class);
                                assertNotNull(route.decodeReservation());
                                assertEquals(
                                        1,
                                        fixture.decode.resourceSnapshot().activeDispatchPermits());
                                // Selection and Decode preparation completed, but Prefill admission
                                // has not started. Retirement wins this ordering deterministically.
                                if (attempts.getAndIncrement() == 0) {
                                    assertSame(fixture.prefill, route.prefillEndpoint());
                                    fixture.prefill.close();
                                    replacement.set(fixture.installReplacementPrefill());
                                } else {
                                    assertSame(replacement.get(), route.prefillEndpoint());
                                    if (replacementAlsoRetires) replacement.get().close();
                                }
                                return call.callRealMethod();
                            })
                    .when(fixture.requests)
                    .currentRoutingContext(any());

            var context = fixture.context(9011L);
            var response =
                    SchedulerTestServices.entry(fixture.scheduler)
                            .submit(context)
                            .get(2L, TimeUnit.SECONDS);
            SchedulerTestSupport.awaitRequestWork(fixture.runtime);

            assertEquals(!replacementAlsoRetires, response.isSuccess());
            assertEquals(2, attempts.get());
            fixture.assertNoPrefillOwnership();
            if (replacementAlsoRetires) {
                assertEquals(RoleType.PREFILL.getErrorType().getErrorCode(), response.getCode());
                assertNull(context.snapshot().delivery());
                assertNull(EndpointTestSupport.decodeReservation(fixture.decode, 9011L));
                assertEquals(0, fixture.decode.resourceSnapshot().activeDispatchPermits());
                assertEquals(
                        0, SchedulerTestSupport.repository(fixture.requests).liveRequestCount());
            } else {
                assertNotNull(context.snapshot().delivery());
                assertSame(replacement.get(), context.snapshot().route().prefillEndpoint());
                assertEquals(1, replacement.get().stats().locallyOwnedRequests());
            }
            assertEquals(
                    replacementAlsoRetires ? 0 : 1,
                    fixture.decode.routingView().engineCapacityUsed());
            assertEquals(
                    replacementAlsoRetires ? 0L : 48L,
                    EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()));
            verify(fixture.decode, times(replacementAlsoRetires ? 2 : 1))
                    .release(any(), eq(DecodeResources.ReleaseReason.LOCAL_ROLLBACK));
            verify(fixture.requests, times(2)).currentRoutingContext(any());
            verify(fixture.prefillSelector, times(2))
                    .selectAndRegister(
                            any(),
                            any(),
                            eq(RoleType.PREFILL),
                            any(),
                            org.mockito.ArgumentMatchers.anyList(),
                            org.mockito.ArgumentMatchers.any(java.util.function.LongSupplier.class),
                            org.mockito.ArgumentMatchers.any());
        }
    }

    @ParameterizedTest
    @CsvSource({"false,false", "false,true", "true,true"})
    void decodeRetirementDuringAdmissionReselectsBeforeReturningRoute(
            boolean queued, boolean afterBinding) throws Exception {
        try (Fixture fixture = new Fixture(16, 1, queued)) {
            var worker = Fixture.worker(RoleType.DECODE, "127.0.0.4");
            var replacement =
                    EndpointTestSupport.decode(
                            worker, SchedulerTestSupport.repository(fixture.requests));
            try {
                fixture.applyWorkerStatus(replacement, Map.of(), Map.of());
                org.mockito.stubbing.Answer<Object> retireDuringAdmission =
                        call -> {
                            Object bound = afterBinding ? call.callRealMethod() : null;
                            fixture.decode.close();
                            org.mockito.Mockito.doAnswer(
                                            selection ->
                                                    SelectionTestSupport.registerResult(
                                                            selection,
                                                            RoutingResult.success(
                                                                    replacement.routingView())))
                                    .when(fixture.decodeSelector)
                                    .selectAndRegister(
                                            any(),
                                            any(),
                                            any(),
                                            org.mockito.ArgumentMatchers.anyList(),
                                            any(),
                                            org.mockito.ArgumentMatchers.any(),
                                            org.mockito.ArgumentMatchers.any());
                            doAnswer(selection -> replacement)
                                    .when(fixture.endpoints)
                                    .selectDecodeEndpoint(any());
                            doAnswer(
                                            snapshot -> {
                                                snapshot.getArgument(1, List.class)
                                                        .add(replacement.routingView());
                                                return null;
                                            })
                                    .when(fixture.endpoints)
                                    .decodeRoutingSnapshot(any(), any());
                            return afterBinding ? bound : call.callRealMethod();
                        };
                if (afterBinding)
                    doAnswer(retireDuringAdmission)
                            .when(fixture.requests)
                            .currentRoutingContext(any());
                else
                    doAnswer(retireDuringAdmission)
                            .when(fixture.decode)
                            .acquireDispatchPermit(any(), any());

                var context = fixture.context(9015L);
                var response =
                        SchedulerTestServices.entry(fixture.scheduler)
                                .submit(context)
                                .get(2L, TimeUnit.SECONDS);
                SchedulerTestSupport.awaitRequestWork(fixture.runtime);

                assertTrue(response.isSuccess());
                assertSame(replacement, context.snapshot().route().decodeEndpoint());
                assertEquals(0, fixture.decode.routingView().engineCapacityUsed());
                assertEquals(0, fixture.decode.resourceSnapshot().activeDispatchPermits());
                assertEquals(1, replacement.routingView().engineCapacityUsed());
                assertEquals(
                        48L,
                        EndpointTestSupport.expectedReservedKv(replacement.resourceSnapshot()));
                assertEquals(1L, fixture.prefill.admissionSummary(0, 0).occupiedRequests());
                verify(fixture.prefillSelector, times(2))
                        .selectAndRegister(
                                any(),
                                any(),
                                eq(RoleType.PREFILL),
                                any(),
                                org.mockito.ArgumentMatchers.anyList(),
                                org.mockito.ArgumentMatchers.any(
                                        java.util.function.LongSupplier.class),
                                org.mockito.ArgumentMatchers.any());
            } finally {
                replacement.close();
            }
        }
    }

    @Test
    void cancellationAfterRollbackPreventsEndpointReselection() throws Exception {
        try (Fixture fixture = new Fixture()) {
            var context = fixture.context(9014L);
            var admissions = new java.util.concurrent.atomic.AtomicInteger();
            doAnswer(
                            call -> {
                                fixture.prefill.close();
                                fixture.installReplacementPrefill();
                                return call.callRealMethod();
                            })
                    .when(fixture.requests)
                    .currentRoutingContext(any());
            doAnswer(
                            call -> {
                                if (admissions.getAndIncrement() == 1) {
                                    assertNull(context.snapshot().route());
                                    assertEquals(
                                            0, fixture.decode.routingView().engineCapacityUsed());
                                    fixture.requests.cancel(
                                            "9014", 0L, CancelReason.CLIENT_CANCELLED);
                                }
                                return call.callRealMethod();
                            })
                    .when(fixture.requests)
                    .tryBeginRouteSelection(context);

            var response =
                    SchedulerTestServices.entry(fixture.scheduler)
                            .submit(context)
                            .get(2L, TimeUnit.SECONDS);
            SchedulerTestSupport.awaitRequestWork(fixture.runtime);

            assertFalse(response.isSuccess());
            assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), response.getCode());
            assertEquals(2, admissions.get());
            assertNull(context.snapshot().delivery());
            fixture.assertNoPrefillOwnership();
            assertEquals(0, fixture.decode.resourceSnapshot().activeDispatchPermits());
            assertEquals(0, SchedulerTestSupport.repository(fixture.requests).liveRequestCount());
            verify(fixture.prefillSelector)
                    .selectAndRegister(
                            any(),
                            any(),
                            eq(RoleType.PREFILL),
                            any(),
                            org.mockito.ArgumentMatchers.anyList(),
                            org.mockito.ArgumentMatchers.any(java.util.function.LongSupplier.class),
                            org.mockito.ArgumentMatchers.any());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void retirementAfterPrefillReservationRollsBackBothEndpointsBeforeRetry(
            boolean drainBeforeCommit) throws Exception {
        try (Fixture fixture = new Fixture()) {
            var context = spy(fixture.context(9013L));
            var retirement = new AtomicReference<EndpointRegistry.Retirement>();
            var replacement = new AtomicReference<PrefillEndpoint>();
            doAnswer(
                            call -> {
                                var operation = call.callRealMethod();
                                if (retirement.get() != null) return operation;
                                assertNull(
                                        context.snapshot().route(),
                                        "a prepared route is still local to selection");
                                assertEquals(
                                        1L,
                                        fixture.prefill.admissionSummary(0, 0).occupiedRequests());
                                assertEquals(
                                        1,
                                        fixture.decode.resourceSnapshot().activeDispatchPermits());
                                // Close admission without draining existing records: commit must
                                // reject, and request rollback must release both preparations.
                                var worker = fixture.prefill.getStatus();
                                worker.lock.lock();
                                try {
                                    retirement.set(
                                            fixture.endpoints.beginRetirement(
                                                    RoleType.PREFILL,
                                                    fixture.prefill.ipPort(),
                                                    worker));
                                    assertNotNull(retirement.get());
                                } finally {
                                    worker.lock.unlock();
                                }
                                if (drainBeforeCommit) fixture.prefill.close();
                                replacement.set(fixture.installReplacementPrefill());
                                return operation;
                            })
                    .when(fixture.scheduler)
                    .currentRoutingContext(any());
            try {
                var response =
                        SchedulerTestServices.entry(fixture.scheduler)
                                .submit(context)
                                .get(2L, TimeUnit.SECONDS);
                SchedulerTestSupport.awaitRequestWork(fixture.runtime);

                assertTrue(response.isSuccess(), response::getErrorMessage);
                assertNotNull(context.snapshot().delivery());
                assertSame(replacement.get(), context.snapshot().route().prefillEndpoint());
                fixture.assertNoPrefillOwnership();
                assertNotNull(EndpointTestSupport.decodeReservation(fixture.decode, 9013L));
                assertEquals(1, fixture.decode.routingView().engineCapacityUsed());
                assertEquals(
                        48L,
                        EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()));
                assertEquals(
                        1, SchedulerTestSupport.repository(fixture.requests).liveRequestCount());
                verify(fixture.decode)
                        .release(any(), eq(DecodeResources.ReleaseReason.LOCAL_ROLLBACK));
                verify(context, never()).publishRoute(any(), any());
                verify(context)
                        .recordDirectDeliveryStarted(
                                any(), any(), org.mockito.ArgumentMatchers.anyLong());
                verify(fixture.requests, times(2)).currentRoutingContext(any());
            } finally {
                if (retirement.get() != null) {
                    EndpointTestSupport.completeRetirement(
                            retirement.get(),
                            mock(org.flexlb.cache.match.CacheAwareService.class),
                            org.slf4j.LoggerFactory.getLogger(getClass()));
                }
            }
        }
    }

    @Test
    void retirementBeforeQueuePublicationReleasesDecodeAndReplansOnReplacementEndpoint()
            throws Exception {
        try (Fixture fixture = new Fixture(16, 1, true)) {
            var attempts = new java.util.concurrent.atomic.AtomicInteger();
            var replacement = new AtomicReference<PrefillEndpoint>();
            doAnswer(
                            call -> {
                                var route = call.getArgument(0, RouteResult.class);
                                if (attempts.getAndIncrement() == 0) {
                                    assertSame(fixture.prefill, route.prefillEndpoint());
                                    assertNotNull(route.decodeReservation());
                                    fixture.prefill.close();
                                    var worker = Fixture.worker(RoleType.PREFILL, "127.0.0.3");
                                    worker.lock.lock();
                                    try {
                                        replacement.set(
                                                (PrefillEndpoint)
                                                        fixture.endpoints.publishPreparedEndpoint(
                                                                worker.getLogicalIpPort(),
                                                                worker,
                                                                worker.prepareNewStatus(
                                                                        WorkerStatusTestSupport
                                                                                .freezeStatusResponse(
                                                                                        worker,
                                                                                        Fixture
                                                                                                .status(
                                                                                                        worker,
                                                                                                        Map
                                                                                                                .of(),
                                                                                                        Map
                                                                                                                .of())))));
                                    } finally {
                                        worker.lock.unlock();
                                    }
                                    fixture.bindPrefill(replacement.get());
                                    org.mockito.Mockito.doAnswer(
                                                    selection ->
                                                            SelectionTestSupport.registerResult(
                                                                    selection,
                                                                    fixture.prefillSelection(
                                                                            replacement.get())))
                                            .when(fixture.prefillSelector)
                                            .selectAndRegister(
                                                    any(),
                                                    any(),
                                                    eq(RoleType.PREFILL),
                                                    any(),
                                                    org.mockito.ArgumentMatchers.anyList(),
                                                    org.mockito.ArgumentMatchers.any(
                                                            java.util.function.LongSupplier.class),
                                                    org.mockito.ArgumentMatchers.any());
                                } else {
                                    assertSame(replacement.get(), route.prefillEndpoint());
                                    // Decode capacity is one: the retry can reach here only after
                                    // rolling back the failed publication's reservation.
                                    assertEquals(
                                            1, fixture.decode.resourceSnapshot().reservedCount());
                                }
                                return call.callRealMethod();
                            })
                    .when(fixture.requests)
                    .currentRoutingContext(any());

            var context = fixture.context(9012L);
            var response =
                    SchedulerTestServices.entry(fixture.scheduler)
                            .submit(context)
                            .get(3L, TimeUnit.SECONDS);
            SchedulerTestSupport.awaitRequestWork(fixture.runtime);

            assertTrue(response.isSuccess());
            assertEquals(2, attempts.get());
            assertSame(replacement.get(), context.snapshot().route().prefillEndpoint());
            assertNotNull(context.snapshot().delivery());
            fixture.assertNoPrefillOwnership();
            assertEquals(1, replacement.get().stats().locallyOwnedRequests());
            assertEquals(1, fixture.decode.routingView().engineCapacityUsed());
            assertEquals(
                    48L, EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()));
            assertEquals(0, fixture.decode.resourceSnapshot().queuedCount());
            verify(fixture.decode).release(any(), eq(DecodeResources.ReleaseReason.LOCAL_ROLLBACK));
        }
    }

    @Test
    void schedulerRetriesOneGenerationHandoffFailureBeforeCommitting() throws Exception {
        try (Fixture fixture = new Fixture()) {
            var captures = new java.util.concurrent.atomic.AtomicInteger();
            doAnswer(call -> captures.getAndIncrement() == 0 ? null : fixture.decode)
                    .when(fixture.endpoints)
                    .selectDecodeEndpoint(any());
            var response =
                    SchedulerTestServices.entry(fixture.scheduler)
                            .submit(fixture.context(9010L))
                            .get(2L, TimeUnit.SECONDS);
            assertTrue(response.isSuccess());
            assertEquals(2, captures.get());
            assertEquals(1L, fixture.prefill.admissionSummary(0, 0).occupiedRequests());
            assertEquals(1, fixture.decode.routingView().engineCapacityUsed());
            fixture.assertNoWaitingQueue();
        }
    }

    @Test
    void completeSubmissionRejectsAnotherAdmissionBeforeResourcePreparation() throws Exception {
        try (Fixture fixture = new Fixture()) {
            var context = fixture.context(9001L);
            var future =
                    new ResponseCompletionExecutor.RequestFuture((kind, result, failure) -> false);
            assertEquals(
                    RequestRepository.RegistrationResult.REGISTERED,
                    fixture.requests.repository.register(
                            context,
                            fixture.requests,
                            future,
                            60_000L,
                            fixture.requests.schedulingTimeoutError()));
            var admission = SchedulerTestSupport.claim(fixture.requests, "9001", future);
            var foreign = new RoutingContext(context, null);
            var route = mock(RouteResult.class);
            when(route.ctx()).thenReturn(context);
            try {
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                ((DirectRequestScheduler) fixture.requests)
                                        .selectAndPublish(
                                                foreign, () -> RoutingResult.success(route)));
                assertSame(admission, context.snapshot().routingContext());
                assertEquals(
                        1,
                        org.flexlb.balance.scheduler.SchedulerTestServices
                                .pendingDeliveryRequestCount(fixture.requests.repository));
                assertEquals(0L, fixture.prefill.queueDepth());
                assertEquals(
                        0L,
                        fixture.prefill
                                .admissionSummary(0, fixture.prefill.requestLimit)
                                .occupiedRequests());
                assertEquals(0, fixture.prefill.stats().batchCount());
                assertEquals(0, fixture.decode.routingView().engineCapacityUsed());
            } finally {
                AbstractRequestScheduler.finishRouting(admission);
                fixture.requests.cancel("9001", 0L, CancelReason.CLIENT_CANCELLED);
            }
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2})
    void concurrentDirectSelectionsRegisterWithinCurrentCapacity(int prefillCapacity)
            throws Exception {
        try (Fixture fixture = new Fixture(prefillCapacity, 2);
                var executor = Executors.newFixedThreadPool(2)) {
            var service = SchedulingTestConfig.configService(fixture.config);
            var cache = mock(org.flexlb.cache.match.CacheAwareService.class);
            when(cache.findMatchingEngines(any()))
                    .thenReturn(
                            org.flexlb.cache.domain.CacheMatchResult.empty(
                                    org.flexlb.cache.domain.CacheMatchSource.LOCAL_SYNC));
            org.springframework.test.util.ReflectionTestUtils.setField(
                    fixture.router,
                    "prefillSelector",
                    new CostBasedPrefillStrategy(cache, service));
            org.springframework.test.util.ReflectionTestUtils.setField(
                    fixture.router,
                    "decodeSelector",
                    new DecodeSelector(service, SchedulingTestConfig.capacity(fixture.config)));
            var first =
                    executor.submit(
                            () ->
                                    SchedulerTestServices.entry(fixture.scheduler)
                                            .submit(fixture.context(201L))
                                            .get(2L, TimeUnit.SECONDS));
            var second =
                    executor.submit(
                            () ->
                                    SchedulerTestServices.entry(fixture.scheduler)
                                            .submit(fixture.context(202L))
                                            .get(2L, TimeUnit.SECONDS));
            var responses =
                    List.of(first.get(3L, TimeUnit.SECONDS), second.get(3L, TimeUnit.SECONDS));

            assertEquals(prefillCapacity, responses.stream().filter(Response::isSuccess).count());
            for (var response : responses) {
                if (!response.isSuccess()) {
                    assertEquals(
                            StrategyErrorType.PRIORITY_ADMISSION_REJECTED.getErrorCode(),
                            response.getCode());
                    assertEquals(
                            org.flexlb.dao.loadbalance.AdmissionRejectReason.SAME_PRIORITY_AHEAD,
                            response.getAdmissionRejectReason());
                }
            }
            assertEquals(
                    prefillCapacity,
                    fixture.prefill
                            .admissionSummary(0, fixture.prefill.requestLimit)
                            .occupiedRequests());
            assertEquals(prefillCapacity, fixture.decode.routingView().engineCapacityUsed());
            assertEquals(
                    48L * prefillCapacity,
                    EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()));
            fixture.assertNoWaitingQueue();
            verify(fixture.requests, times(prefillCapacity)).currentRoutingContext(any());
        }
    }

    @Test
    void directAdmissionDoesNotRetireSelectedEndpoint() {
        try (Fixture fixture = new Fixture()) {
            var selectedEndpoint = fixture.prefill;
            var context = fixture.context(100L);
            var response =
                    org.junit.jupiter.api.Assertions.assertDoesNotThrow(
                            () ->
                                    SchedulerTestServices.entry(fixture.scheduler)
                                            .submit(context)
                                            .get(2, TimeUnit.SECONDS));
            assertTrue(response.isSuccess());
            assertNotNull(context.snapshot().route().prefillReservation());
            assertEquals(1, fixture.prefill.stats().locallyOwnedRequests());
            assertEquals(1, fixture.decode.routingView().engineCapacityUsed());
            fixture.assertNoWaitingQueue();
            org.junit.jupiter.api.Assertions.assertDoesNotThrow(
                    () -> SchedulerTestServices.throwIfFailed(fixture.requests.taskPool));
            assertFalse(selectedEndpoint.isRetired());
        }
    }

    @Test
    void directRouteOwnsPrefillAndDecodeUntilEngineTerminalWithoutEnteringWaitingQueues()
            throws Exception {
        try (Fixture fixture = new Fixture()) {
            Response response =
                    SchedulerTestServices.entry(fixture.scheduler)
                            .submit(fixture.context(101L))
                            .get(2L, TimeUnit.SECONDS);

            assertTrue(response.isSuccess());
            assertFalse(response.isEnqueuedByMaster());
            fixture.assertNoWaitingQueue();
            assertEquals(
                    1,
                    fixture.prefill
                            .admissionSummary(0, fixture.prefill.requestLimit)
                            .occupiedRequests());
            assertEquals(0, fixture.prefill.stats().batchCount());
            assertEquals(1, fixture.decode.routingView().engineCapacityUsed());
            assertEquals(
                    48L, EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()));
            assertEquals(0, fixture.decode.resourceSnapshot().queuedCount());
            assertEquals(1, SchedulerTestSupport.repository(fixture.requests).liveRequestCount());
            assertEquals(
                    RequestState.Phase.ACKNOWLEDGED,
                    SchedulerTestSupport.repository(fixture.requests)
                            .getRequestState("101", 0L)
                            .state());
            var reservation = EndpointTestSupport.decodeReservation(fixture.decode, 101L);
            assertNotNull(reservation);
            assertThrows(
                    IllegalStateException.class,
                    () ->
                            fixture.decode
                                    .release(
                                            reservation,
                                            DecodeResources.ReleaseReason.LOCAL_ROLLBACK)
                                    .status());

            fixture.applyWorkerStatus(
                    fixture.prefill, Map.of(), Map.of("101", task(101L, TaskPhase.RUNNING)));
            assertEquals(
                    0L,
                    fixture.prefill
                            .admissionSummary(0, fixture.prefill.requestLimit)
                            .occupiedRequests());
            assertEquals(
                    48L,
                    EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()),
                    "Prefill completion must retain Decode ownership until its own observation");
            fixture.applyWorkerStatus(
                    fixture.decode, Map.of("101", task(101L, TaskPhase.RUNNING)), Map.of());
            assertTrue(fixture.decode.ownership(reservation).confirmed());
            fixture.applyWorkerStatus(
                    fixture.decode, Map.of(), Map.of("101", task(101L, TaskPhase.RUNNING)));
            assertEquals(0, fixture.decode.routingView().engineCapacityUsed());
            assertEquals(0, SchedulerTestSupport.repository(fixture.requests).liveRequestCount());
            fixture.assertNoWaitingQueue();
        }
    }

    @Test
    void fullDecodeRejectsImmediatelyAndReleasesUncommittedResources() throws Exception {
        try (Fixture fixture = new Fixture()) {
            DecodeResources.RequestReservation occupant;
            {
                var selectedEndpoint = fixture.decode;

                assertNotNull(selectedEndpoint);
                occupant =
                        org.flexlb.balance.scheduler.SelectionTestSupport.registerQueuedDecode(
                                        fixture.decode, String.valueOf(999L), 32L, 48L, 50)
                                .value();
            }
            assertNotNull(occupant);
            var acquired =
                    fixture.decode.acquireDispatchPermit(
                            occupant, new DecodeResources.AdmissionCapacity(1L, 90L));
            assertEquals(
                    DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED, acquired.status());
            assertEquals(
                    DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED,
                    acquired.value().dispatch().status());

            Response response =
                    assertTimeoutPreemptively(
                            Duration.ofSeconds(2),
                            () ->
                                    SchedulerTestServices.entry(fixture.scheduler)
                                            .submit(fixture.context(102L))
                                            .get(2L, TimeUnit.SECONDS));

            assertFalse(response.isSuccess());
            assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), response.getCode());
            fixture.assertNoPrefillOwnership();
            assertNull(EndpointTestSupport.decodeReservation(fixture.decode, 102L));
            assertEquals(occupant, EndpointTestSupport.decodeReservation(fixture.decode, 999L));
            assertEquals(1, fixture.decode.routingView().engineCapacityUsed());
            assertEquals(
                    48L, EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()));
            assertEquals(0, fixture.decode.resourceSnapshot().queuedCount());
            assertEquals(0, SchedulerTestSupport.repository(fixture.requests).liveRequestCount());
        }
    }

    @Test
    void decodeTerminalBetweenAcceptedPreparationAndContextBindingCannotPublishASuccessfulRoute()
            throws Exception {
        try (Fixture fixture = new Fixture()) {
            AtomicBoolean raced = new AtomicBoolean();
            doAnswer(
                            call -> {
                                DecodeResources.RequestReservation reservation =
                                        call.getArgument(0);
                                assertEquals("103", reservation.requestId());
                                fixture.assertItemNotBound(103L);
                                fixture.applyWorkerStatus(
                                        fixture.decode,
                                        Map.of("103", task(103L, TaskPhase.KV_ALLOCATED)),
                                        Map.of());
                                var acquired =
                                        (org.flexlb.balance.endpoint.DecodeResources.Acquisition<
                                                        org.flexlb.balance.endpoint.DecodeResources
                                                                .EngineDispatchPermitAcquireStatus,
                                                        DecodeEndpoint.EngineDispatchPermit>)
                                                call.callRealMethod();
                                assertEquals(
                                        DecodeResources.EngineDispatchPermitAcquireStatus
                                                .ALREADY_ACCEPTED,
                                        acquired.status());
                                assertNotNull(
                                        acquired.value(),
                                        "accepted preparation must retain an exact handoff capability");
                                fixture.applyWorkerStatus(
                                        fixture.decode,
                                        Map.of(),
                                        Map.of("103", task(103L, TaskPhase.RUNNING)));
                                fixture.assertItemNotBound(103L);
                                raced.set(true);
                                return acquired;
                            })
                    .when(fixture.decode)
                    .acquireDispatchPermit(any(), any());

            Response response =
                    SchedulerTestServices.entry(fixture.scheduler)
                            .submit(fixture.context(103L))
                            .get(2L, TimeUnit.SECONDS);

            assertTrue(raced.get());
            assertFalse(
                    response.isSuccess(),
                    "an already ended request must not receive a new successful route");
            fixture.assertNoPrefillOwnership();
            assertNull(EndpointTestSupport.decodeReservation(fixture.decode, 103L));
            assertEquals(0, fixture.decode.routingView().engineCapacityUsed());
            assertEquals(
                    0L, EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()));
            assertEquals(0, SchedulerTestSupport.repository(fixture.requests).liveRequestCount());
        }
    }

    @Test
    void decodeTerminalAfterRouteClaimSettlesResourcesBeforeLateAddressPublication()
            throws Exception {
        try (Fixture fixture = new Fixture()) {
            AtomicBoolean raced = new AtomicBoolean();
            AtomicReference<Delivery.Member> claimed = new AtomicReference<>();
            doAnswer(
                            call -> {
                                var fact =
                                        call.getArgument(0, ContextResult.DeliveryEvidence.class);
                                Delivery.Member claim = fact.delivery();
                                claimed.set(claim);
                                RequestContext context = claim.route().ctx();
                                assertEquals("105", context.data().getRequestId());
                                assertEquals(
                                        Delivery.SendMode.ROUTE_DECISION,
                                        context.snapshot().delivery().identity().kind());
                                synchronized (context) {
                                    assertFalse(
                                            context.snapshot().delivery().send()
                                                    == ContextResult.SendState.DELIVERED);
                                }
                                assertFalse(claim.completion().toCompletableFuture().isDone());

                                fixture.applyWorkerStatus(
                                        fixture.decode,
                                        Map.of(),
                                        Map.of("105", task(105L, TaskPhase.RUNNING)));

                                Response terminal =
                                        context.data().getFuture().get(2L, TimeUnit.SECONDS);
                                assertTrue(terminal.isSuccess());
                                assertEquals(
                                        ContextResult.Phase.CLOSED, context.snapshot().phase());
                                assertEquals(
                                        RequestState.Phase.COMPLETED,
                                        context.snapshot().terminalRecord().state());
                                assertTrue(claim.completion().toCompletableFuture().isDone());
                                synchronized (context) {
                                    assertFalse(
                                            context.snapshot().delivery().send()
                                                    == ContextResult.SendState.DELIVERED,
                                            "a worker terminal must not fabricate address publication");
                                }
                                fixture.assertNoPrefillOwnership();
                                assertNull(
                                        EndpointTestSupport.decodeReservation(
                                                fixture.decode, 105L));
                                assertEquals(0, fixture.decode.routingView().engineCapacityUsed());
                                assertEquals(
                                        0L,
                                        EndpointTestSupport.expectedReservedKv(
                                                fixture.decode.resourceSnapshot()));
                                assertEquals(0, fixture.requests.repository.liveRequestCount());

                                call.callRealMethod();

                                assertSame(terminal, context.data().getFuture().join());
                                assertEquals(
                                        ContextResult.Phase.CLOSED, context.snapshot().phase());
                                synchronized (context) {
                                    assertFalse(
                                            context.snapshot().delivery().send()
                                                    == ContextResult.SendState.DELIVERED);
                                }
                                raced.set(true);
                                return null;
                            })
                    .when(fixture.requests)
                    .onDeliveryResult(
                            org.mockito.ArgumentMatchers.any(ContextResult.DeliveryEvidence.class));

            try {
                Response response =
                        SchedulerTestServices.entry(fixture.scheduler)
                                .submit(fixture.context(105L))
                                .get(2L, TimeUnit.SECONDS);

                SchedulerTestSupport.awaitRequestWork(fixture.runtime);
                assertTrue(raced.get());
                assertTrue(response.isSuccess());
                fixture.assertNoPrefillOwnership();
                assertEquals(0, fixture.requests.repository.liveRequestCount());
            } finally {
                // A broken sender gate must fail this test without hanging runtime shutdown.
                Delivery.Member claim = claimed.get();
                if (claim != null && !claim.completion().toCompletableFuture().isDone()) {
                    claim.route()
                            .ctx()
                            .data()
                            .getRequestScheduler()
                            .cancel(claim.owner(), 0L, CancelReason.CLIENT_CANCELLED);
                    fixture.requests.scheduleClose(claim.route().ctx());
                }
            }
        }
    }

    @Test
    void directCommitKeepsPrimaryFailureWhenPermitCleanupAlsoFails() throws Exception {
        try (Fixture fixture = new Fixture()) {
            var prefill = spy(fixture.prefill);
            var context = fixture.context(104L);
            var future =
                    RequestProtocolTestSupport.submitForManualPlacement(fixture.scheduler, context);
            var primary = new IllegalStateException("Prefill commit unavailable");
            var cleanup = new IllegalStateException("Decode cleanup observer failed");
            org.mockito.Mockito.doThrow(primary).when(prefill).commitDirect(any());
            doAnswer(
                            call -> {
                                call.callRealMethod();
                                throw cleanup;
                            })
                    .when(fixture.decode)
                    .dispatch(any(), eq(DecodeResources.DispatchOutcome.ABANDONED));
            try {
                var selectedRoles = new java.util.ArrayList<RoleSelection>();

                selectedRoles.add(
                        SelectionTestSupport.metadata(
                                prefill, Fixture.metadata(prefill, 104L), 30_000L));
                selectedRoles.add(
                        SelectionTestSupport.metadata(
                                fixture.decode, Fixture.metadata(fixture.decode, 104L), -1L));

                var admission =
                        SchedulerTestServices.registerSelection(context, selectedRoles).value();
                var mutation =
                        SchedulerTestSupport.claim(
                                SchedulerTestServices.lifecycle(fixture.scheduler), "104", future);
                try (var completion = RequestProtocolTestSupport.finishOnExit(mutation)) {
                    var thrown =
                            assertThrows(
                                    IllegalStateException.class,
                                    () ->
                                            fixture.scheduler.dispatcher.dispatchSelectedRoute(
                                                    admission));
                    org.junit.jupiter.api.Assertions.assertSame(primary, thrown);
                    assertEquals(List.of(cleanup), List.of(thrown.getSuppressed()));
                    verify(fixture.decode)
                            .dispatch(any(), eq(DecodeResources.DispatchOutcome.ABANDONED));
                    assertEquals(0, fixture.decode.resourceSnapshot().activeDispatchPermits());
                    AbstractRequestScheduler.finishUnpublishedRoute(admission, null);
                }
            } finally {
                SchedulerTestServices.lifecycle(fixture.scheduler)
                        .cancel("104", 0L, CancelReason.CLIENT_CANCELLED);
            }
            assertNull(EndpointTestSupport.decodeReservation(fixture.decode, 104L));
            fixture.assertNoPrefillOwnership();
        }
    }

    @Test
    void predictionFailureAfterDirectCommitSettlesBothLedgers() throws Exception {
        try (Fixture fixture = new Fixture()) {
            var context = fixture.context(109L);
            var prefill = spy(fixture.prefill);
            doAnswer(
                            call -> {
                                var work =
                                        spy(
                                                (org.flexlb.balance.projection.RequestTimeEstimation
                                                                .WorkSnapshot)
                                                        call.callRealMethod());
                                org.mockito.Mockito.doThrow(
                                                new IllegalStateException("prediction failed"))
                                        .when(work)
                                        .calculate();
                                return work;
                            })
                    .when(prefill)
                    .commitDirect(any());
            fixture.bindPrefill(prefill);
            assertFalse(
                    SchedulerTestServices.entry(fixture.scheduler)
                            .submit(context)
                            .get(2, TimeUnit.SECONDS)
                            .isSuccess());
            SchedulerTestSupport.awaitRequestWork(fixture.runtime);
            assertNull(EndpointTestSupport.decodeReservation(fixture.decode, 109L));
            assertEquals(0, fixture.decode.resourceSnapshot().activeDispatchPermits());
            fixture.assertNoPrefillOwnership();
            assertEquals(0, SchedulerTestSupport.repository(fixture.requests).liveRequestCount());
        }
    }

    @Test
    void cancellationAfterPrefillCommitRejectsClaimAndSettlesBothLedgers() throws Exception {
        try (Fixture fixture = new Fixture()) {
            var context = fixture.context(108L);
            var prefill = spy(fixture.prefill);
            doAnswer(
                            call -> {
                                var handoff = call.callRealMethod();
                                assertEquals(
                                        1,
                                        fixture.decode.resourceSnapshot().activeDispatchPermits());
                                SchedulerTestServices.lifecycle(fixture.scheduler)
                                        .cancel("108", 0L, CancelReason.CLIENT_CANCELLED);
                                return handoff;
                            })
                    .when(prefill)
                    .commitDirect(any());
            fixture.bindPrefill(prefill);
            var selected = fixture.prefillSelection(prefill);
            org.mockito.Mockito.doAnswer(
                            call -> SelectionTestSupport.registerResult(call, selected))
                    .when(fixture.prefillSelector)
                    .selectAndRegister(
                            eq(context.data().getRequest()),
                            eq(context.data().getSchedulingMetadata()),
                            eq(RoleType.PREFILL),
                            any(),
                            org.mockito.ArgumentMatchers.anyList(),
                            org.mockito.ArgumentMatchers.any(java.util.function.LongSupplier.class),
                            org.mockito.ArgumentMatchers.any());
            assertFalse(
                    SchedulerTestServices.entry(fixture.scheduler)
                            .submit(context)
                            .get(2, TimeUnit.SECONDS)
                            .isSuccess());
            SchedulerTestSupport.awaitRequestWork(fixture.runtime);
            assertNull(EndpointTestSupport.decodeReservation(fixture.decode, 108L));
            assertEquals(0, fixture.decode.resourceSnapshot().activeDispatchPermits());
            fixture.assertNoPrefillOwnership();
            assertEquals(0, SchedulerTestSupport.repository(fixture.requests).liveRequestCount());
        }
    }

    @ParameterizedTest
    @CsvSource({"false,false", "true,false", "false,true", "true,true"})
    void queuePublicationRetainsCapacityAcrossWakeFailureAndRepeatedConsumption(
            boolean separateDecode, boolean wakeFails) throws Exception {
        try (Fixture fixture = new Fixture(16, 1, true)) {
            var config = fixture.config;
            var prefill = spy(fixture.prefill);
            var wakeFailure = new IllegalStateException("wake failed after queue publication");
            if (wakeFails) {
                org.mockito.Mockito.doThrow(wakeFailure)
                        .doCallRealMethod()
                        .when(prefill)
                        .signal(org.flexlb.balance.endpoint.PrefillEndpoint.Signal.INPUT);
            }
            try (var queue =
                    SchedulerTestServices.queued(
                            config,
                            fixture.router,
                            mock(DecodePreemption.class),
                            fixture.runtime.newLifecycle(
                                    config,
                                    SchedulingTestConfig.capacity(config),
                                    org.mockito.Mockito.mock(
                                            org.flexlb.service.TheoryCacheHitReporter.class)),
                            new CapacityNotifier(),
                            new OrderedRequestQueue(config.isPriorityOrdering()),
                            org.flexlb.balance.planner.BuiltinGroupPlanner.from(config))) {
                var context = RequestProtocolTestSupport.context(config, 109L);
                var future = SchedulerTestServices.lifecycle(queue).submit(context, () -> {});
                var handle =
                        SchedulerTestSupport.claim(
                                SchedulerTestServices.lifecycle(queue), "109", future);
                try (var finish = RequestProtocolTestSupport.finishOnExit(handle)) {
                    assertNotNull(handle);
                    var selectedRoles = new java.util.ArrayList<RoleSelection>();

                    selectedRoles.add(
                            SelectionTestSupport.metadata(
                                    prefill, Fixture.metadata(prefill, 109L), 30_000L));
                    if (separateDecode) {
                        selectedRoles.add(
                                SelectionTestSupport.metadata(
                                        fixture.decode,
                                        Fixture.metadata(fixture.decode, 109L),
                                        -1L));
                    }

                    var selected =
                            SchedulerTestServices.registerSelection(context, selectedRoles).value();
                    var plan = RequestProtocolTestSupport.plan(context, selected);
                    try (plan) {
                        if (wakeFails) {
                            org.junit.jupiter.api.Assertions.assertSame(
                                    wakeFailure,
                                    assertThrows(
                                            IllegalStateException.class,
                                            () -> SchedulerTestSupport.enqueueRoute(queue, plan)));
                        } else {
                            assertEquals(
                                    RoutingResult.Status.SUCCESS,
                                    SchedulerTestSupport.enqueueRoute(queue, plan).status());
                        }
                        var exact = context.snapshot().route();
                        assertNotNull(exact);
                        assertEquals(ContextResult.Phase.OPEN, context.snapshot().phase());
                        assertNotNull(context.snapshot().route());
                        assertNull(context.snapshot().delivery());
                        assertEquals(
                                RoutingResult.Status.CLOSED,
                                SchedulerTestSupport.enqueueRoute(queue, plan).status());
                        org.junit.jupiter.api.Assertions.assertSame(
                                exact, context.snapshot().route());
                        verify(prefill).enqueueRoute(eq(exact), any());
                        verify(fixture.decode, never())
                                .release(any(), eq(DecodeResources.ReleaseReason.LOCAL_ROLLBACK));
                        assertEquals(1L, prefill.queueDepth());
                    }
                    assertEquals(
                            separateDecode ? 1 : 0,
                            fixture.decode.resourceSnapshot().queuedCount());
                }
            }
        }
    }

    @Test
    void firstQueueActivationPreservesExistingSelectionRegistrations() throws Exception {
        try (Fixture fixture = new Fixture(4, 2, true)) {
            assertFalse(fixture.prefill.queueEnabled());
            var first =
                    RequestExecutionTestSupport.registerPrefillRoute(
                            fixture.context(101L),
                            fixture.prefill,
                            null,
                            null,
                            30_000L,
                            System.currentTimeMillis());
            var second =
                    RequestExecutionTestSupport.registerPrefillRoute(
                            fixture.context(102L),
                            fixture.prefill,
                            null,
                            null,
                            40_000L,
                            System.currentTimeMillis());
            var firstRecord = first.prefillReservation();
            var secondRecord = second.prefillReservation();
            assertEquals(
                    2L,
                    fixture.prefill
                            .admissionSummary(0, fixture.prefill.requestLimit)
                            .occupiedRequests());
            try {
                assertTrue(
                        fixture.prefill.enqueueRoute(
                                first,
                                QueueExecutionSettings.capture(
                                        fixture.config,
                                        org.flexlb.balance.planner.BuiltinGroupPlanner.from(
                                                fixture.config))));
                assertTrue(fixture.prefill.queueEnabled());
                assertSame(firstRecord, first.prefillReservation());
                assertSame(secondRecord, second.prefillReservation());
                assertSame(
                        first,
                        org.springframework.test.util.ReflectionTestUtils.getField(
                                firstRecord, "route"));
                assertSame(
                        second,
                        org.springframework.test.util.ReflectionTestUtils.getField(
                                secondRecord, "route"));
                assertEquals(30_000L, first.prefillWorkMs());
                assertEquals(40_000L, second.prefillWorkMs());
                assertEquals(
                        2L,
                        fixture.prefill
                                .admissionSummary(0, fixture.prefill.requestLimit)
                                .occupiedRequests());
                assertEquals(1L, fixture.prefill.queueDepth());
                assertTrue(fixture.prefill.releaseRequest(first));
                assertEquals(
                        1L,
                        fixture.prefill
                                .admissionSummary(0, fixture.prefill.requestLimit)
                                .occupiedRequests());
                assertSame(
                        second,
                        org.springframework.test.util.ReflectionTestUtils.getField(
                                secondRecord, "route"));
            } finally {
                fixture.prefill.rollback(firstRecord);
                fixture.prefill.rollback(secondRecord);
            }
            assertEquals(
                    0L,
                    fixture.prefill
                            .admissionSummary(0, fixture.prefill.requestLimit)
                            .occupiedRequests());
            assertEquals(0L, fixture.prefill.queueDepth());
        }
    }

    private static TaskInfo task(long requestId, TaskPhase phase) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(String.valueOf(requestId));
        task.setInputLength(32L);
        task.setPhase(phase);
        return task;
    }

    /**
     * Assembles direct admission with real endpoint ledgers and controllable selection and retirement events.
     */
    private static final class Fixture implements AutoCloseable {
        final CapacityNotifier placement = new CapacityNotifier();
        private final FlexlbConfig config = SchedulingTestConfig.newConfig();
        private final AbstractRequestScheduler requests;
        private final EndpointRegistry endpoints;
        private final DeliveryCoordinator queuedDelivery;
        private final PrefillEndpoint prefill;
        private final DecodeEndpoint decode;
        private final org.flexlb.balance.strategy.PrefillSelectionStrategy prefillSelector;
        private final org.flexlb.balance.strategy.DecodeSelectionStrategy decodeSelector;
        private final AbstractRequestScheduler scheduler;
        private final RequestRouteSelector router;
        private final SchedulerTestServices runtime;

        private Fixture() {
            this(16, 1);
        }

        private Fixture(int prefillCapacity, int decodeCapacity) {
            this(prefillCapacity, decodeCapacity, false);
        }

        private Fixture(int prefillCapacity, int decodeCapacity, boolean queued) {
            this(prefillCapacity, decodeCapacity, queued, false);
        }

        private Fixture(int prefillCapacity, int decodeCapacity, boolean queued, boolean withVit) {
            if (queued) SchedulingTestConfig.useFifoQueue(config);
            else config.setScheduler(ConfigTestSupport.direct());
            SchedulingTestConfig.useNonBatchDispatcher(config);
            config.getDispatcher().setMaxInflightPerPrefillWorker(prefillCapacity);
            config.getRouter()
                    .getRoles()
                    .getDecode()
                    .getAvailability()
                    .setMaxEngineRequests((long) decodeCapacity);
            ConfigService service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            var reporter = mock(DeliveryMetricsReporter.class);
            var requestReporter = mock(RequestSchedulerReporter.class);

            queuedDelivery = spy(DeliveryTestSupport.route(reporter));
            var repository = new RequestRepository();
            endpoints =
                    spy(
                            new EndpointRegistry(
                                    service,
                                    repository,
                                    reporter,
                                    queuedDelivery,
                                    placement,
                                    org.flexlb.balance.planner.BuiltinGroupPlanner.from(
                                            service.loadBalanceConfig()),
                                    () ->
                                            org.flexlb.balance.endpoint.OrderedPrefillRequestQueue
                                                    .fifo(16)));
            runtime =
                    spy(
                            new SchedulerTestServices(
                                    repository,
                                    endpoints,
                                    reporter,
                                    requestReporter,
                                    queuedDelivery,
                                    service,
                                    mock(org.flexlb.engine.grpc.EngineGrpcClient.class)));
            prefillSelector =
                    mock(
                            org.flexlb.balance.strategy.PrefillSelectionStrategy.class,
                            org.mockito.Mockito.CALLS_REAL_METHODS);
            decodeSelector =
                    mock(
                            org.flexlb.balance.strategy.DecodeSelectionStrategy.class,
                            org.mockito.Mockito.CALLS_REAL_METHODS);
            var model = mock(ModelMetaConfig.class);
            when(model.requiredRoles())
                    .thenReturn(
                            withVit
                                    ? List.of(RoleType.PREFILL, RoleType.DECODE, RoleType.VIT)
                                    : List.of(RoleType.PREFILL, RoleType.DECODE));
            router =
                    new RequestRouteSelector(
                            endpoints,
                            prefillSelector,
                            decodeSelector,
                            mock(VitWorkerSelector.class),
                            org.mockito.Mockito.mock(
                                    org.flexlb.balance.strategy.EncoderSelectionStrategy.class),
                            org.mockito.Mockito.mock(
                                    org.flexlb.cache.match.CacheAwareService.class),
                            null,
                            mock(org.flexlb.service.monitor.EngineHealthReporter.class),
                            mock(org.flexlb.cache.telemetry.CacheMetricsReporter.class),
                            service,
                            model);
            requests =
                    SchedulerTestSupport.create(
                            runtime,
                            service,
                            reporter,
                            mock(TheoryCacheHitReporter.class),
                            router,
                            mock(DecodePreemption.class),
                            placement);
            scheduler =
                    (AbstractRequestScheduler)
                            org.flexlb.balance.scheduler.SchedulerTestServices.scheduler(requests);
            WorkerStatus worker = worker(RoleType.PREFILL, "127.0.0.1");
            worker.lock.lock();
            try {
                var response = status(worker, Map.of(), Map.of());
                prefill =
                        (PrefillEndpoint)
                                endpoints.publishPreparedEndpoint(
                                        worker.getLogicalIpPort(),
                                        worker,
                                        worker.prepareNewStatus(
                                                WorkerStatusTestSupport.freezeStatusResponse(
                                                        worker, response)));
            } finally {
                worker.lock.unlock();
            }
            decode =
                    spy(
                            EndpointTestSupport.decode(
                                    worker(RoleType.DECODE, "127.0.0.2"), repository));
            applyWorkerStatus(decode, Map.of(), Map.of());
            bindPrefill(prefill);
            org.mockito.Mockito.doAnswer(
                            call ->
                                    SelectionTestSupport.registerResult(
                                            call, prefillSelection(prefill)))
                    .when(prefillSelector)
                    .selectAndRegister(
                            any(),
                            any(),
                            eq(RoleType.PREFILL),
                            any(),
                            org.mockito.ArgumentMatchers.anyList(),
                            org.mockito.ArgumentMatchers.any(java.util.function.LongSupplier.class),
                            org.mockito.ArgumentMatchers.any());
            org.mockito.Mockito.doAnswer(
                            call ->
                                    SelectionTestSupport.registerResult(
                                            call, RoutingResult.success(decode.routingView())))
                    .when(decodeSelector)
                    .selectAndRegister(
                            any(),
                            org.mockito.ArgumentMatchers.any(),
                            any(),
                            org.mockito.ArgumentMatchers.anyList(),
                            org.mockito.ArgumentMatchers.any(),
                            org.mockito.ArgumentMatchers.any(),
                            org.mockito.ArgumentMatchers.any());
            doAnswer(
                            call -> {
                                call.getArgument(1, List.class).add(decode.routingView());
                                return null;
                            })
                    .when(endpoints)
                    .decodeRoutingSnapshot(any(), any());
            doAnswer(call -> decode).when(endpoints).selectDecodeEndpoint(any());
        }

        @Override
        public void close() {
            try {
                decode.close();
            } finally {
                SchedulerTestSupport.shutdown(runtime);
            }
        }

        private PrefillEndpoint installReplacementPrefill() {
            var worker = worker(RoleType.PREFILL, "127.0.0.3");
            PrefillEndpoint replacement;
            worker.lock.lock();
            try {
                var observation =
                        WorkerStatusTestSupport.freezeStatusResponse(
                                worker, status(worker, Map.of(), Map.of()));
                replacement =
                        (PrefillEndpoint)
                                endpoints.publishPreparedEndpoint(
                                        worker.getLogicalIpPort(),
                                        worker,
                                        worker.prepareNewStatus(observation));
            } finally {
                worker.lock.unlock();
            }
            bindPrefill(replacement);
            org.mockito.Mockito.doAnswer(
                            call ->
                                    SelectionTestSupport.registerResult(
                                            call, prefillSelection(replacement)))
                    .when(prefillSelector)
                    .selectAndRegister(
                            any(),
                            any(),
                            eq(RoleType.PREFILL),
                            any(),
                            org.mockito.ArgumentMatchers.anyList(),
                            org.mockito.ArgumentMatchers.any(java.util.function.LongSupplier.class),
                            org.mockito.ArgumentMatchers.any());
            return replacement;
        }

        private static WorkerStatus worker(RoleType role, String ip) {
            return WorkerStatus.createDiscovered(role, "g1", ip, 8080, 8081, "test");
        }

        private static WorkerStatusResponse status(
                WorkerStatus worker,
                Map<String, TaskInfo> running,
                Map<String, TaskInfo> finished) {
            var response = new WorkerStatusResponse();
            response.setRole(worker.getRole());
            response.setAlive(true);
            response.setStatusVersion(
                    Math.max(1L, worker.appliedStatusCursor().statusVersion() + 1L));
            response.setLatestFinishedVersion(
                    Math.max(0L, worker.appliedStatusCursor().latestFinishedTaskVersion())
                            + (finished.isEmpty() ? 0L : 1L));
            response.setRunningTaskInfo(running);
            response.setRunningQueryLen((long) running.size());
            response.setFinishedTaskInfo(finished);
            if (worker.getRole() == RoleType.DECODE) {
                response.setTotalKvCacheTokens(10_000L);
                response.setAvailableKvCacheTokens(10_000L - running.size() * 32L);
            }
            return response;
        }

        private void bindPrefill(PrefillEndpoint selected) {
            String address = selected.ipPort();
            org.mockito.Mockito.doReturn(
                            List.of(new EndpointRegistry.PrefillRoutingEntry(address, selected)))
                    .when(endpoints)
                    .prefillRoutingSnapshot(RoleType.PREFILL);
            doAnswer(call -> selected).when(endpoints).selectEndpoint(RoleType.PREFILL, address);
        }

        private RoutingResult<
                        org.flexlb.balance.strategy.PrefillSelectionStrategy.Selection<
                                org.flexlb.balance.strategy.PrefillSelectionStrategy
                                        .CandidateInput>,
                        CapacityKey>
                prefillSelection(PrefillEndpoint selected) {
            return RoutingResult.success(
                    new org.flexlb.balance.strategy.PrefillSelectionStrategy.Selection<>(
                            selected.ipPort(),
                            selected.getStatus().getGenerationId(),
                            org.flexlb.balance.endpoint.EndpointTestSupport.versions(selected)
                                    .ownershipVersion(),
                            30_000L,
                            30_000L,
                            0L,
                            0L,
                            0L,
                            null,
                            1,
                            new EndpointRegistry.PrefillRoutingEntry(selected.ipPort(), selected),
                            null,
                            new EndpointRegistry.PrefillRoutingEntry(selected.ipPort(), selected)
                                    .projectionInputs()));
        }

        private RequestContext context(long requestId) {
            var context = RequestProtocolTestSupport.context(config, requestId);
            context.data().setRequest(context.data().getRequest().toBuilder().seqLen(32L).build());
            context.data()
                    .setRequest(context.data().getRequest().toBuilder().maxNewTokens(16).build());
            return context;
        }

        private void assertNoPrefillOwnership() {
            assertNoWaitingQueue();
            assertEquals(0L, prefill.admissionSummary(0, prefill.requestLimit).occupiedRequests());
            assertEquals(0, prefill.stats().batchCount());
        }

        private void assertNoWaitingQueue() {
            assertEquals(
                    0,
                    org.flexlb.balance.scheduler.SchedulerTestServices.pendingDeliveryRequestCount(
                            SchedulerTestSupport.repository(requests)));
            assertEquals(0, prefill.queueDepth());
            assertEquals(0, prefill.queueDepth());
        }

        private void assertItemNotBound(long requestId) {
            RequestContext requestContext =
                    SchedulerTestSupport.repository(requests).findActive(String.valueOf(requestId));
            assertNotNull(requestContext);
            synchronized (requestContext) {
                assertNull(
                        requestContext.snapshot().route(),
                        "this Engine observation must precede item binding");
            }
        }

        private void applyWorkerStatus(
                WorkerEndpoint endpoint,
                Map<String, TaskInfo> running,
                Map<String, TaskInfo> finished) {
            WorkerStatus worker = endpoint.getStatus();
            org.flexlb.balance.endpoint.WorkerEndpoint.StatusChange projection;
            worker.lock.lock();
            try {
                var response = status(worker, running, finished);
                projection =
                        endpoint.applyPreparedStatus(
                                worker,
                                worker.prepareNewStatus(
                                        WorkerStatusTestSupport.freezeStatusResponse(
                                                worker, response)));
            } finally {
                worker.lock.unlock();
            }
            endpoint.publishStatus(projection);
            SchedulerTestSupport.awaitRequestWork(runtime);
        }

        private static ServerStatus metadata(WorkerEndpoint endpoint, long requestId) {
            var result = new ServerStatus();
            result.setSuccess(true);
            result.setRequestId(String.valueOf(requestId));
            result.setRole(endpoint.getStatus().getRole());
            result.setGroup("g1");
            result.setServerIp(endpoint.getIp());
            result.setHttpPort(endpoint.getHttpPort());
            result.setGrpcPort(8081);
            if (endpoint instanceof PrefillEndpoint) {
                DebugInfo debug = new DebugInfo();
                debug.setHitCacheLen(8L);
                result.setDebugInfo(debug);
                result.setPrefillTime(30_000L);
            }
            return result;
        }
    }
}
