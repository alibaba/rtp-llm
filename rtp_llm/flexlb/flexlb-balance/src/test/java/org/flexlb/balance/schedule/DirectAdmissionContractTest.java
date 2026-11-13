package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.balance.schedule.RequestContext.SelectionAttempt;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeStrategy;
import org.flexlb.balance.strategy.EncoderStrategy;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.balance.strategy.RoleSelection;
import org.flexlb.balance.strategy.RoutingResult;
import org.flexlb.balance.strategy.VitStrategy;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.service.ModelMetaConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.AdmissionRejectReason;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.CapacityNotifier;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.EndpointTestSupport;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.status.DebugInfo;
import org.flexlb.worker.status.ServerStatus;
import org.flexlb.worker.status.TaskInfo;
import org.flexlb.worker.status.TaskPhase;
import org.flexlb.worker.status.WorkerStatus;
import org.flexlb.worker.status.WorkerStatusResponse;
import org.flexlb.worker.status.WorkerStatusTestSupport;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.time.Duration;
import java.util.List;
import java.util.Map;
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
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * DIRECT uses real admission ledgers and lifecycle while bypassing waiting queues.
 */
class DirectAdmissionContractTest {

    @Test
    void queuedSelectionReturnsRouteWithoutWaitingInPrefillQueue() throws Exception {
        try (Fixture fixture = new Fixture(16, 1, true)) {
            var context = fixture.context(9115L);
            var future =
                    RequestProtocolTestSupport.submitForManualPlacement(fixture.scheduler, context);

            var result =
                    SchedulerTestSupport.selectAndPublish(
                            fixture.scheduler, context, () -> fixture.router.select(context, null));

            assertEquals(RoutingResult.Status.SUCCESS, result.status());
            assertTrue(future.get(2, TimeUnit.SECONDS).isSuccess());

            assertSame(result, context.snapshot().route());
            assertNotNull(context.snapshot().committedRoute());
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
            assertNull(context.currentSelectionAttempt());
            assertNull(context.snapshot().route());
            assertFalse(future.isDone());
            var retry =
                    SchedulerTestSupport.selectAndPublish(
                            fixture.scheduler, context, RequestResult::retry);
            assertEquals(RoutingResult.Status.RETRY, retry.status());
            assertNull(context.currentSelectionAttempt());
            fixture.scheduler.cancel(context, CancelReason.CLIENT_CANCELLED);
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
                                        EndpointTestSupport.decodeRegistration(
                                                fixture.decode, 9112L));
                                fixture.scheduler.cancel(context, CancelReason.CLIENT_CANCELLED);
                                return selected;
                            });
            assertEquals(RoutingResult.Status.CLOSED, result.status());
            assertFalse(future.get(2, TimeUnit.SECONDS).isSuccess());
            SchedulerTestSupport.awaitRequestWork(fixture.runtime);
            assertNull(context.currentSelectionAttempt());
            assertNull(EndpointTestSupport.decodeRegistration(fixture.decode, 9112L));
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
            var route = selected;
            assertNotNull(EndpointTestSupport.decodeRegistration(fixture.decode, 9100L));
            assertTrue(fixture.prefill.releaseRequest(route));
            assertTrue(fixture.prefill.acceptsNewRequests());
            var publication =
                    SchedulerTestSupport.selectAndPublish(fixture.scheduler, routing, () -> route);

            assertEquals(RoutingResult.Status.REJECTED, publication.status());
            assertEquals(
                    StrategyErrorType.DISPATCH_FAILED.getErrorCode(),
                    publication.failure().getCode());
            assertEquals(
                    AdmissionRejectReason.UNSPECIFIED,
                    publication.failure().getAdmissionRejectReason());
            String message = publication.failure().getErrorMessage();
            assertTrue(message.contains("Selected Prefill registration was not valid for commit"));
            assertFalse(message.contains("capacity exhausted"));
            assertFalse(message.contains("priority attribution"));
            assertNull(context.snapshot().route());
            assertNull(context.currentSelectionAttempt());
            assertNull(EndpointTestSupport.decodeRegistration(fixture.decode, 9100L));

            assertEquals(0, fixture.decode.selectionSnapshot().usage().occupiedRequests());

            assertEquals(
                    0L,
                    org.flexlb.worker.endpoint.EndpointTestSupport.occupiedRequestCount(
                            fixture.prefill));

            assertFalse(future.isDone(), "the scheduler still owns the blocked/rejected decision");
            fixture.scheduler.cancel(context, CancelReason.CLIENT_CANCELLED);
            assertFalse(future.get(2, TimeUnit.SECONDS).isSuccess());
            SchedulerTestSupport.awaitRequestWork(fixture.runtime);
            fixture.assertNoPrefillOwnership();
            assertEquals(0, fixture.requests.repository.liveRequestCount());
        }
    }

    @Test
    void failedUnpublishedRollbackRemainsReachableUntilResourcesAreReleased() throws Exception {
        var fixture = new Fixture();
        var failure = new IllegalStateException("temporary cleanup failure");
        try {
            var context = fixture.context(9099L);
            var future =
                    RequestProtocolTestSupport.submitForManualPlacement(fixture.scheduler, context);
            var operation = SchedulerTestSupport.claim(fixture.scheduler, "9099", future);
            assertNotNull(operation);
            var route = fixture.router.select(context, null);
            assertNotNull(route);
            org.mockito.Mockito.doThrow(failure)
                    .doCallRealMethod()
                    .when(fixture.decode)
                    .rollbackSelection(route.decodeRegistration());
            EndpointTestSupport.closeAndWait(fixture.prefill);
            assertSame(
                    failure,
                    assertThrows(
                            IllegalStateException.class,
                            () ->
                                    SchedulerTestSupport.selectAndPublish(
                                            fixture.scheduler, operation, () -> route)));
            assertFalse(future.get(2, TimeUnit.SECONDS).isSuccess());
            SchedulerTestSupport.awaitRequestWork(fixture.runtime);
            assertNull(EndpointTestSupport.decodeRegistration(fixture.decode, 9099L));
            fixture.assertNoPrefillOwnership();
            assertEquals(0, fixture.requests.repository.liveRequestCount());
        } finally {
            assertSame(failure, assertThrows(IllegalStateException.class, fixture::close));
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void vitRetirementDuringAdmissionReplansTheWholeRoute(boolean queued) throws Exception {
        try (Fixture fixture = new Fixture(16, 1, queued, true)) {
            var first = new WorkerEndpoint(Fixture.worker(RoleType.VIT, "vit-first"));
            var replacement = new WorkerEndpoint(Fixture.worker(RoleType.VIT, "vit-replacement"));
            var current = new AtomicReference<>(first);
            doAnswer(call -> List.of(current.get()))
                    .when(fixture.endpoints)
                    .endpoints(eq(RoleType.VIT), any());
            doAnswer(call -> current.get())
                    .when(fixture.endpoints)
                    .getEndpoint(eq(RoleType.VIT), any());
            doAnswer(
                            call -> {
                                if (current.get() == first) {
                                    org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                                            first);
                                    current.set(replacement);
                                }
                                return call.callRealMethod();
                            })
                    .when(fixture.requests)
                    .currentSelectionAttempt(any());
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
                assertEquals(
                        1L,
                        org.flexlb.worker.endpoint.EndpointTestSupport.occupiedRequestCount(
                                fixture.prefill));
                assertEquals(1, fixture.decode.selectionSnapshot().usage().occupiedRequests());
                assertEquals(
                        48L,
                        EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()));
                verify(fixture.decode).rollbackSelection(any());
                verify(fixture.requests, times(2)).currentSelectionAttempt(any());
            } finally {
                org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(first);
                org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(replacement);
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
                                var route = call.getArgument(0, RequestResult.class);
                                assertNotNull(route.decodeRegistration());

                                // Selection and Decode preparation completed, but Prefill admission
                                // has not started. WorkerCleanup wins this ordering
                                // deterministically.
                                if (attempts.getAndIncrement() == 0) {
                                    assertSame(fixture.prefill, route.prefillEndpoint());
                                    org.flexlb.worker.endpoint.EndpointTestSupport.closeAndWait(
                                            fixture.prefill);
                                    replacement.set(fixture.installReplacementPrefill());
                                } else {
                                    assertSame(replacement.get(), route.prefillEndpoint());
                                    if (replacementAlsoRetires)
                                        org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                                                replacement.get());
                                }
                                return call.callRealMethod();
                            })
                    .when(fixture.requests)
                    .currentSelectionAttempt(any());

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
                assertEquals(
                        org.flexlb.worker.endpoint.StrategyErrorType.forRole(RoleType.PREFILL)
                                .getErrorCode(),
                        response.getCode());
                assertNull(context.snapshot().committedRoute());
                assertNull(EndpointTestSupport.decodeRegistration(fixture.decode, 9011L));

                assertEquals(
                        0, SchedulerTestSupport.repository(fixture.requests).liveRequestCount());
            } else {
                assertNotNull(context.snapshot().committedRoute());
                assertSame(replacement.get(), context.snapshot().route().prefillEndpoint());
                assertEquals(1, replacement.get().stats().locallyOwnedRequests());
            }
            assertEquals(
                    replacementAlsoRetires ? 0 : 1,
                    fixture.decode.selectionSnapshot().usage().occupiedRequests());
            assertEquals(
                    replacementAlsoRetires ? 0L : 48L,
                    EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()));
            verify(fixture.decode, times(replacementAlsoRetires ? 2 : 1)).rollbackSelection(any());
            verify(fixture.requests, times(2)).currentSelectionAttempt(any());
            verify(fixture.prefillStrategy, times(2))
                    .chooseBest(
                            any(),
                            org.mockito.ArgumentMatchers.anyList(),
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
                            org.flexlb.worker.endpoint.EndpointTestSupport.closeAndWait(
                                    fixture.decode);
                            org.mockito.Mockito.doAnswer(
                                            selection ->
                                                    SelectionTestSupport.selectionResult(
                                                            selection,
                                                            RoutingResult.success(
                                                                    replacement
                                                                            .selectionSnapshot())))
                                    .when(fixture.decodeStrategy)
                                    .chooseBest(any(), org.mockito.ArgumentMatchers.anyList());
                            doAnswer(selection -> replacement)
                                    .when(fixture.endpoints)
                                    .getEndpoint(
                                            org.mockito.ArgumentMatchers.eq(
                                                    org.flexlb.config.service.RoleType.DECODE),
                                            org.mockito.ArgumentMatchers.anyString());
                            doAnswer(
                                            snapshot -> {
                                                var capturedViews =
                                                        new java.util.ArrayList<
                                                                org.flexlb.worker.endpoint
                                                                        .DecodeEndpoint
                                                                        .SelectionSnapshot>();

                                                capturedViews.add(replacement.selectionSnapshot());
                                                return org.flexlb.worker.endpoint
                                                        .EndpointTestSupport.decodeCandidates(
                                                        capturedViews);
                                            })
                                    .when(fixture.endpoints)
                                    .endpoints(
                                            org.mockito.ArgumentMatchers.eq(
                                                    org.flexlb.config.service.RoleType.DECODE),
                                            org.mockito.ArgumentMatchers.any());
                            return afterBinding ? bound : call.callRealMethod();
                        };
                if (afterBinding)
                    doAnswer(retireDuringAdmission)
                            .when(fixture.requests)
                            .currentSelectionAttempt(any());
                else
                    doAnswer(retireDuringAdmission)
                            .when(fixture.decode)
                            .tryRegisterSelection(
                                    any(),
                                    org.mockito.ArgumentMatchers.anyString(),
                                    org.mockito.ArgumentMatchers.anyLong(),
                                    org.mockito.ArgumentMatchers.anyLong(),
                                    org.mockito.ArgumentMatchers.anyInt(),
                                    any());

                var context = fixture.context(9015L);
                var response =
                        SchedulerTestServices.entry(fixture.scheduler)
                                .submit(context)
                                .get(2L, TimeUnit.SECONDS);
                SchedulerTestSupport.awaitRequestWork(fixture.runtime);

                assertTrue(response.isSuccess());
                assertSame(replacement, context.snapshot().route().decodeEndpoint());
                assertEquals(0, fixture.decode.selectionSnapshot().usage().occupiedRequests());

                assertEquals(1, replacement.selectionSnapshot().usage().occupiedRequests());
                assertEquals(
                        48L,
                        EndpointTestSupport.expectedReservedKv(replacement.resourceSnapshot()));
                assertEquals(
                        1L,
                        org.flexlb.worker.endpoint.EndpointTestSupport.occupiedRequestCount(
                                fixture.prefill));
                verify(fixture.prefillStrategy, times(2))
                        .chooseBest(
                                any(),
                                org.mockito.ArgumentMatchers.anyList(),
                                org.mockito.ArgumentMatchers.any());
            } finally {
                org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(replacement);
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
                                org.flexlb.worker.endpoint.EndpointTestSupport.closeAndWait(
                                        fixture.prefill);
                                fixture.installReplacementPrefill();
                                return call.callRealMethod();
                            })
                    .when(fixture.requests)
                    .currentSelectionAttempt(any());
            doAnswer(
                            call -> {
                                if (admissions.getAndIncrement() == 1) {
                                    assertNull(context.snapshot().route());
                                    assertEquals(
                                            0,
                                            fixture.decode
                                                    .selectionSnapshot()
                                                    .usage()
                                                    .occupiedRequests());
                                    fixture.requests.cancel("9014", CancelReason.CLIENT_CANCELLED);
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
            assertNull(context.snapshot().committedRoute());
            fixture.assertNoPrefillOwnership();

            assertEquals(0, SchedulerTestSupport.repository(fixture.requests).liveRequestCount());
            verify(fixture.prefillStrategy)
                    .chooseBest(
                            any(),
                            org.mockito.ArgumentMatchers.anyList(),
                            org.mockito.ArgumentMatchers.any());
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
                                var route = call.getArgument(0, RequestResult.class);
                                if (attempts.getAndIncrement() == 0) {
                                    assertSame(fixture.prefill, route.prefillEndpoint());
                                    assertNotNull(route.decodeRegistration());
                                    org.flexlb.worker.endpoint.EndpointTestSupport.closeAndWait(
                                            fixture.prefill);
                                    var worker = Fixture.worker(RoleType.PREFILL, "127.0.0.3");
                                    worker.lock.lock();
                                    try {
                                        replacement.set(
                                                (PrefillEndpoint)
                                                        fixture.endpoints
                                                                .updateEndpoint(
                                                                        worker.getLogicalIpPort(),
                                                                        worker,
                                                                        worker.prepareStatus(
                                                                                WorkerStatusTestSupport
                                                                                        .freezeStatusResponse(
                                                                                                worker,
                                                                                                Fixture
                                                                                                        .status(
                                                                                                                worker,
                                                                                                                Map
                                                                                                                        .of(),
                                                                                                                Map
                                                                                                                        .of()))))
                                                                .endpoint());
                                    } finally {
                                        worker.lock.unlock();
                                    }
                                    fixture.bindPrefill(replacement.get());
                                    org.mockito.Mockito.doAnswer(
                                                    selection ->
                                                            SelectionTestSupport.selectionResult(
                                                                    selection,
                                                                    fixture.prefillSelection(
                                                                            replacement.get())))
                                            .when(fixture.prefillStrategy)
                                            .chooseBest(
                                                    any(),
                                                    org.mockito.ArgumentMatchers.anyList(),
                                                    org.mockito.ArgumentMatchers.any());
                                } else {
                                    assertSame(replacement.get(), route.prefillEndpoint());
                                    // Decode capacity is one: the retry can reach here only after
                                    // rolling back the failed publication's reservation.
                                    assertEquals(
                                            1,
                                            fixture
                                                    .decode
                                                    .resourceSnapshot()
                                                    .requests()
                                                    .values()
                                                    .stream()
                                                    .filter(
                                                            snapshotRequest ->
                                                                    !snapshotRequest
                                                                            .phase()
                                                                            .isEngineConfirmed())
                                                    .count());
                                }
                                return call.callRealMethod();
                            })
                    .when(fixture.requests)
                    .currentSelectionAttempt(any());

            var context = fixture.context(9012L);
            var response =
                    SchedulerTestServices.entry(fixture.scheduler)
                            .submit(context)
                            .get(3L, TimeUnit.SECONDS);
            SchedulerTestSupport.awaitRequestWork(fixture.runtime);

            assertTrue(response.isSuccess());
            assertEquals(2, attempts.get());
            assertSame(replacement.get(), context.snapshot().route().prefillEndpoint());
            assertNotNull(context.snapshot().committedRoute());
            fixture.assertNoPrefillOwnership();
            assertEquals(1, replacement.get().stats().locallyOwnedRequests());
            assertEquals(1, fixture.decode.selectionSnapshot().usage().occupiedRequests());
            assertEquals(
                    48L, EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()));
            assertEquals(
                    1,
                    fixture.decode.resourceSnapshot().requests().values().stream()
                            .filter(snapshotRequest -> !snapshotRequest.phase().isEngineConfirmed())
                            .count());
            verify(fixture.decode).rollbackSelection(any());
        }
    }

    @Test
    void schedulerRetriesOneGenerationHandoffFailureBeforeCommitting() throws Exception {
        try (Fixture fixture = new Fixture()) {
            var captures = new java.util.concurrent.atomic.AtomicInteger();
            doAnswer(call -> captures.getAndIncrement() == 0 ? null : fixture.decode)
                    .when(fixture.endpoints)
                    .getEndpoint(
                            org.mockito.ArgumentMatchers.eq(
                                    org.flexlb.config.service.RoleType.DECODE),
                            org.mockito.ArgumentMatchers.anyString());
            var response =
                    SchedulerTestServices.entry(fixture.scheduler)
                            .submit(fixture.context(9010L))
                            .get(2L, TimeUnit.SECONDS);
            assertTrue(response.isSuccess());
            assertEquals(2, captures.get());
            assertEquals(
                    1L,
                    org.flexlb.worker.endpoint.EndpointTestSupport.occupiedRequestCount(
                            fixture.prefill));
            assertEquals(1, fixture.decode.selectionSnapshot().usage().occupiedRequests());
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
            var foreign = new SelectionAttempt(context, Long.MAX_VALUE);
            var route = mock(RequestResult.class);
            when(route.ctx()).thenReturn(context);
            try {
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                SchedulerTestSupport.selectAndPublish(
                                        fixture.requests, foreign, () -> route));
                assertEquals(admission, context.currentSelectionAttempt());
                assertEquals(
                        1,
                        org.flexlb.balance.schedule.SchedulerTestServices
                                .pendingDeliveryRequestCount(fixture.requests.repository));

                assertEquals(
                        0L,
                        org.flexlb.worker.endpoint.EndpointTestSupport.occupiedRequestCount(
                                fixture.prefill));

                assertEquals(0, fixture.decode.selectionSnapshot().usage().occupiedRequests());
            } finally {
                admission
                        .requestContext()
                        .getRequestScheduler()
                        .finishSelectionWithoutCommit(admission, null);
                fixture.requests.cancel("9001", CancelReason.CLIENT_CANCELLED);
            }
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
            assertNotNull(context.snapshot().route().prefillRegistration());
            assertEquals(1, fixture.prefill.stats().locallyOwnedRequests());
            assertEquals(1, fixture.decode.selectionSnapshot().usage().occupiedRequests());
            fixture.assertNoWaitingQueue();
            org.junit.jupiter.api.Assertions.assertDoesNotThrow(
                    () -> SchedulerTestServices.throwIfFailed(fixture.requests.taskPool));
            assertFalse(selectedEndpoint.hasStoppedAcceptingRequests());
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
                    org.flexlb.worker.endpoint.EndpointTestSupport.occupiedRequestCount(
                            fixture.prefill));

            assertEquals(1, fixture.decode.selectionSnapshot().usage().occupiedRequests());
            assertEquals(
                    48L, EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()));
            assertEquals(
                    1,
                    fixture.decode.resourceSnapshot().requests().values().stream()
                            .filter(snapshotRequest -> !snapshotRequest.phase().isEngineConfirmed())
                            .count());
            assertEquals(1, SchedulerTestSupport.repository(fixture.requests).liveRequestCount());
            assertEquals(
                    RequestResult.State.ACKNOWLEDGED,
                    SchedulerTestSupport.repository(fixture.requests)
                            .getRequestState("101")
                            .state());
            var reservation = EndpointTestSupport.decodeRegistration(fixture.decode, 101L);
            assertNotNull(reservation);

            fixture.applyWorkerStatus(
                    fixture.prefill, Map.of(), Map.of("101", task(101L, TaskPhase.RUNNING)));
            assertEquals(
                    0L,
                    org.flexlb.worker.endpoint.EndpointTestSupport.occupiedRequestCount(
                            fixture.prefill));
            assertEquals(
                    48L,
                    EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()),
                    "Prefill completion must retain Decode ownership until its own observation");
            fixture.applyWorkerStatus(
                    fixture.decode, Map.of("101", task(101L, TaskPhase.RUNNING)), Map.of());
            assertEquals(
                    org.flexlb.worker.endpoint.DecodeResources.RegistrationState.CONFIRMED,
                    fixture.decode.registrationState(reservation));
            fixture.applyWorkerStatus(
                    fixture.decode, Map.of(), Map.of("101", task(101L, TaskPhase.RUNNING)));
            assertEquals(0, fixture.decode.selectionSnapshot().usage().occupiedRequests());
            assertEquals(0, SchedulerTestSupport.repository(fixture.requests).liveRequestCount());
            fixture.assertNoWaitingQueue();
        }
    }

    @Test
    void fullDecodeRejectsImmediatelyAndReleasesUncommittedResources() throws Exception {
        try (Fixture fixture = new Fixture()) {
            DecodeEndpoint.RequestRegistration occupant;
            {
                var selectedEndpoint = fixture.decode;

                assertNotNull(selectedEndpoint);
                occupant =
                        org.flexlb.balance.schedule.SelectionTestSupport.registerQueuedDecode(
                                        fixture.decode, String.valueOf(999L), 32L, 48L, 50)
                                .value();
            }
            assertNotNull(occupant);
            Response response =
                    assertTimeoutPreemptively(
                            Duration.ofSeconds(2),
                            () ->
                                    SchedulerTestServices.entry(fixture.scheduler)
                                            .submit(fixture.context(102L))
                                            .get(2L, TimeUnit.SECONDS));

            assertFalse(response.isSuccess());
            assertEquals(
                    StrategyErrorType.PRIORITY_ADMISSION_REJECTED.getErrorCode(),
                    response.getCode());
            assertEquals(
                    org.flexlb.worker.endpoint.AdmissionRejectReason.SAME_PRIORITY_AHEAD,
                    response.getAdmissionRejectReason());
            fixture.assertNoPrefillOwnership();
            assertNull(EndpointTestSupport.decodeRegistration(fixture.decode, 102L));
            assertEquals(occupant, EndpointTestSupport.decodeRegistration(fixture.decode, 999L));
            assertEquals(1, fixture.decode.selectionSnapshot().usage().occupiedRequests());
            assertEquals(
                    48L, EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()));
            assertEquals(
                    1,
                    fixture.decode.resourceSnapshot().requests().values().stream()
                            .filter(snapshotRequest -> !snapshotRequest.phase().isEngineConfirmed())
                            .count());
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
                                var acquired =
                                        (WorkerEndpoint.Registration<
                                                        DecodeEndpoint.RequestRegistration>)
                                                call.callRealMethod();
                                assertNotNull(acquired.value());
                                assertEquals("103", acquired.value().requestId());
                                fixture.assertItemNotBound(103L);
                                fixture.applyWorkerStatus(
                                        fixture.decode,
                                        Map.of("103", task(103L, TaskPhase.KV_ALLOCATED)),
                                        Map.of());
                                fixture.applyWorkerStatus(
                                        fixture.decode,
                                        Map.of(),
                                        Map.of("103", task(103L, TaskPhase.RUNNING)));
                                fixture.assertItemNotBound(103L);
                                raced.set(true);
                                return acquired;
                            })
                    .when(fixture.decode)
                    .tryRegisterSelection(
                            any(),
                            org.mockito.ArgumentMatchers.anyString(),
                            org.mockito.ArgumentMatchers.anyLong(),
                            org.mockito.ArgumentMatchers.anyLong(),
                            org.mockito.ArgumentMatchers.anyInt(),
                            any());

            Response response =
                    SchedulerTestServices.entry(fixture.scheduler)
                            .submit(fixture.context(103L))
                            .get(2L, TimeUnit.SECONDS);

            assertTrue(raced.get());
            assertFalse(
                    response.isSuccess(),
                    "an already ended request must not receive a new successful route");
            fixture.assertNoPrefillOwnership();
            assertNull(EndpointTestSupport.decodeRegistration(fixture.decode, 103L));
            assertEquals(0, fixture.decode.selectionSnapshot().usage().occupiedRequests());
            assertEquals(
                    0L, EndpointTestSupport.expectedReservedKv(fixture.decode.resourceSnapshot()));
            assertEquals(0, SchedulerTestSupport.repository(fixture.requests).liveRequestCount());
        }
    }

    @Test
    void directCommitFailureAndRollbackFailureRemainSeparatelyObservable() throws Exception {
        try (Fixture fixture = new Fixture()) {
            var prefill = spy(fixture.prefill);
            var context = fixture.context(104L);
            var future =
                    RequestProtocolTestSupport.submitForManualPlacement(fixture.scheduler, context);
            var primary = new IllegalStateException("Prefill commit unavailable");
            var cleanup = new IllegalStateException("Decode cleanup observer failed");
            org.mockito.Mockito.doThrow(primary).when(prefill).commitSelection(any());
            doAnswer(
                            call -> {
                                boolean released = (boolean) call.callRealMethod();
                                if (released) throw cleanup;
                                return false;
                            })
                    .when(fixture.decode)
                    .rollbackSelection(any());
            try {
                var selectedRoles = new java.util.ArrayList<RoleSelection>();

                selectedRoles.add(
                        SelectionTestSupport.metadata(
                                prefill, Fixture.metadata(prefill, 104L), 30_000L));
                selectedRoles.add(
                        SelectionTestSupport.metadata(
                                fixture.decode, Fixture.metadata(fixture.decode, 104L), -1L));

                var admission = SchedulerTestServices.registerSelection(context, selectedRoles);
                var mutation =
                        SchedulerTestSupport.claim(
                                SchedulerTestServices.lifecycle(fixture.scheduler), "104", future);
                try (var completion = RequestProtocolTestSupport.finishOnExit(mutation)) {
                    var thrown =
                            assertThrows(
                                    IllegalStateException.class,
                                    () -> fixture.scheduler.publishSelectedRoute(admission));
                    org.junit.jupiter.api.Assertions.assertSame(primary, thrown);
                    assertEquals(0, thrown.getSuppressed().length);
                    org.junit.jupiter.api.Assertions.assertSame(
                            cleanup,
                            AbstractRequestScheduler.finishUnpublishedRoute(admission, null));
                    verify(fixture.decode).rollbackSelection(any());
                }
            } finally {
                SchedulerTestServices.lifecycle(fixture.scheduler)
                        .cancel("104", CancelReason.CLIENT_CANCELLED);
            }
            assertNull(EndpointTestSupport.decodeRegistration(fixture.decode, 104L));
            fixture.assertNoPrefillOwnership();
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

                                SchedulerTestServices.lifecycle(fixture.scheduler)
                                        .cancel("108", CancelReason.CLIENT_CANCELLED);
                                return handoff;
                            })
                    .when(prefill)
                    .commitSelection(any());

            fixture.bindPrefill(prefill);
            var selected = fixture.prefillSelection(prefill);
            org.mockito.Mockito.doAnswer(
                            call -> SelectionTestSupport.selectionResult(call, selected))
                    .when(fixture.prefillStrategy)
                    .chooseBest(
                            eq(context.getRequest()),
                            org.mockito.ArgumentMatchers.anyList(),
                            org.mockito.ArgumentMatchers.any());
            assertFalse(
                    SchedulerTestServices.entry(fixture.scheduler)
                            .submit(context)
                            .get(2, TimeUnit.SECONDS)
                            .isSuccess());
            SchedulerTestSupport.awaitRequestWork(fixture.runtime);
            assertNull(EndpointTestSupport.decodeRegistration(fixture.decode, 108L));

            fixture.assertNoPrefillOwnership();
            assertEquals(0, SchedulerTestSupport.repository(fixture.requests).liveRequestCount());
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
     * Assembles direct admission with real endpoint ledgers and controllable selection and cleanup events.
     */
    private static final class Fixture implements AutoCloseable {
        final CapacityNotifier placement = new CapacityNotifier();
        private final FlexlbConfig config = SchedulingTestConfig.newConfig();
        private final AbstractRequestScheduler requests;
        private final EndpointRegistry endpoints;
        private final EngineCancellation queuedDelivery;
        private final PrefillEndpoint prefill;
        private final DecodeEndpoint decode;
        private final CostBasedPrefillStrategy prefillStrategy;
        private final DecodeStrategy decodeStrategy;
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
                                    placement,
                                    org.mockito.Mockito.mock(
                                            org.flexlb.cache.match.CacheAwareService.class)));
            runtime =
                    spy(
                            new SchedulerTestServices(
                                    repository,
                                    endpoints,
                                    reporter,
                                    requestReporter,
                                    queuedDelivery,
                                    service,
                                    mock(org.flexlb.worker.rpc.EngineGrpcClient.class)));
            prefillStrategy =
                    org.mockito.Mockito.spy(
                            new CostBasedPrefillStrategy(
                                    service,
                                    endpoints,
                                    org.mockito.Mockito.mock(
                                            org.flexlb.cache.match.CacheAwareService.class),
                                    org.mockito.Mockito.mock(
                                            org.flexlb.cache.metrics.CacheMetricsReporter.class)));
            decodeStrategy =
                    org.mockito.Mockito.spy(
                            new DecodeStrategy(
                                    service,
                                    SchedulingTestConfig.capacity(service.loadBalanceConfig()),
                                    endpoints));
            var model = mock(ModelMetaConfig.class);
            when(model.requiredRoles())
                    .thenReturn(
                            withVit
                                    ? List.of(RoleType.PREFILL, RoleType.DECODE, RoleType.VIT)
                                    : List.of(RoleType.PREFILL, RoleType.DECODE));
            router =
                    new RequestRouteSelector(
                            endpoints,
                            prefillStrategy,
                            decodeStrategy,
                            new VitStrategy(
                                    org.mockito.Mockito.mock(
                                            org.flexlb.cache.VitCacheDirectory.class),
                                    endpoints),
                            org.mockito.Mockito.mock(EncoderStrategy.class),
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
                            org.flexlb.balance.schedule.SchedulerTestServices.scheduler(requests);
            WorkerStatus worker = worker(RoleType.PREFILL, "127.0.0.1");
            worker.lock.lock();
            try {
                var response = status(worker, Map.of(), Map.of());
                prefill =
                        (PrefillEndpoint)
                                endpoints
                                        .updateEndpoint(
                                                worker.getLogicalIpPort(),
                                                worker,
                                                worker.prepareStatus(
                                                        WorkerStatusTestSupport
                                                                .freezeStatusResponse(
                                                                        worker, response)))
                                        .endpoint();
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
                                    SelectionTestSupport.selectionResult(
                                            call, prefillSelection(prefill)))
                    .when(prefillStrategy)
                    .chooseBest(
                            any(),
                            org.mockito.ArgumentMatchers.anyList(),
                            org.mockito.ArgumentMatchers.any());
            org.mockito.Mockito.doAnswer(
                            call ->
                                    SelectionTestSupport.selectionResult(
                                            call,
                                            RoutingResult.success(decode.selectionSnapshot())))
                    .when(decodeStrategy)
                    .chooseBest(any(), org.mockito.ArgumentMatchers.anyList());
            doAnswer(
                            call -> {
                                var capturedViews =
                                        new java.util.ArrayList<
                                                org.flexlb.worker.endpoint.DecodeEndpoint
                                                        .SelectionSnapshot>();

                                capturedViews.add(decode.selectionSnapshot());
                                return org.flexlb.worker.endpoint.EndpointTestSupport
                                        .decodeCandidates(capturedViews);
                            })
                    .when(endpoints)
                    .endpoints(
                            org.mockito.ArgumentMatchers.eq(
                                    org.flexlb.config.service.RoleType.DECODE),
                            org.mockito.ArgumentMatchers.any());
            doAnswer(call -> decode)
                    .when(endpoints)
                    .getEndpoint(
                            org.mockito.ArgumentMatchers.eq(
                                    org.flexlb.config.service.RoleType.DECODE),
                            org.mockito.ArgumentMatchers.anyString());
        }

        @Override
        public void close() {
            try {
                org.flexlb.worker.endpoint.EndpointTestSupport.closeAndWait(decode);
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
                                endpoints
                                        .updateEndpoint(
                                                worker.getLogicalIpPort(),
                                                worker,
                                                worker.prepareStatus(observation))
                                        .endpoint();
            } finally {
                worker.lock.unlock();
            }
            bindPrefill(replacement);
            org.mockito.Mockito.doAnswer(
                            call ->
                                    SelectionTestSupport.selectionResult(
                                            call, prefillSelection(replacement)))
                    .when(prefillStrategy)
                    .chooseBest(
                            any(),
                            org.mockito.ArgumentMatchers.anyList(),
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
            org.mockito.Mockito.doReturn(List.of(selected))
                    .when(endpoints)
                    .<org.flexlb.worker.endpoint.PrefillEndpoint>endpoints(
                            org.mockito.ArgumentMatchers.eq(RoleType.PREFILL),
                            org.mockito.ArgumentMatchers.any());
            doAnswer(call -> selected).when(endpoints).getEndpoint(RoleType.PREFILL, address);
        }

        private RoutingResult<CostBasedPrefillStrategy.Selection, CapacityKey> prefillSelection(
                PrefillEndpoint selected) {
            return RoutingResult.success(
                    new CostBasedPrefillStrategy.Selection(
                            selected.ipPort(),
                            selected.getStatus().getGenerationId(),
                            30_000L,
                            0L,
                            0L,
                            0L,
                            null,
                            selected,
                            null,
                            selected.selectionSnapshot()));
        }

        private RequestContext context(long requestId) {
            var context = RequestProtocolTestSupport.context(config, requestId);
            context.setRequest(context.getRequest().toBuilder().seqLen(32L).build());
            context.setRequest(context.getRequest().toBuilder().maxNewTokens(16).build());
            return context;
        }

        private void assertNoPrefillOwnership() {
            assertNoWaitingQueue();
            assertEquals(
                    0L,
                    org.flexlb.worker.endpoint.EndpointTestSupport.occupiedRequestCount(prefill));
        }

        private void assertNoWaitingQueue() {
            assertEquals(
                    0,
                    org.flexlb.balance.schedule.SchedulerTestServices.pendingDeliveryRequestCount(
                            SchedulerTestSupport.repository(requests)));
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
            WorkerEndpoint.StatusChange projection;
            worker.lock.lock();
            try {
                var response = status(worker, running, finished);
                projection =
                        endpoint.applyWorkerStatus(
                                worker,
                                worker.prepareStatus(
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
