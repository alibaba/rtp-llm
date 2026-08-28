package org.flexlb.httpserver;

import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.DeliveryClaimKind;
import org.flexlb.balance.scheduler.RequestState;

import org.flexlb.config.ConfigService;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.function.BiConsumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class FlexlbServiceCancelTest {

    private RouteService routeService;
    private LBStatusConsistencyService consistencyService;
    private FlexlbGrpcForwarder forwarder;
    private FlexlbServiceImpl service;

    @BeforeEach
    void setUp() {
        routeService = mock(RouteService.class);
        consistencyService = mock(LBStatusConsistencyService.class);
        forwarder = mock(FlexlbGrpcForwarder.class);
        service = new FlexlbServiceImpl(
                routeService,
                consistencyService,
                mock(EngineHealthReporter.class),
                mock(ActiveRequestCounter.class),
                forwarder,
                mock(ConfigService.class),
                mock(BatchSchedulerReporter.class),
                mock(ServerScheduleLatencyRecorder.class),
                mock(RequestSchedulerReporter.class));
    }

    @Test
    void localCancelReturnsAuthoritativePendingLifecycle() {
        RequestState pending = snapshot(
                101L, RequestState.Phase.CANCEL_REQUESTED, 301L);
        when(routeService.cancelRequest(
                101L, 301L, CancelReason.DEADLINE_EXCEEDED))
                .thenReturn(pending);
        StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> observer =
                mock(StreamObserver.class);

        service.cancel(cancelRequest(
                101L,
                301L,
                FlexlbScheduleProtocol.CancelReasonPB
                        .CANCEL_REASON_DEADLINE_EXCEEDED), observer);

        org.mockito.ArgumentCaptor<FlexlbScheduleProtocol.FlexlbCancelResponsePB> response =
                org.mockito.ArgumentCaptor.forClass(
                        FlexlbScheduleProtocol.FlexlbCancelResponsePB.class);
        verify(observer).onNext(response.capture());
        verify(observer).onCompleted();
        verify(observer, never()).onError(any());
        assertTrue(response.getValue().getFound());
        assertEquals(
                FlexlbScheduleProtocol.RequestStatePB
                        .REQUEST_STATE_CANCEL_REQUESTED,
                response.getValue().getLifecycle().getState());
        assertEquals(301L, response.getValue().getLifecycle().getBatchId());
    }

    @Test
    void localCancelPreservesExistingTerminalLifecycle() {
        RequestState completed = snapshot(
                102L, RequestState.Phase.COMPLETED, 0);
        when(routeService.cancelRequest(
                102L, 0, CancelReason.CLIENT_CANCELLED))
                .thenReturn(completed);
        StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> observer =
                mock(StreamObserver.class);

        service.cancel(cancelRequest(
                102L,
                0,
                FlexlbScheduleProtocol.CancelReasonPB
                        .CANCEL_REASON_CLIENT_CANCELLED), observer);

        org.mockito.ArgumentCaptor<FlexlbScheduleProtocol.FlexlbCancelResponsePB> response =
                org.mockito.ArgumentCaptor.forClass(
                        FlexlbScheduleProtocol.FlexlbCancelResponsePB.class);
        verify(observer).onNext(response.capture());
        assertTrue(response.getValue().getFound());
        assertEquals(
                FlexlbScheduleProtocol.RequestStatePB.REQUEST_STATE_COMPLETED,
                response.getValue().getLifecycle().getState());
    }

    @Test
    void unknownRequestIsTheOnlyLocalNotFoundResponse() {
        when(routeService.cancelRequest(
                103L, 0, CancelReason.CLIENT_CANCELLED))
                .thenReturn(null);
        StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> observer =
                mock(StreamObserver.class);

        service.cancel(FlexlbScheduleProtocol.FlexlbCancelRequestPB.newBuilder()
                .setRequestId(103L)
                .build(), observer);

        org.mockito.ArgumentCaptor<FlexlbScheduleProtocol.FlexlbCancelResponsePB> response =
                org.mockito.ArgumentCaptor.forClass(
                        FlexlbScheduleProtocol.FlexlbCancelResponsePB.class);
        verify(observer).onNext(response.capture());
        verify(observer).onCompleted();
        assertFalse(response.getValue().getFound());
        assertFalse(response.getValue().hasLifecycle());
    }

    @Test
    void pendingForwardReturnsImmediatelyAndMasterNotFoundIsAuthoritative() {
        when(consistencyService.isNeedConsistency()).thenReturn(true);
        when(consistencyService.isMaster()).thenReturn(false);
        CompletableFuture<FlexlbGrpcForwarder.CancelForwardResult> pending =
                new CompletableFuture<>();
        when(forwarder.forwardCancelToMaster(any())).thenReturn(pending);
        StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> observer =
                mock(StreamObserver.class);
        FlexlbScheduleProtocol.FlexlbCancelRequestPB request =
                cancelRequest(
                        104L,
                        0,
                        FlexlbScheduleProtocol.CancelReasonPB
                                .CANCEL_REASON_CLIENT_CANCELLED);

        assertTimeoutPreemptively(
                Duration.ofSeconds(1),
                () -> service.cancel(request, observer));

        verify(observer, never()).onNext(any());
        FlexlbScheduleProtocol.FlexlbCancelResponsePB masterResponse =
                FlexlbScheduleProtocol.FlexlbCancelResponsePB.newBuilder()
                        .setFound(false)
                        .build();
        assertTrue(pending.complete(
                FlexlbGrpcForwarder.CancelForwardResult.forwarded(
                        masterResponse, "10.0.0.2:7001")));

        verify(observer, times(1)).onNext(masterResponse);
        verify(observer, times(1)).onCompleted();
        verify(routeService, never()).cancelRequest(
                anyLong(), anyLong(), any(CancelReason.class));
    }

    @Test
    void attemptedForwardFailureNeverFallsBackLocally() {
        when(consistencyService.isNeedConsistency()).thenReturn(true);
        when(consistencyService.isMaster()).thenReturn(false);
        when(forwarder.forwardCancelToMaster(any())).thenReturn(
                CompletableFuture.completedFuture(
                        FlexlbGrpcForwarder.CancelForwardResult.failed(
                                "UNAVAILABLE", "10.0.0.2:7001")));
        StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> observer =
                mock(StreamObserver.class);

        service.cancel(cancelRequest(
                105L,
                0,
                FlexlbScheduleProtocol.CancelReasonPB
                        .CANCEL_REASON_CLIENT_CANCELLED), observer);

        org.mockito.ArgumentCaptor<Throwable> error =
                org.mockito.ArgumentCaptor.forClass(Throwable.class);
        verify(observer).onError(error.capture());
        assertEquals(Status.Code.UNAVAILABLE,
                Status.fromThrowable(error.getValue()).getCode());
        verify(observer, never()).onNext(any());
        verify(routeService, never()).cancelRequest(
                anyLong(), anyLong(), any(CancelReason.class));
    }

    @Test
    void noSelectedMasterFallsBackBeforeAnyRpcWasAttempted() {
        when(consistencyService.isNeedConsistency()).thenReturn(true);
        when(consistencyService.isMaster()).thenReturn(false);
        when(forwarder.forwardCancelToMaster(any())).thenReturn(
                CompletableFuture.completedFuture(
                        FlexlbGrpcForwarder.CancelForwardResult.noMaster()));
        when(routeService.cancelRequest(
                106L, 0, CancelReason.CLIENT_CANCELLED))
                .thenReturn(snapshot(106L, RequestState.Phase.CANCELLED, 0));
        StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> observer =
                mock(StreamObserver.class);

        service.cancel(cancelRequest(
                106L,
                0,
                FlexlbScheduleProtocol.CancelReasonPB
                        .CANCEL_REASON_CLIENT_CANCELLED), observer);

        verify(routeService).cancelRequest(
                106L, 0, CancelReason.CLIENT_CANCELLED);
        verify(observer).onNext(any());
        verify(observer).onCompleted();
    }

    @Test
    @SuppressWarnings("unchecked")
    void callbackRegistrationAfterCompletionStillRespondsExactlyOnce() {
        when(consistencyService.isNeedConsistency()).thenReturn(true);
        when(consistencyService.isMaster()).thenReturn(false);
        CompletionStage<FlexlbGrpcForwarder.CancelForwardResult> unusualStage =
                mock(CompletionStage.class);
        FlexlbScheduleProtocol.FlexlbCancelResponsePB masterResponse =
                FlexlbScheduleProtocol.FlexlbCancelResponsePB.newBuilder()
                        .setFound(true)
                        .setLifecycle(FlexlbScheduleProtocol.RequestLifecyclePB.newBuilder()
                                .setRequestId(107L)
                                .setState(FlexlbScheduleProtocol.RequestStatePB
                                        .REQUEST_STATE_CANCEL_REQUESTED))
                        .build();
        FlexlbGrpcForwarder.CancelForwardResult result =
                FlexlbGrpcForwarder.CancelForwardResult.forwarded(
                        masterResponse, "10.0.0.2:7001");
        when(unusualStage.whenComplete(any())).thenAnswer(invocation -> {
            BiConsumer<FlexlbGrpcForwarder.CancelForwardResult, Throwable> callback =
                    invocation.getArgument(0);
            callback.accept(result, null);
            throw new IllegalStateException("registration failed after callback");
        });
        when(forwarder.forwardCancelToMaster(any())).thenReturn(unusualStage);
        StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> observer =
                mock(StreamObserver.class);

        service.cancel(cancelRequest(
                107L,
                0,
                FlexlbScheduleProtocol.CancelReasonPB
                        .CANCEL_REASON_CLIENT_CANCELLED), observer);

        verify(observer, times(1)).onNext(masterResponse);
        verify(observer, times(1)).onCompleted();
        verify(observer, never()).onError(any());
    }

    private static FlexlbScheduleProtocol.FlexlbCancelRequestPB cancelRequest(
            long requestId,
            long batchId,
            FlexlbScheduleProtocol.CancelReasonPB reason) {
        return FlexlbScheduleProtocol.FlexlbCancelRequestPB.newBuilder()
                .setRequestId(requestId)
                .setBatchId(batchId)
                .setReason(reason)
                .build();
    }

    private static RequestState snapshot(
            long requestId,
            RequestState.Phase state,
            long batchId) {
        return new RequestState(
                requestId, state,
                batchId > 0 ? DeliveryClaimKind.BATCH_ENQUEUE : DeliveryClaimKind.NONE,
                batchId, 1L, 2L, state.name());
    }
}
