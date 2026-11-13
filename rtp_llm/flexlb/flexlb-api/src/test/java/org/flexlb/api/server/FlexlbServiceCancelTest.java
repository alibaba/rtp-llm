package org.flexlb.api.server;

import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import org.flexlb.api.ServerScheduleLatencyRecorder;
import org.flexlb.api.protocol.FlexlbScheduleProtocol;
import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.schedule.AbstractRequestScheduler;
import org.flexlb.balance.schedule.CancelReason;
import org.flexlb.balance.schedule.RequestResult;
import org.flexlb.config.ConfigService;
import org.flexlb.master.MasterStatusService;
import org.flexlb.worker.status.ServiceMetricsReporter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.function.BiConsumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks local and forwarded cancellation results while preserving the authoritative request
 * lifecycle.
 */
class FlexlbServiceCancelTest {

    private AbstractRequestScheduler requestScheduler;
    private final AbstractRequestScheduler requestState = mock(AbstractRequestScheduler.class);
    private MasterStatusService consistencyService;
    private FlexlbGrpcForwarder forwarder;
    private FlexlbServiceImpl service;

    @BeforeEach
    void setUp() {
        requestScheduler = mock(AbstractRequestScheduler.class);
        consistencyService = mock(MasterStatusService.class);
        forwarder = mock(FlexlbGrpcForwarder.class);
        service =
                FlexlbServiceTestSupport.create(
                        requestScheduler,
                        org.flexlb.balance.schedule.SchedulerTestSupport.repository(requestState),
                        consistencyService,
                        mock(ServiceMetricsReporter.class),
                        forwarder,
                        mock(ConfigService.class),
                        mock(DeliveryMetricsReporter.class),
                        mock(ServerScheduleLatencyRecorder.class),
                        mock(RequestSchedulerReporter.class));
    }

    @Test
    void localCancelReturnsAuthoritativePendingLifecycle() {
        RequestResult pending = snapshot(101L, RequestResult.State.CANCEL_REQUESTED);
        when(requestScheduler.cancel(
                        "101",
                        CancelReason.DEADLINE_EXCEEDED,
                        org.flexlb.balance.schedule.RequestPhase.GENERATION))
                .thenReturn(pending);
        StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> observer =
                mock(StreamObserver.class);

        service.cancel(
                cancelRequest(
                        101L,
                        FlexlbScheduleProtocol.CancelReasonPB.CANCEL_REASON_DEADLINE_EXCEEDED),
                observer);

        org.mockito.ArgumentCaptor<FlexlbScheduleProtocol.FlexlbCancelResponsePB> response =
                org.mockito.ArgumentCaptor.forClass(
                        FlexlbScheduleProtocol.FlexlbCancelResponsePB.class);
        verify(observer).onNext(response.capture());
        verify(observer).onCompleted();
        verify(observer, never()).onError(any());
        assertTrue(response.getValue().getFound());
        assertEquals(
                FlexlbScheduleProtocol.RequestStatePB.REQUEST_STATE_CANCEL_REQUESTED,
                response.getValue().getLifecycle().getState());
    }

    @Test
    void localCancelPreservesExistingTerminalLifecycle() {
        RequestResult completed = snapshot(102L, RequestResult.State.COMPLETED);
        when(requestScheduler.cancel(
                        "102",
                        CancelReason.CLIENT_CANCELLED,
                        org.flexlb.balance.schedule.RequestPhase.GENERATION))
                .thenReturn(completed);
        StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> observer =
                mock(StreamObserver.class);

        service.cancel(
                cancelRequest(
                        102L, FlexlbScheduleProtocol.CancelReasonPB.CANCEL_REASON_CLIENT_CANCELLED),
                observer);

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
        when(requestScheduler.cancel(
                        "103",
                        CancelReason.CLIENT_CANCELLED,
                        org.flexlb.balance.schedule.RequestPhase.GENERATION))
                .thenReturn(null);
        StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> observer =
                mock(StreamObserver.class);

        service.cancel(
                FlexlbScheduleProtocol.FlexlbCancelRequestPB.newBuilder()
                        .setRequestId(103L)
                        .build(),
                observer);

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
                        104L, FlexlbScheduleProtocol.CancelReasonPB.CANCEL_REASON_CLIENT_CANCELLED);

        assertTimeoutPreemptively(Duration.ofSeconds(1), () -> service.cancel(request, observer));

        verify(observer, never()).onNext(any());
        FlexlbScheduleProtocol.FlexlbCancelResponsePB masterResponse =
                FlexlbScheduleProtocol.FlexlbCancelResponsePB.newBuilder().setFound(false).build();
        assertTrue(
                pending.complete(
                        FlexlbGrpcForwarder.CancelForwardResult.forwarded(
                                masterResponse, "10.0.0.2:7001")));

        verify(observer, times(1)).onNext(masterResponse);
        verify(observer, times(1)).onCompleted();
        verify(requestScheduler)
                .cancel(
                        "104",
                        CancelReason.CLIENT_CANCELLED,
                        org.flexlb.balance.schedule.RequestPhase.GENERATION);
    }

    @Test
    void attemptedForwardFailureNeverRetriesLocally() {
        when(consistencyService.isNeedConsistency()).thenReturn(true);
        when(consistencyService.isMaster()).thenReturn(false);
        when(forwarder.forwardCancelToMaster(any()))
                .thenReturn(
                        CompletableFuture.completedFuture(
                                FlexlbGrpcForwarder.CancelForwardResult.failed(
                                        "UNAVAILABLE", "10.0.0.2:7001")));
        StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> observer =
                mock(StreamObserver.class);

        service.cancel(
                cancelRequest(
                        105L, FlexlbScheduleProtocol.CancelReasonPB.CANCEL_REASON_CLIENT_CANCELLED),
                observer);

        org.mockito.ArgumentCaptor<Throwable> error =
                org.mockito.ArgumentCaptor.forClass(Throwable.class);
        verify(observer).onError(error.capture());
        assertEquals(Status.Code.UNAVAILABLE, Status.fromThrowable(error.getValue()).getCode());
        verify(observer, never()).onNext(any());
        verify(requestScheduler, times(1))
                .cancel(
                        "105",
                        CancelReason.CLIENT_CANCELLED,
                        org.flexlb.balance.schedule.RequestPhase.GENERATION);
    }

    @Test
    void noSelectedMasterReturnsInitialLocalMissWithoutRetry() {
        when(consistencyService.isNeedConsistency()).thenReturn(true);
        when(consistencyService.isMaster()).thenReturn(false);
        when(forwarder.forwardCancelToMaster(any()))
                .thenReturn(
                        CompletableFuture.completedFuture(
                                FlexlbGrpcForwarder.CancelForwardResult.noMaster()));
        StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> observer =
                mock(StreamObserver.class);

        service.cancel(
                cancelRequest(
                        106L, FlexlbScheduleProtocol.CancelReasonPB.CANCEL_REASON_CLIENT_CANCELLED),
                observer);

        verify(requestScheduler, times(1))
                .cancel(
                        "106",
                        CancelReason.CLIENT_CANCELLED,
                        org.flexlb.balance.schedule.RequestPhase.GENERATION);
        org.mockito.ArgumentCaptor<FlexlbScheduleProtocol.FlexlbCancelResponsePB> response =
                org.mockito.ArgumentCaptor.forClass(
                        FlexlbScheduleProtocol.FlexlbCancelResponsePB.class);
        verify(observer).onNext(response.capture());
        verify(observer).onCompleted();
        assertFalse(response.getValue().getFound());
        assertFalse(response.getValue().hasLifecycle());
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
                        .setLifecycle(
                                FlexlbScheduleProtocol.RequestLifecyclePB.newBuilder()
                                        .setRequestId(107L)
                                        .setState(
                                                FlexlbScheduleProtocol.RequestStatePB
                                                        .REQUEST_STATE_CANCEL_REQUESTED))
                        .build();
        FlexlbGrpcForwarder.CancelForwardResult result =
                FlexlbGrpcForwarder.CancelForwardResult.forwarded(masterResponse, "10.0.0.2:7001");
        when(unusualStage.whenComplete(any()))
                .thenAnswer(
                        invocation -> {
                            BiConsumer<FlexlbGrpcForwarder.CancelForwardResult, Throwable>
                                    callback = invocation.getArgument(0);
                            callback.accept(result, null);
                            throw new IllegalStateException("registration failed after callback");
                        });
        when(forwarder.forwardCancelToMaster(any())).thenReturn(unusualStage);
        StreamObserver<FlexlbScheduleProtocol.FlexlbCancelResponsePB> observer =
                mock(StreamObserver.class);

        service.cancel(
                cancelRequest(
                        107L, FlexlbScheduleProtocol.CancelReasonPB.CANCEL_REASON_CLIENT_CANCELLED),
                observer);

        verify(observer, times(1)).onNext(masterResponse);
        verify(observer, times(1)).onCompleted();
        verify(observer, never()).onError(any());
    }

    private static FlexlbScheduleProtocol.FlexlbCancelRequestPB cancelRequest(
            long requestId, FlexlbScheduleProtocol.CancelReasonPB reason) {
        return FlexlbScheduleProtocol.FlexlbCancelRequestPB.newBuilder()
                .setRequestId(requestId)
                .setReason(reason)
                .build();
    }

    private static RequestResult snapshot(long requestId, RequestResult.State phase) {
        return new RequestResult(
                String.valueOf(requestId),
                requestId,
                org.flexlb.balance.schedule.RequestPhase.GENERATION,
                phase,
                1000L,
                1001L,
                "test");
    }
}
