package org.flexlb.service.monitor;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.TtftStagesConfig;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.client.RequestTimingClient;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.timing.grpc.PullRequestTimingsRequest;
import org.flexlb.timing.grpc.PullRequestTimingsResponse;
import org.flexlb.timing.grpc.RequestTimingEventKind;
import org.flexlb.timing.grpc.RequestTimingRecord;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.mockito.ArgumentCaptor;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class RequestTimingPollerTest {
    private final ConfigService configs = mock(ConfigService.class);
    private final FlexlbConfig config = new FlexlbConfig();
    private final LBStatusConsistencyService leadership = mock(LBStatusConsistencyService.class);
    private final RequestTimingClient client = mock(RequestTimingClient.class);
    private final RequestTimingDiscovery discovery = mock(RequestTimingDiscovery.class);
    private final GrpcTarget target = new GrpcTarget("daemon", 8080);
    private RequestTimingPoller poller;
    private ScheduledThreadPoolExecutor executor;

    @BeforeEach
    void setUp() {
        when(configs.loadBalanceConfig()).thenReturn(config);
        when(leadership.isNeedConsistency()).thenReturn(true);
        when(leadership.isMaster()).thenReturn(true);
        when(leadership.getLeadershipGeneration()).thenReturn(1L);
        when(discovery.sources()).thenReturn(Set.of(new RequestTimingSettings.Source(target, "frontend")));
        poller = new RequestTimingPoller(configs, leadership, client, mock(FlexMonitor.class), discovery);
        executor = (ScheduledThreadPoolExecutor) ReflectionTestUtils.getField(poller, "executor");
    }

    @AfterEach
    void close() throws Exception {
        poller.close();
        executor.awaitTermination(5, TimeUnit.SECONDS);
    }

    @Test
    void doesNotPullWithoutConfirmedLeadershipOrWhenDisabled() throws Exception {
        when(leadership.isMaster()).thenReturn(false);
        tick();
        when(leadership.isMaster()).thenReturn(true);
        config.getObservability().getTtftStages().setEnabled(false);
        tick();
        verifyNoInteractions(discovery, client);
    }

    @Test
    void hasOnlyOneOutstandingPullAndAdvancesCursorAfterValidResponse() throws Exception {
        CompletableFuture<PullRequestTimingsResponse> reply = new CompletableFuture<>();
        when(client.pull(any(), any(), anyLong())).thenReturn(reply, new CompletableFuture<>());
        tick();
        tick();
        verify(client, times(1)).pull(eq(target), any(), eq(800L));
        reply.complete(response("epoch-a", 4, false));
        drain();
        tick();
        ArgumentCaptor<PullRequestTimingsRequest> requests = ArgumentCaptor.forClass(PullRequestTimingsRequest.class);
        verify(client, times(2)).pull(eq(target), requests.capture(), eq(800L));
        assertEquals("", requests.getAllValues().get(0).getEpoch());
        assertEquals("epoch-a", requests.getAllValues().get(1).getEpoch());
        assertEquals(4, requests.getAllValues().get(1).getAfterSequence());
    }

    @Test
    void clearsMissingSourceHistoryBeforeAddingGapRecords() throws Exception {
        CompletableFuture<PullRequestTimingsResponse> reply = new CompletableFuture<>();
        when(client.pull(any(), any(), anyLong())).thenReturn(reply);
        tick();
        TtftStageAggregator aggregator = mock(TtftStageAggregator.class);
        ReflectionTestUtils.setField(poller, "aggregator", aggregator);
        reply.complete(response("epoch-b", 4, true));
        drain();
        var order = org.mockito.Mockito.inOrder(aggregator);
        order.verify(aggregator).clearSource("daemon/frontend/daemon:8080");
        order.verify(aggregator)
                .addTiming("daemon/frontend/daemon:8080", "request", "frontend", "deployment", 1_000_000L, null);
    }

    @Test
    void discardsOldLeadershipRepliesEvenWhenNoFollowerTickWasObserved() throws Exception {
        CompletableFuture<PullRequestTimingsResponse> old = new CompletableFuture<>();
        CompletableFuture<PullRequestTimingsResponse> current = new CompletableFuture<>();
        when(client.pull(any(), any(), anyLong())).thenReturn(old, current);
        tick();
        when(leadership.getLeadershipGeneration()).thenReturn(2L);
        tick();
        TtftStageAggregator aggregator = mock(TtftStageAggregator.class);
        ReflectionTestUtils.setField(poller, "aggregator", aggregator);
        old.complete(response("old-epoch", 4, false));
        drain();
        verifyNoInteractions(aggregator);
        current.complete(response("new-epoch", 4, false));
        drain();
        verify(aggregator).addTiming(any(), eq("request"), eq("frontend"), eq("deployment"), anyLong(), any());
    }

    @Test
    void disableDiscardsInFlightRepliesAndClosesChannels() throws Exception {
        CompletableFuture<PullRequestTimingsResponse> reply = new CompletableFuture<>();
        when(client.pull(any(), any(), anyLong())).thenReturn(reply);
        tick();
        TtftStageAggregator aggregator = mock(TtftStageAggregator.class);
        ReflectionTestUtils.setField(poller, "aggregator", aggregator);
        config.getObservability().getTtftStages().setEnabled(false);
        tick();
        reply.complete(response("epoch", 4, false));
        drain();
        verify(aggregator).resetAll();
        verify(aggregator, never()).addTiming(any(), any(), any(), any(), anyLong(), any());
        verify(client).remove(target);
    }

    @Test
    void wrongRoleIsSkippedWhileCursorStillAdvances() throws Exception {
        CompletableFuture<PullRequestTimingsResponse> reply = new CompletableFuture<>();
        when(client.pull(any(), any(), anyLong())).thenReturn(reply, new CompletableFuture<>());
        tick();
        TtftStageAggregator aggregator = mock(TtftStageAggregator.class);
        ReflectionTestUtils.setField(poller, "aggregator", aggregator);
        var wrongRole = response("epoch", 4, false).toBuilder();
        wrongRole.setRecords(0, wrongRole.getRecords(0).toBuilder().setRole("prefill"));
        reply.complete(wrongRole.build());
        drain();
        tick();
        ArgumentCaptor<PullRequestTimingsRequest> requests = ArgumentCaptor.forClass(PullRequestTimingsRequest.class);
        verify(client, times(2)).pull(any(), requests.capture(), anyLong());
        assertEquals(4, requests.getAllValues().get(1).getAfterSequence());
        verify(aggregator, never()).addTiming(any(), any(), any(), any(), anyLong(), any());
    }

    @Test
    void forwardsFinishedFailedPrefillAttemptWithoutRequiringFirstToken() throws Exception {
        when(client.pull(any(), any(), anyLong())).thenReturn(new CompletableFuture<>());
        tick();
        TtftStageAggregator aggregator = mock(TtftStageAggregator.class);
        ReflectionTestUtils.setField(poller, "aggregator", aggregator);
        WorkerStatus worker = mock(WorkerStatus.class);
        when(worker.getMetricIpPort()).thenReturn("engine:8001");
        when(worker.getGenerationId()).thenReturn(3L);
        WorkerStatus.StatusObservation status = mock(WorkerStatus.StatusObservation.class);
        WorkerStatus.TaskObservation failed = mock(WorkerStatus.TaskObservation.class);
        when(failed.endTimeMs()).thenReturn(1234L);
        when(status.role()).thenReturn(RoleType.PREFILL);
        when(status.finishedTasks()).thenReturn(Map.of("attempt", failed));
        poller.addWorkerStatus(worker, status);
        drain();
        verify(aggregator).addWorkerStatus("worker/engine:8001/3", failed);
    }

    @Test
    void equivalentHotUpdateKeepsInFlightPullAndContinuesCollection() throws Exception {
        CompletableFuture<PullRequestTimingsResponse> first = new CompletableFuture<>();
        when(client.pull(any(), any(), anyLong())).thenReturn(first, new CompletableFuture<>());
        tick();
        TtftStagesConfig replacement = new TtftStagesConfig();
        replacement.setEnabled(true);
        config.getObservability().setTtftStages(replacement);
        tick();
        first.complete(response("epoch", 4, false));
        drain();
        tick();
        ArgumentCaptor<PullRequestTimingsRequest> requests = ArgumentCaptor.forClass(PullRequestTimingsRequest.class);
        verify(client, times(2)).pull(any(), requests.capture(), anyLong());
        assertEquals(4, requests.getAllValues().get(1).getAfterSequence());
        verify(client, never()).remove(any());
    }

    @Test
    void drainsAtMostEightPagesPerTickAndPublishesReadyRequestsIndependently() throws Exception {
        when(client.pull(any(), any(), anyLong())).thenAnswer(invocation -> {
            PullRequestTimingsRequest request = invocation.getArgument(1);
            return CompletableFuture.completedFuture(
                    response("epoch", request.getAfterSequence() + 1, false).toBuilder().setHasMore(true).build());
        });
        tick();
        for (int page = 0; page < 10; page++) {
            drain();
        }
        verify(client, times(8)).pull(any(), any(), anyLong());
        TtftStageAggregator aggregator = mock(TtftStageAggregator.class);
        ReflectionTestUtils.setField(poller, "aggregator", aggregator);
        tick();
        for (int page = 0; page < 10; page++) {
            drain();
        }
        verify(client, times(16)).pull(any(), any(), anyLong());
        verify(aggregator, times(1)).publishReady();
    }

    @Test
    void rejectsUnannouncedSequenceGapWithoutAdvancingCursor() throws Exception {
        CompletableFuture<PullRequestTimingsResponse> first = new CompletableFuture<>();
        CompletableFuture<PullRequestTimingsResponse> skipped = new CompletableFuture<>();
        when(client.pull(any(), any(), anyLong())).thenReturn(first, skipped, new CompletableFuture<>());
        tick();
        first.complete(response("epoch", 4, false));
        drain();
        tick();
        skipped.complete(response("epoch", 6, false));
        drain();
        tick();
        ArgumentCaptor<PullRequestTimingsRequest> requests = ArgumentCaptor.forClass(PullRequestTimingsRequest.class);
        verify(client, times(3)).pull(any(), requests.capture(), anyLong());
        assertEquals(4, requests.getAllValues().get(2).getAfterSequence());
    }

    @Test
    void pendingAndFailedSourcesDoNotGatePeriodicPublication() throws Exception {
        CompletableFuture<PullRequestTimingsResponse> pending = new CompletableFuture<>();
        when(client.pull(any(), any(), anyLong())).thenReturn(pending,
                CompletableFuture.failedFuture(new IllegalStateException("daemon timeout")));
        tick();
        TtftStageAggregator aggregator = mock(TtftStageAggregator.class);
        ReflectionTestUtils.setField(poller, "aggregator", aggregator);
        tick();
        pending.completeExceptionally(new IllegalStateException("daemon timeout"));
        drain();
        tick();
        drain();
        verify(aggregator, times(2)).publishReady();
        verify(aggregator, never()).resetAll();
    }

    @Test
    void addingSourcePreservesExistingCursorAndStartsOnlyNewSourceAtZero() throws Exception {
        var original = new RequestTimingSettings.Source(target, "frontend");
        var added = new RequestTimingSettings.Source(new GrpcTarget("new-prefill", 8080), "prefill");
        CompletableFuture<PullRequestTimingsResponse> first = new CompletableFuture<>();
        when(client.pull(eq(target), any(), anyLong())).thenReturn(first, new CompletableFuture<>());
        when(client.pull(eq(added.target()), any(), anyLong())).thenReturn(new CompletableFuture<>());
        tick();
        first.complete(response("epoch", 4, false));
        drain();
        when(discovery.sources()).thenReturn(Set.of(original, added));
        tick();
        ArgumentCaptor<PullRequestTimingsRequest> oldRequests =
                ArgumentCaptor.forClass(PullRequestTimingsRequest.class);
        verify(client, times(2)).pull(eq(target), oldRequests.capture(), anyLong());
        assertEquals(4, oldRequests.getAllValues().get(1).getAfterSequence());
        ArgumentCaptor<PullRequestTimingsRequest> newRequest = ArgumentCaptor.forClass(PullRequestTimingsRequest.class);
        verify(client).pull(eq(added.target()), newRequest.capture(), anyLong());
        assertEquals(0, newRequest.getValue().getAfterSequence());
        assertEquals("", newRequest.getValue().getEpoch());
        verify(client, never()).remove(target);
    }

    @Test
    void removedSourceReplyCannotPopulateNewCursorAfterSameAddressReturns() throws Exception {
        var original = new RequestTimingSettings.Source(target, "frontend");
        var removed = new RequestTimingSettings.Source(new GrpcTarget("prefill", 8080), "prefill");
        when(discovery.sources()).thenReturn(Set.of(original, removed));
        when(client.pull(eq(target), any(), anyLong())).thenReturn(new CompletableFuture<>());
        CompletableFuture<PullRequestTimingsResponse> stale = new CompletableFuture<>();
        CompletableFuture<PullRequestTimingsResponse> replacement = new CompletableFuture<>();
        when(client.pull(eq(removed.target()), any(), anyLong())).thenReturn(stale, replacement);
        tick();
        TtftStageAggregator aggregator = mock(TtftStageAggregator.class);
        ReflectionTestUtils.setField(poller, "aggregator", aggregator);
        when(discovery.sources()).thenReturn(Set.of(original));
        tick();
        verify(client).remove(removed.target());
        verify(client, never()).remove(target);
        when(discovery.sources()).thenReturn(Set.of(original, removed));
        tick();
        verify(aggregator, never()).resetAll();
        verify(aggregator, never()).clearSource(any());
        stale.complete(prefillResponse("retired", 7));
        drain();
        verify(aggregator, never()).addTiming(any(), any(), any(), any(), anyLong(), any());
        replacement.complete(prefillResponse("new", 1));
        drain();
        verify(aggregator).addTiming(eq(removed.key()), eq("request"), eq("prefill"), eq("deployment"), eq(1_000_000L),
                eq(null));
    }

    @Test
    void skipsInvalidDispatchEventsAcrossPagesAndPreservesStartAndFailureTimes() throws Exception {
        CompletableFuture<PullRequestTimingsResponse> first = new CompletableFuture<>();
        CompletableFuture<PullRequestTimingsResponse> second = new CompletableFuture<>();
        when(client.pull(any(), any(), anyLong())).thenReturn(first, second, new CompletableFuture<>());
        tick();
        TtftStageAggregator aggregator = mock(TtftStageAggregator.class);
        ReflectionTestUtils.setField(poller, "aggregator", aggregator);
        first.complete(PullRequestTimingsResponse.newBuilder()
                .setEpoch("epoch")
                .setNextSequence(3)
                .setHasMore(true)
                .addRecords(frontendRecord(1, "valid-first"))
                .addRecords(frontendRecord(2, "wrong-role").setRole("prefill"))
                .addRecords(frontendRecord(3, "valid-second"))
                .build());
        drain();
        second.complete(PullRequestTimingsResponse.newBuilder()
                .setEpoch("epoch")
                .setNextSequence(19)
                .addRecords(frontendRecord(4, ""))
                .addRecords(frontendRecord(5, "missing-deployment").setDeploymentName(" "))
                .addRecords(frontendRecord(6, "zero-start").setDispatchStartUs(0))
                .addRecords(frontendRecord(7, "negative-start").setDispatchStartUs(-1))
                .addRecords(frontendRecord(8, "completed-with-p-receipt").setReceivedTimeUs(1_000_001))
                .addRecords(frontendRecord(9, "zero-failure").setEventKind(RequestTimingEventKind.DISPATCH_FAILED)
                        .setFailedAtUs(0))
                .addRecords(frontendRecord(10, "negative-failure").setEventKind(RequestTimingEventKind.DISPATCH_FAILED)
                        .setFailedAtUs(-1))
                .addRecords(
                        frontendRecord(11, "failure-before-start").setEventKind(RequestTimingEventKind.DISPATCH_FAILED)
                                .setFailedAtUs(999_999))
                .addRecords(frontendRecord(12, "valid-last").setDispatchStartUs(Long.MAX_VALUE))
                .addRecords(frontendRecord(13, "failed-at-start").setEventKind(RequestTimingEventKind.DISPATCH_FAILED)
                        .setFailedAtUs(1_000_000))
                .addRecords(frontendRecord(14, "failed-later").setEventKind(RequestTimingEventKind.DISPATCH_FAILED)
                        .setFailedAtUs(1_010_000))
                .addRecords(frontendRecord(15, "failure-with-p-receipt")
                        .setEventKind(RequestTimingEventKind.DISPATCH_FAILED)
                        .setFailedAtUs(1_010_000)
                        .setReceivedTimeUs(1_000_001))
                .addRecords(frontendRecord(16, "completed-with-failure").setFailedAtUs(1_010_000))
                .addRecords(
                        frontendRecord(17, "failure-without-start").setEventKind(RequestTimingEventKind.DISPATCH_FAILED)
                                .clearDispatchStartUs()
                                .setFailedAtUs(1_010_000))
                .addRecords(frontendRecord(18, "unknown-event").setEventKindValue(99))
                .addRecords(frontendRecord(19, "missing-event").clearEventKind())
                .build());
        drain();
        tick();
        ArgumentCaptor<String> ids = ArgumentCaptor.forClass(String.class);
        verify(aggregator, times(5)).addTiming(any(), ids.capture(), eq("frontend"), eq("deployment"), anyLong(),
                any());
        assertEquals(List.of("valid-first", "valid-second", "valid-last", "failed-at-start", "failed-later"),
                ids.getAllValues());
        verify(aggregator).addTiming("daemon/frontend/daemon:8080", "valid-first", "frontend", "deployment", 1_000_000L,
                null);
        verify(aggregator).addTiming("daemon/frontend/daemon:8080", "failed-at-start", "frontend", "deployment",
                1_000_000L, 1_000_000L);
        verify(aggregator).addTiming("daemon/frontend/daemon:8080", "failed-later", "frontend", "deployment",
                1_000_000L, 1_010_000L);
        ArgumentCaptor<PullRequestTimingsRequest> requests = ArgumentCaptor.forClass(PullRequestTimingsRequest.class);
        verify(client, times(3)).pull(any(), requests.capture(), anyLong());
        assertEquals(List.of(0L, 3L, 19L),
                requests.getAllValues().stream().map(PullRequestTimingsRequest::getAfterSequence).toList());
        assertEquals("epoch", requests.getAllValues().get(2).getEpoch());
    }

    @Test
    void prefillAcceptsOnlyReceiptEventsWithoutDispatchOrFailureFields() throws Exception {
        when(discovery.sources()).thenReturn(Set.of(new RequestTimingSettings.Source(target, "prefill")));
        CompletableFuture<PullRequestTimingsResponse> reply = new CompletableFuture<>();
        when(client.pull(any(), any(), anyLong())).thenReturn(reply, new CompletableFuture<>());
        tick();
        TtftStageAggregator aggregator = mock(TtftStageAggregator.class);
        ReflectionTestUtils.setField(poller, "aggregator", aggregator);
        reply.complete(PullRequestTimingsResponse.newBuilder()
                .setEpoch("prefill-epoch")
                .setNextSequence(6)
                .addRecords(prefillRecord(1, "receipt-with-dispatch").setDispatchStartUs(1_000_000))
                .addRecords(prefillRecord(2, "receipt-with-failure").setFailedAtUs(1_010_000))
                .addRecords(prefillRecord(3, "wrong-event").setEventKind(RequestTimingEventKind.DISPATCH_COMPLETED))
                .addRecords(prefillRecord(4, "valid-prefill"))
                .addRecords(prefillRecord(5, "zero-received").setReceivedTimeUs(0))
                .addRecords(prefillRecord(6, "negative-received").setReceivedTimeUs(-1))
                .build());
        drain();
        tick();
        verify(aggregator, times(1)).addTiming(any(), any(), any(), any(), anyLong(), any());
        verify(aggregator).addTiming("daemon/prefill/daemon:8080", "valid-prefill", "prefill", "deployment", 1_000_000L,
                null);
        ArgumentCaptor<PullRequestTimingsRequest> requests = ArgumentCaptor.forClass(PullRequestTimingsRequest.class);
        verify(client, times(2)).pull(any(), requests.capture(), anyLong());
        assertEquals(6, requests.getAllValues().get(1).getAfterSequence());
    }

    @ParameterizedTest
    @MethodSource("invalidPages")
    void rejectsEntireInvalidPageBeforeAddingAnyBusinessRecords(PullRequestTimingsResponse invalid) throws Exception {
        CompletableFuture<PullRequestTimingsResponse> reply = new CompletableFuture<>();
        when(client.pull(any(), any(), anyLong())).thenReturn(reply, new CompletableFuture<>());
        tick();
        TtftStageAggregator aggregator = mock(TtftStageAggregator.class);
        ReflectionTestUtils.setField(poller, "aggregator", aggregator);
        reply.complete(invalid);
        drain();
        tick();
        verify(aggregator, never()).addTiming(any(), any(), any(), any(), anyLong(), any());
        verify(aggregator, never()).clearSource(any());
        ArgumentCaptor<PullRequestTimingsRequest> requests = ArgumentCaptor.forClass(PullRequestTimingsRequest.class);
        verify(client, times(2)).pull(any(), requests.capture(), anyLong());
        assertEquals(0, requests.getAllValues().get(1).getAfterSequence());
        assertEquals("", requests.getAllValues().get(1).getEpoch());
    }

    private static Stream<PullRequestTimingsResponse> invalidPages() {
        return Stream.of(
                PullRequestTimingsResponse.newBuilder()
                        .setEpoch(" ")
                        .setNextSequence(1)
                        .addRecords(frontendRecord(1, "valid"))
                        .build(),
                PullRequestTimingsResponse.newBuilder()
                        .setEpoch("epoch")
                        .setNextSequence(3)
                        .setGap(true)
                        .addRecords(frontendRecord(1, "valid"))
                        .addRecords(frontendRecord(3, "late-gap"))
                        .build(),
                PullRequestTimingsResponse.newBuilder()
                        .setEpoch("epoch")
                        .setNextSequence(1)
                        .addRecords(frontendRecord(1, "valid"))
                        .addRecords(frontendRecord(1, "duplicate-sequence"))
                        .build(),
                PullRequestTimingsResponse.newBuilder()
                        .setEpoch("epoch")
                        .setNextSequence(2)
                        .addRecords(frontendRecord(1, "valid"))
                        .build(),
                PullRequestTimingsResponse.newBuilder()
                        .setEpoch("epoch")
                        .setNextSequence(-1)
                        .addRecords(frontendRecord(-1, "unsigned-overflow"))
                        .build(),
                PullRequestTimingsResponse.newBuilder().setEpoch("epoch").setNextSequence(0).setHasMore(true).build());
    }

    @Test
    void changedEpochClearsSourceThenSkipsBadRecordAndAcceptsNewSequence() throws Exception {
        CompletableFuture<PullRequestTimingsResponse> initial = new CompletableFuture<>();
        CompletableFuture<PullRequestTimingsResponse> restarted = new CompletableFuture<>();
        when(client.pull(any(), any(), anyLong())).thenReturn(initial, restarted, new CompletableFuture<>());
        tick();
        initial.complete(response("old-epoch", 4, false));
        drain();
        tick();
        TtftStageAggregator aggregator = mock(TtftStageAggregator.class);
        ReflectionTestUtils.setField(poller, "aggregator", aggregator);
        restarted.complete(PullRequestTimingsResponse.newBuilder()
                .setEpoch("new-epoch")
                .setNextSequence(2)
                .addRecords(frontendRecord(1, "bad").clearDispatchStartUs())
                .addRecords(frontendRecord(2, "valid-new"))
                .build());
        drain();
        var order = org.mockito.Mockito.inOrder(aggregator);
        order.verify(aggregator).clearSource("daemon/frontend/daemon:8080");
        order.verify(aggregator)
                .addTiming("daemon/frontend/daemon:8080", "valid-new", "frontend", "deployment", 1_000_000L, null);
        tick();
        ArgumentCaptor<PullRequestTimingsRequest> requests = ArgumentCaptor.forClass(PullRequestTimingsRequest.class);
        verify(client, times(3)).pull(any(), requests.capture(), anyLong());
        assertEquals("new-epoch", requests.getAllValues().get(2).getEpoch());
        assertEquals(2, requests.getAllValues().get(2).getAfterSequence());
    }

    private static RequestTimingRecord.Builder frontendRecord(long sequence, String requestId) {
        return RequestTimingRecord.newBuilder()
                .setSequence(sequence)
                .setRequestId(requestId)
                .setRole("frontend")
                .setDeploymentName("deployment")
                .setEventKind(RequestTimingEventKind.DISPATCH_COMPLETED)
                .setDispatchStartUs(1_000_000);
    }

    private static RequestTimingRecord.Builder prefillRecord(long sequence, String requestId) {
        return RequestTimingRecord.newBuilder()
                .setSequence(sequence)
                .setRequestId(requestId)
                .setRole("prefill")
                .setDeploymentName("deployment")
                .setEventKind(RequestTimingEventKind.PREFILL_RECEIVED)
                .setReceivedTimeUs(1_000_000);
    }

    private PullRequestTimingsResponse prefillResponse(String epoch, long sequence) {
        var response = response(epoch, sequence, false).toBuilder();
        response.setRecords(0, prefillRecord(sequence, "request"));
        return response.build();
    }

    private PullRequestTimingsResponse response(String epoch, long sequence, boolean gap) {
        return PullRequestTimingsResponse.newBuilder()
                .setEpoch(epoch)
                .setNextSequence(sequence)
                .setGap(gap)
                .addRecords(RequestTimingRecord.newBuilder()
                        .setSequence(sequence)
                        .setRequestId("request")
                        .setRole("frontend")
                        .setDeploymentName("deployment")
                        .setEventKind(RequestTimingEventKind.DISPATCH_COMPLETED)
                        .setDispatchStartUs(1_000_000))
                .build();
    }

    private void tick() throws Exception {
        executor.submit(poller::tick).get(5, TimeUnit.SECONDS);
    }

    private void drain() throws Exception {
        executor.submit(() -> {}).get(5, TimeUnit.SECONDS);
    }
}
