package org.flexlb.balance.scheduler;

import com.google.protobuf.ByteString;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.OrderedPrefillRequestQueue;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.prediction.PrefillTimePredictor.PrefillBatchFeatures;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot;
import org.flexlb.balance.scheduler.ResponseCompletionExecutor.RequestFuture;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;

import static org.flexlb.balance.scheduler.ContextResult.DeliveryEvidence;
import static org.flexlb.balance.scheduler.ContextResult.Prediction;
import static org.flexlb.balance.scheduler.ContextResult.PreparationFailure;
import static org.flexlb.balance.scheduler.ContextResult.RouteDelivered;
import static org.flexlb.balance.scheduler.ContextResult.RoutingCompletion;
import static org.flexlb.balance.scheduler.ContextResult.TransportResult;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isA;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.anyInt;
import static org.mockito.Mockito.anyLong;
import static org.mockito.Mockito.anyString;
import static org.mockito.Mockito.argThat;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * The production Worker -> Dispatcher -> Runtime path; only protocol owners and transport are simulated.
 */
@Timeout(15)
class BatchDeliveryTest {
    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void queueSignalFailureAfterHandoffDoesNotRepeatCleanupOrFailDeliveredRequests(boolean batch) {
        try (var f = new LinearDeliveryFixture(batch)) {
            var first = f.item(1L, 10L);
            var second = f.item(2L, 20L);
            doThrow(new IllegalStateException("queue signal failed"))
                    .when(f.prefill)
                    .signal(PrefillEndpoint.Signal.PLACEMENT);
            assertDoesNotThrow(() -> f.process(List.of(first, second), OptionalLong.of(30L)));
            f.drain();
            assertEquals(List.of(1L, 2L), f.claimIds());
            if (batch) {
                assertEquals(Delivery.Result.Status.DELIVERED, f.result(1L).status());
                assertEquals(Delivery.Result.Status.DELIVERED, f.result(2L).status());
            } else {
                verify(f.owner, times(2)).onDeliveryResult(isA(RouteDelivered.class));
                verifyNoInteractions(f.grpc);
            }
            assertTrue(f.preparationFailures.isEmpty());
            verify(f.permits.get(1L), times(1)).release();
            verify(f.permits.get(2L), times(1)).release();
        }
    }

    @Test
    void failedResourceCommitReportsPreparationOnceAndDoesNotSend() {
        try (var f =
                new LinearDeliveryFixture(
                        true,
                        config ->
                                config.decisionPolicy()
                                        .setType(
                                                org.flexlb.config.DecisionPolicyConfig.Type
                                                        .SINGLE))) {
            f.item(1L, 10L);
            var failure = new IllegalStateException("Decode transfer failed");
            doThrow(failure).when(f.permits.get(1L)).dispatch();
            RequestExecutionTestSupport.schedule(f.worker);
            f.drain();
            assertEquals(Set.of(1L), f.preparationFailures.keySet());
            verify(f.owner, times(1)).recordPreparationFailure(any(PreparationFailure.class));
            verify(f.owner, never()).onDeliveryResult(any(DeliveryEvidence.class));
            verify(f.permits.get(1L), times(1)).release();
            verifyNoInteractions(f.grpc);
        }
    }

    @Test
    void predictionIsAvailableBeforeRpcAndOnlyRpcCompletionReportsAResult() {
        try (var f =
                new LinearDeliveryFixture(
                        true,
                        config ->
                                config.decisionPolicy()
                                        .setType(
                                                org.flexlb.config.DecisionPolicyConfig.Type
                                                        .SINGLE))) {
            f.item(1L, 10L);
            doAnswer(
                            call -> {
                                assertTrue(f.predictions.containsKey(1L));
                                verify(f.owner, never())
                                        .onDeliveryResult(any(DeliveryEvidence.class));
                                return CompletableFuture.completedFuture(
                                        LinearDeliveryFixture.ack(call.getArgument(2)));
                            })
                    .when(f.grpc)
                    .batchEnqueueAsync(anyString(), anyInt(), any());
            RequestExecutionTestSupport.schedule(f.worker);
            f.drain();
            assertEquals(Delivery.Result.Status.DELIVERED, f.result(1L).status());
            assertEquals(1, f.completionCounts.get(1L));
            verify(f.owner, times(1)).onDeliveryResult(isA(TransportResult.class));
            verify(f.owner, never()).recordPreparationFailure(any());
        }
    }

    @Test
    void rpcInvocationFailureReportsOneUncertainResultWithoutPreparationFailure() {
        try (var f =
                new LinearDeliveryFixture(
                        true,
                        config ->
                                config.decisionPolicy()
                                        .setType(
                                                org.flexlb.config.DecisionPolicyConfig.Type
                                                        .SINGLE))) {
            f.item(1L, 10L);
            doThrow(new IllegalStateException("RPC invocation failed"))
                    .when(f.grpc)
                    .batchEnqueueAsync(anyString(), anyInt(), any());
            RequestExecutionTestSupport.schedule(f.worker);
            f.drain();
            assertEquals(Delivery.Result.Status.UNCERTAIN, f.result(1L).status());
            assertEquals(1, f.completionCounts.get(1L));
            verify(f.owner, times(1)).onDeliveryResult(isA(TransportResult.class));
            verify(f.owner, never()).recordPreparationFailure(any());
        }
    }

    @Test
    void deliveryTelemetrySaturatesTotalsAndDoesNotNeedACopiedRouteList() {
        var reporter = mock(DeliveryMetricsReporter.class);
        var dispatcher =
                new DeliveryCoordinator(
                        new org.flexlb.balance.scheduler.RequestRepository(),
                        null,
                        DeliveryTestSupport.service(SchedulingTestConfig.batchConfig()),
                        reporter,
                        new org.flexlb.balance.scheduler.SchedulerConfiguration()
                                .decodeAdmissionCapacity(
                                        DeliveryTestSupport.service(
                                                SchedulingTestConfig.batchConfig())),
                        org.mockito.Mockito.mock(
                                org.flexlb.balance.scheduler.DeliveryExecution.class));
        var item = mock(RouteResult.class);
        var endpoint = mock(PrefillEndpoint.class);
        when(item.prefillEndpoint()).thenReturn(endpoint);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(item.enqueuedAtMs()).thenReturn(Long.MAX_VALUE);
        when(item.hitCache()).thenReturn(Long.MAX_VALUE);
        when(item.seqLen()).thenReturn(Long.MAX_VALUE);
        when(item.priority()).thenReturn(17);
        var claim = mock(Delivery.Member.class);
        when(claim.route()).thenReturn(item);
        var batch = mock(Delivery.class);
        batch.batchId = 71L;
        batch.reason = "batch_full";
        when(endpoint.queueDepth()).thenReturn(3);
        ReflectionTestUtils.invokeMethod(
                new DeliveryDispatcher(mock(RequestRepository.class), null, null, null, reporter),
                "reportDelivery",
                batch,
                List.of(claim, claim),
                7L);
        verify(reporter, org.mockito.Mockito.times(2))
                .reportBatchWaitTimeMs("PREFILL", "127.0.0.1", 0L, 17);
        verify(reporter)
                .reportBatchDelivery(
                        "PREFILL",
                        "127.0.0.1",
                        71L,
                        "batch_full",
                        3,
                        2,
                        Long.MAX_VALUE,
                        Long.MAX_VALUE,
                        7L);
    }

    @Test
    void singleDecisionStillUsesBatchOwnershipAndEngineAcknowledgement() {
        try (var f =
                new LinearDeliveryFixture(
                        true,
                        config ->
                                config.decisionPolicy()
                                        .setType(
                                                org.flexlb.config.DecisionPolicyConfig.Type
                                                        .SINGLE))) {
            var first = f.item(1L, 10L);
            f.item(2L, 20L);
            RequestExecutionTestSupport.schedule(f.worker);
            f.drain();
            assertEquals(List.of(first), f.committedItems);
            assertEquals(List.of(1L), f.sentIds());
            assertEquals(1, f.sent.size());
            assertEquals(701L, f.claims.getFirst().batchId());
            assertEquals(Delivery.SendMode.BATCH_ENQUEUE, f.claims.getFirst().kind());
            assertEquals(Delivery.Result.Status.DELIVERED, f.result(1L).status());
            verify(f.runtime.deliveries()).tryReserveDelivery();
            verify(f.prefill).batchCapacityAvailable(anyInt());
            verify(f.owner, never()).onDeliveryResult(isA(RouteDelivered.class));
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void byteBudgetSplitsLargeImageRequestsButPreservesTextBatchCount(boolean images)
            throws Exception {
        try (var f = new LinearDeliveryFixture(true)) {
            var template = EngineRpcService.GenerateInputPB.newBuilder().addTokenIds(129264);
            if (images) {
                template.addMultimodalInputs(
                        EngineRpcService.MultimodalInputPB.newBuilder()
                                .setMultimodalTensor(
                                        EngineRpcService.TensorPB.newBuilder()
                                                .setBf16Data(
                                                        ByteString.copyFrom(
                                                                new byte[20 * 1024 * 1024]))));
            }
            ByteString body = template.build().toByteString();
            List<RouteResult> remaining = new ArrayList<>();
            for (long id = 1; id <= 64; id++) {
                remaining.add(
                        f.item(
                                id,
                                100,
                                EngineRpcService.GenerateInputPB.newBuilder()
                                        .setRequestId(id)
                                        .build()
                                        .toByteString()
                                        .concat(body)));
            }
            var ids = new java.util.concurrent.atomic.AtomicLong(701L);
            var execution = f.runtime.deliveries();
            doAnswer(call -> ids.getAndIncrement()).when(execution).nextBatchId();
            int maxImageBatchCount =
                    (org.flexlb.constant.GrpcConstants.MAX_MESSAGE_SIZE - 16)
                            / (20 * 1024 * 1024 + 1024);
            List<Integer> expectedCounts = new ArrayList<>();
            while (!remaining.isEmpty()) {

                assertNull(
                        f.process(remaining, OptionalLong.of(9999L)),
                        "size boundary must leave suffix queued");
                int count =
                        images ? Math.min(maxImageBatchCount, remaining.size()) : remaining.size();
                assertEquals(remaining.subList(0, count), f.committedItems);
                if (count < remaining.size()) {
                    assertEquals(count * 100L, f.committedPrediction);
                }
                int completed = 64 - remaining.size() + count;
                RequestProtocolTestSupport.awaitCondition(() -> f.results.size() == completed);
                expectedCounts.add(count);
                remaining = new ArrayList<>(remaining.subList(count, remaining.size()));
            }
            f.drain();
            assertEquals(
                    expectedCounts,
                    f.sent.stream()
                            .map(
                                    batch ->
                                            batch.getDpSlotsList().stream()
                                                    .mapToInt(slot -> slot.getRequestsCount())
                                                    .sum())
                            .toList());
            assertEquals(
                    f.sent.size(),
                    f.sent.stream().map(batch -> batch.getBatchId()).distinct().count());
            assertEquals(64, f.sentIds().size());
            assertEquals(64, new HashSet<>(f.sentIds()).size());
            f.sent.forEach(
                    batch -> {
                        assertTrue(
                                batch.getSerializedSize()
                                        <= org.flexlb.constant.GrpcConstants.MAX_MESSAGE_SIZE);
                        batch.getDpSlotsList()
                                .forEach(
                                        slot ->
                                                slot.getRequestsList()
                                                        .forEach(
                                                                member -> {
                                                                    var input = member.getInput();
                                                                    assertEquals(
                                                                            images ? 1 : 0,
                                                                            input
                                                                                    .getMultimodalInputsCount());
                                                                    if (images) {
                                                                        assertEquals(
                                                                                20 * 1024 * 1024,
                                                                                input.getMultimodalInputs(
                                                                                                0)
                                                                                        .getMultimodalTensor()
                                                                                        .getBf16Data()
                                                                                        .size());
                                                                    }
                                                                }));
                    });
            assertTrue(
                    f.results.values().stream()
                            .allMatch(
                                    result -> result.status() == Delivery.Result.Status.DELIVERED));
        }
    }

    @Test
    void queuedBatchStillRechecksCurrentDispatcherConfigurationBeforeRpc() {
        try (var f = new LinearDeliveryFixture(true)) {
            var item = f.item(1, 10);
            f.config.getDispatcher().setType(DispatcherConfig.Type.NON_BATCH);
            assertEquals(
                    DispatcherConfig.Type.BATCH,
                    f.dispatcher.mode(),
                    "topology mode remains frozen");
            f.process(List.of(item), OptionalLong.of(10));
            f.drain();
            assertEquals(Delivery.Result.Status.NOT_SENT, f.result(1).status());
            verifyNoInteractions(f.grpc);
        }
    }

    @Test
    void orderedCapacityPrefixCommitsOnceAndRepredictsOnlyThatPrefix() {
        try (var f = new LinearDeliveryFixture(true)) {
            var first = f.item(1, 10);
            var second = f.item(2, 20);
            var third = f.item(3, 30);
            f.blockDecode(2);
            f.process(List.of(first, second, third), OptionalLong.of(999));
            f.drain();
            assertEquals(List.of(first), f.committedItems);
            assertEquals(10, f.committedPrediction);
            assertEquals(List.of(1L), f.sentIds());
            assertEquals(List.of(1L), f.claimIds());
            verify(f.decode, never()).acquireDispatchPermit(eq(third.decodeReservation()), any());

            assertEquals(Delivery.Result.Status.DELIVERED, f.result(1).status());
        }
    }

    @Test
    void unavailableHeadReturnsExactBoundaryAndRollsBackItsTemporaryReservation() {
        try (var f = new LinearDeliveryFixture(true)) {
            var first = f.item(1, 10);
            f.blockDecode(1);
            var boundary = f.process(List.of(first), OptionalLong.empty());
            assertNotNull(boundary);
            assertEquals(CapacityBoundary.Resource.DECODE_ENGINE, boundary.resource());
            assertTrue(boundary.unavailable());
            assertTrue(f.committedItems.isEmpty());

            assertTrue(f.runtime.deliveries().deliveryCapacityAvailable());
            verifyNoInteractions(f.grpc);
        }
    }

    @Test
    void firstMemberAdmissionFailureReturnsBoundaryInsteadOfFailingTheWorker() {
        try (var f = new LinearDeliveryFixture(true)) {
            var item = f.item(1, 10);
            var failure = new IllegalStateException("prefill reservation unavailable");
            doThrow(failure).when(f.prefill).batchCapacityAvailable(anyInt());
            var boundary = assertDoesNotThrow(() -> f.process(List.of(item), OptionalLong.empty()));
            assertEquals(CapacityBoundary.Status.FAILED, boundary.status());
            assertSame(failure, boundary.cause());
            assertTrue(f.runtime.deliveries().deliveryCapacityAvailable());
            assertTrue(f.committedItems.isEmpty());
            verifyNoInteractions(f.grpc);
        }
    }

    @Test
    void predictionFailurePreservesPrimaryCauseAndAttemptsAllPreparationCleanup() {
        try (var f = new LinearDeliveryFixture(true)) {
            var first = f.item(1, 10);
            var second = f.item(2, 20);
            var primary = new IllegalStateException("prediction failed");
            var cleanup = new IllegalStateException("permit cleanup failed");
            doThrow(primary).when(f.evaluator).predictBatchMs(any());
            doThrow(cleanup).when(f.permits.get(1L)).release();
            assertSame(
                    primary,
                    assertThrows(
                            IllegalStateException.class,
                            () -> f.process(List.of(first, second), OptionalLong.empty())));
            assertTrue(Arrays.asList(primary.getSuppressed()).contains(cleanup));
            verify(f.permits.get(2L)).release();

            assertTrue(f.runtime.deliveries().deliveryCapacityAvailable());
            verifyNoInteractions(f.grpc);
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {0, 1, 2})
    void lostClaimExcludesOnlyThatMemberAndUsesTheFrozenEvaluator(int lostIndex) throws Exception {
        try (var f = new LinearDeliveryFixture(true)) {
            var items = List.of(f.item(1, 10), f.item(2, 20), f.item(3, 30));
            try (var blocker = f.blockExecution()) {
                f.process(items, OptionalLong.of(999));
                f.lostClaims.add(Long.parseLong(items.get(lostIndex).requestId()));
                // Changing the endpoint predictor after queue commit must not change the captured
                // evaluator.
                var changed = mock(PrefillTimePredictor.class);
                when(changed.evaluator())
                        .thenThrow(new AssertionError("captured evaluator was discarded"));
                ReflectionTestUtils.setField(f.prefill, "predictor", changed);
            }
            f.drain();
            var expected =
                    items.stream()
                            .filter(
                                    item ->
                                            !f.lostClaims.contains(
                                                    Long.parseLong(item.requestId())))
                            .toList();
            assertEquals(
                    expected.stream().map(route -> Long.parseLong(route.requestId())).toList(),
                    f.sentIds());
            long prediction = expected.stream().mapToLong(RouteResult::seqLen).sum();
            expected.forEach(
                    item ->
                            assertEquals(
                                    prediction,
                                    f.predictions
                                            .get(Long.parseLong(item.requestId()))
                                            .longValue()));
            assertFalse(f.results.containsKey(Long.parseLong(items.get(lostIndex).requestId())));
            verify(f.permits.get(Long.parseLong(items.get(lostIndex).requestId()))).release();
            verify(f.evaluator).predictBatchMs(argThat(features -> features.items().size() == 2));
        }
    }

    @Test
    void completedRpcPublishesOutsideEndpointEventsAndContextMonitor() {
        try (var f = new LinearDeliveryFixture(true)) {
            var first = f.item(1, 10);
            var second = f.item(2, 20);
            f.beforeCompletion =
                    claim -> {
                        assertFalse(f.prefill.isRetired());
                        assertFalse(Thread.holdsLock(claim.route().ctx()));
                        assertFalse(f.queueLock.isHeldByCurrentThread());
                    };
            f.process(List.of(first, second), OptionalLong.of(30));
            f.drain();
            assertEquals(List.of(1L, 2L), f.sentIds());
            assertEquals(2, f.results.size());
        }
    }

    @Test
    void payloadFailureNeverCommitsDeliveryOrTransfersDecodeCapacity() {
        try (var f = new LinearDeliveryFixture(true)) {
            var item = f.item(1, 10, ByteString.EMPTY);
            f.process(List.of(item), OptionalLong.of(10));
            f.drain();
            assertTrue(f.preparationFailures.containsKey(1L));
            assertFalse(f.results.containsKey(1L));
            assertNull(item.ctx().snapshot().delivery());
            verify(f.permits.get(1L), never()).dispatch();
            verify(f.permits.get(1L)).release();
            verifyNoInteractions(f.grpc);
            assertFalse(f.prefill.isRetired());
        }
    }

    @ParameterizedTest
    @ValueSource(
            strings = {
                "THROW",
                "NULL_FUTURE",
                "EXCEPTIONAL_FUTURE",
                "NULL_RESPONSE",
                "WRONG_BATCH"
            })
    void failuresAfterRpcInvocationRemainUncertain(String outcome) {
        try (var f = new LinearDeliveryFixture(true)) {
            doAnswer(
                            call -> {
                                f.sent.add(call.getArgument(2));
                                return switch (outcome) {
                                    case "THROW" ->
                                            throw new IllegalStateException(
                                                    "invoked transport failed");
                                    case "NULL_FUTURE" -> null;
                                    case "EXCEPTIONAL_FUTURE" ->
                                            CompletableFuture.failedFuture(
                                                    new TimeoutException("remote timeout"));
                                    case "NULL_RESPONSE" -> CompletableFuture.completedFuture(null);
                                    case "WRONG_BATCH" ->
                                            CompletableFuture.completedFuture(
                                                    EngineRpcService.EnqueueBatchResponsePB
                                                            .newBuilder()
                                                            .setBatchId(702)
                                                            .build());
                                    default -> throw new AssertionError(outcome);
                                };
                            })
                    .when(f.grpc)
                    .batchEnqueueAsync(anyString(), anyInt(), any());
            f.process(List.of(f.item(1, 10), f.item(2, 20)), OptionalLong.of(30));
            f.drain();
            assertEquals(2, f.results.size());
            f.results
                    .values()
                    .forEach(
                            result ->
                                    assertEquals(
                                            Delivery.Result.Status.UNCERTAIN, result.status()));
            assertEquals(Map.of(1L, 1, 2L, 1), f.completionCounts);
            verify(f.grpc, times(1)).batchEnqueueAsync(anyString(), anyInt(), any());
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {0, 1, 2, 3, 4, 5, 6, 7})
    void finalCommitFiltersEveryRejectedSubsetWithoutChangingOtherResults(int refusedMask) {
        try (var f = new LinearDeliveryFixture(true)) {
            var items = List.of(f.item(1, 10), f.item(2, 20), f.item(3, 30));
            for (int i = 0; i < 3; i++) {
                if ((refusedMask & (1 << i)) != 0) {
                    f.lostClaims.add((long) i + 1);
                }
            }
            f.process(items, OptionalLong.of(60));
            f.drain();
            var expected =
                    items.stream()
                            .filter(
                                    item ->
                                            !f.lostClaims.contains(
                                                    Long.parseLong(item.requestId())))
                            .map(route -> Long.parseLong(route.requestId()))
                            .toList();
            assertEquals(expected, f.sentIds());
            for (var item : items) {
                long id = Long.parseLong(item.requestId());
                if (f.lostClaims.contains(id)) {
                    assertFalse(f.results.containsKey(id));
                    assertNull(item.ctx().snapshot().delivery());
                    verify(f.permits.get(id), never()).dispatch();
                    verify(f.permits.get(id)).release();
                } else {
                    assertEquals(Delivery.Result.Status.DELIVERED, f.result(id).status());
                    assertEquals(1, f.completionCounts.get(id));
                }
            }
            if (expected.isEmpty()) {
                verifyNoInteractions(f.grpc);
            }
        }
    }

    @Test
    void failedFactIsNotReplayedOrReclassifiedAndDoesNotSuppressItsSiblings() {
        try (var f = new LinearDeliveryFixture(true)) {
            f.expectedShutdownFailure = new IllegalStateException("owner callback failed");
            f.throwCompletionFor = 1;
            f.process(List.of(f.item(1, 10), f.item(2, 20)), OptionalLong.of(30));
            f.drain();
            assertEquals(Delivery.Result.Status.DELIVERED, f.result(1).status());
            assertEquals(Delivery.Result.Status.DELIVERED, f.result(2).status());
            assertEquals(Map.of(1L, 1, 2L, 1), f.completionCounts);
            verify(f.grpc, times(1)).batchEnqueueAsync(anyString(), anyInt(), any());
        }
    }

    @Test
    void transportCompletionDoesNotStartAnotherStopOperation() {
        try (var f = new LinearDeliveryFixture(true)) {
            var dispatcher = f.dispatcher;
            ReflectionTestUtils.setField(f.worker, "dispatcher", dispatcher);
            f.beforeCompletion =
                    claim ->
                            claim.owner()
                                    .requestCancel(
                                            new ContextResult.Cancellation(
                                                    0L,
                                                    CancelReason.CLIENT_CANCELLED,
                                                    false,
                                                    System.currentTimeMillis(),
                                                    org.flexlb.dao.loadbalance.StrategyErrorType
                                                            .REQUEST_CANCELLED,
                                                    "client cancelled"));
            f.process(List.of(f.item(1L, 10L)), OptionalLong.of(10L));
            f.drain();
            assertEquals(Map.of(1L, 1), f.completionCounts);
            assertNotNull(f.claims.getFirst().owner().snapshot().delivery().abandonment());
            verify(dispatcher, never()).stop(any());
        }
    }

    @Test
    void temporaryCleanupFailureCannotDiscardAnAcceptedRpcResponse() {
        try (var f = new LinearDeliveryFixture(true)) {
            var first = f.item(1, 10);
            var second = f.item(2, 20);
            var permitFailure = new IllegalStateException("permit cleanup failed");
            f.expectedShutdownFailure = permitFailure;
            doThrow(permitFailure).when(f.permits.get(1L)).release();
            f.process(List.of(first, second), OptionalLong.of(30));
            f.drain();
            assertEquals(Delivery.Result.Status.DELIVERED, f.result(1).status());
            assertEquals(Delivery.Result.Status.DELIVERED, f.result(2).status());
            verify(f.permits.get(2L)).release();
        }
    }

    @Test
    void executorRejectionAfterDispatcherEntryTerminalizesEveryCommittedOwnerExactlyOnce() {
        try (var f = new LinearDeliveryFixture(true)) {
            var first = f.item(1, 10);
            var second = f.item(2, 20);
            ((ThreadPoolExecutor)
                            ReflectionTestUtils.getField(
                                    f.runtime.deliveries(), "dispatchExecutor"))
                    .shutdown();
            assertDoesNotThrow(() -> f.process(List.of(first, second), OptionalLong.of(30)));
            f.drain();
            assertEquals(Set.of(1L, 2L), f.preparationFailures.keySet());
            assertTrue(f.results.isEmpty());
            verify(f.owner, times(1))
                    .recordPreparationFailure(
                            argThat(
                                    e ->
                                            e instanceof PreparationFailure failure
                                                    && failure.route() == first));
            verify(f.owner, times(1))
                    .recordPreparationFailure(
                            argThat(
                                    e ->
                                            e instanceof PreparationFailure failure
                                                    && failure.route() == second));
            verify(f.permits.get(1L), times(1)).release();
            verify(f.permits.get(2L), times(1)).release();

            verifyNoInteractions(f.grpc);
        }
    }
}

/**
 * No transaction/state-machine fake: calls real preparation, commitment orchestration and asynchronous transport.
 */
final class LinearDeliveryFixture implements AutoCloseable {
    final FlexlbConfig config;
    final DeliveryMetricsReporter reporter = mock(DeliveryMetricsReporter.class);
    final EngineGrpcClient grpc = mock(EngineGrpcClient.class);
    final PrefillEndpoint prefill = mock(PrefillEndpoint.class);
    final DecodeEndpoint decode =
            org.flexlb.balance.scheduler.SchedulerTestSupport.mockDecodeEndpoint();
    final AbstractRequestScheduler owner;
    final WorkSnapshot work = mock(WorkSnapshot.class);
    final ReentrantLock queueLock = new ReentrantLock();
    final PrefillEndpoint queueOwnership =
            org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                    queueLock, OrderedPrefillRequestQueue.fifo(16), System::currentTimeMillis);

    final List<String> events = new CopyOnWriteArrayList<>();
    final List<Delivery.Member> claims = new CopyOnWriteArrayList<>();
    final List<EngineRpcService.EnqueueBatchRequestPB> sent = new CopyOnWriteArrayList<>();
    final Map<Long, DecodeEndpoint.EngineDispatchPermit> permits = new ConcurrentHashMap<>();
    final Map<Long, Long> predictions = new ConcurrentHashMap<>();
    final Map<Long, Delivery.Result> results = new ConcurrentHashMap<>();
    final Map<Long, Integer> completionCounts = new ConcurrentHashMap<>();
    final Map<Long, String> preparationFailures = new ConcurrentHashMap<>();
    final Set<Long> lostClaims = ConcurrentHashMap.newKeySet();
    final Set<Long> blockedDecode = ConcurrentHashMap.newKeySet();
    final SchedulerTestServices runtime;
    final DeliveryCoordinator dispatcher;
    final PrefillQueueScheduler worker;
    final PrefillTimePredictor.Evaluator evaluator =
            spy(
                    new PrefillTimePredictor.Evaluator() {
                        @Override
                        public long estimateMs(long totalTokens, long hitTokens) {
                            return totalTokens - hitTokens;
                        }

                        @Override
                        public double predictBatchMs(PrefillBatchFeatures features) {
                            return features.items().stream()
                                    .mapToLong(PrefillBatchFeatures.Item::seqLen)
                                    .sum();
                        }
                    });

    volatile List<RouteResult> committedItems = List.of();

    volatile long committedPrediction;

    volatile long[] committedRoutePredictions;

    volatile long throwCompletionFor = -1;

    RuntimeException expectedShutdownFailure;
    volatile Consumer<Delivery.Member> beforeCompletion = ignored -> {};

    final List<DecodeEndpoint> realEndpoints = new ArrayList<>();

    LinearDeliveryFixture(boolean batch) {
        this(batch, config -> {});
    }

    LinearDeliveryFixture(boolean batch, Consumer<FlexlbConfig> configure) {
        config = spy(SchedulingTestConfig.batchConfig());
        if (!batch) {
            config.getDispatcher().setType(DispatcherConfig.Type.NON_BATCH);
        }
        configure.accept(config);
        var predictor = mock(PrefillTimePredictor.class);
        when(predictor.evaluator()).thenReturn(evaluator);
        ReflectionTestUtils.setField(prefill, "predictor", predictor);
        var sizing = spy(config.getInternalRuntime());
        doReturn(1).when(sizing).getBatchDispatchThreads();
        doReturn(8).when(sizing).getBatchDispatchQueueCapacity();
        doReturn(1).when(sizing).getBatchDispatchCompletionThreads();
        doReturn(sizing).when(config).getInternalRuntime();
        var service = SchedulingTestConfig.configService(config);
        dispatcher =
                spy(
                        new DeliveryCoordinator(
                                new org.flexlb.balance.scheduler.RequestRepository(),
                                grpc,
                                service,
                                reporter,
                                new org.flexlb.balance.scheduler.SchedulerConfiguration()
                                        .decodeAdmissionCapacity(service),
                                org.mockito.Mockito.mock(
                                        org.flexlb.balance.scheduler.DeliveryExecution.class)));
        runtime =
                spy(
                        new SchedulerTestServices(
                                new RequestRepository(),
                                mock(EndpointRegistry.class),
                                reporter,
                                mock(RequestSchedulerReporter.class),
                                dispatcher,
                                service,
                                mock(EngineGrpcClient.class)));
        owner =
                spy(
                        runtime.newLifecycle(
                                config,
                                SchedulingTestConfig.capacity(config),
                                mock(org.flexlb.service.TheoryCacheHitReporter.class)));
        var execution = spy(runtime.deliveries());
        org.flexlb.balance.scheduler.SchedulerTestServices.setDependency(
                runtime, "deliveries", execution);
        doReturn(701L).when(execution).nextBatchId();
        org.flexlb.balance.scheduler.SchedulerTestServices.bind(owner, runtime);
        org.flexlb.balance.endpoint.EndpointTestSupport.stubResources(prefill, queueOwnership);
        org.mockito.Mockito.doAnswer(call -> !prefill.isRetired())
                .when(prefill)
                .acceptsNewRequests();
        org.mockito.Mockito.doAnswer(call -> !decode.isRetired()).when(decode).acceptsNewRequests();
        org.mockito.Mockito.doReturn("127.0.0.2").when(decode).getIp();
        worker =
                org.flexlb.balance.endpoint.EndpointTestSupport.attach(
                        org.flexlb.balance.endpoint.EndpointTestSupport.worker(
                                "contract",
                                prefill,
                                QueueExecutionSettings.capture(
                                        config,
                                        org.flexlb.balance.planner.BuiltinGroupPlanner.from(
                                                config)),
                                dispatcher,
                                queueOwnership,
                                SchedulerTestSupport.queueDependencies(runtime)));
        when(prefill.getIp()).thenReturn("127.0.0.1");
        when(prefill.getGrpcPort()).thenReturn(8090);
        doReturn(true).when(prefill).batchCapacityAvailable(anyInt());
        when(prefill.dequeueForDelivery(
                        any(org.flexlb.balance.endpoint.PrefillEndpoint.DeliverySelection.class)))
                .thenAnswer(
                        call -> {
                            assertFalse(queueLock.isHeldByCurrentThread());
                            var plan = call.getArgument(0, PrefillEndpoint.DeliverySelection.class);
                            committedItems = List.copyOf(plan.items());
                            if (plan.batch()) {
                                committedPrediction = plan.predictedMs();
                            } else {
                                committedRoutePredictions =
                                        Arrays.copyOf(plan.predictions(), committedItems.size());
                            }
                            for (var member : committedItems) {
                                assertTrue(queueOwnership.releaseQueuedRequest(member));
                            }
                            return org.flexlb.balance.delivery.CapacityBoundary.Attempt.accepted(
                                    work);
                        });
        when(work.calculate())
                .thenAnswer(
                        call -> {
                            assertFalse(queueLock.isHeldByCurrentThread());
                            return new WorkSnapshot(0, List.of(), List.of(), 0);
                        });
        doAnswer(
                        call -> {
                            Delivery.Member member = call.getArgument(0);
                            RouteResult item = member.route();
                            events.add("claim-" + Long.parseLong(item.requestId()));
                            if (lostClaims.contains(Long.parseLong(item.requestId()))) {
                                return null;
                            }
                            var accepted = (Delivery.Member) call.callRealMethod();
                            if (accepted != null) claims.add(accepted);
                            return accepted;
                        })
                .when(org.flexlb.balance.scheduler.DeliveryTestSupport.transport(owner.dispatcher))
                .commit(
                        any(Delivery.Member.class),
                        any(),
                        anyLong(),
                        org.mockito.ArgumentMatchers.isNull());
        org.mockito.stubbing.Answer<Object> deliveryObservation =
                call -> {
                    DeliveryEvidence evidence = call.getArgument(0);
                    if (evidence instanceof Prediction prediction) {
                        predictions.put(
                                Long.parseLong(prediction.delivery().route().requestId()),
                                duration(prediction.predictedAtMs(), prediction.nowMs()));
                    } else if (evidence instanceof RouteDelivered delivered) {
                        predictions.put(
                                Long.parseLong(delivered.delivery().route().requestId()),
                                duration(delivered.predictedAtMs(), delivered.nowMs()));
                    } else if (evidence instanceof TransportResult result) {
                        var claim = result.delivery();
                        beforeCompletion.accept(claim);
                        results.put(Long.parseLong(claim.route().requestId()), result.result());
                        completionCounts.merge(
                                Long.parseLong(claim.route().requestId()), 1, Integer::sum);
                        events.add("complete-" + Long.parseLong(claim.route().requestId()));
                        if (Long.parseLong(claim.route().requestId()) == throwCompletionFor) {
                            throw expectedShutdownFailure;
                        }
                    } else if (evidence instanceof PreparationFailure failed) {
                        preparationFailures.put(
                                Long.parseLong(failed.route().requestId()), failed.detail());
                    }
                    return null;
                };
        doAnswer(deliveryObservation)
                .when(owner)
                .onDeliveryResult(any(ContextResult.DeliveryEvidence.class));
        doAnswer(deliveryObservation).when(owner).recordDeliveryPrediction(any(Prediction.class));
        doAnswer(deliveryObservation)
                .when(owner)
                .recordPreparationFailure(any(PreparationFailure.class));
        when(decode.acquireDispatchPermit(any(), any()))
                .thenAnswer(
                        call -> {
                            DecodeResources.RequestReservation handle = call.getArgument(0);
                            return blockedDecode.contains(Long.parseLong(handle.requestId()))
                                    ? new org.flexlb.balance.endpoint.DecodeResources.Acquisition<>(
                                            DecodeResources.EngineDispatchPermitAcquireStatus
                                                    .CAPACITY_FULL,
                                            null)
                                    : new org.flexlb.balance.endpoint.DecodeResources.Acquisition<>(
                                            DecodeResources.EngineDispatchPermitAcquireStatus
                                                    .ACQUIRED,
                                            permits.get(Long.parseLong(handle.requestId())));
                        });
        when(grpc.batchEnqueueAsync(anyString(), anyInt(), any()))
                .thenAnswer(
                        call -> {
                            var request =
                                    call.getArgument(
                                            2, EngineRpcService.EnqueueBatchRequestPB.class);
                            assertFalse(queueLock.isHeldByCurrentThread());
                            claims.forEach(
                                    claim -> assertFalse(Thread.holdsLock(claim.route().ctx())));
                            assertFalse(
                                    prefill.isRetired(),
                                    "sending does not retire the selected endpoint");
                            sent.add(request);
                            return CompletableFuture.completedFuture(ack(request));
                        });
    }

    @Override
    public void close() {
        realEndpoints.forEach(
                endpoint -> {
                    endpoint.close();
                    org.flexlb.balance.endpoint.EndpointTestSupport.awaitRetirement(endpoint);
                });
        runtime.deliveries().awaitDeliveries();
        owner.closeExecutors();
        assertSame(expectedShutdownFailure, SchedulerTestServices.failure(runtime));
        runtime.closeTimers();
    }

    long duration(long atMs, long nowMs) {
        return Math.round(
                (atMs - nowMs - 10_000L)
                        / config.getRequestLifecycle().getDecision().getLifetime());
    }

    DecodeEndpoint realDecode() {
        var endpoint =
                EndpointTestSupport.decode(
                        WorkerStatus.createDiscovered(
                                RoleType.DECODE, "contract", "127.0.0.1", 8000, 8001, "contract"),
                        new RequestRepository());
        realEndpoints.add(endpoint);
        return endpoint;
    }

    RouteResult realItem(DecodeEndpoint endpoint, long id, long seqLen) {
        DecodeResources.RequestReservation handle;
        {
            var selectedEndpoint = endpoint;

            handle =
                    endpoint.tryRegisterSelection(
                                    endpoint.routingView(),
                                    String.valueOf(id),
                                    100,
                                    200,
                                    50,
                                    SchedulingTestConfig.capacity(config))
                            .value();
        }
        assertNotNull(handle);
        return item(id, seqLen, endpoint, handle);
    }

    RouteResult item(long id, long seqLen) {
        return item(id, seqLen, (ByteString) null);
    }

    RouteResult item(long id, long seqLen, ByteString payload) {
        var permit = mock(DecodeEndpoint.EngineDispatchPermit.class);
        when(permit.dispatch())
                .thenReturn(
                        org.flexlb.balance.endpoint.DecodeResources.ResourceChange.of(
                                DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED,
                                false));
        when(permit.belongsTo(
                        decode, new DecodeResources.RequestReservation(1, String.valueOf(id), id)))
                .thenReturn(true);
        permits.put(id, permit);
        return item(
                id,
                seqLen,
                decode,
                new DecodeResources.RequestReservation(1, String.valueOf(id), id),
                payload);
    }

    RouteResult item(
            long id,
            long seqLen,
            DecodeEndpoint endpoint,
            DecodeResources.RequestReservation handle) {
        return item(id, seqLen, endpoint, handle, null);
    }

    RouteResult item(
            long id,
            long seqLen,
            DecodeEndpoint endpoint,
            DecodeResources.RequestReservation handle,
            ByteString payload) {
        var context = new RequestContext(config);
        context.data()
                .setRequest(
                        Request.builder()
                                .requestId(String.valueOf(id))
                                .seqLen(seqLen)
                                .rawPriority(50)
                                .build());
        context.data()
                .setGenerateInputPb(
                        payload != null
                                ? payload
                                : EngineRpcService.GenerateInputPB.newBuilder()
                                        .setRequestId(id)
                                        .setGenerateConfig(
                                                EngineRpcService.GenerateConfigPB.newBuilder())
                                        .build()
                                        .toByteString());
        context.data()
                .setSchedulingMetadata(
                        org.flexlb.dao.SchedulingMetadata.explicit(50, Long.MAX_VALUE));
        assertEquals(
                RequestRepository.RegistrationResult.REGISTERED,
                runtime.requests()
                        .register(
                                context,
                                owner,
                                new RequestFuture((kind, response, failure) -> false),
                                60_000L,
                                org.flexlb.dao.loadbalance.StrategyErrorType.BATCH_SLO_EXPIRED));
        long now = System.currentTimeMillis();
        var route =
                RequestExecutionTestSupport.registerPrefillRoute(
                        context, prefill, endpoint, handle, 0L, now);
        var placement = (RoutingContext) context.beginRouteSelection();
        context.publishRoute(placement, route);
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.completeRouting(
                                placement, RoutingCompletion.FINISHED, now));
        assertSame(
                route,
                org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(queueOwnership, route),
                "the selected route has a real exact queue owner before preparation");
        return route;
    }

    void blockDecode(long id) {
        blockedDecode.add(id);
    }

    CapacityBoundary process(List<RouteResult> items, OptionalLong prediction) {
        return worker.processSelection(items, "contract", evaluator, prediction);
    }

    void drain() {
        runtime.deliveries().awaitDeliveries();
    }

    Delivery.Result result(long id) {
        return Objects.requireNonNull(results.get(id), "missing result " + id);
    }

    List<Long> claimIds() {
        return claims.stream().map(claim -> Long.parseLong(claim.route().requestId())).toList();
    }

    List<Long> sentIds() {
        return sent.stream()
                .flatMap(request -> request.getDpSlotsList().stream())
                .flatMap(slot -> slot.getRequestsList().stream())
                .map(member -> member.getInput().getRequestId())
                .toList();
    }

    static EngineRpcService.EnqueueBatchResponsePB ack(
            EngineRpcService.EnqueueBatchRequestPB request) {
        var reply =
                EngineRpcService.EnqueueBatchResponsePB.newBuilder()
                        .setBatchId(request.getBatchId());
        request.getDpSlotsList()
                .forEach(
                        slot ->
                                slot.getRequestsList()
                                        .forEach(
                                                member ->
                                                        reply.addSuccesses(
                                                                EngineRpcService
                                                                        .EnqueueBatchSuccessPB
                                                                        .newBuilder()
                                                                        .setRequestId(
                                                                                member.getInput()
                                                                                        .getRequestId()))));
        return reply.build();
    }

    AutoCloseable blockExecution() throws Exception {
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        runtime.deliveries()
                .tryReserveDelivery()
                .value()
                .submit(
                        () -> {
                            entered.countDown();
                            try {
                                assertTrue(release.await(5, TimeUnit.SECONDS));
                            } catch (InterruptedException interrupted) {
                                Thread.currentThread().interrupt();
                                throw new AssertionError(interrupted);
                            }
                        },
                        failure -> {
                            throw new AssertionError(failure);
                        });
        assertTrue(entered.await(5, TimeUnit.SECONDS));
        return release::countDown;
    }
}
