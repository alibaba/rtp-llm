package org.flexlb.balance.scheduler;

import com.google.protobuf.ByteString;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.*;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.config.*;
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

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;

import static org.flexlb.balance.scheduler.ContextResult.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/** The production Worker -> Dispatcher -> Runtime path; only protocol owners and transport are simulated. */
@Timeout(15)
class BatchDeliveryTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void byteBudgetSplitsLargeImageRequestsButPreservesTextBatchCount(boolean images) throws Exception {
        try (var f = new LinearDeliveryFixture(true)) {
            var template = EngineRpcService.GenerateInputPB.newBuilder().addTokenIds(129264);
            if (images) {
                template.addMultimodalInputs(EngineRpcService.MultimodalInputPB.newBuilder()
                        .setMultimodalTensor(EngineRpcService.TensorPB.newBuilder()
                                .setBf16Data(ByteString.copyFrom(new byte[20 * 1024 * 1024]))));
            }
            ByteString body = template.build().toByteString();
            List<RequestRoute> remaining = new ArrayList<>();
            for (long id = 1; id <= 64; id++) {
                remaining.add(f.item(id, 100, EngineRpcService.GenerateInputPB.newBuilder()
                        .setRequestId(id).build().toByteString().concat(body)));
            }
            var ids = new java.util.concurrent.atomic.AtomicLong(701L);
            var execution = f.runtime.deliveries();
            doAnswer(call -> ids.getAndIncrement()).when(execution).nextBatchId();
            int maxImageBatchCount = (org.flexlb.constant.GrpcConstants.MAX_MESSAGE_SIZE - 16)
                    / (20 * 1024 * 1024 + 1024);
            List<Integer> expectedCounts = new ArrayList<>();
            while (!remaining.isEmpty()) {
                f.handoffClosed.set(false);
                assertNull(f.process(remaining, OptionalLong.of(9999L)), "size boundary must leave suffix queued");
                int count = images ? Math.min(maxImageBatchCount, remaining.size()) : remaining.size();
                assertEquals(remaining.subList(0, count), f.committedItems);
                if (count < remaining.size()) { assertEquals(count * 100L, f.committedPrediction); }
                int completed = 64 - remaining.size() + count;
                RequestProtocolTestSupport.awaitCondition(() -> f.results.size() == completed);
                expectedCounts.add(count);
                remaining = new ArrayList<>(remaining.subList(count, remaining.size()));
            }
            f.drain();
            assertEquals(expectedCounts, f.sent.stream().map(batch -> batch.getDpSlotsList().stream()
                    .mapToInt(slot -> slot.getRequestsCount()).sum()).toList());
            assertEquals(f.sent.size(), f.sent.stream().map(batch -> batch.getBatchId()).distinct().count());
            assertEquals(64, f.sentIds().size());
            assertEquals(64, new HashSet<>(f.sentIds()).size());
            f.sent.forEach(batch -> {
                assertTrue(batch.getSerializedSize() <= org.flexlb.constant.GrpcConstants.MAX_MESSAGE_SIZE);
                batch.getDpSlotsList().forEach(slot -> slot.getRequestsList().forEach(member -> {
                    var input = member.getInput();
                    assertEquals(images ? 1 : 0, input.getMultimodalInputsCount());
                    if (images) {
                        assertEquals(20 * 1024 * 1024,
                                input.getMultimodalInputs(0).getMultimodalTensor().getBf16Data().size());
                    }
                }));
            });
            assertTrue(f.results.values().stream().allMatch(result -> result.status() == DeliveryResult.Status.DELIVERED));
        }
    }

    @Test
    void queuedBatchStillRechecksCurrentDispatcherConfigurationBeforeRpc() {
        try (var f = new LinearDeliveryFixture(true)) {
            var item = f.item(1, 10);
            f.config.getDispatcher().setType(DispatcherConfig.Type.NON_BATCH);
            assertEquals(DispatcherConfig.Type.BATCH, f.dispatcher.mode(), "topology mode remains frozen");
            f.process(List.of(item), OptionalLong.of(10));
            f.drain();
            assertEquals(DeliveryResult.Status.NOT_SENT, f.result(1).status());
            verifyNoInteractions(f.grpc);
            verify(f.handoff).close();
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
            verify(f.prefill, never()).release(any(PrefillState.Reservation.class), isNull());
            assertEquals(DeliveryResult.Status.DELIVERED, f.result(1).status());
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
            verify(f.prefill).release(f.reservation, null);
            assertTrue(f.runtime.deliveries().deliveryCapacityAvailable());
            verifyNoInteractions(f.grpc);
        }
    }

    @Test
    void firstMemberAdmissionFailureReturnsBoundaryInsteadOfFailingTheWorker() {
        try (var f = new LinearDeliveryFixture(true)) {
            var item = f.item(1, 10);
            var failure = new IllegalStateException("prefill reservation unavailable");
            when(f.prefill.reserveBatch(any(), anyLong(), anyInt())).thenThrow(failure);
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
            assertSame(primary, assertThrows(IllegalStateException.class,
                    () -> f.process(List.of(first, second), OptionalLong.empty())));
            assertTrue(Arrays.asList(primary.getSuppressed()).contains(cleanup));
            verify(f.permits.get(2L)).release();
            verify(f.prefill).release(f.reservation, null);
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
                f.lostClaims.add(items.get(lostIndex).requestId());
                // Changing the endpoint predictor after queue commit must not change the captured evaluator.
                var changed = mock(PrefillTimePredictor.class);
                when(changed.evaluator()).thenThrow(new AssertionError("captured evaluator was discarded"));
                ReflectionTestUtils.setField(f.prefill, "predictor", changed);
            }
            f.drain();
            var expected = items.stream().filter(item -> !f.lostClaims.contains(item.requestId())).toList();
            assertEquals(expected.stream().map(RequestRoute::requestId).toList(), f.sentIds());
            long prediction = expected.stream().mapToLong(RequestRoute::seqLen).sum();
            expected.forEach(item -> assertEquals(prediction, f.predictions.get(item.requestId()).longValue()));
            assertFalse(f.results.containsKey(items.get(lostIndex).requestId()));
            verify(f.permits.get(items.get(lostIndex).requestId())).release();
            verify(f.evaluator).predictBatchMs(argThat(features -> features.items().size() == 2));
        }
    }

    @Test
    void completedRpcPublishesAfterHandoffCloseAndNeverUnderOwnerLocks() {
        try (var f = new LinearDeliveryFixture(true)) {
            var first = f.item(1, 10);
            var second = f.item(2, 20);
            f.beforeCompletion = claim -> {
                assertTrue(f.handoffClosed.get());
                assertFalse(Thread.holdsLock(claim.item().ctx()));
                assertFalse(f.queueLock.isHeldByCurrentThread());
            };
            f.process(List.of(first, second), OptionalLong.of(30));
            f.drain();
            assertEquals(List.of(1L, 2L), f.sentIds());
            assertEquals(2, f.results.size());
            verify(f.handoff, times(1)).close();
            assertTrue(f.events.indexOf("handoff-close") < f.events.indexOf("complete-1"));
        }
    }

    @Test
    void payloadFailureAfterClaimIsNotSentAndNeverInvokesRpc() {
        try (var f = new LinearDeliveryFixture(true)) {
            var item = f.item(1, 10, ByteString.EMPTY);
            f.process(List.of(item), OptionalLong.of(10));
            f.drain();
            assertEquals(DeliveryResult.Status.NOT_SENT, f.result(1).status());
            verifyNoInteractions(f.grpc);
            assertTrue(f.handoffClosed.get());
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"THROW", "NULL_FUTURE", "EXCEPTIONAL_FUTURE", "NULL_RESPONSE", "WRONG_BATCH"})
    void failuresAfterRpcInvocationRemainUncertain(String outcome) {
        try (var f = new LinearDeliveryFixture(true)) {
            doAnswer(call -> {
                f.sent.add(call.getArgument(2));
                return switch (outcome) {
                    case "THROW" -> throw new IllegalStateException("invoked transport failed");
                    case "NULL_FUTURE" -> null;
                    case "EXCEPTIONAL_FUTURE" -> CompletableFuture.failedFuture(new TimeoutException("remote timeout"));
                    case "NULL_RESPONSE" -> CompletableFuture.completedFuture(null);
                    case "WRONG_BATCH" -> CompletableFuture.completedFuture(
                            EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(702).build());
                    default -> throw new AssertionError(outcome);
                };
            }).when(f.grpc).batchEnqueueAsync(anyString(), anyInt(), any());
            f.process(List.of(f.item(1, 10), f.item(2, 20)), OptionalLong.of(30));
            f.drain();
            assertEquals(2, f.results.size());
            f.results.values().forEach(result -> assertEquals(DeliveryResult.Status.UNCERTAIN, result.status()));
            assertEquals(Map.of(1L, 1, 2L, 1), f.completionCounts);
            verify(f.grpc, times(1)).batchEnqueueAsync(anyString(), anyInt(), any());
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {0, 1, 2, 3, 4, 5, 6, 7})
    void sendBoundaryFiltersEveryCancelledSubsetWithoutChangingOtherResults(int refusedMask) {
        try (var f = new LinearDeliveryFixture(true)) {
            var items = List.of(f.item(1, 10), f.item(2, 20), f.item(3, 30));
            for (int i = 0; i < 3; i++) { if ((refusedMask & (1 << i)) != 0) { f.refusedSends.add((long) i + 1); } }
            f.process(items, OptionalLong.of(60));
            f.drain();
            var expected = items.stream().filter(item -> !f.refusedSends.contains(item.requestId())).map(RequestRoute::requestId).toList();
            assertEquals(expected, f.sentIds());
            for (var item : items) {
                assertEquals(f.refusedSends.contains(item.requestId()) ? DeliveryResult.Status.NOT_SENT
                        : DeliveryResult.Status.DELIVERED, f.result(item.requestId()).status());
                assertEquals(1, f.completionCounts.get(item.requestId()));
            }
            if (expected.isEmpty()) { verifyNoInteractions(f.grpc); }
        }
    }

    @Test
    void failedFactIsRetriedWithoutReclassifyingOrSuppressingItsSiblings() {
        try (var f = new LinearDeliveryFixture(true)) {
            f.expectedShutdownFailure = new IllegalStateException("owner callback failed");
            f.throwCompletionFor = 1;
            f.process(List.of(f.item(1, 10), f.item(2, 20)), OptionalLong.of(30));
            f.drain();
            assertEquals(DeliveryResult.Status.DELIVERED, f.result(1).status());
            assertEquals(DeliveryResult.Status.DELIVERED, f.result(2).status());
            assertEquals(Map.of(1L, 3, 2L, 1), f.completionCounts);
            verify(f.grpc, times(1)).batchEnqueueAsync(anyString(), anyInt(), any());
        }
    }

    @Test
    void temporaryCleanupFailureCannotDiscardAnAcceptedRpcResponse() {
        try (var f = new LinearDeliveryFixture(true)) {
            var first = f.item(1, 10);
            var second = f.item(2, 20);
            var permitFailure = new IllegalStateException("permit cleanup failed");
            var handoffFailure = new IllegalStateException("handoff close failed");
            f.expectedShutdownFailure = permitFailure;
            doThrow(permitFailure).when(f.permits.get(1L)).release();
            doAnswer(call -> { f.handoffClosed.set(true); throw handoffFailure; })
                    .when(f.handoff).close();
            f.process(List.of(first, second), OptionalLong.of(30));
            f.drain();
            assertEquals(DeliveryResult.Status.DELIVERED, f.result(1).status());
            assertEquals(DeliveryResult.Status.DELIVERED, f.result(2).status());
            verify(f.permits.get(2L)).release();
            verify(f.handoff).close();
            assertTrue(Arrays.asList(permitFailure.getSuppressed()).contains(handoffFailure));
        }
    }

    @Test
    void executorRejectionAfterDispatcherEntryTerminalizesEveryCommittedOwnerExactlyOnce() {
        try (var f = new LinearDeliveryFixture(true)) {
            var first = f.item(1, 10);
            var second = f.item(2, 20);
            ((ThreadPoolExecutor) ReflectionTestUtils.getField(f.runtime.deliveries(), "dispatchExecutor")).shutdown();
            assertDoesNotThrow(() -> f.process(List.of(first, second), OptionalLong.of(30)));
            f.drain();
            assertEquals(Set.of(1L, 2L), f.preparationFailures.keySet());
            assertTrue(f.results.isEmpty());
            verify(f.owner, times(1)).onDeliveryEvidence(isNull(), argThat(e -> e instanceof PreparationFailure failure && failure.route() == first));
            verify(f.owner, times(1)).onDeliveryEvidence(isNull(), argThat(e -> e instanceof PreparationFailure failure && failure.route() == second));
            verify(f.permits.get(1L), times(1)).release();
            verify(f.permits.get(2L), times(1)).release();
            verify(f.handoff, times(1)).close();
            verifyNoInteractions(f.grpc);
        }
    }
}

/** No transaction/state-machine fake: calls real preparation, commitment orchestration and asynchronous transport. */
final class LinearDeliveryFixture implements AutoCloseable {
    final FlexlbConfig config;
    final DeliveryMetricsReporter reporter = mock(DeliveryMetricsReporter.class);
    final EngineGrpcClient grpc = mock(EngineGrpcClient.class);
    final PrefillEndpoint prefill = mock(PrefillEndpoint.class);
    final DecodeEndpoint decode = org.flexlb.balance.scheduler.SchedulerTestSupport.mockDecodeEndpoint();
    final AbstractRequestScheduler owner = mock(AbstractRequestScheduler.class);
    final PrefillState.BatchReservation reservation = mock(PrefillState.BatchReservation.class);
    final PrefillState.CommittedHandoff handoff = mock(PrefillState.CommittedHandoff.class);
    final ReentrantLock queueLock = new ReentrantLock();
    final PrefillState queueOwnership = new PrefillState(queueLock, PrefillActiveIndex.ordered(16, WorkerBatcher.PRIORITY_QUEUE_ORDER), System::currentTimeMillis);
    final AtomicBoolean handoffClosed = new AtomicBoolean();
    final List<String> events = new CopyOnWriteArrayList<>();
    final List<DeliveryClaim> claims = new CopyOnWriteArrayList<>();
    final List<EngineRpcService.EnqueueBatchRequestPB> sent = new CopyOnWriteArrayList<>();
    final Map<Long, DecodeEndpoint.EngineDispatchPermit> permits = new ConcurrentHashMap<>();
    final Map<Long, Long> predictions = new ConcurrentHashMap<>();
    final Map<Long, DeliveryResult> results = new ConcurrentHashMap<>();
    final Map<Long, Integer> completionCounts = new ConcurrentHashMap<>();
    final Map<Long, String> preparationFailures = new ConcurrentHashMap<>();
    final Set<Long> lostClaims = ConcurrentHashMap.newKeySet();
    final Set<Long> refusedSends = ConcurrentHashMap.newKeySet();
    final Set<Long> blockedDecode = ConcurrentHashMap.newKeySet();
    final SchedulerRuntime runtime;
    final DeliveryDispatcher dispatcher;
    final WorkerBatcher worker;
    final PrefillTimePredictor.Evaluator evaluator = spy(new PrefillTimePredictor.Evaluator() {
        @Override public long estimateMs(long totalTokens, long hitTokens) { return totalTokens - hitTokens; }
        @Override public double predictBatchMs(PrefillBatchFeatures features) {
            return features.items().stream().mapToLong(PrefillBatchFeatures.Item::seqLen).sum();
        }
    });
    volatile List<RequestRoute> committedItems = List.of();
    volatile long committedPrediction;
    volatile long[] committedRoutePredictions;
    volatile long throwCompletionFor = -1;
    RuntimeException expectedShutdownFailure;
    volatile Consumer<DeliveryClaim> beforeCompletion = ignored -> {};
    final List<DecodeEndpoint> realEndpoints = new ArrayList<>();

    LinearDeliveryFixture(boolean batch) {
        config = spy(SchedulingTestConfig.batchConfig());
        if (!batch) { config.getDispatcher().setType(DispatcherConfig.Type.NON_BATCH); }
        var sizing = spy(config.getInternalRuntime());
        doReturn(1).when(sizing).getBatchDispatchThreads();
        doReturn(8).when(sizing).getBatchDispatchQueueCapacity();
        doReturn(1).when(sizing).getBatchDispatchCompletionThreads();
        doReturn(sizing).when(config).getInternalRuntime();
        var service = SchedulingTestConfig.configService(config);
        dispatcher = new DeliveryDispatcher(grpc, service, reporter);
        runtime = spy(new SchedulerRuntime(new RequestRepository(), mock(EndpointRegistry.class), reporter,
                mock(RequestSchedulerReporter.class), dispatcher, service, mock(EngineGrpcClient.class)));
        var execution = spy(runtime.deliveries());
        ReflectionTestUtils.setField(runtime, "deliveries", execution);
        doReturn(701L).when(execution).nextBatchId();
        ReflectionTestUtils.setField(owner, "runtime", runtime);
        worker = new WorkerBatcher("contract", prefill, QueueExecutionSettings.capture(config), dispatcher,
                queueOwnership);
        when(prefill.getIp()).thenReturn("127.0.0.1");
        when(prefill.getGrpcPort()).thenReturn(8090);
        when(prefill.reserveBatch(any(), anyLong(), anyInt())).thenReturn(
                new PrefillState.ReservationResult<>(PrefillState.CapacityStatus.ACQUIRED, reservation));
        when(prefill.commitLocked(any(PrefillState.CommitPlan.class))).thenAnswer(call -> {
            assertTrue(queueLock.isHeldByCurrentThread());
            var plan = call.getArgument(0, PrefillState.CommitPlan.class);
            committedItems = List.copyOf(plan.items());
            if (plan.batchReservation() != null) { committedPrediction = plan.predictions()[0]; }
            else { committedRoutePredictions = plan.predictions(); }
            return handoff;
        });
        var work = mock(PrefillState.WorkCapture.class);
        when(handoff.precedingWork()).thenReturn(work);
        when(work.materialize()).thenAnswer(call -> {
            assertFalse(queueLock.isHeldByCurrentThread());
            return new WorkSnapshot(0, List.of(), List.of(), 0);
        });
        doAnswer(call -> { events.add("handoff-close"); handoffClosed.set(true); return null; }).when(handoff).close();
        when(owner.claimDelivery(any(), any(), anyLong(), any())).thenAnswer(call -> {
            RequestRoute item = call.getArgument(0);
            events.add("claim-" + item.requestId());
            if (lostClaims.contains(item.requestId())) { return null; }
            DecodeEndpoint.EngineDispatchPermit permit = call.getArgument(3);
            if (permit != null && permit.dispatch().status() != DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED) { return null; }
            var claim = (DeliveryClaim) ((Granted<?>) item.ctx().beginDelivery(item, call.getArgument(1),
                    call.getArgument(2), System.currentTimeMillis())).value();
            item.ctx().commitDelivery(claim, true, System.currentTimeMillis());
            claims.add(claim);
            return claim;
        });
        when(owner.tryStartSend(any())).thenAnswer(call -> !refusedSends.contains(((DeliveryClaim) call.getArgument(0)).item().requestId()));
        doAnswer(call -> {
            DeliveryEvidence evidence = call.getArgument(1);
            if (evidence instanceof Prediction prediction) {
                predictions.put(prediction.delivery().item().requestId(), duration(prediction.predictedAtMs(), prediction.nowMs()));
            } else if (evidence instanceof RouteDelivered delivered) {
                predictions.put(delivered.delivery().item().requestId(), duration(delivered.predictedAtMs(), delivered.nowMs()));
            } else if (evidence instanceof TransportResult result) {
                var claim = result.delivery();
                beforeCompletion.accept(claim);
                results.put(claim.item().requestId(), result.result());
                completionCounts.merge(claim.item().requestId(), 1, Integer::sum);
                events.add("complete-" + claim.item().requestId());
                if (claim.item().requestId() == throwCompletionFor) { throw expectedShutdownFailure; }
            } else if (evidence instanceof PreparationFailure failed) {
                preparationFailures.put(failed.route().requestId(), failed.detail());
            }
            return null;
        }).when(owner).onDeliveryEvidence(any(), any());
        when(decode.acquireDispatchPermit(any(), any())).thenAnswer(call -> {
            DecodeResources.ReservationHandle handle = call.getArgument(0);
            return blockedDecode.contains(handle.requestId())
                    ? new org.flexlb.balance.endpoint.DecodeResources.Acquisition<>(DecodeResources.EngineDispatchPermitAcquireStatus.CAPACITY_FULL, null)
                    : new org.flexlb.balance.endpoint.DecodeResources.Acquisition<>(DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED, permits.get(handle.requestId()));
        });
        when(grpc.batchEnqueueAsync(anyString(), anyInt(), any())).thenAnswer(call -> {
            var request = call.getArgument(2, EngineRpcService.EnqueueBatchRequestPB.class);
            assertFalse(queueLock.isHeldByCurrentThread());
            claims.forEach(claim -> assertFalse(Thread.holdsLock(claim.item().ctx())));
            assertFalse(handoffClosed.get(), "generation handoff must cover RPC invocation");
            sent.add(request);
            return CompletableFuture.completedFuture(ack(request));
        });
    }

    RequestRoute item(long id, long seqLen) { return item(id, seqLen, (ByteString) null); }

    RequestRoute item(long id, long seqLen, ByteString payload) {
        var permit = mock(DecodeEndpoint.EngineDispatchPermit.class);
        when(permit.dispatch()).thenReturn(org.flexlb.balance.endpoint.DecodeResources.ResourceChange.of(DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED, false));
        permits.put(id, permit);
        return item(id, seqLen, decode, new DecodeResources.ReservationHandle(1, id, id), payload);
    }

    RequestRoute item(long id, long seqLen, DecodeEndpoint endpoint, DecodeResources.ReservationHandle handle) {
        return item(id, seqLen, endpoint, handle, null);
    }

    RequestRoute item(long id, long seqLen, DecodeEndpoint endpoint, DecodeResources.ReservationHandle handle, ByteString payload) {
        var context = new RequestContext(config);
        context.data().setRequest(Request.builder().requestId(id).seqLen(seqLen).rawPriority(50).build());
        context.data().setFuture(new CompletableFuture<>());
        context.data().setGenerateInputPb(payload != null ? payload : EngineRpcService.GenerateInputPB.newBuilder().setRequestId(id)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()).build().toByteString());
        context.data().register(owner, new RequestFuture((kind, response, failure) -> false));
        long now = System.currentTimeMillis();
        context.register(id, now, Long.MAX_VALUE, 60_000L);
        var route = SchedulingTestConfig.createRoute(context, null, null, null, prefill, endpoint, handle, now);
        var placement = (PlacementToken) ((Granted<?>) context.beginPlacement(null)).value();
        context.bindPlacement(placement, route, PublicationResult.PREPARED);
        context.bindPlacement(placement, route, PublicationResult.PUBLISHED);
        context.completePlacement(placement, PublicationResult.PUBLISHED, now);
        assertSame(route, queueOwnership.updateQueue(PrefillState.QueueChange.ENQUEUE, route, 0L, now),
                "the selected route has a real exact queue owner before preparation");
        return route;
    }

    long duration(long atMs, long nowMs) {
        return Math.round((atMs - nowMs - 10_000L) / config.getRequestLifecycle().getDecision().getLifetime());
    }

    DecodeEndpoint realDecode() {
        var endpoint = EndpointTestSupport.decode(WorkerStatus.createDiscovered(RoleType.DECODE,
                "contract", "127.0.0.1", 8000, 8001, "contract"), new RequestRepository());
        realEndpoints.add(endpoint);
        return endpoint;
    }

    RequestRoute realItem(DecodeEndpoint endpoint, long id, long seqLen) {
        DecodeResources.ReservationHandle handle;
        try (var pin = endpoint.tryPinGeneration()) {
            handle = endpoint.tryReserveQueuedRequest(pin, id, 100, 200, 50, null).handle();
        }
        assertNotNull(handle);
        return item(id, seqLen, endpoint, handle);
    }

    void blockDecode(long id) { blockedDecode.add(id); }
    CapacityBoundary process(List<RequestRoute> items, OptionalLong prediction) {
        return worker.processSelection(items, "contract", evaluator, prediction);
    }
    void drain() { runtime.deliveries().awaitDeliveries(); }
    DeliveryResult result(long id) { return Objects.requireNonNull(results.get(id), "missing result " + id); }
    List<Long> claimIds() { return claims.stream().map(claim -> claim.item().requestId()).toList(); }
    List<Long> sentIds() {
        return sent.stream().flatMap(request -> request.getDpSlotsList().stream())
                .flatMap(slot -> slot.getRequestsList().stream()).map(member -> member.getInput().getRequestId()).toList();
    }
    static EngineRpcService.EnqueueBatchResponsePB ack(EngineRpcService.EnqueueBatchRequestPB request) {
        var reply = EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(request.getBatchId());
        request.getDpSlotsList().forEach(slot -> slot.getRequestsList().forEach(member -> reply.addSuccesses(
                EngineRpcService.EnqueueBatchSuccessPB.newBuilder().setRequestId(member.getInput().getRequestId()))));
        return reply.build();
    }
    AutoCloseable blockExecution() throws Exception {
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        runtime.deliveries().tryReserveDelivery().value().submit(() -> {
            entered.countDown();
            try { assertTrue(release.await(5, TimeUnit.SECONDS)); }
            catch (InterruptedException interrupted) { Thread.currentThread().interrupt(); throw new AssertionError(interrupted); }
        }, failure -> { throw new AssertionError(failure); });
        assertTrue(entered.await(5, TimeUnit.SECONDS));
        return release::countDown;
    }
    @Override public void close() {
        realEndpoints.forEach(endpoint -> { endpoint.close(); endpoint.awaitRetirement(); });
        if (expectedShutdownFailure == null) { runtime.shutdown(); }
        else { assertSame(expectedShutdownFailure, assertThrows(RuntimeException.class, runtime::shutdown)); }
    }
}
