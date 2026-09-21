package org.flexlb.balance.scheduler;

import com.google.protobuf.ByteString;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.TestBatchSubmission;
import org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.TestContext;
import org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.TestEndpointCapabilities;
import org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.TestRequestRegistry;
import org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.TestTelemetry;
import org.flexlb.config.ConfigService;
import org.flexlb.constant.GrpcConstants;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.unavailable;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.mock;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;

/** Final batch admission, transport handoff, and completion-correlation contract. */
class BatchDeliveryStrategyTest {

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void byteBudgetSplitsLargeImageRequestsButPreservesTextBatchCount(boolean images) throws Exception {
        Fixture fixture = new Fixture(701L);
        var config = SchedulingTestConfig.batchConfig();
        ConfigService configs = mock(ConfigService.class);
        when(configs.loadBalanceConfig()).thenReturn(config);
        EngineGrpcClient client = mock(EngineGrpcClient.class);
        DefaultBatchDispatcher dispatcher = new DefaultBatchDispatcher(client, configs, null, 1, 1);
        when(fixture.capabilities.prefill().getIp()).thenReturn("127.0.0.1");
        when(fixture.capabilities.prefill().getGrpcPort()).thenReturn(8090);
        ServerStatus prefill = new ServerStatus();
        prefill.setRole(RoleType.PREFILL);
        prefill.setServerIp("127.0.0.1");
        prefill.setHttpPort(8080);
        prefill.setGrpcPort(8090);
        var template = EngineRpcService.GenerateInputPB.newBuilder().addTokenIds(129264);
        if (images) {
            template.addMultimodalInputs(EngineRpcService.MultimodalInputPB.newBuilder()
                    .setMultimodalTensor(EngineRpcService.TensorPB.newBuilder()
                            .setBf16Data(ByteString.copyFrom(new byte[20 * 1024 * 1024]))));
        }
        ByteString sharedBody = template.build().toByteString();
        List<ScheduledRequest> remaining = new ArrayList<>();
        for (long id = 1; id <= 64; id++) {
            ScheduledRequest item = fixture.item(id);
            BalanceContext context = new BalanceContext(config);
            Request request = new Request();
            request.setRequestId(id);
            context.setRequest(request);
            // Rope concatenation shares the large immutable payload across all 64 inputs.
            context.setGenerateInputPb(EngineRpcService.GenerateInputPB.newBuilder()
                    .setRequestId(id).build().toByteString().concat(sharedBody));
            when(item.ctx()).thenReturn(context);
            when(item.prefill()).thenReturn(prefill);
            when(item.batchPayloadSizeUpperBound()).thenCallRealMethod();
            remaining.add(item);
        }
        List<Integer> rpcCounts = new CopyOnWriteArrayList<>();
        List<Long> rpcBatchIds = new CopyOnWriteArrayList<>();
        List<Long> sentIds = new CopyOnWriteArrayList<>();
        when(client.batchEnqueueAsync(anyString(), anyInt(), any())).thenAnswer(call -> {
            EngineRpcService.EnqueueBatchRequestPB batch = call.getArgument(2);
            assertTrue(batch.getSerializedSize() <= GrpcConstants.MAX_MESSAGE_SIZE);
            var response = EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(batch.getBatchId());
            int count = 0;
            for (var slot : batch.getDpSlotsList()) {
                for (var external : slot.getRequestsList()) {
                    var input = external.getInput();
                    sentIds.add(input.getRequestId());
                    assertEquals(images ? 1 : 0, input.getMultimodalInputsCount());
                    if (images) {
                        assertEquals(20 * 1024 * 1024,
                                input.getMultimodalInputs(0).getMultimodalTensor().getBf16Data().size());
                    }
                    response.addSuccesses(EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                            .setRequestId(input.getRequestId()));
                    count++;
                }
            }
            rpcCounts.add(count);
            rpcBatchIds.add(batch.getBatchId());
            return CompletableFuture.completedFuture(response.build());
        });
        try {
            while (!remaining.isEmpty()) {
                assertEquals("COMMITTED", fixture.context.deliver(fixture.strategy, remaining,
                        "payload_budget", 0, OptionalLong.of(9999L)));
                var batch = fixture.submission.command();
                int count = batch.exactItems().size();
                assertEquals(images ? Math.min(12, remaining.size()) : 64, count);
                assertNull(fixture.context.committedBoundary(), "size boundary must leave suffix queued");
                if (count < remaining.size()) {
                    assertEquals(count * 100L, batch.predictedMs(), "repredict the admitted prefix");
                }
                CountDownLatch completed = new CountDownLatch(count);
                var permit = dispatcher.tryPrepareSubmission();
                assertTrue(permit.accepted());
                permit.value().submit(sender -> sender.sendBatch(batch.exactItems(), batch.batchId(), batch.predictedMs(),
                        batch.decisionReason(), (item, result) -> {
                            try {
                                fixture.submission.complete(item, result);
                            } finally {
                                completed.countDown();
                            }
                        }));
                assertTrue(completed.await(10, TimeUnit.SECONDS));
                remaining = new ArrayList<>(remaining.subList(count, remaining.size()));
                fixture.correlationId++;
            }
            assertEquals(images ? List.of(12, 12, 12, 12, 12, 4) : List.of(64), rpcCounts);
            assertEquals(rpcCounts.size(), new HashSet<>(rpcBatchIds).size());
            assertEquals(64, sentIds.size());
            assertEquals(64, new HashSet<>(sentIds).size());
            assertEquals(64, fixture.slots.completions().size());
            assertTrue(fixture.slots.completions().stream()
                    .allMatch(event -> event.completion().status() == DeliveryResult.Status.DELIVERED));
        } finally {
            dispatcher.shutdown();
        }
    }

    @Test
    void preparedPredictionAndExactBatchReachTransportOnce() {
        Fixture fixture = new Fixture(701L);
        ScheduledRequest first = fixture.item(1L);
        ScheduledRequest second = fixture.item(2L);
        String result = fixture.context.deliver(
                fixture.strategy, List.of(first, second), "fixed_window", 3,
                OptionalLong.of(83L));

        assertEquals("COMMITTED", result);
        DeliveryStrategyTestSupport.SubmittedBatch command =
                fixture.submission.command();
        assertEquals(List.of(first, second), command.exactItems());
        assertEquals(701L, command.batchId());
        assertEquals(83L, command.predictedMs());
        assertEquals("fixed_window", command.decisionReason());
        assertTrue(fixture.slots.identities().stream().allMatch(identity ->
                identity.kind() == DeliveryClaimKind.BATCH_ENQUEUE
                        && identity.correlationId() == 701L));
        assertEquals(1, fixture.telemetry.batches().size());
        DeliveryStrategyTestSupport.BatchTelemetry telemetry =
                fixture.telemetry.batches().getFirst();
        assertEquals(701L, telemetry.batchId());
        assertEquals(List.of(first, second), telemetry.dispatched());
        assertEquals(83L, telemetry.predictedMs());
        assertEquals(3, telemetry.remainingQueueDepth());
        verify(fixture.capabilities.batchReservation()).commit(
                org.mockito.ArgumentMatchers.eq(List.of(first, second)),
                org.mockito.ArgumentMatchers.eq(83L));
        verify(fixture.capabilities.permit(first)).dispatch();
        verify(fixture.capabilities.permit(second)).dispatch();
        assertEquals(1, fixture.capabilities.handoffs().size());
        fixture.capabilities.handoffs().forEach(handoff -> verify(handoff).close());
        assertEquals(1, fixture.submission.totalCloseCount());
    }

    @Test
    void missingPlannedPredictionUsesFrozenEvaluatorForCommittedBatch() {
        Fixture fixture = new Fixture(702L);
        ScheduledRequest first = fixture.item(1L);
        ScheduledRequest second = fixture.item(2L);

        fixture.context.deliver(
                fixture.strategy, List.of(first, second),
                "predict", 0, OptionalLong.empty());

        assertEquals(200L, fixture.submission.command().predictedMs());
        assertEquals(200L, fixture.telemetry.batches()
                .getFirst().predictedMs());
        verify(fixture.capabilities.batchReservation()).commit(
                org.mockito.ArgumentMatchers.eq(List.of(first, second)),
                org.mockito.ArgumentMatchers.eq(200L));
    }

    @Test
    void nonPositiveBatchIdClosesSubmissionBeforeEndpointOwnership() {
        Fixture fixture = new Fixture(0L);
        ScheduledRequest item = fixture.item(1L);

        String result = fixture.context.deliver(
                fixture.strategy, List.of(item),
                "bad-id", 0, OptionalLong.empty());

        assertEquals("BOUNDARY", result);
        assertSame(item, fixture.context.emptyBoundary().item());
        RuntimeException failure = assertInstanceOf(
                RuntimeException.class,
                fixture.context.emptyBoundary().result().cause());
        assertTrue(failure.getMessage().contains("batch id supplier"));
        assertEquals(1, fixture.submission.closeCount());
        assertTrue(fixture.slots.committed().isEmpty());
    }

    @Test
    void unavailableSubmissionReturnsExactHeadBoundaryBeforeAdmission() {
        Fixture fixture = new Fixture(701L);
        ScheduledRequest head = fixture.item(1L);
        CapacityBoundary unavailable = unavailable();
        fixture.submission.prepareBoundary(unavailable);

        String result = fixture.context.deliver(
                fixture.strategy, List.of(head),
                "submission-full", 0,
                OptionalLong.empty());

        assertEquals("BOUNDARY", result);
        assertSame(head, fixture.context.emptyBoundary().item());
        assertSame(unavailable, fixture.context.emptyBoundary().result());
        assertTrue(fixture.slots.committed().isEmpty());
    }

    @Test
    void unavailableAdmissionClosesPreparedSubmissionAndReturnsBoundary() {
        Fixture fixture = new Fixture(701L);
        ScheduledRequest head = fixture.item(1L);
        fixture.capabilities.rejectPermitAt(0);

        String result = fixture.context.deliver(
                fixture.strategy, List.of(head),
                "admission-full", 0,
                OptionalLong.empty());

        assertEquals("BOUNDARY", result);
        assertSame(head, fixture.context.emptyBoundary().item());
        assertEquals(CapacityBoundary.Status.UNAVAILABLE,
                fixture.context.emptyBoundary().result().status());
        assertEquals(1, fixture.submission.closeCount());
        verify(fixture.capabilities.batchReservation()).close();
        assertTrue(fixture.slots.committed().isEmpty());
    }

    @Test
    void unavailableSuffixSubmitsLargestAdmittedPrefixAndRepredictsIt() {
        Fixture fixture = new Fixture(701L);
        ScheduledRequest first = fixture.item(1L);
        ScheduledRequest second = fixture.item(2L);
        fixture.capabilities.rejectPermitAt(1);

        String result = fixture.context.deliver(
                fixture.strategy, List.of(first, second),
                "prefix", 1, OptionalLong.of(999L));

        assertEquals("COMMITTED", result);
        assertEquals(List.of(first),
                fixture.submission.command().exactItems());
        assertEquals(100L, fixture.submission.command().predictedMs());
        assertSame(second, fixture.context.committedBoundary().item());
        assertEquals(CapacityBoundary.Status.UNAVAILABLE,
                fixture.context.committedBoundary().result().status());
        assertEquals(List.of(first), fixture.slots.committed());
    }

    @Test
    void synchronousTransportCompletionWaitsForCapabilityHandoffClose() {
        Fixture fixture = new Fixture(701L);
        ScheduledRequest first = fixture.item(1L);
        ScheduledRequest second = fixture.item(2L);
        fixture.submission.completeSynchronously(
                first, DeliveryResult.delivered());
        fixture.submission.completeSynchronously(
                second, DeliveryResult.delivered());
        fixture.slots.beforeCompletion(() -> {
            fixture.capabilities.handoffs()
                    .forEach(handoff -> verify(handoff).close());
            assertEquals(1, fixture.submission.totalCloseCount(),
                    "transport preparation must close before callbacks open");
        });

        fixture.context.deliver(
                fixture.strategy, List.of(first, second),
                "gate", 0, OptionalLong.of(20L));

        assertEquals(List.of(
                        new DeliveryStrategyTestSupport.CompletionEvent(
                                first,
                                DeliveryResult.delivered()),
                        new DeliveryStrategyTestSupport.CompletionEvent(
                                second,
                                DeliveryResult.delivered())),
                fixture.slots.completions());
    }

    @Test
    void callbackForUnsubmittedIdentityFailsClosed() {
        Fixture fixture = new Fixture(701L);
        ScheduledRequest canonical = fixture.item(1L);
        ScheduledRequest lookalike = DeliveryStrategyTestSupport.item(
                canonical.requestId(), canonical.priority(),
                canonical.enqueuedAtMs(), canonical.seqLen(),
                canonical.hitCache());
        fixture.context.deliver(
                fixture.strategy, List.of(canonical),
                "identity-fence", 0,
                OptionalLong.empty());

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> fixture.submission.complete(
                        lookalike,
                        DeliveryResult.delivered()));

        assertTrue(failure.getMessage().contains("unsubmitted identity"));
        assertTrue(fixture.slots.completions().isEmpty());
    }

    @Test
    void timeoutAndUncertainTransportOutcomesReachExactClaims() {
        Fixture fixture = new Fixture(701L);
        ScheduledRequest first = fixture.item(1L);
        ScheduledRequest second = fixture.item(2L);
        fixture.context.deliver(
                fixture.strategy, List.of(first, second),
                "outcomes", 0, OptionalLong.empty());
        RuntimeException timeout = new RuntimeException("timeout");
        RuntimeException uncertain = new RuntimeException("uncertain");

        fixture.submission.complete(
                first, DeliveryResult.timedOut(timeout));
        fixture.submission.complete(
                second, DeliveryResult.uncertain(uncertain));

        assertEquals(2, fixture.slots.completions().size());
        DeliveryResult timedOut =
                fixture.slots.completions().get(0).completion();
        DeliveryResult unresolved =
                fixture.slots.completions().get(1).completion();
        assertEquals(DeliveryResult.Status.TIMED_OUT,
                timedOut.status());
        assertEquals(DeliveryResult.Status.UNCERTAIN,
                unresolved.status());
        assertSame(timeout, timedOut.cause());
        assertSame(uncertain, unresolved.cause());
    }

    @Test
    void lostClaimExcludesOnlyThatMemberFromSubmittedBatch() {
        Fixture fixture = new Fixture(701L);
        ScheduledRequest first = fixture.item(1L);
        ScheduledRequest second = fixture.item(2L);
        fixture.slots.commitLostFor(first);
        fixture.capabilities.precedingWork(new WorkSnapshot(1_000L, List.of(new WorkSnapshot.RequestWork(99L, WorkSnapshot.Phase.COMMITTED, 25L)), List.of(), 0L));

        fixture.context.deliver(
                fixture.strategy, List.of(first, second),
                "claim-race", 0,
                OptionalLong.of(999L));

        assertEquals(List.of(second),
                fixture.submission.command().exactItems());
        assertEquals(100L, fixture.submission.command().predictedMs());
        DeliveryStrategyTestSupport.BatchTelemetry telemetry =
                fixture.telemetry.batches().getFirst();
        assertEquals(List.of(second), telemetry.dispatched());
        assertEquals(100L, telemetry.predictedMs());
        assertEquals(Map.of(second, 100L),
                fixture.slots.unstartedWorkMs(),
                "cancelled members must not inflate the delivered batch lifetime");
        assertEquals(125L, fixture.slots.remainingWorkMsAt(second, 1_000L).orElseThrow());
    }

    @Test
    void zeroPredictionStillSubmitsTheBatch() {
        Fixture fixture = new Fixture(701L);
        ScheduledRequest item = fixture.item(1L);

        fixture.context.deliver(fixture.strategy, List.of(item),
                "zero", 0, OptionalLong.of(0L));

        assertEquals(List.of(item), fixture.submission.command().exactItems());
        assertEquals(Map.of(item, 0L), fixture.slots.unstartedWorkMs());
    }

    @Test
    void unknownPrecedingWorkStillSubmitsWithUnknownRemainingTime() {
        Fixture fixture = new Fixture(701L);
        ScheduledRequest item = fixture.item(1L);
        fixture.capabilities.precedingWork(new WorkSnapshot(2_000L, List.of(), List.of(), 1L));

        fixture.context.deliver(fixture.strategy, List.of(item),
                "unknown", 0, OptionalLong.of(100L));

        assertEquals(List.of(item), fixture.submission.command().exactItems());
        assertEquals(Map.of(item, 100L),
                fixture.slots.unstartedWorkMs());
        assertTrue(fixture.slots.remainingWorkMsAt(item, 2_000L).isEmpty());
    }

    private static final class Fixture {
        private final TestBatchSubmission submission =
                new TestBatchSubmission();
        private final TestEndpointCapabilities capabilities =
                new TestEndpointCapabilities();
        private final TestRequestRegistry slots = new TestRequestRegistry();
        private final TestTelemetry telemetry = new TestTelemetry();
        private final TestContext context = new TestContext();
        private long correlationId;
        private final BatchDeliveryStrategy strategy;

        private Fixture(long correlationId) {
            this.correlationId = correlationId;
            this.strategy = new BatchDeliveryStrategy(
                    submission::tryPrepareSubmission,
                    () -> this.correlationId,
                    slots.requests(),
                    telemetry.metrics());
        }

        private ScheduledRequest item(long requestId) {
            ScheduledRequest item = DeliveryStrategyTestSupport.item(requestId);
            capabilities.bind(item);
            return item;
        }
    }
}
