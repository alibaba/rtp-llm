package org.flexlb.mockengine;

import io.grpc.Context;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.MultimodalRpcServiceGrpc;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.jupiter.api.Assertions.*;
import static org.flexlb.mockengine.MockEngineTestSupport.*;

/** Behavioral vectors from PrefillBatchRpcServerTest, including both sides of the terminal-cause race. */
@Timeout(15)
class MockBatchAdmissionTest {
    @TempDir Path directory;
    private MockPerformanceModel model(int cap) throws Exception {
        var model = performanceModel(directory, "500", 1);
        model.setPrefillBatchPolicy(new MockPrefillBatchPolicy(cap, 10000, 10000, 1, false, 0, 0, false));
        model.pdPolicy = new MockPdPolicy(0, 5000, 10000, 1, 5000, 10000, 128, 2000);
        return model;
    }
    private static EngineRpcService.GenerateInputPB typed(long id, int vitPort, int decodePort) {
        var input = MockV41ImageTest.input(MockV41ImageTest.image("a".repeat(64)), vitPort).toBuilder().setRequestId(id);
        input.getGenerateConfigBuilder().addRoleAddrs(EngineRpcService.RoleAddrPB.newBuilder().setRoleStr("DECODE").setGrpcPort(decodePort));
        return input.build();
    }
    private static void success(StreamObserver<EngineRpcService.MultimodalOutputPB> observer) throws Exception {
        var result = MockV41Image.compute(MockV41ImageTest.image("a".repeat(64)), MockV41ImageTest.config(), 1 << 20);
        observer.onNext(EngineRpcService.MultimodalOutputPB.newBuilder().addSplitSize(5).setMultimodalEmbedding(result.embedding()).build());
        observer.onCompleted();
    }
    private static void await(java.util.function.BooleanSupplier done) throws Exception {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (!done.getAsBoolean() && System.nanoTime() < deadline) Thread.sleep(1);
        assertTrue(done.getAsBoolean());
    }
    private static void answers(EngineRpcService.EnqueueBatchResponsePB response, int expected) {
        assertEquals(expected, response.getSuccessesCount() + response.getErrorsCount());
        var ids = new java.util.HashSet<Long>();
        response.getSuccessesList().forEach(value -> assertTrue(ids.add(value.getRequestId()), "duplicate ACK"));
        response.getErrorsList().forEach(value -> assertTrue(ids.add(value.getRequestId()), "success/error overlap"));
    }
    @Test void localDpSlotsFlattenIntoOneGroupAndEnvelopeOverridesInputMetadata() throws Exception {
        try (var c = MockEngineTestCluster.create(model(4), 62880, 1, 0)) {
            var p = c.prefill(0);
            p.batchAdmissionObserver = phase -> {
                if (!phase.equals("prepared")) return;
                try {
                    var registry = p.getClass().getDeclaredField("batchAdmissions"); registry.setAccessible(true);
                    for (var value : ((java.util.Map<?, ?>)registry.get(p)).values()) {
                        var field = value.getClass().getDeclaredField("input"); field.setAccessible(true);
                        var in = (EngineRpcService.GenerateInputPB)field.get(value);
                        assertEquals(2, in.getGroupSize()); assertEquals(77, in.getGroupId().getValue());
                    }
                } catch (ReflectiveOperationException error) { throw new AssertionError(error); }
            };
            var in = input(1, 10).toBuilder().setGroupSize(999).setGroupId(com.google.protobuf.Int64Value.of(99)).build();
            var ack = enqueueAndFetch(p, batch(77, slot(0, in), slot(0, input(2, 10))));
            answers(ack, 2); assertEquals(2, ack.getSuccessesCount());
            assertEquals(1, ((Number)p.getSnapshot().get("prefill_batches")).longValue());
        }
    }
    @Test void splitSlotsCanPrepareConcurrently() throws Exception {
        var entered = new CountDownLatch(2); var release = new CountDownLatch(1);
        Server vit = ServerBuilder.forPort(0).addService(new MultimodalRpcServiceGrpc.MultimodalRpcServiceImplBase() {
            public void remoteMultimodalEmbedding(EngineRpcService.MultimodalInputsPB input, StreamObserver<EngineRpcService.MultimodalOutputPB> observer) {
                entered.countDown(); try { release.await(); success(observer); } catch (Exception error) { observer.onError(error); }
            }
        }).build().start();
        try (var c = MockEngineTestCluster.create(model(4), 62880, 1, 1)) {
            var request = batch(77, slot(0, typed(1, vit.getPort(), c.decode(0).getGrpcPort())),
                    slot(0, typed(2, vit.getPort(), c.decode(0).getGrpcPort())));
            var ack = CompletableFuture.supplyAsync(() -> enqueueAndFetch(c.prefill(0), request));
            try { assertTrue(entered.await(3, TimeUnit.SECONDS)); } finally { release.countDown(); }
            answers(ack.get(5, TimeUnit.SECONDS), 2);
            assertEquals(2, ack.get().getSuccessesCount());
        } finally { release.countDown(); vit.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
    }
    @Test void failedPrepareIsRemovedBeforeMixedExecutionModeCheck() throws Exception {
        var calls = new AtomicInteger();
        Server vit = ServerBuilder.forPort(0).addService(new MultimodalRpcServiceGrpc.MultimodalRpcServiceImplBase() {
            public void remoteMultimodalEmbedding(EngineRpcService.MultimodalInputsPB input, StreamObserver<EngineRpcService.MultimodalOutputPB> observer) {
                calls.incrementAndGet(); observer.onError(Status.INVALID_ARGUMENT.withDescription("bad image").asRuntimeException());
            }
        }).build().start();
        try (var c = MockEngineTestCluster.create(model(4), 62880, 1, 1)) {
            var bad = typed(1, vit.getPort(), c.decode(0).getGrpcPort()).toBuilder();
            bad.getGenerateConfigBuilder().setMaxNewTokens(0);
            var ack = enqueueAndFetch(c.prefill(0), batch(77, slot(0, bad.build(), inputWithDecode(2, 10, c.decode(0).getGrpcPort()))));
            answers(ack, 2); assertEquals(1, calls.get());
            assertEquals(2, ack.getSuccesses(0).getRequestId()); assertEquals(1, ack.getErrors(0).getRequestId());
        } finally { vit.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
    }
    @Test void remainingMixedModesRejectTogetherAcrossDpSlots() throws Exception {
        try (var c = MockEngineTestCluster.create(model(4), 62880, 1, 1)) {
            var only = input(1, 10).toBuilder(); only.getGenerateConfigBuilder().setMaxNewTokens(0);
            var ack = enqueue(c.prefill(0), batch(77, slot(0, only.build()), slot(0, inputWithDecode(2, 10, c.decode(0).getGrpcPort()))));
            answers(ack, 2); assertEquals(0, ack.getSuccessesCount());
            ack.getErrorsList().forEach(error -> assertEquals(605, error.getErrorInfo().getErrorCode()));
            assertEquals(0, c.decode(0).getOccupiedKvTokens()); assertFalse(c.prefill(0).hasInflightWork());
        }
    }
    @Test void inputLengthCheckFollowsPrepareAndOnlyRejectsInvalidMember() throws Exception {
        var model = model(4);
        model.setDecodeReserveStep(1);
        model.setPrefillBatchPolicy(new MockPrefillBatchPolicy(4, 10000, 20, 1, false, 0, 0, false));
        try (var c = MockEngineTestCluster.create(model, 62880, 1, 1)) {
            var field = c.decode(0).getClass().getDeclaredField("performance"); field.setAccessible(true);
            ((MockPerformanceModel)field.get(c.decode(0))).setPrefillBatchPolicy(
                    new MockPrefillBatchPolicy(4, 10000, 10000, 1, false, 0, 0, false));
            c.prefill(0).batchAdmissionObserver = phase -> {
                if (phase.equals("prepared")) assertTrue(c.decode(0).getOccupiedKvTokens() > 0, "D allocated before P input-length check");
            };
            var ack = enqueueAndFetch(c.prefill(0), batch(77, slot(0, inputWithDecode(1, 30, c.decode(0).getGrpcPort()),
                    inputWithDecode(2, 10, c.decode(0).getGrpcPort()))));
            answers(ack, 2); assertEquals(1, ack.getSuccessesCount(), ack.toString());
            assertEquals(2, ack.getSuccesses(0).getRequestId());
            assertEquals(511, ack.getErrors(0).getErrorInfo().getErrorCode());
        }
    }
    @Test void cancelBeforeEnqueueOrPublicationProducesOnlyAnErrorForThatMember() throws Exception {
        for (String checkpoint : List.of("prepared", "before_enqueue", "enqueueing", "after_enqueue", "before_publish")) {
            try (var c = MockEngineTestCluster.create(model(4), 62880, 1, 1)) {
                var p = c.prefill(0); var once = new java.util.concurrent.atomic.AtomicBoolean();
                p.batchAdmissionObserver = phase -> {
                    if (phase.equals(checkpoint) && once.compareAndSet(false, true)) assertTrue(p.cancelRequest(1).found());
                };
                var ack = enqueueAndFetch(p, batch(77, slot(0, inputWithDecode(1, 10, c.decode(0).getGrpcPort()),
                        inputWithDecode(2, 10, c.decode(0).getGrpcPort()))));
                answers(ack, 2); assertEquals(2, ack.getSuccesses(0).getRequestId(), checkpoint);
                assertEquals(1, ack.getErrors(0).getRequestId()); assertEquals(8429, ack.getErrors(0).getErrorInfo().getErrorCode());
                await(() -> !p.hasInflightWork() && c.decode(0).getOccupiedKvTokens() == 0);
                var terminal = workerStatus(p, 0).getFinishedTaskListList().stream().filter(task -> task.getRequestId() == 1).toList();
                assertEquals(1, terminal.size(), checkpoint);
                assertEquals(77, terminal.getFirst().getBatchId(), checkpoint);
                assertEquals(EngineRpcService.PriorityPreemptionProgressPB.PRIORITY_PREEMPTION_CANCELED,
                        terminal.getFirst().getPriorityPreemptionProgress(), checkpoint);
                assertFalse(p.isLeakDetected());
            }
        }
    }
    @Test void allCancelledBeforeEnqueueCannotBeAckedAsSuccess() throws Exception {
        try (var c = MockEngineTestCluster.create(model(4), 62880, 1, 1)) {
            var p = c.prefill(0);
            p.batchAdmissionObserver = phase -> { if (phase.equals("before_enqueue")) { p.cancelRequest(1); p.cancelRequest(2); } };
            var ack = enqueue(p, batch(77, slot(0, inputWithDecode(1, 10, c.decode(0).getGrpcPort()), inputWithDecode(2, 10, c.decode(0).getGrpcPort()))));
            answers(ack, 2); assertEquals(0, ack.getSuccessesCount());
            assertEquals(0, c.decode(0).getOccupiedKvTokens()); assertFalse(p.hasInflightWork());
        }
    }
    @Test void terminalPrepareFailureWinsLateCancelEvenWhileSiblingStillPrepares() throws Exception {
        var entered = new CountDownLatch(1); var release = new CountDownLatch(1);
        Server vit = ServerBuilder.forPort(0).addService(new MultimodalRpcServiceGrpc.MultimodalRpcServiceImplBase() {
            public void remoteMultimodalEmbedding(EngineRpcService.MultimodalInputsPB input, StreamObserver<EngineRpcService.MultimodalOutputPB> observer) {
                if (input.getRequestId() == 1) { observer.onError(MockRpcErrors.error(Status.INVALID_ARGUMENT, 902, "bad image")); return; }
                entered.countDown(); try { release.await(); success(observer); } catch (Exception error) { observer.onError(error); }
            }
        }).build().start();
        try (var c = MockEngineTestCluster.create(model(4), 62880, 1, 1)) {
            var p = c.prefill(0);
            var response = CompletableFuture.supplyAsync(() -> enqueueAndFetch(p, batch(77, slot(0,
                    typed(1, vit.getPort(), c.decode(0).getGrpcPort()), typed(2, vit.getPort(), c.decode(0).getGrpcPort())))));
            try {
                assertTrue(entered.await(3, TimeUnit.SECONDS));
                await(() -> "rejected".equals(p.getRequestStates().get(1L)));
                assertFalse(p.cancelRequest(1).found(), "terminal prepare failure cannot become accepted priority Cancel");
            } finally { release.countDown(); }
            var ack = response.get(5, TimeUnit.SECONDS); answers(ack, 2);
            assertEquals(902, ack.getErrors(0).getErrorInfo().getErrorCode()); assertEquals(2, ack.getSuccesses(0).getRequestId());
        } finally { release.countDown(); vit.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
    }
    @Test void stoppedBatchAnswersEachMemberAndInvalidRankDoesNotBlockLocalMembers() throws Exception {
        try (var c = MockEngineTestCluster.create(model(4), 62880, 1, 0)) {
            var p = c.prefill(0); p.setStopped(true);
            var ack = enqueue(p, batch(77, slot(0, input(1, 10)), slot(1, input(2, 10))));
            answers(ack, 2); assertEquals(Set.of(14L, 3L), ack.getErrorsList().stream().map(error -> error.getErrorInfo().getErrorCode()).collect(java.util.stream.Collectors.toSet()));
            p.setStopped(false);
            var mixedRank = enqueueAndFetch(p, batch(78, slot(1, input(3, 10)), slot(0, input(4, 10))));
            answers(mixedRank, 2); assertEquals(4, mixedRank.getSuccesses(0).getRequestId());
            assertEquals(3, mixedRank.getErrors(0).getErrorInfo().getErrorCode());
        }
    }
    @Test void requestDeadlineBeforeAckRejectsPublicationAndReleasesPreparedResources() throws Exception {
        try (var c = MockEngineTestCluster.create(model(4), 62880, 1, 1)) {
            var p = c.prefill(0);
            p.batchAdmissionObserver = phase -> {
                if (phase.equals("before_publish")) try { Thread.sleep(150); } catch (InterruptedException error) { throw new AssertionError(error); }
            };
            var input = inputWithDecode(1, 10, c.decode(0).getGrpcPort()).toBuilder(); input.getGenerateConfigBuilder().setTimeoutMs(100);
            var ack = enqueue(p, batch(77, slot(0, input.build())));
            answers(ack, 1); assertEquals(0, ack.getSuccessesCount()); assertEquals(4, ack.getErrors(0).getErrorInfo().getErrorCode());
            await(() -> !p.hasInflightWork() && c.decode(0).getOccupiedKvTokens() == 0);
        }
    }
    @Test void batchRpcContextCancellationDoesNotCancelAcceptedLogicalRequests() throws Exception {
        try (var c = MockEngineTestCluster.create(model(4), 62880, 1, 0); var context = Context.current().withCancellation()) {
            c.prefill(0).batchAdmissionObserver = phase -> { if (phase.equals("prepared")) context.cancel(null); };
            var response = new java.util.concurrent.atomic.AtomicReference<EngineRpcService.EnqueueBatchResponsePB>();
            context.run(() -> response.set(enqueueAndFetch(c.prefill(0), batch(77, slot(0, input(1, 10))))));
            answers(response.get(), 1); assertEquals(1, response.get().getSuccessesCount());
        }
    }
    @Test void missingInputAndDuplicateIdsStillAnswerTheEntireEnvelope() throws Exception {
        try (var c = MockEngineTestCluster.create(model(4), 62880, 1, 0)) {
            var p = c.prefill(0);
            var partial = slot(0, input(1, 10)).toBuilder().addRequests(EngineRpcService.EnqueueBatchExternalInputPB.getDefaultInstance()).build();
            var ack = enqueueAndFetch(p, batch(77, partial));
            answers(ack, 2); assertEquals(1, ack.getSuccessesCount());
            assertEquals(0, ack.getErrors(0).getRequestId()); assertEquals(3, ack.getErrors(0).getErrorInfo().getErrorCode());
            var duplicate = enqueue(p, batch(78, slot(0, input(2, 10)), slot(0, input(2, 10))));
            assertEquals(0, duplicate.getSuccessesCount()); assertEquals(2, duplicate.getErrorsCount());
            duplicate.getErrorsList().forEach(error -> assertEquals(6, error.getErrorInfo().getErrorCode()));
            assertEquals(0, enqueue(p, batch(79)).getSuccessesCount());
        }
    }

    @Test void unifiedGroupFallbackUsesSurvivingRequestCount() throws Exception {
        try (var c = MockEngineTestCluster.create(model(1), 62880, 1, 0)) {
            var p = c.prefill(0);
            var fallback = new java.util.concurrent.atomic.AtomicLong();
            var field = p.getClass().getDeclaredField("eventMetricReporter"); field.setAccessible(true);
            field.set(p, (java.util.function.Consumer<java.util.Map<String, Number>>)values -> {
                var value = values.get("rtp_llm_scheduler_group_fallback_acc");
                if (value != null) fallback.addAndGet(value.longValue());
            });
            var ack = enqueueAndFetch(p, batch(77, slot(0, input(1, 10)), slot(0, input(2, 10))));
            answers(ack, 2); assertEquals(2, ack.getSuccessesCount());
            assertEquals(1, fallback.get(), "one flattened two-member group falls back exactly once");
            await(() -> !p.hasInflightWork());
            var invalid = input(3, 10).toBuilder(); invalid.getGenerateConfigBuilder().setMaxNewTokens(-1);
            var partial = enqueueAndFetch(p, batch(78, slot(0, invalid.build()), slot(0, input(4, 10))));
            answers(partial, 2); assertEquals(1, partial.getSuccessesCount()); assertEquals(4, partial.getSuccesses(0).getRequestId());
            assertEquals(1, fallback.get(), "the surviving singleton must not inherit the original group's two-member fallback");
        }
    }

    @Test void cancellationAfterPublishedSuccessDoesNotRewriteItsAck() throws Exception {
        try (var c = MockEngineTestCluster.create(model(4), 62880, 1, 1)) {
            var p = c.prefill(0); var publish = new AtomicInteger();
            p.batchAdmissionObserver = phase -> {
                if (phase.equals("before_publish") && publish.incrementAndGet() == 2) assertTrue(p.cancelRequest(1).found());
            };
            var ack = enqueueAndFetch(p, batch(77, slot(0, inputWithDecode(1, 10, c.decode(0).getGrpcPort()), inputWithDecode(2, 10, c.decode(0).getGrpcPort()))));
            answers(ack, 2); assertEquals(2, ack.getSuccessesCount(), "published success precedes the later Cancel");
            await(() -> !p.hasInflightWork() && c.decode(0).getOccupiedKvTokens() == 0);
        }
    }

    @Test void priorityCancelOnlyChangesMixedModeMembershipBeforeR1() throws Exception {
        for (String phase : List.of("before_enqueue", "enqueueing")) {
            try (var c = MockEngineTestCluster.create(model(4), 62880, 1, 1)) {
                var p = c.prefill(0);
                p.batchAdmissionObserver = reached -> { if (phase.equals(reached)) p.cancelRequest(1); };
                var only = input(1, 10).toBuilder(); only.getGenerateConfigBuilder().setMaxNewTokens(0);
                var ack = enqueueAndFetch(p, batch(77, slot(0, only.build(), inputWithDecode(2, 10, c.decode(0).getGrpcPort()))));
                answers(ack, 2);
                if (phase.equals("before_enqueue")) {
                    assertEquals(1, ack.getSuccessesCount(), ack.toString()); assertEquals(2, ack.getSuccesses(0).getRequestId());
                } else {
                    assertEquals(0, ack.getSuccessesCount(), ack.toString());
                    assertTrue(ack.getErrorsList().stream().anyMatch(error -> error.getRequestId() == 2 && error.getErrorInfo().getErrorCode() == 605));
                }
                assertTrue(ack.getErrorsList().stream().anyMatch(error -> error.getRequestId() == 1 && error.getErrorInfo().getErrorCode() == 8429));
            }
        }
    }

    @Test void preparingCancelReportsCancelingUntilItsOwnOperationReturns() throws Exception {
        var entered = new CountDownLatch(1); var release = new CountDownLatch(1);
        Server vit = ServerBuilder.forPort(0).addService(new MultimodalRpcServiceGrpc.MultimodalRpcServiceImplBase() {
            public void remoteMultimodalEmbedding(EngineRpcService.MultimodalInputsPB input, StreamObserver<EngineRpcService.MultimodalOutputPB> observer) {
                entered.countDown(); try { release.await(); success(observer); } catch (Exception error) { observer.onError(error); }
            }
        }).build().start();
        try (var c = MockEngineTestCluster.create(model(4), 62880, 1, 1)) {
            var p = c.prefill(0);
            var response = CompletableFuture.supplyAsync(() -> enqueue(p, batch(77, slot(0, typed(1, vit.getPort(), c.decode(0).getGrpcPort())))));
            try {
                assertTrue(entered.await(3, TimeUnit.SECONDS)); assertTrue(p.cancelRequest(1).found());
                var status = workerStatus(p, 0);
                assertTrue(status.getRunningTaskInfoList().stream().anyMatch(task -> task.getRequestId() == 1
                        && task.getPriorityPreemptionProgress() == EngineRpcService.PriorityPreemptionProgressPB.PRIORITY_PREEMPTION_CANCELING));
                assertTrue(status.getFinishedTaskListList().stream().noneMatch(task -> task.getRequestId() == 1));
            } finally { release.countDown(); }
            var ack = response.get(5, TimeUnit.SECONDS); answers(ack, 1);
            assertEquals(8429, ack.getErrors(0).getErrorInfo().getErrorCode());
            var terminal = workerStatus(p, 0).getFinishedTaskListList().stream().filter(task -> task.getRequestId() == 1).toList();
            assertEquals(1, terminal.size());
            assertEquals(77, terminal.getFirst().getBatchId());
            assertEquals(EngineRpcService.PriorityPreemptionProgressPB.PRIORITY_PREEMPTION_CANCELED, terminal.getFirst().getPriorityPreemptionProgress());
        } finally { release.countDown(); vit.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
    }

    @Test void groupAdmissionExceptionRejectsUnansweredSlotsWithoutStrandingDecode() throws Exception {
        try (var c = MockEngineTestCluster.create(model(4), 62880, 1, 1)) {
            var p = c.prefill(0);
            p.batchAdmissionObserver = phase -> { if (phase.equals("enqueueing")) throw new IllegalStateException("engine failure"); };
            var ack = enqueue(p, batch(77, slot(0, inputWithDecode(1, 10, c.decode(0).getGrpcPort()), inputWithDecode(2, 10, c.decode(0).getGrpcPort()))));
            answers(ack, 2); assertEquals(0, ack.getSuccessesCount());
            ack.getErrorsList().forEach(error -> assertEquals(13, error.getErrorInfo().getErrorCode()));
            assertEquals(0, c.decode(0).getOccupiedKvTokens()); assertFalse(p.hasInflightWork());
        }
    }

    @Test void shutdownSettlesAQueuedPrepareRatherThanLeavingItsBatchJoinPending() throws Exception {
        var model = model(4);
        model.pdPolicy = new MockPdPolicy(0, 5000, 10000, 1, 5000, 10000, 1, 2000);
        var entered = new CountDownLatch(1); var release = new CountDownLatch(1);
        try (var c = MockEngineTestCluster.create(model, 62880, 1, 0)) {
            var p = c.prefill(0);
            var field = p.getClass().getDeclaredField("prepareExecutor"); field.setAccessible(true);
            var executor = (java.util.concurrent.ThreadPoolExecutor)field.get(p);
            executor.execute(() -> { entered.countDown(); try { release.await(); } catch (InterruptedException expected) { Thread.currentThread().interrupt(); } });
            assertTrue(entered.await(3, TimeUnit.SECONDS));
            var response = CompletableFuture.supplyAsync(() -> enqueue(p, batch(77, slot(0, input(1, 10)))));
            try {
                await(() -> executor.getQueue().size() == 1);
                p.shutdown();
                var ack = response.get(3, TimeUnit.SECONDS); answers(ack, 1);
                assertEquals(14, ack.getErrors(0).getErrorInfo().getErrorCode());
                assertFalse(p.hasInflightWork());
            } finally { release.countDown(); }
        } finally { release.countDown(); }
    }

    @Test void readyMemberCancellationFinishesWithoutWaitingForItsPreparingSibling() throws Exception {
        var entered = new CountDownLatch(1); var release = new CountDownLatch(1);
        Server vit = ServerBuilder.forPort(0).addService(new MultimodalRpcServiceGrpc.MultimodalRpcServiceImplBase() {
            public void remoteMultimodalEmbedding(EngineRpcService.MultimodalInputsPB input, StreamObserver<EngineRpcService.MultimodalOutputPB> observer) {
                try {
                    if (input.getRequestId() == 2) { entered.countDown(); release.await(); }
                    success(observer);
                } catch (Exception error) { observer.onError(error); }
            }
        }).build().start();
        try (var c = MockEngineTestCluster.create(model(4), 62880, 1, 1)) {
            var p = c.prefill(0);
            var response = CompletableFuture.supplyAsync(() -> enqueueAndFetch(p, batch(77, slot(0,
                    typed(1, vit.getPort(), c.decode(0).getGrpcPort()), typed(2, vit.getPort(), c.decode(0).getGrpcPort())))));
            try {
                assertTrue(entered.await(3, TimeUnit.SECONDS));
                await(() -> c.decode(0).getOccupiedKvTokens() > 0);
                assertTrue(p.cancelRequest(1).found());
                await(() -> workerStatus(p, 0).getFinishedTaskListList().stream().anyMatch(task -> task.getRequestId() == 1));
                assertFalse(response.isDone(), "the sibling is still preparing");
                assertEquals(0, c.decode(0).getOccupiedKvTokens(), "the ready member's D reservation was released independently");
            } finally { release.countDown(); }
            var ack = response.get(5, TimeUnit.SECONDS); answers(ack, 2);
            assertEquals(2, ack.getSuccesses(0).getRequestId()); assertEquals(8429, ack.getErrors(0).getErrorInfo().getErrorCode());
        } finally { release.countDown(); vit.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
    }

    @Test void preparePoolOverloadOnlyRejectsTheUnsubmittedMember() throws Exception {
        var model = model(4);
        model.pdPolicy = new MockPdPolicy(0, 5000, 10000, 1, 5000, 10000, 1, 2000);
        var entered = new CountDownLatch(1); var release = new CountDownLatch(1);
        try (var c = MockEngineTestCluster.create(model, 62880, 1, 0)) {
            var p = c.prefill(0);
            var field = p.getClass().getDeclaredField("prepareExecutor"); field.setAccessible(true);
            var executor = (java.util.concurrent.ThreadPoolExecutor)field.get(p);
            executor.execute(() -> { entered.countDown(); try { release.await(); } catch (InterruptedException expected) { Thread.currentThread().interrupt(); } });
            assertTrue(entered.await(3, TimeUnit.SECONDS));
            var response = CompletableFuture.supplyAsync(() -> enqueueAndFetch(p, batch(77, slot(0, input(1, 10), input(2, 10)))));
            try {
                await(() -> "rejected".equals(p.getRequestStates().get(2L)));
                assertFalse(p.cancelRequest(2).found(), "failed submission already owns its terminal cause");
            } finally { release.countDown(); }
            var ack = response.get(3, TimeUnit.SECONDS); answers(ack, 2);
            assertEquals(1, ack.getSuccesses(0).getRequestId()); assertEquals(2, ack.getErrors(0).getRequestId());
            assertEquals(13, ack.getErrors(0).getErrorInfo().getErrorCode());
        } finally { release.countDown(); }
    }

    @Test void schedulerRejectionOnlyOwnsTheTerminalCauseAfterR2() throws Exception {
        for (String phase : List.of("after_enqueue", "before_publish")) {
            var model = model(4); model.setDecodeReserveStep(1);
            model.setPrefillBatchPolicy(new MockPrefillBatchPolicy(4, 10000, 20, 1, false, 0, 0, false));
            try (var c = MockEngineTestCluster.create(model, 62880, 1, 1)) {
                var p = c.prefill(0);
                var field = c.decode(0).getClass().getDeclaredField("performance"); field.setAccessible(true);
                ((MockPerformanceModel)field.get(c.decode(0))).setPrefillBatchPolicy(
                        new MockPrefillBatchPolicy(4, 10000, 10000, 1, false, 0, 0, false));
                p.batchAdmissionObserver = reached -> {
                    if (phase.equals(reached)) assertEquals(phase.equals("after_enqueue"), p.cancelRequest(1).found(), phase);
                };
                var ack = enqueueAndFetch(p, batch(77, slot(0, inputWithDecode(1, 30, c.decode(0).getGrpcPort()),
                        inputWithDecode(2, 10, c.decode(0).getGrpcPort()))));
                answers(ack, 2); assertEquals(2, ack.getSuccesses(0).getRequestId());
                assertEquals(phase.equals("after_enqueue") ? 8429 : 511, ack.getErrors(0).getErrorInfo().getErrorCode(), phase);
                await(() -> !p.hasInflightWork() && c.decode(0).getOccupiedKvTokens() == 0);
            }
        }
    }

}
