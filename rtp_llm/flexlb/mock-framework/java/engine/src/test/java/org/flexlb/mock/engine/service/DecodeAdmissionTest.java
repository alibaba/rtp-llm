package org.flexlb.mock.engine.service;

import org.flexlb.mock.protocol.grpc.MockDecodeServiceGrpc;
import org.flexlb.mock.protocol.grpc.MockEngineProtocol.DecodeFrame;
import org.flexlb.mock.protocol.grpc.MockEngineProtocol.MockDecodePromptPB;
import org.flexlb.mock.protocol.PromptCacheKeys.Span;
import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.grpc.*;
import io.grpc.stub.StreamObserver;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.BooleanSupplier;
import java.util.stream.IntStream;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.mock.engine.config.RtpConfig;
import org.flexlb.mock.engine.execution.LLMMockEngine;
import org.flexlb.mock.engine.scheduler.LLMMockEngineTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.flexlb.mock.engine.cache.KvCacheManager.Ready;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Exercises KV admission and cancellation through the Decode RPC.
 */
class DecodeAdmissionTest {
    @TempDir Path directory;

    @Test void allocateAcknowledgementOwnsKvAndCancellationReleasesItWithoutPublishing() throws Exception {
        try (var node = node(false); var call = new Call(node, 0)) {
            call.request.onNext(input("p", 1, 5, 0)); call.ack();
            var status = node.status(); assertEquals(10, status.getAvailableKvCache());
            assertEquals(TaskPhase.TASK_PHASE_KV_ALLOCATED, status.getRunningTaskInfo(0).getPhase());
            assertEquals(1, status.getRunningTaskInfoList().stream().filter(task -> task.getIsWaiting()).count()); assertEquals(0, status.getRunningTaskInfoList().stream().filter(task -> !task.getIsWaiting()).count());
            call.close(); await(() -> node.status().getAvailableKvCache() == 16 && node.status().getRunningTaskInfoList().stream().filter(task -> task.getIsWaiting()).count() == 0);
            assertEquals(8100, node.status().getFinishedTaskList(0).getErrorInfo().getErrorCode());
            assertEquals(0, node.node.cacheManager().snapshot(-1, true).keys().length);
        }
    }

    @Test void temporaryPressureWaitsPastOldRetryBudgetAndWakesOnRelease() throws Exception {
        try (var node = node(false); var owner = assertInstanceOf(Ready.class, node.node.cacheManager().prepare(12, List.of())).lease(); var call = new Call(node, 0)) {
            call.request.onNext(input("p", 1, 5, 0));
            await(() -> node.status().getRunningTaskInfoList().stream().filter(task -> task.getIsWaiting()).count() == 1);
            assertEquals(TaskPhase.TASK_PHASE_RECEIVED, node.status().getRunningTaskInfo(0).getPhase());
            assertNull(call.replies.poll(200, TimeUnit.MILLISECONDS)); assertEquals(4, node.status().getAvailableKvCache());
            owner.close(); call.ack(); assertEquals(10, node.status().getAvailableKvCache());
        }
    }

    @Test void waitingDeadlineHasNative603AndCannotAllocateAfterExpiry() throws Exception {
        try (var node = node(false); var owner = assertInstanceOf(Ready.class, node.node.cacheManager().prepare(12, List.of())).lease(); var call = new Call(node, 0)) {
            call.request.onNext(input("p", 1, 5, 50));
            assertEquals(603, engineCode(call.error(Status.Code.DEADLINE_EXCEEDED)));
            owner.close(); Thread.sleep(100);
            assertEquals(16, node.status().getAvailableKvCache()); assertEquals(0, node.status().getRunningTaskInfoList().stream().filter(task -> task.getIsWaiting()).count());
            assertEquals(1, node.status().getFinishedTaskListCount()); assertTrue(call.replies.isEmpty());
        }
    }

    @Test void requestBudgetAlsoExpiresAnAlreadyAcknowledgedReservation() throws Exception {
        try (var node = node(false); var call = new Call(node, 0)) {
            call.request.onNext(input("p", 1, 5, 300)); call.ack();
            assertEquals(603, engineCode(call.error(Status.Code.DEADLINE_EXCEEDED)));
            assertEquals(16, node.status().getAvailableKvCache());
        }
    }

    @Test void rpcDeadlineReleasesAnUnboundedRequestReservation() throws Exception {
        try (var node = node(false); var call = new Call(node, 500)) {
            call.request.onNext(input("p", 1, 5, 0)); call.ack();
            call.error(Status.Code.DEADLINE_EXCEEDED);
            await(() -> node.status().getRunningTaskInfoList().stream().filter(task -> task.getIsWaiting()).count() == 0);
            assertEquals(16, node.status().getAvailableKvCache());
            assertEquals(603, node.status().getFinishedTaskList(0).getErrorInfo().getErrorCode());
        }
    }

    @Test void permanentCapacityFailureReturns602RatherThanWaitingForever() throws Exception {
        try (var node = node(false); var call = new Call(node, 0)) {
            call.request.onNext(input("p", 1, 19, 0));
            assertEquals(602, engineCode(call.error(Status.Code.RESOURCE_EXHAUSTED)));
            assertEquals(16, node.status().getAvailableKvCache()); assertEquals(0, node.status().getRunningTaskInfoList().stream().filter(task -> task.getIsWaiting()).count());
        }
    }

    @Test void compactDecodeSharesPromptBlocksWhenFirstMallocReuseIsEnabled() throws Exception {
        for (boolean disabled : List.of(false, true)) {
            try (var node = node(disabled); var first = new Call(node, 0); var second = new Call(node, 0)) {
                var request = input("p", 1, 5, 0);
                var prompt = request.getPrompt();
                try (var lease = assertInstanceOf(Ready.class, node.node.cacheManager().prepare(6, List.of())).lease()) {
                    lease.publish(prompt.getBlockKeysList(), 5);
                }
                first.request.onNext(request); first.ack();
                assertEquals(10, node.status().getAvailableKvCache());

                second.request.onNext(input("p", 2, 5, 0)); second.ack();
                var status = node.status();
                assertEquals(2, status.getRunningTaskInfoCount());
                for (var task : status.getRunningTaskInfoList())
                    assertEquals(disabled ? 0 : 4, task.getPrefixLength());
                assertEquals(disabled ? 4 : 8, status.getAvailableKvCache());

                first.close(); second.close();
                await(() -> node.status().getRunningTaskInfoCount() == 0);
                assertEquals(Set.copyOf(prompt.getBlockKeysList()), cacheKeys(node.node));
            }
        }
    }

    @Test void duplicateIdentityDoesNotReplaceOwnerButDifferentClientCanUseSameId() throws Exception {
        try (var node = node(false); var first = new Call(node, 0); var duplicate = new Call(node, 0); var other = new Call(node, 0)) {
            first.request.onNext(input("p1", 1, 5, 0)); first.ack();
            duplicate.request.onNext(input("p1", 1, 5, 0)); duplicate.error(Status.Code.ALREADY_EXISTS);
            other.request.onNext(input("p2", 1, 5, 0)); other.ack();
            assertEquals(2, node.status().getRunningTaskInfoList().stream().filter(task -> task.getIsWaiting()).count()); assertEquals(4, node.status().getAvailableKvCache());
        }
    }

    @Test void completedDecodePublishesOnlyCompletePromptBlocks() throws Exception {
        for (boolean media : List.of(false, true)) {
            try (var node = node(false); var call = new Call(node, 0)) {
                var promptBuilder = promptInput(71, 3, 3000).toBuilder();
                promptBuilder.getGenerateConfigBuilder().setMaxNewTokens(5).setIgnoreEos(true);
                var spans = new ArrayList<Span>();
                if (media) {
                    promptBuilder.setTokenIds(2, -200);
                    spans.add(new Span(2, List.of(777)));
                }
                var prompt = promptBuilder.build();
                var prepared = new org.flexlb.mock.protocol.MultimodalProtocol.Prepared(prompt, spans);
                var promptCacheKeys = org.flexlb.mock.protocol.MultimodalProtocol.computePromptKeys(prepared, 2, 0, 1);
                call.request.onNext(allocate("p", prompt, promptCacheKeys.toDecodePrompt())); call.ack();
                call.request.onNext(DecodeFrame.newBuilder().setRequestId(71).setStage(RemoteStage.LOAD).build());
                assertInstanceOf(GenerateOutputsPB.class, call.replies.poll(3, TimeUnit.SECONDS));
                call.request.onNext(DecodeFrame.newBuilder().setRequestId(71).setClientId("p")
                    .setStage(RemoteStage.GENERATE).setFirstGenerateTokenId(7).build());
                Object reply;
                do {
                    reply = call.replies.poll(3, TimeUnit.SECONDS);
                    assertNotNull(reply);
                    assertTrue(reply instanceof GenerateOutputsPB || "completed".equals(reply), String.valueOf(reply));
                } while (!"completed".equals(reply));
                var all = prompt.toBuilder().addAllTokenIds(Collections.nCopies(5, 7)).build();
                var full = new org.flexlb.mock.protocol.MultimodalProtocol.Prepared(all, spans);
                var allKeys = org.flexlb.mock.protocol.MultimodalProtocol.computePromptKeys(full, 2, 0, 1).keys();
                await(() -> cacheKeys(node.node).equals(Set.copyOf(promptCacheKeys.keys())));
                try (var lease = assertInstanceOf(Ready.class, node.node.cacheManager().prepare(8, allKeys)).lease()) {
                    assertNotNull(lease);
                    assertEquals(2, lease.reusedTokens());
                }
            }
        }
    }

    static DecodeFrame allocate(String client, GenerateInputPB input) {
        var sequence = org.flexlb.mock.protocol.MultimodalProtocol.computePromptKeys(
            org.flexlb.mock.protocol.MultimodalProtocol.text(input), 2, 0, 1);
        return allocate(client, input, sequence.toDecodePrompt());
    }

    @Test void loadUsesTheExistingConnectionAndNeedsNoCallbackAddress() throws Exception {
        try (var node = node(false); var call = new Call(node, 0)) {
            call.request.onNext(input("p", 1, 5, 0)); call.ack();
            call.request.onNext(DecodeFrame.newBuilder().setRequestId(1).setStage(RemoteStage.LOAD).build());
            var loaded = call.replies.poll(3, TimeUnit.SECONDS);
            assertInstanceOf(GenerateOutputsPB.class, loaded);
            assertEquals(0, ((GenerateOutputsPB) loaded).getErrorInfo().getErrorCodeValue());
            assertEquals(10, node.status().getAvailableKvCache());
            call.close();
            await(() -> node.status().getAvailableKvCache() == 16);
            assertEquals(1, node.status().getFinishedTaskListCount());
        }
    }

    @Test void explicitPrefillHandoffRejectsUncomputedTokenAndWrongProducer() throws Exception {
        for (boolean wrongProducer : List.of(false,true)) {
            try (var node = node(false); var call = new Call(node, 0)) {
                call.request.onNext(input("p", 1, 5, 0)); call.ack();
                call.request.onNext(DecodeFrame.newBuilder().setRequestId(1).setStage(RemoteStage.LOAD).build());
                assertInstanceOf(GenerateOutputsPB.class,call.replies.poll(3,TimeUnit.SECONDS));
                call.request.onNext(DecodeFrame.newBuilder().setRequestId(1).setStage(RemoteStage.GENERATE)
                    .setClientId(wrongProducer ? "" : "p")
                    .setComputedKvEnd(wrongProducer ? 5 : 6).build());
                call.error(Status.Code.INVALID_ARGUMENT);
                await(() -> node.status().getAvailableKvCache() == 16);
            }
        }
    }

    @Test void invalidFirstFrameHas605AndHalfCloseReleasesAnAcceptedReservation() throws Exception {
        try (var node = node(false); var invalid = new Call(node, 0); var valid = new Call(node, 0)) {
            invalid.request.onNext(DecodeFrame.newBuilder().setStage(RemoteStage.ALLOCATE)
                .setClientId("p").setRequestId(1).setInput(promptInput(1, 5, 0)).build());
            assertEquals(605, engineCode(invalid.error(Status.Code.INVALID_ARGUMENT)));
            valid.request.onNext(input("p", 1, 5, 0)); valid.ack(); valid.request.onCompleted();
            valid.error(Status.Code.CANCELLED);
            assertEquals(16, node.status().getAvailableKvCache());
            assertEquals(1, node.status().getLatestFinishedVersion());
        }
    }

    @Test void cancellingCapacityWaitCannotResurrectAnAllocationOrReportForwardWork() throws Exception {
        try (var node = node(false); var owner = assertInstanceOf(Ready.class, node.node.cacheManager().prepare(12, List.of())).lease(); var call = new Call(node, 0)) {
            call.request.onNext(input("p", 1, 5, 0)); await(() -> node.status().getRunningTaskInfoList().stream().filter(task -> task.getIsWaiting()).count() == 1);
            call.close(); await(() -> node.status().getRunningTaskInfoList().stream().filter(task -> task.getIsWaiting()).count() == 0);
            owner.close(); Thread.sleep(100);
            assertEquals(16, node.status().getAvailableKvCache()); assertEquals(1, node.status().getFinishedTaskListCount());
            try (var text = java.net.URI.create((String) node.node.endpoints().get("metrics")).toURL().openStream()) {
                var metrics = new String(text.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
                assertEquals(0, LocalRpcTest.sample(metrics, "rtp_llm_model_forward_seconds_count", ""));
                assertEquals(1, LocalRpcTest.sample(metrics, "rtp_llm_framework_requests_total", "outcome=\"cancelled\""));
            }
        }
    }

    @Test void nodeCloseReleasesAcknowledgedAndWaitingReservations() throws Exception {
        try (var node = node(false); var first = new Call(node, 0); var waiting = new Call(node, 0)) {
            first.request.onNext(input("p", 1, 9, 0)); first.ack();
            waiting.request.onNext(input("p", 2, 9, 0)); await(() -> node.status().getRunningTaskInfoList().stream().filter(task -> task.getIsWaiting()).count() == 2);
            node.node.close();
            assertEquals(0, node.node.cacheManager().snapshot(0, false).heldBlocks());
            assertEquals(8, node.node.cacheManager().snapshot(0, false).availableBlocks());
            assertEquals(0, node.node.cacheManager().snapshot(-1, true).keys().length);
        }
    }

    private static Set<Long> cacheKeys(LLMMockEngine node) {
        return new HashSet<>(Arrays.stream(node.cacheManager().snapshot(-1, true).keys()).boxed().toList());
    }

    private static DecodeFrame input(String client, long id, int length, int timeout) {
        return allocate(client, promptInput(id, length, timeout));
    }

    private static DecodeFrame allocate(String client, GenerateInputPB input, MockDecodePromptPB prompt) {
        return DecodeFrame.newBuilder().setStage(RemoteStage.ALLOCATE).setRequestId(input.getRequestId())
            .setClientId(client).setInput(input.toBuilder().clearTokenIds()).setPrompt(prompt).build();
    }

    private static GenerateInputPB promptInput(long id, int length, int timeout) {
        return GenerateInputPB.newBuilder().setRequestId(id).addAllTokenIds(IntStream.rangeClosed(1, length).boxed().toList())
            .setGenerateConfig(GenerateConfigPB.newBuilder().setMaxNewTokens(4).setReuseCache(true).setTimeoutMs(timeout)).build();
    }

    private Harness node(boolean disableReuse) throws Exception {
        var json = new ObjectMapper(); ObjectNode root = json.valueToTree(LLMMockEngineTest.configuration("decode"));
        ((ObjectNode) root.path("component").path("config")).put("seq_size_per_block", 2).put("reuse_cache", true)
            .put("disable_decode_first_malloc_device_reuse", disableReuse);
        var path = directory.resolve("config.json"); json.writeValue(path.toFile(), root);
        return new Harness(LLMMockEngine.start(RtpConfig.load(path)));
    }

    private static void await(BooleanSupplier condition) throws Exception {
        long end = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
        while (!condition.getAsBoolean()) { if (System.nanoTime() >= end) fail("Condition timed out"); Thread.sleep(5); }
    }

    private static long engineCode(Throwable error) throws Exception {
        var trailers = Status.trailersFromThrowable(error);
        return ErrorDetailsPB.parseFrom(trailers.get(Metadata.Key.of("grpc-status-details-bin", Metadata.BINARY_BYTE_MARSHALLER))).getErrorCode();
    }

    /**
     * Owns the local Decode engine and the channel used to call it.
     */
    private static final class Harness implements AutoCloseable {
        final LLMMockEngine node;
        final ManagedChannel channel;

        Harness(LLMMockEngine node) { this.node = node; channel = ManagedChannelBuilder.forTarget(node.rpcAddress()).usePlaintext().build(); }

        @Override public void close() throws Exception { channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); node.close(); }

        WorkerStatusPB status() { return RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3, TimeUnit.SECONDS).getWorkerStatus(StatusVersionPB.getDefaultInstance()); }
    }

    /**
     * Drives one Decode RPC and exposes its responses to tests.
     */
    private static final class Call implements AutoCloseable {
        final Context.CancellableContext context = Context.current().withCancellation();
        final BlockingQueue<Object> replies = new LinkedBlockingQueue<>();
        final StreamObserver<DecodeFrame> request;

        Call(Harness node, long rpcMs) throws Exception {
            request = context.call(() -> {
                var stub = MockDecodeServiceGrpc.newStub(node.channel);
                if (rpcMs > 0) stub = stub.withDeadlineAfter(rpcMs, TimeUnit.MILLISECONDS);
                return stub.remoteGenerate(new StreamObserver<>() {
                    @Override public void onNext(GenerateOutputsPB value) { replies.add(value); }
                    @Override public void onError(Throwable error) { replies.add(error); }
                    @Override public void onCompleted() { replies.add("completed"); }
                });
            });
        }

        @Override public void close() { context.cancel(null); }

        void ack() throws Exception {
            var value = replies.poll(3, TimeUnit.SECONDS);
            assertInstanceOf(GenerateOutputsPB.class, value);
            assertFalse(((GenerateOutputsPB) value).hasFlattenOutput());
            assertTrue(((GenerateOutputsPB) value).getSupportsPrefillCompletion());
        }

        Throwable error(Status.Code expected) throws Exception {
            var value = replies.poll(3, TimeUnit.SECONDS); assertInstanceOf(Throwable.class, value);
            assertEquals(expected, Status.fromThrowable((Throwable) value).getCode()); return (Throwable) value;
        }
    }
}
