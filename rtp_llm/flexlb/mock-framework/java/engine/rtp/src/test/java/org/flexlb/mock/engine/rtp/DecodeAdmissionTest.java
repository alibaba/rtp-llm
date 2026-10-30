package org.flexlb.mock.engine.rtp;

import static org.junit.jupiter.api.Assertions.*;
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
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class DecodeAdmissionTest {
    @TempDir Path directory;
    private static GenerateRequestPB input(String client, long id, int length, int timeout) {
        return GenerateRequestPB.newBuilder().setStage(RemoteStage.ALLOCATE).setRequestId(id).setClientId(client)
            .setInput(GenerateInputPB.newBuilder().setRequestId(id).addAllTokenIds(IntStream.rangeClosed(1, length).boxed().toList())
                .setGenerateConfig(GenerateConfigPB.newBuilder().setMaxNewTokens(4).setReuseCache(true).setTimeoutMs(timeout))).build();
    }
    private Harness node(boolean disableReuse) throws Exception {
        var json = new ObjectMapper(); ObjectNode root = json.valueToTree(RtpNodeTest.configuration("decode"));
        ((ObjectNode) root.path("component").path("config")).put("seq_size_per_block", 2).put("reuse_cache", true)
            .put("disable_decode_first_malloc_device_reuse", disableReuse);
        var path = directory.resolve("config.json"); json.writeValue(path.toFile(), root);
        return new Harness(RtpNode.start(RtpConfig.load(path)));
    }
    private static void await(BooleanSupplier condition) throws Exception {
        long end = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
        while (!condition.getAsBoolean()) { if (System.nanoTime() >= end) fail("Condition timed out"); Thread.sleep(5); }
    }
    private static final class Harness implements AutoCloseable {
        final RtpNode node;
        final ManagedChannel channel;
        Harness(RtpNode node) { this.node = node; channel = ManagedChannelBuilder.forTarget(node.rpcAddress()).usePlaintext().build(); }
        WorkerStatusPB status() { return RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3, TimeUnit.SECONDS).getWorkerStatus(StatusVersionPB.getDefaultInstance()); }
        @Override public void close() throws Exception { channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); node.close(); }
    }
    private static final class Call implements AutoCloseable {
        final Context.CancellableContext context = Context.current().withCancellation();
        final BlockingQueue<Object> replies = new LinkedBlockingQueue<>();
        final StreamObserver<GenerateRequestPB> request;
        Call(Harness node, long rpcMs) throws Exception {
            request = context.call(() -> {
                var stub = RpcServiceGrpc.newStub(node.channel);
                if (rpcMs > 0) stub = stub.withDeadlineAfter(rpcMs, TimeUnit.MILLISECONDS);
                return stub.remoteGenerate(new StreamObserver<>() {
                    @Override public void onNext(GenerateOutputsPB value) { replies.add(value); }
                    @Override public void onError(Throwable error) { replies.add(error); }
                    @Override public void onCompleted() { replies.add("completed"); }
                });
            });
        }
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
        @Override public void close() { context.cancel(null); }
    }
    private static long engineCode(Throwable error) throws Exception {
        var trailers = Status.trailersFromThrowable(error);
        return ErrorDetailsPB.parseFrom(trailers.get(Metadata.Key.of("grpc-status-details-bin", Metadata.BINARY_BYTE_MARSHALLER))).getErrorCode();
    }

    @Test void allocateAcknowledgementOwnsKvAndCancellationReleasesItWithoutPublishing() throws Exception {
        try (var node = node(false); var call = new Call(node, 0)) {
            call.request.onNext(input("p", 1, 5, 0)); call.ack();
            var status = node.status(); assertEquals(10, status.getAvailableKvCache());
            assertEquals(TaskPhase.TASK_PHASE_KV_ALLOCATED, status.getRunningTaskInfo(0).getPhase());
            assertEquals(1, status.getWaitingQueryLen()); assertEquals(0, status.getRunningQueryLen());
            call.close(); await(() -> node.status().getAvailableKvCache() == 16 && node.status().getWaitingQueryLen() == 0);
            assertEquals(8100, node.status().getFinishedTaskList(0).getErrorInfo().getErrorCode());
            assertTrue(node.node.cache().snapshot(-1, true).keys().isEmpty());
        }
    }

    @Test void temporaryPressureWaitsPastOldRetryBudgetAndWakesOnRelease() throws Exception {
        try (var node = node(false); var owner = node.node.cache().acquire(12, List.of()).lease(); var call = new Call(node, 0)) {
            call.request.onNext(input("p", 1, 5, 0));
            await(() -> node.status().getWaitingQueryLen() == 1);
            assertEquals(TaskPhase.TASK_PHASE_RECEIVED, node.status().getRunningTaskInfo(0).getPhase());
            assertNull(call.replies.poll(200, TimeUnit.MILLISECONDS)); assertEquals(4, node.status().getAvailableKvCache());
            owner.close(); call.ack(); assertEquals(10, node.status().getAvailableKvCache());
        }
    }

    @Test void waitingDeadlineHasNative603AndCannotAllocateAfterExpiry() throws Exception {
        try (var node = node(false); var owner = node.node.cache().acquire(12, List.of()).lease(); var call = new Call(node, 0)) {
            call.request.onNext(input("p", 1, 5, 50));
            assertEquals(603, engineCode(call.error(Status.Code.DEADLINE_EXCEEDED)));
            owner.close(); Thread.sleep(100);
            assertEquals(16, node.status().getAvailableKvCache()); assertEquals(0, node.status().getWaitingQueryLen());
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
            await(() -> node.status().getWaitingQueryLen() == 0);
            assertEquals(16, node.status().getAvailableKvCache());
            assertEquals(603, node.status().getFinishedTaskList(0).getErrorInfo().getErrorCode());
        }
    }

    @Test void permanentCapacityFailureReturns602RatherThanWaitingForever() throws Exception {
        try (var node = node(false); var call = new Call(node, 0)) {
            call.request.onNext(input("p", 1, 19, 0));
            assertEquals(602, engineCode(call.error(Status.Code.RESOURCE_EXHAUSTED)));
            assertEquals(16, node.status().getAvailableKvCache()); assertEquals(0, node.status().getWaitingQueryLen());
        }
    }

    @Test void firstMallocReuseHonorsNativePolicyAndDoesNotPublishUncomputedKeys() throws Exception {
        for (boolean disabled : List.of(false, true)) {
            try (var node = node(disabled); var call = new Call(node, 0)) {
                var input = input("p", 1, 5, 0);
                var keys = PrefixKeys.completeBlocks(input.getInput(), 2);
                try (var lease = node.node.cache().acquire(6, List.of()).lease()) {
                    lease.publish(keys, 5);
                }
                call.request.onNext(input); call.ack();
                assertEquals(disabled ? 0 : 4, node.status().getRunningTaskInfo(0).getPrefixLength());
                call.close(); await(() -> node.status().getWaitingQueryLen() == 0);
                assertEquals(Set.copyOf(keys), node.node.cache().snapshot(-1, true).keys());
            }
        }
    }

    @Test void duplicateIdentityDoesNotReplaceOwnerButDifferentClientCanUseSameId() throws Exception {
        try (var node = node(false); var first = new Call(node, 0); var duplicate = new Call(node, 0); var other = new Call(node, 0)) {
            first.request.onNext(input("p1", 1, 5, 0)); first.ack();
            duplicate.request.onNext(input("p1", 1, 5, 0)); duplicate.error(Status.Code.ALREADY_EXISTS);
            other.request.onNext(input("p2", 1, 5, 0)); other.ack();
            assertEquals(2, node.status().getWaitingQueryLen()); assertEquals(4, node.status().getAvailableKvCache());
        }
    }

    @Test void loadWithoutSourceFailsAndReleasesReservationInsteadOfInventingKv() throws Exception {
        try (var node = node(false); var call = new Call(node, 0)) {
            call.request.onNext(input("p", 1, 5, 0)); call.ack();
            call.request.onNext(GenerateRequestPB.newBuilder().setRequestId(1).setStage(RemoteStage.LOAD).build());
            call.error(Status.Code.FAILED_PRECONDITION);
            assertEquals(16, node.status().getAvailableKvCache()); assertEquals(1, node.status().getFinishedTaskListCount());
        }
    }

    @Test void invalidFirstFrameHas605AndHalfCloseReleasesAnAcceptedReservation() throws Exception {
        try (var node = node(false); var invalid = new Call(node, 0); var valid = new Call(node, 0)) {
            invalid.request.onNext(GenerateRequestPB.getDefaultInstance());
            assertEquals(605, engineCode(invalid.error(Status.Code.INVALID_ARGUMENT)));
            valid.request.onNext(input("p", 1, 5, 0)); valid.ack(); valid.request.onCompleted();
            valid.error(Status.Code.CANCELLED);
            assertEquals(16, node.status().getAvailableKvCache());
            assertEquals(1, node.status().getLatestFinishedVersion());
        }
    }

    @Test void cancellingCapacityWaitCannotResurrectAnAllocationOrReportForwardWork() throws Exception {
        try (var node = node(false); var owner = node.node.cache().acquire(12, List.of()).lease(); var call = new Call(node, 0)) {
            call.request.onNext(input("p", 1, 5, 0)); await(() -> node.status().getWaitingQueryLen() == 1);
            call.close(); await(() -> node.status().getWaitingQueryLen() == 0);
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
            waiting.request.onNext(input("p", 2, 9, 0)); await(() -> node.status().getWaitingQueryLen() == 2);
            node.node.close();
            assertEquals(0, node.node.cache().snapshot(0, false).heldBlocks());
            assertEquals(8, node.node.cache().snapshot(0, false).availableBlocks());
            assertTrue(node.node.cache().snapshot(-1, true).keys().isEmpty());
        }
    }
}
