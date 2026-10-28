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
import org.flexlb.mock.engine.rtp.transfer.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Two independent node owners, real network RPCs, no in-process transfer bypass. */
class PrefillTransferTest {
    @TempDir Path directory;
    private static GenerateInputPB input(long id, int length, int timeout) {
        return GenerateInputPB.newBuilder().setRequestId(id)
            .addAllTokenIds(IntStream.rangeClosed(1, length).boxed().toList())
            .setGenerateConfig(GenerateConfigPB.newBuilder().setMaxNewTokens(4).setTimeoutMs(timeout).setReuseCache(true)).build();
    }
    private static TransferRequest transfer(GenerateInputPB input, int block) {
        return TransferRequest.newBuilder().setClientId("client").setInput(input).setBlockSize(block).build();
    }
    private Harness node(String role, int block, int millis) throws Exception {
        var json = new ObjectMapper(); ObjectNode root = json.valueToTree(RtpNodeTest.configuration(role));
        ((ObjectNode) root.path("component").path("config")).put("seq_size_per_block", block).put("reuse_cache", true).put("load_cache_timeout_ms", 1200);
        ((ObjectNode) root.path("environment").path("compute").path(role)).put("expression", Integer.toString(millis));
        var path = directory.resolve(role + ".json"); json.writeValue(path.toFile(), root);
        return new Harness(RtpNode.start(RtpConfig.load(path)));
    }
    static void await(BooleanSupplier condition) throws Exception {
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
    private static final class Replies<T> implements StreamObserver<T>, AutoCloseable {
        final Context.CancellableContext context = Context.current().withCancellation();
        final BlockingQueue<Object> values = new LinkedBlockingQueue<>();
        @Override public void onNext(T value) { values.add(value); }
        @Override public void onError(Throwable error) { values.add(error); }
        @Override public void onCompleted() { values.add("completed"); }
        Object next() throws Exception { var value = values.poll(3, TimeUnit.SECONDS); assertNotNull(value); return value; }
        void error(Status.Code expected) throws Exception { var value = next(); assertInstanceOf(Throwable.class, value); assertEquals(expected, Status.fromThrowable((Throwable) value).getCode()); }
        @Override public void close() { context.cancel(null); }
    }
    private Replies<TransferMetadata> prepare(Harness p, GenerateInputPB input) throws Exception {
        var replies = new Replies<TransferMetadata>();
        replies.context.run(() -> PrefillTransferGrpc.newStub(p.channel).withDeadlineAfter(3, TimeUnit.SECONDS).prepare(transfer(input, 2), replies));
        await(() -> p.status().getRunningTaskInfoCount() == 1);
        return replies;
    }
    private StreamObserver<GenerateRequestPB> allocate(Harness p, Harness d, GenerateInputPB input, Replies<GenerateOutputsPB> replies) throws Exception {
        var request = replies.context.call(() -> RpcServiceGrpc.newStub(d.channel).remoteGenerate(replies));
        request.onNext(GenerateRequestPB.newBuilder().setStage(RemoteStage.ALLOCATE).setRequestId(input.getRequestId()).setClientId("client")
            .setInput(input).addPeerAddrs(p.node.rpcAddress()).build());
        assertInstanceOf(GenerateOutputsPB.class, replies.next()); return request;
    }
    private static GenerateRequestPB load(long id) { return GenerateRequestPB.newBuilder().setStage(RemoteStage.LOAD).setRequestId(id).build(); }

    @Test void loadWaitsForRealForwardThenReleasesPButKeepsDReservation() throws Exception {
        try (var p = node("prefill", 2, 300); var d = node("decode", 2, 1); var source = prepare(p, input(1, 5, 2000)); var replies = new Replies<GenerateOutputsPB>()) {
            var request = allocate(p, d, input(1, 5, 2000), replies); request.onNext(load(1));
            assertNull(replies.values.poll(80, TimeUnit.MILLISECONDS));
            assertEquals(10, p.status().getAvailableKvCache()); assertEquals(10, d.status().getAvailableKvCache());
            assertInstanceOf(TransferMetadata.class, source.next());
            var ack = (GenerateOutputsPB) replies.next(); assertTrue(ack.hasErrorInfo()); assertEquals(0, ack.getErrorInfo().getErrorCodeValue());
            assertEquals("completed", source.next()); await(() -> p.status().getAvailableKvCache() == 16);
            assertEquals(10, d.status().getAvailableKvCache()); assertEquals(0, d.status().getRunningTaskInfo(0).getPrefixLength());
            assertEquals(0, d.status().getRunningQueryLen()); assertTrue(d.node.cache().snapshot(-1, true).keys().isEmpty());
            request.onNext(GenerateRequestPB.newBuilder().setRequestId(1).setStage(RemoteStage.GENERATE).build()); replies.error(Status.Code.UNIMPLEMENTED);
            assertEquals(16, d.status().getAvailableKvCache());
        }
    }

    @Test void readyDataRemainsPinnedWithoutAReader() throws Exception {
        try (var p = node("prefill", 2, 1); var source = prepare(p, input(1, 5, 2000))) {
            var metadata = (TransferMetadata) source.next(); assertEquals(5, metadata.getInputTokens()); assertEquals(2, metadata.getCacheKeysCount());
            assertEquals(10, p.status().getAvailableKvCache());
            assertEquals(TaskPhase.TASK_PHASE_KV_ALLOCATED, p.status().getRunningTaskInfo(0).getPhase());
            assertTrue(p.status().getRunningTaskInfo(0).getExecutionTimeMs() > 0);
            source.close(); await(() -> p.status().getAvailableKvCache() == 16 && p.status().getRunningTaskInfoCount() == 0);
            assertTrue(p.node.cache().snapshot(-1, true).keys().isEmpty());
        }
    }

    @Test void sourceCancellationBeforeForwardFailsLoadAndReleasesBothOwners() throws Exception {
        try (var p = node("prefill", 2, 1000); var d = node("decode", 2, 1); var source = prepare(p, input(1, 5, 2000)); var replies = new Replies<GenerateOutputsPB>()) {
            var request = allocate(p, d, input(1, 5, 2000), replies); request.onNext(load(1)); source.close();
            var value = replies.next(); assertInstanceOf(Throwable.class, value);
            await(() -> p.status().getAvailableKvCache() == 16 && d.status().getAvailableKvCache() == 16);
            assertTrue(p.node.cache().snapshot(-1, true).keys().isEmpty()); assertTrue(d.node.cache().snapshot(-1, true).keys().isEmpty());
        }
    }

    @Test void loadCannotUseAnUnrelatedCachedPrefixOrMissingRequest() throws Exception {
        try (var p = node("prefill", 2, 1); var d = node("decode", 2, 1); var replies = new Replies<GenerateOutputsPB>()) {
            p.node.cache().acquire(2, List.of()).lease().complete(PrefixKeys.completeBlocks(input(1, 5, 2000), 2));
            var request = allocate(p, d, input(1, 5, 2000), replies); request.onNext(load(1)); replies.error(Status.Code.NOT_FOUND);
            assertEquals(16, d.status().getAvailableKvCache());
        }
    }

    @Test void mismatchedPromptFailsWithoutConsumingSource() throws Exception {
        try (var p = node("prefill", 2, 1); var d = node("decode", 2, 1); var source = prepare(p, input(1, 5, 2000)); var replies = new Replies<GenerateOutputsPB>()) {
            assertInstanceOf(TransferMetadata.class, source.next());
            var request = allocate(p, d, input(1, 4, 2000), replies); request.onNext(load(1)); replies.error(Status.Code.FAILED_PRECONDITION);
            assertEquals(10, p.status().getAvailableKvCache()); assertEquals(16, d.status().getAvailableKvCache());
        }
    }

    @Test void mismatchedBlockSizeFailsWithoutConsumingSource() throws Exception {
        try (var p = node("prefill", 2, 1); var d = node("decode", 4, 1); var source = prepare(p, input(1, 5, 2000)); var replies = new Replies<GenerateOutputsPB>()) {
            source.next(); var request = allocate(p, d, input(1, 5, 2000), replies); request.onNext(load(1)); replies.error(Status.Code.FAILED_PRECONDITION);
            assertEquals(10, p.status().getAvailableKvCache()); assertEquals(32, d.status().getAvailableKvCache());
        }
    }

    @Test void dDeadlineWhileWaitingForPDoesNotAcknowledgeOrCommitData() throws Exception {
        try (var p = node("prefill", 2, 700); var d = node("decode", 2, 1); var source = prepare(p, input(1, 5, 2000)); var replies = new Replies<GenerateOutputsPB>()) {
            var request = allocate(p, d, input(1, 5, 250), replies); request.onNext(load(1)); replies.error(Status.Code.DEADLINE_EXCEEDED);
            assertEquals(16, d.status().getAvailableKvCache()); assertInstanceOf(Throwable.class, source.next());
            await(() -> p.status().getAvailableKvCache() == 16); assertTrue(d.node.cache().snapshot(-1, true).keys().isEmpty());
            assertTrue(replies.values.isEmpty());
        }
    }

    @Test void cancellingDecodeDuringReadCancelsPForwardAndReleasesBoth() throws Exception {
        try (var p = node("prefill", 2, 1000); var d = node("decode", 2, 1); var source = prepare(p, input(1, 5, 2000)); var replies = new Replies<GenerateOutputsPB>()) {
            var request = allocate(p, d, input(1, 5, 2000), replies); request.onNext(load(1));
            // Wait until the D has opened the reader, then cancel its owning native RPC.
            Thread.sleep(80); replies.close();
            source.error(Status.Code.CANCELLED);
            await(() -> p.status().getAvailableKvCache() == 16 && d.status().getAvailableKvCache() == 16);
            assertTrue(p.node.cache().snapshot(-1, true).keys().isEmpty());
        }
    }

    @Test void nodeCloseDuringLoadTerminatesSourceAndD() throws Exception {
        try (var p = node("prefill", 2, 1000); var d = node("decode", 2, 1); var source = prepare(p, input(1, 5, 2000)); var replies = new Replies<GenerateOutputsPB>()) {
            var request = allocate(p, d, input(1, 5, 2000), replies); request.onNext(load(1)); p.node.close();
            assertInstanceOf(Throwable.class, replies.next()); await(() -> d.status().getAvailableKvCache() == 16);
        }
    }

    @Test void readySourceExpiresWithoutAReaderAndCannotBeReadAfterExpiry() throws Exception {
        try (var p = node("prefill", 2, 1); var source = prepare(p, input(1, 5, 250))) {
            source.next(); source.error(Status.Code.DEADLINE_EXCEEDED);
            assertEquals(16, p.status().getAvailableKvCache()); assertTrue(p.node.cache().snapshot(-1, true).keys().isEmpty());
            var error = assertThrows(StatusRuntimeException.class, () -> PrefillTransferGrpc.newBlockingStub(p.channel)
                .withDeadlineAfter(1, TimeUnit.SECONDS).read(transfer(input(1, 5, 250), 2)));
            assertEquals(Status.Code.NOT_FOUND, error.getStatus().getCode());
        }
    }

    @Test void aSecondLoadDoesNotProduceAnotherAcknowledgementOrRetainDResources() throws Exception {
        try (var p = node("prefill", 2, 1); var d = node("decode", 2, 1); var source = prepare(p, input(1, 5, 2000)); var replies = new Replies<GenerateOutputsPB>()) {
            source.next(); var request = allocate(p, d, input(1, 5, 2000), replies); request.onNext(load(1));
            assertInstanceOf(GenerateOutputsPB.class, replies.next()); assertEquals("completed", source.next());
            request.onNext(load(1)); replies.error(Status.Code.FAILED_PRECONDITION);
            assertEquals(16, d.status().getAvailableKvCache()); assertEquals(1, d.status().getFinishedTaskListCount());
        }
    }

    @Test void retainedSourceRequiresBoundedLifetimeAndDuplicateDoesNotReplaceOwner() throws Exception {
        try (var p = node("prefill", 2, 1)) {
            var invalid = new Replies<TransferMetadata>();
            PrefillTransferGrpc.newStub(p.channel).prepare(transfer(input(1, 5, 0), 2), invalid); invalid.error(Status.Code.INVALID_ARGUMENT);
            try (var source = prepare(p, input(1, 5, 2000)); var duplicate = new Replies<TransferMetadata>()) {
                source.next(); PrefillTransferGrpc.newStub(p.channel).prepare(transfer(input(1, 5, 2000), 2), duplicate);
                duplicate.error(Status.Code.ALREADY_EXISTS); assertEquals(10, p.status().getAvailableKvCache());
            }
        }
    }
}
