package org.flexlb.mock.engine.rtp;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.grpc.*;
import io.grpc.stub.StreamObserver;
import io.grpc.netty.NettyServerBuilder;
import java.net.InetSocketAddress;
import org.flexlb.mock.cache.KvCache;
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
    private static TransferRequest transfer(GenerateInputPB input) {
        return TransferRequest.newBuilder().setClientId("client").setInput(input).build();
    }
    private Harness node(String role, int block, int millis) throws Exception {
        var json = new ObjectMapper(); ObjectNode root = json.valueToTree(RtpNodeTest.configuration(role));
        ((ObjectNode) root.path("component").path("config")).put("seq_size_per_block", block).put("reuse_cache", true);
        ((ObjectNode) root.path("component").path("config").path("compute").path("forward")).put("expression", Integer.toString(millis));
        var path = directory.resolve(role + ".json"); json.writeValue(path.toFile(), root);
        return new Harness(RtpConfig.load(path));
    }
    static void await(BooleanSupplier condition) throws Exception {
        long end = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
        while (!condition.getAsBoolean()) { if (System.nanoTime() >= end) fail("Condition timed out"); Thread.sleep(5); }
    }
    private static final class Harness implements AutoCloseable {
        final KvCache cache;
        final RtpRpcService service;
        final PrefillTransfer transfer;
        final Server server;
        final ManagedChannel channel;
        Harness(RtpConfig config) throws Exception {
            cache = new KvCache(config.blockSize(), config.deviceBlocks(), config.reserveBlocks(), config.evictionPolicy());
            service = new RtpRpcService(config, cache, RtpScheduler.factory(config));
            transfer = (PrefillTransfer) service.transfer();
            var builder = NettyServerBuilder.forAddress(new InetSocketAddress("127.0.0.1", 0)).addService(service);
            if (transfer != null) builder.addService(transfer);
            server = builder.build().start();
            service.rpcAddress(rpcAddress());
            channel = ManagedChannelBuilder.forTarget(rpcAddress()).usePlaintext().build();
        }
        String rpcAddress() { return "127.0.0.1:" + server.getPort(); }
        WorkerStatusPB status() { return RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3, TimeUnit.SECONDS).getWorkerStatus(StatusVersionPB.getDefaultInstance()); }
        @Override public void close() throws Exception {
            service.close(); server.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); cache.close();
        }
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
    private PrefillTransfer.Source prepare(Harness p, GenerateInputPB input) throws Exception {
        var source = p.transfer.start(transfer(input), Long.MAX_VALUE);
        await(() -> p.status().getRunningTaskInfoCount() == 1);
        return source;
    }
    private void ready(PrefillTransfer.Source source) throws Exception {
        assertTrue(source.ready().toCompletableFuture().get(3, TimeUnit.SECONDS).status().isOk());
    }
    private StreamObserver<GenerateRequestPB> allocate(Harness p, Harness d, GenerateInputPB input, Replies<GenerateOutputsPB> replies) throws Exception {
        var request = replies.context.call(() -> RpcServiceGrpc.newStub(d.channel).remoteGenerate(replies));
        request.onNext(GenerateRequestPB.newBuilder().setStage(RemoteStage.ALLOCATE).setRequestId(input.getRequestId()).setClientId("client")
            .setInput(input).addPeerAddrs(p.rpcAddress()).build());
        assertInstanceOf(GenerateOutputsPB.class, replies.next()); return request;
    }
    private static GenerateRequestPB load(long id) { return GenerateRequestPB.newBuilder().setStage(RemoteStage.LOAD).setRequestId(id).build(); }

    @Test void loadWaitsForForwardAndPRemainsPinnedUntilAcknowledgement() throws Exception {
        try (var p = node("prefill", 2, 300); var d = node("decode", 2, 1); var source = prepare(p, input(1, 5, 2000)); var replies = new Replies<GenerateOutputsPB>()) {
            var request = allocate(p, d, input(1, 5, 2000), replies); request.onNext(load(1));
            assertNull(replies.values.poll(80, TimeUnit.MILLISECONDS));
            assertEquals(10, p.status().getAvailableKvCache()); assertEquals(10, d.status().getAvailableKvCache());
            ready(source);
            var ack = (GenerateOutputsPB) replies.next(); assertTrue(ack.hasErrorInfo()); assertEquals(0, ack.getErrorInfo().getErrorCodeValue());
            assertEquals(10, p.status().getAvailableKvCache());
            assertEquals(1, p.status().getWaitingQueryLen());
            assertEquals(0, p.status().getRunningQueryLen());
            source.acknowledged();
            await(() -> p.status().getRunningTaskInfoCount() == 0); await(() -> p.status().getAvailableKvCache() == 16);
            assertEquals(10, d.status().getAvailableKvCache()); assertEquals(0, d.status().getRunningTaskInfo(0).getPrefixLength());
            assertEquals(0, d.status().getRunningQueryLen()); assertTrue(d.cache.snapshot(-1, true).keys().isEmpty());
            request.onNext(GenerateRequestPB.newBuilder().setRequestId(1).setStage(RemoteStage.GENERATE).setFirstGenerateTokenId(5).build());
            for (int count = 2; count <= 4; count++) {
                var output = (GenerateOutputsPB) replies.next();
                assertEquals(count, output.getFlattenOutput().getAuxInfo(0).getOutputLen());
                assertEquals(count == 4, output.getFlattenOutput().getFinished(0));
            }
            assertEquals("completed", replies.next()); assertEquals(16, d.status().getAvailableKvCache());
        }
    }

    @Test void readyDataRemainsPinnedWithoutAReader() throws Exception {
        try (var p = node("prefill", 2, 1); var source = prepare(p, input(1, 5, 2000))) {
            ready(source);
            assertFalse(source.ready().toCompletableFuture().join().output().getFlattenOutput().getFinished(0));
            assertEquals(10, p.status().getAvailableKvCache());
            assertEquals(TaskPhase.TASK_PHASE_KV_ALLOCATED, p.status().getRunningTaskInfo(0).getPhase());
            assertTrue(p.status().getRunningTaskInfo(0).getExecutionTimeMs() > 0);
            source.close(); await(() -> p.status().getAvailableKvCache() == 16 && p.status().getRunningTaskInfoCount() == 0);
            assertTrue(p.cache.snapshot(-1, true).keys().isEmpty());
        }
    }

    @Test void sourceCancellationBeforeForwardFailsLoadAndReleasesBothOwners() throws Exception {
        try (var p = node("prefill", 2, 1000); var d = node("decode", 2, 1); var source = prepare(p, input(1, 5, 2000)); var replies = new Replies<GenerateOutputsPB>()) {
            var request = allocate(p, d, input(1, 5, 2000), replies); request.onNext(load(1)); source.close();
            var value = replies.next(); assertInstanceOf(Throwable.class, value);
            await(() -> p.status().getAvailableKvCache() == 16 && d.status().getAvailableKvCache() == 16);
            assertTrue(p.cache.snapshot(-1, true).keys().isEmpty()); assertTrue(d.cache.snapshot(-1, true).keys().isEmpty());
        }
    }

    @Test void loadCannotUseAnUnrelatedCachedPrefixOrMissingRequest() throws Exception {
        try (var p = node("prefill", 2, 1); var d = node("decode", 2, 1); var replies = new Replies<GenerateOutputsPB>()) {
            try (var lease = p.cache.acquire(4, List.of()).lease()) {
                lease.publish(PrefixKeys.completeBlocks(input(1, 5, 2000), 2), 4);
            }
            var request = allocate(p, d, input(1, 5, 2000), replies); request.onNext(load(1)); replies.error(Status.Code.NOT_FOUND);
            assertEquals(16, d.status().getAvailableKvCache());
        }
    }

    @Test void mismatchedPromptFailsWithoutConsumingSource() throws Exception {
        try (var p = node("prefill", 2, 1); var d = node("decode", 2, 1); var source = prepare(p, input(1, 5, 2000)); var replies = new Replies<GenerateOutputsPB>()) {
            ready(source);
            var request = allocate(p, d, input(1, 5, 2000).toBuilder().setTokenIds(0, 99).build(), replies); request.onNext(load(1)); replies.error(Status.Code.FAILED_PRECONDITION);
            assertEquals(10, p.status().getAvailableKvCache()); assertEquals(16, d.status().getAvailableKvCache());
        }
    }

    @Test void transferUsesTokenCoverageAndEachEngineOwnsItsBlockGeometry() throws Exception {
        for (int[] shape : new int[][] {{512, 128, 600}, {128, 512, 600}, {2, 3, 5}}) {
            int pBlock = shape[0], dBlock = shape[1], tokens = shape[2];
            var input = input(1, tokens, 2000);
            try (var p = node("prefill", pBlock, 1); var d = node("decode", dBlock, 1);
                    var source = prepare(p, input); var replies = new Replies<GenerateOutputsPB>()) {
                ready(source);
                var request = allocate(p, d, input, replies); request.onNext(load(1));
                var ack = (GenerateOutputsPB) replies.next(); assertEquals(0, ack.getErrorInfo().getErrorCodeValue());
                assertEquals((8 - (tokens + pBlock - 1) / pBlock) * pBlock, p.status().getAvailableKvCache());
                assertEquals((8 - (tokens + dBlock - 1) / dBlock) * dBlock, d.status().getAvailableKvCache());
                assertTrue(d.cache.snapshot(-1, true).keys().isEmpty());
                source.acknowledged();
                await(() -> p.status().getRunningTaskInfoCount() == 0);
                request.onNext(GenerateRequestPB.newBuilder().setRequestId(1).setStage(RemoteStage.GENERATE)
                    .setFirstGenerateTokenId(tokens).build());
                for (int count = 2; count <= 4; count++) {
                    var output = (GenerateOutputsPB) replies.next();
                    assertEquals(count, output.getFlattenOutput().getAuxInfo(0).getOutputLen());
                    assertEquals(count == 4, output.getFlattenOutput().getFinished(0));
                }
                assertEquals("completed", replies.next());
                assertEquals(8 * pBlock, p.status().getAvailableKvCache());
                assertEquals(8 * dBlock, d.status().getAvailableKvCache());
                assertEquals(new HashSet<>(PrefixKeys.completeBlocks(input, pBlock)), p.cache.snapshot(-1, true).keys());
                assertEquals(new HashSet<>(PrefixKeys.completeBlocks(input, dBlock)), d.cache.snapshot(-1, true).keys());
            }
        }
    }

    @Test void incorrectTokenCoverageFailsLoadAndReleasesDecode() throws Exception {
        var peer = NettyServerBuilder.forAddress(new InetSocketAddress("127.0.0.1", 0))
            .addService(new PrefillTransferGrpc.PrefillTransferImplBase() {
                @Override public void read(TransferRequest request, StreamObserver<TransferMetadata> response) {
                    response.onNext(TransferMetadata.newBuilder().setInputTokens(4).setPrefillFinished(false).build());
                    response.onCompleted();
                }
            }).build().start();
        try (var d = node("decode", 3, 1); var replies = new Replies<GenerateOutputsPB>()) {
            var request = replies.context.call(() -> RpcServiceGrpc.newStub(d.channel).remoteGenerate(replies));
            request.onNext(GenerateRequestPB.newBuilder().setStage(RemoteStage.ALLOCATE).setRequestId(1).setClientId("client")
                .setInput(input(1, 5, 2000)).addPeerAddrs("127.0.0.1:" + peer.getPort()).build());
            assertInstanceOf(GenerateOutputsPB.class, replies.next());
            request.onNext(load(1)); replies.error(Status.Code.FAILED_PRECONDITION);
            await(() -> d.status().getAvailableKvCache() == 24 && d.status().getRunningTaskInfoCount() == 0);
            assertTrue(d.cache.snapshot(-1, true).keys().isEmpty());
        } finally { peer.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
    }

    @Test void dDeadlineWhileWaitingForPDoesNotAcknowledgeOrCommitData() throws Exception {
        try (var p = node("prefill", 2, 700); var d = node("decode", 2, 1); var source = prepare(p, input(1, 5, 2000)); var replies = new Replies<GenerateOutputsPB>()) {
            var request = allocate(p, d, input(1, 5, 250), replies); request.onNext(load(1)); replies.error(Status.Code.DEADLINE_EXCEEDED);
            assertEquals(16, d.status().getAvailableKvCache()); assertFalse(source.ready().toCompletableFuture().get(3, TimeUnit.SECONDS).status().isOk());
            await(() -> p.status().getAvailableKvCache() == 16); assertTrue(d.cache.snapshot(-1, true).keys().isEmpty());
            assertTrue(replies.values.isEmpty());
        }
    }

    @Test void cancellingDecodeDuringReadCancelsPForwardAndReleasesBoth() throws Exception {
        try (var p = node("prefill", 2, 1000); var d = node("decode", 2, 1); var source = prepare(p, input(1, 5, 2000)); var replies = new Replies<GenerateOutputsPB>()) {
            var request = allocate(p, d, input(1, 5, 2000), replies); request.onNext(load(1));
            // Wait until the D has opened the reader, then cancel its owning native RPC.
            Thread.sleep(80); replies.close();
            assertEquals(Status.Code.CANCELLED, source.ready().toCompletableFuture().get(3, TimeUnit.SECONDS).status().getCode());
            await(() -> p.status().getAvailableKvCache() == 16 && d.status().getAvailableKvCache() == 16);
            assertTrue(p.cache.snapshot(-1, true).keys().isEmpty());
        }
    }

    @Test void nodeCloseDuringLoadTerminatesSourceAndD() throws Exception {
        try (var p = node("prefill", 2, 1000); var d = node("decode", 2, 1); var source = prepare(p, input(1, 5, 2000)); var replies = new Replies<GenerateOutputsPB>()) {
            var request = allocate(p, d, input(1, 5, 2000), replies); request.onNext(load(1)); p.close();
            assertInstanceOf(Throwable.class, replies.next()); await(() -> d.status().getAvailableKvCache() == 16);
        }
    }

    @Test void readySourceExpiresWithoutAReaderAndCannotBeReadAfterExpiry() throws Exception {
        try (var p = node("prefill", 2, 1); var source = prepare(p, input(1, 5, 250))) {
            ready(source); await(() -> p.status().getRunningTaskInfoCount() == 0);
            assertEquals(603, p.status().getFinishedTaskList(0).getErrorInfo().getErrorCode());
            assertEquals(16, p.status().getAvailableKvCache()); assertTrue(p.cache.snapshot(-1, true).keys().isEmpty());
            var error = assertThrows(StatusRuntimeException.class, () -> PrefillTransferGrpc.newBlockingStub(p.channel)
                .withDeadlineAfter(1, TimeUnit.SECONDS).read(transfer(input(1, 5, 250))));
            assertEquals(Status.Code.NOT_FOUND, error.getStatus().getCode());
        }
    }

    @Test void aSecondLoadDoesNotProduceAnotherAcknowledgementOrRetainDResources() throws Exception {
        try (var p = node("prefill", 2, 1); var d = node("decode", 2, 1); var source = prepare(p, input(1, 5, 2000)); var replies = new Replies<GenerateOutputsPB>()) {
            ready(source); var request = allocate(p, d, input(1, 5, 2000), replies); request.onNext(load(1));
            assertInstanceOf(GenerateOutputsPB.class, replies.next()); source.acknowledged(); await(() -> p.status().getRunningTaskInfoCount() == 0);
            request.onNext(load(1)); replies.error(Status.Code.FAILED_PRECONDITION);
            assertEquals(16, d.status().getAvailableKvCache()); assertEquals(1, d.status().getFinishedTaskListCount());
        }
    }

    @Test void duplicateSourceDoesNotReplaceOwner() throws Exception {
        try (var p = node("prefill", 2, 1); var source = prepare(p, input(1, 5, 2000))) {
            ready(source);
            var error = assertThrows(StatusRuntimeException.class,
                () -> p.transfer.start(transfer(input(1, 5, 2000)), Long.MAX_VALUE));
            assertEquals(Status.Code.ALREADY_EXISTS, error.getStatus().getCode());
            assertEquals(10, p.status().getAvailableKvCache());
        }
    }
}
