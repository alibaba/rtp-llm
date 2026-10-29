package org.flexlb.mock.engine.rtp;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.grpc.*;
import io.grpc.stub.StreamObserver;
import java.nio.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.BooleanSupplier;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class LocalRpcTest {
    @TempDir Path directory;
    private RtpConfig config(String expression) throws Exception {
        var json = new ObjectMapper(); ObjectNode root = json.valueToTree(RtpNodeTest.configuration("prefill"));
        ((ObjectNode) root.path("component").path("config")).put("seq_size_per_block", 2).put("reuse_cache", true);
        ((ObjectNode) root.path("component").path("config").path("compute").path("forward")).put("expression", expression);
        var path = directory.resolve("config.json"); json.writeValue(path.toFile(), root); return RtpConfig.load(path);
    }
    private static void await(BooleanSupplier condition) throws Exception {
        long until = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
        while (!condition.getAsBoolean()) { if (System.nanoTime() >= until) fail("Condition did not become true"); Thread.sleep(5); }
    }

    private static String metrics(RtpNode node) throws Exception {
        try (var input = java.net.URI.create((String) node.endpoints().get("metrics")).toURL().openStream()) {
            return new String(input.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
        }
    }
    static double sample(String text, String name, String label) {
        return text.lines().filter(line -> line.startsWith(name + "{") && line.contains(label))
            .mapToDouble(line -> Double.parseDouble(line.substring(line.lastIndexOf(' ') + 1))).findFirst().orElseThrow();
    }

    @Test void nativeStreamReturnsRealTimedResultWithWireTensorAndCacheReuse() throws Exception {
        try (var node = RtpNode.start(config("totalComputeTokens"))) {
            var channel = ManagedChannelBuilder.forTarget(node.rpcAddress()).usePlaintext().build();
            try {
                var rpc = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3, TimeUnit.SECONDS);
                var outputs = rpc.generateStreamCall(LocalPrefillTest.input(1, 0, true));
                var first = outputs.next(); assertFalse(outputs.hasNext());
                var flatten = first.getFlattenOutput(); assertTrue(flatten.getFinished(0));
                assertEquals(List.of(1L, 1L, 1L), flatten.getOutputIds().getShapeList());
                assertEquals(TensorPB.DataType.INT32, flatten.getOutputIds().getDataType());
                assertEquals(5, ByteBuffer.wrap(flatten.getOutputIds().getInt32Data().toByteArray()).order(ByteOrder.LITTLE_ENDIAN).getInt());
                assertEquals(1, flatten.getAuxInfo(0).getOutputLen());
                var reused = rpc.generateStreamCall(LocalPrefillTest.input(2, 0, true)).next();
                assertEquals(4, reused.getFlattenOutput().getAuxInfo(0).getTotalReuseLen());
                assertEquals(16, rpc.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getAvailableKvCache());
                var text = metrics(node);
                assertEquals(2, sample(text, "mock_engine_requests_finished_total", "outcome=\"success\""));
                assertEquals(10, sample(text, "mock_engine_cache_lookup_tokens_total", ""));
                assertEquals(4, sample(text, "mock_engine_cache_hit_tokens_total", ""));
                assertEquals(6, sample(text, "mock_engine_forward_tokens_total", "kind=\"compute\""));
                assertEquals(2, sample(text, "mock_engine_forward_seconds_count", ""));
                assertEquals(0, sample(text, "mock_engine_requests", "state=\"running\""));
                // The same scrape must not reset event counters.
                assertEquals(4, sample(metrics(node), "mock_engine_cache_hit_tokens_total", ""));
            } finally { channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
        }
    }

    @Test void requestBudgetReturnsNative603DetailsAndReclaimsKv() throws Exception {
        try (var node = RtpNode.start(config("500"))) {
            var channel = ManagedChannelBuilder.forTarget(node.rpcAddress()).usePlaintext().build();
            try {
                var rpc = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3, TimeUnit.SECONDS);
                var error = assertThrows(StatusRuntimeException.class, () -> rpc.generateStreamCall(LocalPrefillTest.input(1, 30, true)).hasNext());
                assertEquals(Status.Code.DEADLINE_EXCEEDED, error.getStatus().getCode());
                var raw = error.getTrailers().get(Metadata.Key.of("grpc-status-details-bin", Metadata.BINARY_BYTE_MARSHALLER));
                assertEquals(603, ErrorDetailsPB.parseFrom(raw).getErrorCode());
                assertEquals(16, rpc.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getAvailableKvCache());
                var text = metrics(node);
                assertEquals(1, sample(text, "mock_engine_requests_finished_total", "outcome=\"timeout\""));
                assertEquals(0, sample(text, "mock_engine_forward_seconds_count", ""));
            } finally { channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
        }
    }

    @Test void grpcCancellationReleasesInFlightLeaseAndDoesNotCommitLater() throws Exception {
        try (var node = RtpNode.start(config("300"))) {
            var channel = ManagedChannelBuilder.forTarget(node.rpcAddress()).usePlaintext().build();
            var context = Context.current().withCancellation();
            try {
                var read = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3, TimeUnit.SECONDS);
                read.checkHealth(EmptyPB.getDefaultInstance());
                var done = new CountDownLatch(1);
                var frames = new java.util.concurrent.atomic.AtomicInteger();
                context.run(() -> RpcServiceGrpc.newStub(channel).generateStreamCall(LocalPrefillTest.input(1, 0, true), new StreamObserver<>() {
                    @Override public void onNext(GenerateOutputsPB output) { frames.incrementAndGet(); }
                    @Override public void onError(Throwable error) { done.countDown(); }
                    @Override public void onCompleted() { done.countDown(); }
                }));
                await(() -> read.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getRunningQueryLen() == 1);
                assertEquals(10, read.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getAvailableKvCache());
                assertEquals(10, sample(metrics(node), "mock_engine_kv_tokens", "state=\"available\""));
                context.cancel(null); assertTrue(done.await(2, TimeUnit.SECONDS));
                await(() -> read.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getRunningQueryLen() == 0);
                Thread.sleep(350);
                assertEquals(0, frames.get());
                assertEquals(16, read.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getAvailableKvCache());
                assertEquals(0, read.getCacheStatus(CacheVersionPB.newBuilder().setNeedCacheKeys(true).setLatestCacheVersion(-1).build()).getCacheKeysCount());
                var text = metrics(node);
                assertEquals(1, sample(text, "mock_engine_requests_finished_total", "outcome=\"cancelled\""));
                assertEquals(0, sample(text, "mock_engine_forward_seconds_count", ""));
                assertEquals(16, sample(text, "mock_engine_kv_tokens", "state=\"available\""));
            } finally { context.cancel(null); channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
        }
    }

    @Test void missingDecodeRouteAndInsufficientMemoryAreErrorsNotSuccessfulFrames() throws Exception {
        try (var node = RtpNode.start(config("1"))) {
            var channel = ManagedChannelBuilder.forTarget(node.rpcAddress()).usePlaintext().build();
            try {
                var rpc = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3, TimeUnit.SECONDS);
                var input = LocalPrefillTest.input(1, 0, true);
                var pd = input.toBuilder().setGenerateConfig(input.getGenerateConfig().toBuilder().setMaxNewTokens(8).setCanUsePdSeparation(true)).build();
                var unsupported = assertThrows(StatusRuntimeException.class, () -> rpc.generateStreamCall(pd).hasNext());
                assertEquals(Status.Code.INVALID_ARGUMENT, unsupported.getStatus().getCode());
                var probabilities = input.toBuilder().setGenerateConfig(input.getGenerateConfig().toBuilder().setReturnSoftmaxProbs(true)).build();
                assertEquals(Status.Code.UNIMPLEMENTED, assertThrows(StatusRuntimeException.class,
                    () -> rpc.generateStreamCall(probabilities).hasNext()).getStatus().getCode());
                var oversized = input.toBuilder().clearTokenIds()
                    .addAllTokenIds(java.util.Collections.nCopies(19, 1)).build();
                var error = assertThrows(StatusRuntimeException.class, () -> rpc.generateStreamCall(oversized).hasNext());
                assertEquals(Status.Code.RESOURCE_EXHAUSTED, error.getStatus().getCode());
                var bytes = error.getTrailers().get(Metadata.Key.of("grpc-status-details-bin", Metadata.BINARY_BYTE_MARSHALLER));
                assertEquals(602, ErrorDetailsPB.parseFrom(bytes).getErrorCode());
                var text = metrics(node);
                assertEquals(1, sample(text, "mock_engine_requests_started_total", ""));
                assertEquals(1, sample(text, "mock_engine_requests_finished_total", "outcome=\"error\""));
                assertEquals(0, sample(text, "mock_engine_cache_lookup_tokens_total", ""));
            } finally { channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
        }
    }

    @Test void rpcDeadlineIndependentlyCancelsARequestWithoutItsOwnBudget() throws Exception {
        try (var node = RtpNode.start(config("3000"))) {
            var channel = ManagedChannelBuilder.forTarget(node.rpcAddress()).usePlaintext().build();
            try {
                var read = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3, TimeUnit.SECONDS);
                read.checkHealth(EmptyPB.getDefaultInstance());
                var done = new CountDownLatch(1);
                var failure = new java.util.concurrent.atomic.AtomicReference<Throwable>();
                RpcServiceGrpc.newStub(channel).withDeadlineAfter(500, TimeUnit.MILLISECONDS)
                    .generateStreamCall(LocalPrefillTest.input(77, 0, true), new StreamObserver<>() {
                        @Override public void onNext(GenerateOutputsPB output) { }
                        @Override public void onError(Throwable error) { failure.set(error); done.countDown(); }
                        @Override public void onCompleted() { done.countDown(); }
                    });
                await(() -> read.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getRunningQueryLen() == 1);
                assertTrue(done.await(2, TimeUnit.SECONDS));
                assertEquals(Status.Code.DEADLINE_EXCEEDED, Status.fromThrowable(failure.get()).getCode());
                await(() -> read.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getRunningQueryLen() == 0);
                var status = read.getWorkerStatus(StatusVersionPB.getDefaultInstance());
                assertEquals(603, status.getFinishedTaskList(0).getErrorInfo().getErrorCode());
                assertEquals(16, status.getAvailableKvCache());
            } finally { channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
        }
    }

    @Test void nativePrefixKeysAgreeWithProductionCppReferenceIncludingNegativeToken() {
        var input = GenerateInputPB.newBuilder().addAllTokenIds(List.of(1, 2, 3, 4, 5, -1)).build();
        assertEquals(List.of(-8366447769517741626L, 455111481605203084L, 3534810267006820515L), PrefixKeys.completeBlocks(input, 2));
    }
}
