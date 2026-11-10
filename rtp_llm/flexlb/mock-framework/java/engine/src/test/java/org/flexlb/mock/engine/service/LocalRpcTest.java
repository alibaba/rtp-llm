package org.flexlb.mock.engine.service;

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
import org.flexlb.mock.engine.config.RtpConfig;
import org.flexlb.mock.engine.execution.LLMMockEngine;
import org.flexlb.mock.engine.scheduler.LLMMockEngineTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Exercises the local RTP RPC surface and its cache key reporting.
 */
class LocalRpcTest {
    @TempDir Path directory;

    @Test void nativeStreamReturnsRealTimedResultWithWireTensorAndCacheReuse() throws Exception {
        try (var node = LLMMockEngine.start(config("totalComputeTokens"))) {
            var channel = ManagedChannelBuilder.forTarget(node.rpcAddress()).usePlaintext().build();
            try {
                var rpc = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3, TimeUnit.SECONDS);
                var outputs = rpc.generateStreamCall(PrefillAdmissionTest.input(1, 0, true));
                var first = outputs.next(); assertFalse(outputs.hasNext());
                var flatten = first.getFlattenOutput(); assertTrue(flatten.getFinished(0));
                assertEquals(List.of(1L, 1L, 1L), flatten.getOutputIds().getShapeList());
                assertEquals(TensorPB.DataType.INT32, flatten.getOutputIds().getDataType());
                assertEquals(5, ByteBuffer.wrap(flatten.getOutputIds().getInt32Data().toByteArray()).order(ByteOrder.LITTLE_ENDIAN).getInt());
                assertEquals(1, flatten.getAuxInfo(0).getOutputLen());
                var reused = rpc.generateStreamCall(PrefillAdmissionTest.input(2, 0, true)).next();
                assertEquals(4, reused.getFlattenOutput().getAuxInfo(0).getTotalReuseLen());
                assertEquals(16, rpc.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getAvailableKvCache());
                rpc.getCacheStatus(CacheVersionPB.getDefaultInstance());
                var text = metrics(node);
                for (String kind : List.of("worker_status", "cache_status")) {
                    assertTrue(sample(text, "rtp_llm_rpc_" + kind + "_qps_total", "role=\"prefill\"") > 0);
                    assertEquals(1, sample(text, "rtp_llm_rpc_" + kind + "_total_rt_us_count", "role=\"prefill\""));
                }
                assertEquals(2, sample(text, "rtp_llm_kv_cache_item_num", "role=\"prefill\""));
                assertEquals(2, sample(text, "rtp_llm_framework_requests_total", "outcome=\"success\""));
                assertEquals(2,sample(text,"rtp_llm_rpc_onflight_request_count",""));
                assertEquals(2,sample(text,"rtp_llm_rpc_onflight_request_sum",""));
                assertEquals(10, sample(text, "rtp_llm_cache_lookup_tokens_total", ""));
                assertEquals(4, sample(text, "rtp_llm_cache_hit_tokens_total", ""));
                assertEquals(6, sample(text, "rtp_llm_forward_tokens_total", "kind=\"compute\""));
                assertEquals(2, sample(text, "rtp_llm_model_forward_seconds_count", ""));
                assertEquals(0, sample(text, "rtp_llm_scheduler_requests", "state=\"running\""));
                // The same scrape must not reset event counters.
                assertEquals(4, sample(metrics(node), "rtp_llm_cache_hit_tokens_total", ""));
            } finally { channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
        }
    }

    static double sample(String text, String name, String label) {
        return text.lines().filter(line -> line.startsWith(name + "{") && line.contains(label))
            .mapToDouble(line -> Double.parseDouble(line.substring(line.lastIndexOf(' ') + 1))).findFirst().orElseThrow();
    }

    @Test void shardedPrefillUsesBaseHashesButVirtualAllocationAndReuseBoundaries() throws Exception {
        try(var node=LLMMockEngine.start(config("1",Map.of("prefill_cp_size",4)))) {
            var channel=ManagedChannelBuilder.forTarget(node.rpcAddress()).usePlaintext().build();
            try {
                var rpc=RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3,TimeUnit.SECONDS);
                var input=PrefillAdmissionTest.input(1,0,true).toBuilder().clearTokenIds()
                    .addAllTokenIds(java.util.stream.IntStream.rangeClosed(1,17).boxed().toList()).build();
                var first=rpc.generateStreamCall(input);while(first.hasNext()) first.next();
                var cache=rpc.getCacheStatus(CacheVersionPB.newBuilder().setNeedCacheKeys(true).setLatestCacheVersion(-1).build());
                var base=org.flexlb.mock.protocol.MediaInput.sequence(org.flexlb.mock.protocol.MediaInput.text(input), 2, 0, 1).keys();
                assertEquals(Set.of(base.get(3),base.get(7)),cache.getCacheKeysMap().keySet());
                assertEquals(8,cache.getBlockSize());assertEquals(64,cache.getTotalKvCache());
                var worker = rpc.getWorkerStatus(StatusVersionPB.getDefaultInstance());
                assertEquals(64, worker.getTotalKvCache());
                assertEquals(4, worker.getTpSize());
                assertEquals(1, worker.getDpSize());
                assertEquals(0, worker.getDpRank());
                var second=rpc.generateStreamCall(input.toBuilder().setRequestId(2).build());
                assertEquals(16,second.next().getFlattenOutput().getAuxInfo(0).getTotalReuseLen());
                assertFalse(second.hasNext());
            } finally {channel.shutdownNow().awaitTermination(3,TimeUnit.SECONDS);}
        }
    }

    @Test void cacheStatusPreservesVersionedKeysAndLiveCapacityOnTheWire() throws Exception {
        try (var node = LLMMockEngine.start(config("1"))) {
            try (var seed = node.cacheManager().acquire(4, List.of()).lease()) {
                assertTrue(seed.publish(List.of(Long.MIN_VALUE, Long.MAX_VALUE), 4));
            }
            var channel = ManagedChannelBuilder.forTarget(node.rpcAddress()).usePlaintext().build();
            try {
                var rpc = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3, TimeUnit.SECONDS);
                var query = CacheVersionPB.newBuilder().setNeedCacheKeys(true).setLatestCacheVersion(-1).build();
                var first = rpc.getCacheStatus(query);
                assertEquals(Map.of(Long.MIN_VALUE, true, Long.MAX_VALUE, true), first.getCacheKeysMap());
                assertEquals(first.getCacheKeysMap(), rpc.getCacheStatus(query).getCacheKeysMap());
                assertEquals(2, first.getBlockSize());
                assertEquals(16, first.getTotalKvCache());
                assertEquals(16, first.getAvailableKvCache());
                try (var held = node.cacheManager().acquire(2, List.of()).lease()) {
                    assertNotNull(held);
                    var unchanged = rpc.getCacheStatus(query.toBuilder().setLatestCacheVersion(first.getVersion()).build());
                    assertEquals(first.getVersion(), unchanged.getVersion());
                    assertEquals(0, unchanged.getCacheKeysCount());
                    assertEquals(14, unchanged.getAvailableKvCache());
                    var capacityOnly = rpc.getCacheStatus(query.toBuilder().setNeedCacheKeys(false).setLatestCacheVersion(12345).build());
                    assertEquals(12345, capacityOnly.getVersion());
                    assertEquals(0, capacityOnly.getCacheKeysCount());
                    assertEquals(14, capacityOnly.getAvailableKvCache());
                }
                assertEquals(16, rpc.getCacheStatus(query).getAvailableKvCache());
            } finally { channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
        }
    }

    @Test void requestBudgetReturnsNative603DetailsAndReclaimsKv() throws Exception {
        try (var node = LLMMockEngine.start(config("500"))) {
            var channel = ManagedChannelBuilder.forTarget(node.rpcAddress()).usePlaintext().build();
            try {
                var rpc = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3, TimeUnit.SECONDS);
                var error = assertThrows(StatusRuntimeException.class, () -> rpc.generateStreamCall(PrefillAdmissionTest.input(1, 30, true)).hasNext());
                assertEquals(Status.Code.DEADLINE_EXCEEDED, error.getStatus().getCode());
                var raw = error.getTrailers().get(Metadata.Key.of("grpc-status-details-bin", Metadata.BINARY_BYTE_MARSHALLER));
                assertEquals(603, ErrorDetailsPB.parseFrom(raw).getErrorCode());
                assertEquals(16, rpc.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getAvailableKvCache());
                var text = metrics(node);
                assertEquals(1, sample(text, "rtp_llm_framework_requests_total", "outcome=\"timeout\""));
                assertEquals(0, sample(text, "rtp_llm_model_forward_seconds_count", ""));
            } finally { channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
        }
    }

    @Test void grpcCancellationReleasesInFlightLeaseAndDoesNotCommitLater() throws Exception {
        try (var node = LLMMockEngine.start(config("300"))) {
            var channel = ManagedChannelBuilder.forTarget(node.rpcAddress()).usePlaintext().build();
            var context = Context.current().withCancellation();
            try {
                var read = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3, TimeUnit.SECONDS);
                read.checkHealth(EmptyPB.getDefaultInstance());
                var done = new CountDownLatch(1);
                var frames = new java.util.concurrent.atomic.AtomicInteger();
                context.run(() -> RpcServiceGrpc.newStub(channel).generateStreamCall(PrefillAdmissionTest.input(1, 0, true), new StreamObserver<>() {
                    @Override public void onNext(GenerateOutputsPB output) { frames.incrementAndGet(); }
                    @Override public void onError(Throwable error) { done.countDown(); }
                    @Override public void onCompleted() { done.countDown(); }
                }));
                await(() -> read.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getRunningTaskInfoList().stream().filter(task -> !task.getIsWaiting()).count() == 1);
                // A rejected second request samples both itself and the still-running first request.
                var rejected=PrefillAdmissionTest.input(2,0,true).toBuilder();
                rejected.getGenerateConfigBuilder().setReturnSoftmaxProbs(true);
                assertThrows(StatusRuntimeException.class,()->read.generateStreamCall(rejected.build()).hasNext());
                assertEquals(1,sample(metrics(node),"rtp_llm_rpc_onflight_request_count",""));
                assertEquals(2,sample(metrics(node),"rtp_llm_rpc_onflight_request_sum",""));
                var nativeStatus = read.getWorkerStatus(StatusVersionPB.getDefaultInstance());
                assertEquals(0, nativeStatus.getWaitingQueryLen());
                assertEquals(0, nativeStatus.getRunningQueryLen());
                assertEquals(-1, nativeStatus.getRunningTaskInfo(0).getExecutionTimeMs());
                assertEquals(0, nativeStatus.getRunningTaskInfo(0).getWaitingTimeMs());
                assertEquals(10, read.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getAvailableKvCache());
                assertEquals(10, sample(metrics(node), "rtp_llm_kv_tokens", "state=\"available\""));
                context.cancel(null); assertTrue(done.await(2, TimeUnit.SECONDS));
                await(() -> read.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getRunningTaskInfoList().stream().filter(task -> !task.getIsWaiting()).count() == 0);
                Thread.sleep(350);
                assertEquals(0, frames.get());
                assertEquals(16, read.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getAvailableKvCache());
                assertEquals(0, read.getCacheStatus(CacheVersionPB.newBuilder().setNeedCacheKeys(true).setLatestCacheVersion(-1).build()).getCacheKeysCount());
                var text = metrics(node);
                assertEquals(1, sample(text, "rtp_llm_framework_requests_total", "outcome=\"cancelled\""));
                assertEquals(2,sample(text,"rtp_llm_rpc_onflight_request_count",""));
                assertEquals(3,sample(text,"rtp_llm_rpc_onflight_request_sum",""));
                assertEquals(0, sample(text, "rtp_llm_model_forward_seconds_count", ""));
                assertEquals(16, sample(text, "rtp_llm_kv_tokens", "state=\"available\""));
                var next=read.generateStreamCall(PrefillAdmissionTest.input(3,0,true));
                while(next.hasNext()) next.next();
                String after=metrics(node);
                assertEquals(3,sample(after,"rtp_llm_rpc_onflight_request_count",""));
                assertEquals(4,sample(after,"rtp_llm_rpc_onflight_request_sum",""));

            } finally { context.cancel(null); channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
        }
    }

    @Test void missingDecodeRouteAndInsufficientMemoryAreErrorsNotSuccessfulFrames() throws Exception {
        try (var node = LLMMockEngine.start(config("1"))) {
            var channel = ManagedChannelBuilder.forTarget(node.rpcAddress()).usePlaintext().build();
            try {
                var rpc = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3, TimeUnit.SECONDS);
                var input = PrefillAdmissionTest.input(1, 0, true);
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
                assertEquals(0, sample(text, "rtp_llm_requests_started_total", ""), "Permanently oversized prompts never enter EngineScheduler");
                assertEquals(0, sample(text, "rtp_llm_framework_requests_total", "outcome=\"error\""));
                assertEquals(0, sample(text, "rtp_llm_cache_lookup_tokens_total", ""));
            } finally { channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
        }
    }

    @Test void rpcDeadlineIndependentlyCancelsARequestWithoutItsOwnBudget() throws Exception {
        try (var node = LLMMockEngine.start(config("3000"))) {
            var channel = ManagedChannelBuilder.forTarget(node.rpcAddress()).usePlaintext().build();
            try {
                var read = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3, TimeUnit.SECONDS);
                read.checkHealth(EmptyPB.getDefaultInstance());
                var done = new CountDownLatch(1);
                var failure = new java.util.concurrent.atomic.AtomicReference<Throwable>();
                RpcServiceGrpc.newStub(channel).withDeadlineAfter(500, TimeUnit.MILLISECONDS)
                    .generateStreamCall(PrefillAdmissionTest.input(77, 0, true), new StreamObserver<>() {
                        @Override public void onNext(GenerateOutputsPB output) { }
                        @Override public void onError(Throwable error) { failure.set(error); done.countDown(); }
                        @Override public void onCompleted() { done.countDown(); }
                    });
                await(() -> read.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getRunningTaskInfoList().stream().filter(task -> !task.getIsWaiting()).count() == 1);
                assertTrue(done.await(2, TimeUnit.SECONDS));
                assertEquals(Status.Code.DEADLINE_EXCEEDED, Status.fromThrowable(failure.get()).getCode());
                await(() -> read.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getRunningTaskInfoList().stream().filter(task -> !task.getIsWaiting()).count() == 0);
                var status = read.getWorkerStatus(StatusVersionPB.getDefaultInstance());
                assertEquals(603, status.getFinishedTaskList(0).getErrorInfo().getErrorCode());
                assertEquals(16, status.getAvailableKvCache());
            } finally { channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
        }
    }

    @Test void nativePrefixKeysAgreeWithProductionCppReferenceIncludingNegativeToken() {
        var input = GenerateInputPB.newBuilder().addAllTokenIds(List.of(1, 2, 3, 4, 5, -1)).build();
        assertEquals(List.of(-8366447769517741626L, 455111481605203084L, 3534810267006820515L), org.flexlb.mock.protocol.MediaInput.sequence(org.flexlb.mock.protocol.MediaInput.text(input), 2, 0, 1).keys());
    }

    private RtpConfig config(String expression) throws Exception { return config(expression,Map.of()); }

    private RtpConfig config(String expression,Map<String,Object> settings) throws Exception {
        var json = new ObjectMapper(); ObjectNode root = json.valueToTree(LLMMockEngineTest.configuration("prefill"));
        ((ObjectNode) root.path("component").path("config")).put("seq_size_per_block", 2).put("reuse_cache", true);
        ((ObjectNode) root.path("component").path("config").path("mock.compute").path("forward")).put("expression", expression);
        ((ObjectNode)root.path("component").path("config")).setAll((ObjectNode)json.valueToTree(settings));
        var path = directory.resolve("config.json"); json.writeValue(path.toFile(), root); return RtpConfig.load(path);
    }

    private static void await(BooleanSupplier condition) throws Exception {
        long until = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
        while (!condition.getAsBoolean()) { if (System.nanoTime() >= until) fail("Condition did not become true"); Thread.sleep(5); }
    }

    private static String metrics(LLMMockEngine node) throws Exception {
        try (var input = java.net.URI.create((String) node.endpoints().get("metrics")).toURL().openStream()) {
            return new String(input.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
        }
    }
}
