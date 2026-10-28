package org.flexlb.mock.engine.rtp;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.*;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.LinkedBlockingQueue;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.mock.engine.rtp.transfer.*;

/** Test-only external native client. No Engine substitution. */
public final class NativeProbe {
    public static void main(String[] args) throws Exception {
        var channel = ManagedChannelBuilder.forTarget(args[0]).usePlaintext().build();
        try {
            var rpc = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3, TimeUnit.SECONDS);
            var worker = rpc.getWorkerStatus(StatusVersionPB.getDefaultInstance());
            var cache = rpc.getCacheStatus(CacheVersionPB.newBuilder().setNeedCacheKeys(true).setLatestCacheVersion(-1).build());
            var health = rpc.checkHealth(EmptyPB.getDefaultInstance());
            var result = new java.util.HashMap<String, Object>(Map.of("health", health.getHealth(), "role", worker.getRole(), "role_type", worker.getRoleType().name(),
                "alive", worker.getAlive(), "total_tokens", worker.getTotalKvCache(), "available_tokens", cache.getAvailableKvCache(),
                "block_size", cache.getBlockSize(), "cache_keys", cache.getCacheKeysCount()));
            if (worker.getRoleType() == RoleTypePB.ROLE_TYPE_PREFILL) {
                var input = GenerateInputPB.newBuilder().setRequestId(42).addAllTokenIds(List.of(1, 2, 3, 4))
                    .setGenerateConfig(GenerateConfigPB.newBuilder().setMaxNewTokens(1)).build();
                var stream = rpc.generateStreamCall(input); var output = stream.next();
                if (stream.hasNext() || !output.getFlattenOutput().getFinished(0)) throw new IllegalStateException("Invalid local generation result");
                result.put("local_output_tokens", output.getFlattenOutput().getAuxInfo(0).getOutputLen());
            } else if (worker.getRoleType() == RoleTypePB.ROLE_TYPE_DECODE) {
                var replies = new LinkedBlockingQueue<Object>();
                var context = Context.current().withCancellation();
                try {
                    var stream = context.call(() -> RpcServiceGrpc.newStub(channel).remoteGenerate(new StreamObserver<>() {
                        @Override public void onNext(GenerateOutputsPB output) { replies.add(output); }
                        @Override public void onError(Throwable error) { replies.add(error); }
                        @Override public void onCompleted() { replies.add("completed"); }
                    }));
                    stream.onNext(GenerateRequestPB.newBuilder().setStage(RemoteStage.ALLOCATE).setRequestId(43).setClientId("native-probe")
                        .setInput(GenerateInputPB.newBuilder().setRequestId(43).addAllTokenIds(List.of(1, 2, 3, 4))
                            .setGenerateConfig(GenerateConfigPB.newBuilder().setMaxNewTokens(4))).build());
                    if (!(replies.poll(2, TimeUnit.SECONDS) instanceof GenerateOutputsPB acknowledgement) || acknowledgement.hasFlattenOutput())
                        throw new IllegalStateException("Missing decode ALLOCATE acknowledgement");
                    var reserved = rpc.getWorkerStatus(StatusVersionPB.getDefaultInstance());
                    if (reserved.getRunningTaskInfoCount() != 1 || reserved.getRunningTaskInfo(0).getPhase() != TaskPhase.TASK_PHASE_KV_ALLOCATED)
                        throw new IllegalStateException("Decode did not hold an allocated reservation");
                    result.put("decode_reserved_tokens", worker.getAvailableKvCache() - reserved.getAvailableKvCache());
                    context.cancel(null);
                    long end = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
                    while (rpc.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getWaitingQueryLen() != 0) {
                        if (System.nanoTime() >= end) throw new IllegalStateException("Decode did not release cancelled reservation");
                        Thread.sleep(5);
                    }
                    result.put("decode_available_after_cancel", rpc.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getAvailableKvCache());
                } finally { context.cancel(null); }
            }
            if (args.length == 3) result.putAll(transfer(args[2], channel));
            new ObjectMapper().writeValue(Path.of(args[1]).toFile(), result);
        } finally { channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
    }
    private static Map<String, Object> transfer(String prefillAddress, ManagedChannel decode) throws Exception {
        var p = ManagedChannelBuilder.forTarget(prefillAddress).usePlaintext().build();
        var source = new LinkedBlockingQueue<Object>(); var replies = new LinkedBlockingQueue<Object>();
        var context = Context.current().withCancellation();
        try {
            var input = GenerateInputPB.newBuilder().setRequestId(44).addAllTokenIds(List.of(1, 2, 3, 4))
                .setGenerateConfig(GenerateConfigPB.newBuilder().setMaxNewTokens(4).setTimeoutMs(2000)).build();
            context.run(() -> PrefillTransferGrpc.newStub(p).prepare(TransferRequest.newBuilder().setClientId("native-probe")
                .setInput(input).setBlockSize(16).build(), new StreamObserver<>() {
                    @Override public void onNext(TransferMetadata value) { source.add(value); }
                    @Override public void onError(Throwable error) { source.add(error); }
                    @Override public void onCompleted() { source.add("completed"); }
                }));
            if (!(source.poll(2, TimeUnit.SECONDS) instanceof TransferMetadata)) throw new IllegalStateException("P source not ready");
            var pRpc = RpcServiceGrpc.newBlockingStub(p).withDeadlineAfter(2, TimeUnit.SECONDS);
            var dRpc = RpcServiceGrpc.newBlockingStub(decode).withDeadlineAfter(2, TimeUnit.SECONDS);
            long held = pRpc.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getAvailableKvCache();
            if (held != 112) throw new IllegalStateException("P data was not pinned until transfer");
            var request = context.call(() -> RpcServiceGrpc.newStub(decode).remoteGenerate(new StreamObserver<>() {
                @Override public void onNext(GenerateOutputsPB value) { replies.add(value); }
                @Override public void onError(Throwable error) { replies.add(error); }
                @Override public void onCompleted() { replies.add("completed"); }
            }));
            request.onNext(GenerateRequestPB.newBuilder().setRequestId(44).setClientId("native-probe").setInput(input)
                .addPeerAddrs(prefillAddress).setStage(RemoteStage.ALLOCATE).build());
            if (!(replies.poll(2, TimeUnit.SECONDS) instanceof GenerateOutputsPB)) throw new IllegalStateException("D allocation failed");
            request.onNext(GenerateRequestPB.newBuilder().setRequestId(44).setStage(RemoteStage.LOAD).build());
            if (!(replies.poll(2, TimeUnit.SECONDS) instanceof GenerateOutputsPB loaded) || !loaded.hasErrorInfo()
                    || loaded.getErrorInfo().getErrorCodeValue() != 0) throw new IllegalStateException("D LOAD was not confirmed");
            if (!"completed".equals(source.poll(2, TimeUnit.SECONDS))) throw new IllegalStateException("P source did not terminate after transfer");
            long pAvailable = pRpc.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getAvailableKvCache();
            if (pAvailable != 128 || dRpc.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getAvailableKvCache() != 112)
                throw new IllegalStateException("Wrong P/D ownership after LOAD");
            context.cancel(null);
            long end = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
            while (dRpc.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getAvailableKvCache() != 128) {
                if (System.nanoTime() >= end) throw new IllegalStateException("D leaked loaded KV");
                Thread.sleep(5);
            }
            return Map.of("transfer_loaded", true, "prefill_available_while_pinned", held,
                "prefill_available_after_load", pAvailable, "decode_available_after_load_cancel", 128);
        } finally { context.cancel(null); p.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
    }

}
