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
        try {
            var input = GenerateInputPB.newBuilder().setRequestId(44).addAllTokenIds(List.of(1, 2, 3, 4))
                .setGenerateConfig(GenerateConfigPB.newBuilder().setMaxNewTokens(4).setTimeoutMs(2000)).build();
            var pRpc = RpcServiceGrpc.newBlockingStub(p).withDeadlineAfter(2, TimeUnit.SECONDS);
            var dRpc = RpcServiceGrpc.newBlockingStub(decode).withDeadlineAfter(2, TimeUnit.SECONDS);
            var nativeInput = input.toBuilder().setRequestId(45);
            nativeInput.getGenerateConfigBuilder().setCanUsePdSeparation(true).addRoleAddrs(RoleAddrPB.newBuilder()
                .setRole(RoleAddrPB.RoleType.DECODE).setIp("127.0.0.1").setGrpcPort(Integer.parseInt(decode.authority().split(":")[1])));
            int count = 0;
            var generated = pRpc.generateStreamCall(nativeInput.build());
            while (generated.hasNext()) {
                var output = generated.next(); count++;
                if (output.getFlattenOutput().getAuxInfo(0).getOutputLen() != count || output.getFlattenOutput().getFinished(0) != (count == 4))
                    throw new IllegalStateException("Invalid automatic P/D output");
            }
            if (count != 4) throw new IllegalStateException("Automatic P/D did not produce four outputs");
            var batched = nativeInput.setRequestId(46).build();
            var ack = pRpc.enqueueBatch(EnqueueBatchRequestPB.newBuilder().setBatchId(7).setFetchAttachTimeoutMs(1000)
                .addDpSlots(EnqueueBatchDpSlotPB.newBuilder().addRequests(EnqueueBatchExternalInputPB.newBuilder().setInput(batched))).build());
            if (ack.getSuccessesCount() != 1 || ack.getErrorsCount() != 0) throw new IllegalStateException("Batch admission failed");
            int batchCount = 0;
            var fetched = pRpc.fetchResponse(FetchRequestPB.newBuilder().setRequestId(46).build());
            while (fetched.hasNext()) {
                var output = fetched.next(); batchCount++;
                if (output.getFlattenOutput().getAuxInfo(0).getOutputLen() != batchCount || output.getFlattenOutput().getFinished(0) != (batchCount == 4))
                    throw new IllegalStateException("Invalid fetched P/D output");
            }
            if (batchCount != 4) throw new IllegalStateException("Fetch did not produce four outputs");
            var cancelled = pRpc.cancel(CancelRequestPB.newBuilder().setRequestId(47).build());
            var fenced = pRpc.enqueueBatch(EnqueueBatchRequestPB.newBuilder().addDpSlots(EnqueueBatchDpSlotPB.newBuilder()
                .addRequests(EnqueueBatchExternalInputPB.newBuilder().setInput(nativeInput.setRequestId(47)))).build());
            if (cancelled.getStatus() != CancelStatusPB.CANCEL_STATUS_TOMBSTONED || fenced.getErrorsCount() != 1
                    || fenced.getErrors(0).getErrorInfo().getErrorCode() != 8429) throw new IllegalStateException("Cancel fence failed");
            long pAvailable = pRpc.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getAvailableKvCache();
            long dAvailable = dRpc.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getAvailableKvCache();
            if (pAvailable != 128 || dAvailable != 128) throw new IllegalStateException("Formal P/D leaked KV");
            return Map.of("batch_output_tokens", batchCount, "cancel_fenced", true, "pd_output_tokens", count,
                "prefill_available_after_generation", pAvailable, "decode_available_after_generation", dAvailable);
        } finally { p.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
    }

}
