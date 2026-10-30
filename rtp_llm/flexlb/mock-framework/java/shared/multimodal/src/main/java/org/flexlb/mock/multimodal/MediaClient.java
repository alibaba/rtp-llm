package org.flexlb.mock.multimodal;

import io.grpc.*;
import java.util.*;
import java.util.concurrent.TimeUnit;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.MultimodalRpcServiceGrpc;

/** Always TCP, including when V shares the caller's JVM. Channels belong to this caller. */
public final class MediaClient implements AutoCloseable {
    private final Map<String,ManagedChannel> channels=new HashMap<>();
    private boolean closed;
    private synchronized ManagedChannel channel(String address) {
        if(closed) throw Status.UNAVAILABLE.withDescription("Vision client closed").asRuntimeException();
        return channels.computeIfAbsent(address,a->ManagedChannelBuilder.forTarget(a).usePlaintext().build());
    }
    public MultimodalOutputPB fetch(GenerateInputPB input,long deadline) {
        long remaining=deadline==Long.MAX_VALUE?TimeUnit.MINUTES.toNanos(2):deadline-System.nanoTime();
        if(remaining<=0) throw Status.DEADLINE_EXCEEDED.withDescription("Request expired preparing vision input").asRuntimeException();
        return MultimodalRpcServiceGrpc.newBlockingStub(channel(MediaInput.address(input))).withDeadlineAfter(remaining,TimeUnit.NANOSECONDS)
            .remoteMultimodalEmbedding(MediaInput.media(input));
    }
    public GenerateInputPB routingInput(GenerateInputPB input,long deadline) {
        long remaining=deadline-System.nanoTime();
        if(remaining<=0) throw Status.DEADLINE_EXCEEDED.withDescription("Request expired acquiring vision hashes").asRuntimeException();
        var request=MultimodalHashRequestPB.newBuilder().addAllKeys(input.getMultimodalInputsList().stream().map(MediaInput::key).distinct().toList())
            .setInputs(MediaInput.media(input)).setTimeoutMs(Math.max(1,TimeUnit.NANOSECONDS.toMillis(remaining))).build();
        var metadata=MultimodalRpcServiceGrpc.newBlockingStub(channel(MediaInput.address(input))).withDeadlineAfter(remaining,TimeUnit.NANOSECONDS).getMultimodalHashes(request);
        return MediaInput.applyHashes(input,metadata);
    }
    public GenerateInputPB prepare(GenerateInputPB input,long deadline) {
        if(!MediaInput.present(input)) return input;
        return MediaInput.apply(input,fetch(input,deadline)).toBuilder().clearMultimodalInputs().clearMultimodalTokenLayout().clearV41Inputs().build();
    }
    @Override public void close() {
        List<ManagedChannel> owned;
        synchronized(this) { if(closed) return;closed=true;owned=List.copyOf(channels.values());channels.clear(); }
        owned.forEach(ManagedChannel::shutdownNow);
        for(var channel:owned) try { channel.awaitTermination(3,TimeUnit.SECONDS); } catch(InterruptedException error) { Thread.currentThread().interrupt();break; }
    }
}
