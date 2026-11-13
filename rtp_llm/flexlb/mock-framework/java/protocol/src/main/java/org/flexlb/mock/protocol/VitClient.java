package org.flexlb.mock.protocol;

import io.grpc.*;
import java.util.*;
import java.util.concurrent.TimeUnit;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.mock.protocol.grpc.MockVitServiceGrpc;
import org.flexlb.mock.protocol.grpc.MockEngineProtocol.ImageSeeds;

/**
 * Owns the TCP channels used by a caller to fetch ViT embedding metadata and routing hashes.
 */
public final class VitClient implements AutoCloseable {
    private final Map<String,ManagedChannel> channels=new HashMap<>();

    private boolean closed;

    private final int imageToken;

    public VitClient(int imageToken) { this.imageToken = imageToken; }

    public MultimodalProtocol.Prepared routingInput(GenerateInputPB input,long deadline) {
        long remaining=deadline-System.nanoTime();
        if(remaining<=0) throw Status.DEADLINE_EXCEEDED.withDescription("Request expired acquiring vision hashes").asRuntimeException();
        var request=MultimodalHashRequestPB.newBuilder().addAllKeys(input.getMultimodalInputsList().stream().map(MultimodalProtocol::key).distinct().toList())
            .setInputs(MultimodalProtocol.media(input)).setTimeoutMs(Math.max(1,TimeUnit.NANOSECONDS.toMillis(remaining))).build();
        checkActive();
        var metadata=MockVitServiceGrpc.newBlockingStub(channel(MultimodalProtocol.address(input))).withDeadlineAfter(remaining,TimeUnit.NANOSECONDS).getHashSeeds(request);
        checkActive();
        return MultimodalProtocol.applyHashes(input,metadata,imageToken);
    }

    public MultimodalProtocol.Prepared prepare(GenerateInputPB input,long deadline) {
        if(!MultimodalProtocol.present(input)) return MultimodalProtocol.text(input);
        return MultimodalProtocol.apply(input,fetch(input,deadline),imageToken);
    }

    @Override public void close() {
        List<ManagedChannel> owned;
        synchronized(this) { if(closed) return;closed=true;owned=List.copyOf(channels.values());channels.clear(); }
        owned.forEach(ManagedChannel::shutdownNow);
        for(var channel:owned) try { channel.awaitTermination(3,TimeUnit.SECONDS); } catch(InterruptedException error) { Thread.currentThread().interrupt();break; }
    }

    private ImageSeeds fetch(GenerateInputPB input,long deadline) {
        long remaining=deadline==Long.MAX_VALUE?Long.MAX_VALUE:deadline-System.nanoTime();
        if(remaining<=0) throw Status.DEADLINE_EXCEEDED.withDescription("Request expired preparing vision input").asRuntimeException();
        checkActive();
        var stub=MockVitServiceGrpc.newBlockingStub(channel(MultimodalProtocol.address(input)));
        if(deadline!=Long.MAX_VALUE) stub=stub.withDeadlineAfter(remaining,TimeUnit.NANOSECONDS);
        var result=stub.prepareImages(MultimodalProtocol.media(input));
        checkActive();
        return result;
    }

    private static void checkActive() {
        var context=Context.current();
        if(context.getDeadline()!=null && context.getDeadline().isExpired())
            throw Status.DEADLINE_EXCEEDED.withDescription("Vision preparation expired").asRuntimeException();
        if(context.isCancelled() || Thread.currentThread().isInterrupted())
            throw Status.CANCELLED.withDescription("Vision preparation cancelled").asRuntimeException();
    }

    private synchronized ManagedChannel channel(String address) {
        if(closed) throw Status.UNAVAILABLE.withDescription("Vision client closed").asRuntimeException();
        return channels.computeIfAbsent(address,a->ManagedChannelBuilder.forTarget(a).usePlaintext().build());
    }
}
