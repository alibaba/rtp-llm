package org.flexlb.mock.protocol;

import io.grpc.*;
import java.util.*;
import java.util.concurrent.TimeUnit;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.MultimodalRpcServiceGrpc;
import org.flexlb.mock.protocol.grpc.MockVitServiceGrpc;
import org.flexlb.mock.protocol.grpc.MockEngineProtocol.ImageSeeds;

/**
 * Owns the TCP channels used by a caller to fetch ViT embedding metadata and routing hashes.
 */
public final class VitClient implements AutoCloseable {
    // Matches Real ViT's maximum encoded hash metadata response.
    private static final int MAX_NATIVE_HASH_RESPONSE_BYTES = 16 * 1024 * 1024;

    private final int imageToken;

    private final Map<String, VitChannel> channels = new HashMap<>();

    private boolean closed;

    public VitClient(int imageToken) { this.imageToken = imageToken; }

    public MultimodalProtocol.Prepared routingInput(GenerateInputPB input,long deadline) {
        long remaining=deadline-System.nanoTime();
        if(remaining<=0) throw Status.DEADLINE_EXCEEDED.withDescription("Request expired acquiring vision hashes").asRuntimeException();
        var request = hashRequest(input, remaining);
        checkActive();
        var metadata=MockVitServiceGrpc.newBlockingStub(channel(MultimodalProtocol.address(input)).channel).withDeadlineAfter(remaining,TimeUnit.NANOSECONDS).getHashSeeds(request);
        checkActive();
        return MultimodalProtocol.applyHashes(input,metadata,imageToken);
    }

    public MultimodalProtocol.Prepared prepare(GenerateInputPB input,long deadline) {
        if(!MultimodalProtocol.present(input)) return MultimodalProtocol.text(input);
        var target = channel(MultimodalProtocol.address(input));
        if (!target.nativePrefill) {
            try {
                var seeds = fetch(input, deadline, target.channel);
                return MultimodalProtocol.apply(input, seeds, imageToken);
            } catch (StatusRuntimeException error) {
                if (error.getStatus().getCode() != Status.Code.UNIMPLEMENTED) throw error;
                target.nativePrefill = true;
            }
        }
        return fetchNativeHashes(input, deadline, target.channel);
    }

    @Override public void close() {
        List<ManagedChannel> owned;
        synchronized(this) {
            if (closed) return;
            closed = true;
            owned = channels.values().stream().map(target -> target.channel).toList();
            channels.clear();
        }
        owned.forEach(ManagedChannel::shutdownNow);
        for(var channel:owned) try { channel.awaitTermination(3,TimeUnit.SECONDS); } catch(InterruptedException error) { Thread.currentThread().interrupt();break; }
    }

    private MultimodalProtocol.Prepared fetchNativeHashes(GenerateInputPB input, long deadline, ManagedChannel channel) {
        long remaining = deadline == Long.MAX_VALUE ? Long.MAX_VALUE : deadline - System.nanoTime();
        if (remaining <= 0) {
            throw Status.DEADLINE_EXCEEDED.withDescription("Request expired acquiring vision hashes").asRuntimeException();
        }
        checkActive();
        var stub = MultimodalRpcServiceGrpc.newBlockingStub(channel);
        if (deadline != Long.MAX_VALUE) stub = stub.withDeadlineAfter(remaining, TimeUnit.NANOSECONDS);
        var metadata = stub.getMultimodalHashes(hashRequest(input, remaining));
        checkActive();
        return MultimodalProtocol.applyHashes(input, metadata, imageToken);
    }

    private static void checkActive() {
        var context=Context.current();
        if(context.getDeadline()!=null && context.getDeadline().isExpired())
            throw Status.DEADLINE_EXCEEDED.withDescription("Vision preparation expired").asRuntimeException();
        if(context.isCancelled() || Thread.currentThread().isInterrupted())
            throw Status.CANCELLED.withDescription("Vision preparation cancelled").asRuntimeException();
    }

    private static MultimodalHashRequestPB hashRequest(GenerateInputPB input, long remaining) {
        var request = MultimodalHashRequestPB.newBuilder()
            .addAllKeys(input.getMultimodalInputsList().stream().map(MultimodalProtocol::key).distinct().toList())
            .setInputs(MultimodalProtocol.media(input));
        if (remaining != Long.MAX_VALUE) {
            request.setTimeoutMs(Math.max(1, TimeUnit.NANOSECONDS.toMillis(remaining)));
        }
        return request.build();
    }

    private ImageSeeds fetch(GenerateInputPB input, long deadline, ManagedChannel channel) {
        long remaining=deadline==Long.MAX_VALUE?Long.MAX_VALUE:deadline-System.nanoTime();
        if(remaining<=0) throw Status.DEADLINE_EXCEEDED.withDescription("Request expired preparing vision input").asRuntimeException();
        checkActive();
        var stub=MockVitServiceGrpc.newBlockingStub(channel);
        if(deadline!=Long.MAX_VALUE) stub=stub.withDeadlineAfter(remaining,TimeUnit.NANOSECONDS);
        var result=stub.prepareImages(MultimodalProtocol.media(input));
        checkActive();
        return result;
    }

    private synchronized VitChannel channel(String address) {
        if(closed) throw Status.UNAVAILABLE.withDescription("Vision client closed").asRuntimeException();
        return channels.computeIfAbsent(address, target -> new VitChannel(ManagedChannelBuilder.forTarget(target)
            .usePlaintext().maxInboundMessageSize(MAX_NATIVE_HASH_RESPONSE_BYTES).build()));
    }

    /**
     * Remembers when the endpoint only offers native Prefill metadata.
     */
    private static final class VitChannel {
        private final ManagedChannel channel;
        private volatile boolean nativePrefill;

        private VitChannel(ManagedChannel channel) { this.channel = channel; }
    }
}
