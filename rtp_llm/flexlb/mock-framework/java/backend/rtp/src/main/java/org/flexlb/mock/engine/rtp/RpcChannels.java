package org.flexlb.mock.engine.rtp;

import io.grpc.*;
import java.util.*;
import java.util.concurrent.TimeUnit;

/** Node-owned connections; request cancellation never shuts down another request's channel. */
final class RpcChannels implements AutoCloseable {
    private final Map<String, ManagedChannel> channels = new HashMap<>();
    private boolean closed;
    synchronized ManagedChannel get(String address) {
        if (closed) throw Status.UNAVAILABLE.withDescription("RPC connections closed").asRuntimeException();
        return channels.computeIfAbsent(address, key -> ManagedChannelBuilder.forTarget(key).usePlaintext().build());
    }
    @Override public void close() {
        List<ManagedChannel> owned;
        synchronized (this) { closed = true; owned = List.copyOf(channels.values()); channels.clear(); }
        owned.forEach(ManagedChannel::shutdownNow);
        for (var channel : owned) try { channel.awaitTermination(3, TimeUnit.SECONDS); }
        catch (InterruptedException error) { Thread.currentThread().interrupt(); return; }
    }
}
