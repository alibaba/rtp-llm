package org.flexlb.engine.grpc.dp;

import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import io.grpc.ManagedChannel;
import io.grpc.netty.NettyChannelBuilder;
import io.netty.buffer.PooledByteBufAllocator;
import io.netty.channel.ChannelOption;
import io.netty.channel.WriteBufferWaterMark;
import io.netty.channel.socket.nio.NioSocketChannel;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Master ↔ engine workers gRPC client for V1-α DP batching paths.
 *
 * <p>All three RPCs ({@code Enqueue} / {@code FetchResponse} / {@code Cancel}) live on
 * the existing {@code RpcService} defined in
 * {@code rtp_llm/cpp/model_rpc/proto/model_rpc_service.proto} (mirrored automatically
 * into this module via the {@code prepare-proto} maven antrun task as
 * {@code engine_rpc_service.proto}). This client only invokes the three new ones; the
 * legacy status RPCs continue to be served by {@link org.flexlb.engine.grpc.EngineGrpcClient}
 * with a different (Blocking) stub flavour.
 *
 * <p>Channels are pooled per {@code ip:port}. We use the async future stub so the caller
 * (DpBatchScheduler flushing a batch on the timer thread) is never blocked on network I/O.
 *
 * <p>Intentionally NOT extending {@link org.flexlb.engine.grpc.AbstractGrpcClient}: that
 * base class is parameterised by a single STUB type and tied to its WORKER_STATUS /
 * CACHE_STATUS service-type taxonomy. For DP batching we want an async future stub,
 * which doesn't fit cleanly. A self-contained pool is simpler than parameterising the
 * base.
 */
@Component
public class DpGrpcClient {

    /** Channel pool keyed by "ip:port". One channel hosts the future stub. */
    private final Map<String, ChannelEntry> channels = new ConcurrentHashMap<>();

    /** Bounded ack wait. Enqueue is fire-and-forget but we still want a deadline so
     *  that DpBatchScheduler can fail the entire batch's futures rather than hang. */
    private static final long DEFAULT_DEADLINE_MS = 500;

    private static final class ChannelEntry {
        final ManagedChannel channel;
        final RpcServiceGrpc.RpcServiceFutureStub stub;

        ChannelEntry(ManagedChannel channel) {
            this.channel = channel;
            this.stub = RpcServiceGrpc.newFutureStub(channel);
        }
    }

    private ChannelEntry getOrCreate(String ip, int port) {
        String key = ip + ":" + port;
        return channels.computeIfAbsent(key, k -> new ChannelEntry(buildChannel(ip, port)));
    }

    private ManagedChannel buildChannel(String ip, int port) {
        return NettyChannelBuilder.forAddress(ip, port)
                .channelType(NioSocketChannel.class)
                .withOption(ChannelOption.TCP_NODELAY, true)
                .withOption(ChannelOption.SO_KEEPALIVE, true)
                .withOption(ChannelOption.ALLOCATOR, PooledByteBufAllocator.DEFAULT)
                .withOption(ChannelOption.CONNECT_TIMEOUT_MILLIS, 200)
                .withOption(ChannelOption.WRITE_BUFFER_WATER_MARK,
                        new WriteBufferWaterMark(64 * 1024, 128 * 1024))
                .maxInboundMessageSize(8 * 1024 * 1024)
                .keepAliveTime(2, TimeUnit.SECONDS)
                .keepAliveTimeout(10, TimeUnit.SECONDS)
                .keepAliveWithoutCalls(true)
                .usePlaintext()
                .build();
    }

    // ============== BatchEnqueue (fire-and-forget) ==============

    public CompletableFuture<EngineRpcService.BatchEnqueueResponsePB> enqueue(
            String ip, int port, EngineRpcService.BatchEnqueueRequestPB request) {
        return enqueue(ip, port, request, DEFAULT_DEADLINE_MS);
    }

    public CompletableFuture<EngineRpcService.BatchEnqueueResponsePB> enqueue(
            String ip, int port, EngineRpcService.BatchEnqueueRequestPB request, long deadlineMs) {
        ChannelEntry entry = getOrCreate(ip, port);
        ListenableFuture<EngineRpcService.BatchEnqueueResponsePB> lf = entry.stub
                .withDeadlineAfter(deadlineMs, TimeUnit.MILLISECONDS)
                .batchEnqueue(request);
        return toCompletable(lf);
    }

    // ============== Cancel (Prefill / Decode share the same RPC; engine-side
    //                       implementations differ in what they release.) ==============

    public CompletableFuture<Void> cancelPrefill(String ip, int port, long requestId) {
        return cancel(ip, port, requestId);
    }

    public CompletableFuture<Void> cancelDecode(String ip, int port, long requestId) {
        return cancel(ip, port, requestId);
    }

    private CompletableFuture<Void> cancel(String ip, int port, long requestId) {
        ChannelEntry entry = getOrCreate(ip, port);
        ListenableFuture<EngineRpcService.EmptyPB> lf = entry.stub
                .withDeadlineAfter(DEFAULT_DEADLINE_MS, TimeUnit.MILLISECONDS)
                .cancel(EngineRpcService.CancelRequestPB.newBuilder().setRequestId(requestId).build());
        return toCompletable(lf).thenApply(e -> null);
    }

    // ============== helpers ==============

    private static <T> CompletableFuture<T> toCompletable(ListenableFuture<T> lf) {
        CompletableFuture<T> cf = new CompletableFuture<>();
        Futures.addCallback(lf, new FutureCallback<T>() {
            @Override
            public void onSuccess(T result) {
                cf.complete(result);
            }
            @Override
            public void onFailure(Throwable t) {
                cf.completeExceptionally(t);
            }
        }, com.google.common.util.concurrent.MoreExecutors.directExecutor());
        return cf;
    }

    @PreDestroy
    public void shutdown() {
        for (ChannelEntry e : channels.values()) {
            try {
                e.channel.shutdown().awaitTermination(2, TimeUnit.SECONDS);
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
        }
    }

    /** Test/observability: how many distinct workers we hold channels to. */
    public int channelCount() {
        return channels.size();
    }
}
