package org.flexlb.mock.engine.service;

import io.grpc.Context;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import java.net.URI;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.flexlb.engine.grpc.EngineRpcService.CancelRequestPB;
import org.flexlb.engine.grpc.EngineRpcService.CancelResponsePB;
import org.flexlb.engine.grpc.RpcServiceGrpc;

/** Routes Master's Decode-owned cancel to the original Prefill owner without scanning workers. */
final class DecodeCancelForwarder implements AutoCloseable {
    private final Map<String, ManagedChannel> channels = new HashMap<>();
    private final ExecutorService calls = Executors.newVirtualThreadPerTaskExecutor();
    private boolean closed;

    void cancel(CancelRequestPB request, StreamObserver<CancelResponsePB> response) {
        final String target;
        try {
            var address = URI.create("http://" + request.getPrefillAddress());
            if (request.getRequestId() <= 0 || address.getHost() == null || address.getPort() <= 0
                    || address.getPort() > 65535 || address.getRawUserInfo() != null
                    || address.getRawQuery() != null || address.getRawFragment() != null
                    || !address.getRawPath().isEmpty()) throw new IllegalArgumentException();
            target = request.getPrefillAddress();
        } catch (IllegalArgumentException error) {
            response.onError(Status.INVALID_ARGUMENT.withDescription("Cancel requires a valid prefill_address and positive request_id").asRuntimeException());
            return;
        }
        var context = Context.current();
        synchronized (this) {
            if (closed) { response.onError(Status.UNAVAILABLE.withDescription("Decode closed").asRuntimeException()); return; }
            var channel = channels.computeIfAbsent(target, value -> ManagedChannelBuilder.forTarget(value).usePlaintext().build());
            calls.execute(context.wrap(() -> {
                final CancelResponsePB result;
                try {
                    // Inherited RPC deadline/cancellation remains in force; cap unbounded callers too.
                    var stub = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(1, TimeUnit.SECONDS);
                    // A wrong route to another Decode must fail rather than forward recursively.
                    result = stub.cancel(request.toBuilder().clearPrefillAddress().build());
                } catch (RuntimeException error) { response.onError(error); return; }
                response.onNext(result);
                response.onCompleted();
            }));
        }
    }

    @Override public synchronized void close() {
        if (closed) return;
        closed = true;
        calls.shutdownNow();
        channels.values().forEach(ManagedChannel::shutdownNow);
        channels.clear();
    }
}
