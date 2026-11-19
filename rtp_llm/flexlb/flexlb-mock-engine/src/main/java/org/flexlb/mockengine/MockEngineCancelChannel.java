package org.flexlb.mockengine;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.engine.grpc.RequestId;

import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Resolves mock Decode cancellation by gRPC port and returns accepted, seen-but-finished or
 * fenced-absent outcomes. Injected RPC faults leave engine state unchanged; successful cancellation
 * still requires worker terminal feedback before scheduler cleanup.
 */
public final class MockEngineCancelChannel implements EngineCancelChannel {

    private final Map<Integer, JavaMockEngineCluster.FastRpcService> services;

    public MockEngineCancelChannel(Map<Integer, JavaMockEngineCluster.FastRpcService> services) {
        this.services = services;
    }

    @Override
    public boolean isSupported(DecodeEndpoint endpoint) {
        return endpoint != null && services.containsKey(endpoint.getGrpcPort());
    }

    @Override
    public CompletableFuture<CancelAck> cancel(
            CancelTarget target, String requestId, long timeoutMs) {
        JavaMockEngineCluster.FastRpcService service =
                target == null ? null : services.get(target.decodeGrpcPort());
        if (service == null || !target.isRoutable()) {
            return CompletableFuture.completedFuture(CancelAck.UNSUPPORTED);
        }
        CompletableFuture<CancelAck> result = new CompletableFuture<>();
        try {
            var request =
                    org.flexlb.engine.grpc.EngineRpcService.CancelRequestPB.newBuilder()
                            .setRequestId(RequestId.toEngineRequestId(requestId))
                            .setPrefillAddress(target.prefillAddress())
                            .build();
            service.cancel(
                    request,
                    new io.grpc.stub.StreamObserver<>() {
                        @Override
                        public void onNext(
                                org.flexlb.engine.grpc.EngineRpcService.CancelResponsePB response) {
                            switch (response.getStatus()) {
                                case CANCEL_STATUS_ACCEPTED -> result.complete(CancelAck.ACCEPTED);
                                case CANCEL_STATUS_NOT_FOUND ->
                                        result.complete(CancelAck.NOT_FOUND);
                                case CANCEL_STATUS_TOMBSTONED ->
                                        result.complete(CancelAck.REQUEST_FENCED);
                                default ->
                                        result.completeExceptionally(
                                                new IllegalStateException(
                                                        "unexpected cancel ack status"));
                            }
                        }

                        @Override
                        public void onError(Throwable failure) {
                            result.completeExceptionally(failure);
                        }

                        @Override
                        public void onCompleted() {
                            if (!result.isDone()) {
                                result.completeExceptionally(
                                        new IllegalStateException("empty cancel response"));
                            }
                        }
                    });
        } catch (RuntimeException failure) {
            result.completeExceptionally(failure);
        }
        return result;
    }
}
