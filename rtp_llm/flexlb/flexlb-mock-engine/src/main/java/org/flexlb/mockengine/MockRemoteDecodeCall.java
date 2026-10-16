package org.flexlb.mockengine;

import io.grpc.Status;
import io.grpc.stub.ServerCallStreamObserver;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;

import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;

/** D-side protocol adapter. Cache admission and execution remain in FastRpcService. */
final class MockRemoteDecodeCall implements StreamObserver<EngineRpcService.GenerateRequestPB> {
    interface Backend {
        Lease allocate(EngineRpcService.GenerateInputPB input, String clientId, Runnable stopped);
    }

    interface Lease {
        boolean start(LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> outputs);
        void cancel();
        void completed();
    }

    private enum Stage { ALLOCATE, LOAD, GENERATE, RUNNING, CLOSED }

    private final Backend backend;
    private final StreamObserver<EngineRpcService.GenerateOutputsPB> response;
    private final Executor executor;
    private CompletableFuture<Void> outputTail = CompletableFuture.completedFuture(null);
    private Stage stage = Stage.ALLOCATE;
    private Lease lease;
    private long requestId;

    MockRemoteDecodeCall(Backend backend, StreamObserver<EngineRpcService.GenerateOutputsPB> response,
                         Executor executor) {
        this.backend = backend;
        this.response = response;
        this.executor = executor;
        if (response instanceof ServerCallStreamObserver<?> server) {
            server.setOnCancelHandler(this::cancel);
        }
    }

    @Override
    public synchronized void onNext(EngineRpcService.GenerateRequestPB request) {
        if (stage == Stage.CLOSED) {
            return;
        }
        try {
            if (stage == Stage.ALLOCATE && request.getStage() == EngineRpcService.RemoteStage.ALLOCATE) {
                if (!request.hasInput() || request.getInput().getRequestId() != request.getRequestId()
                        || request.getClientId().isBlank()) {
                    throw Status.INVALID_ARGUMENT.withDescription("Invalid ALLOCATE identity").asRuntimeException();
                }
                requestId = request.getRequestId();
                lease = backend.allocate(request.getInput(), request.getClientId(),
                        () -> executor.execute(() -> engineStopped()));
                if (lease == null) {
                    throw Status.RESOURCE_EXHAUSTED.withDescription("Decode ALLOCATE rejected")
                            .asRuntimeException();
                }
                stage = Stage.LOAD;
                response.onNext(EngineRpcService.GenerateOutputsPB.getDefaultInstance());
            } else if (request.getRequestId() != requestId) {
                throw Status.INVALID_ARGUMENT.withDescription("RemoteGenerate identity changed")
                        .asRuntimeException();
            } else if (stage == Stage.LOAD && request.getStage() == EngineRpcService.RemoteStage.LOAD) {
                // KV bytes are simulated. The resource lease and network stage
                // are real; no GPU/RDMA transfer performance is claimed here.
                stage = Stage.GENERATE;
                response.onNext(EngineRpcService.GenerateOutputsPB.getDefaultInstance());
            } else if (stage == Stage.GENERATE && request.getStage() == EngineRpcService.RemoteStage.GENERATE) {
                stage = Stage.RUNNING;
                if (!lease.start(new LinkedBlockingQueue<>() {
                    @Override
                    public synchronized boolean offer(EngineRpcService.GenerateOutputsPB output) {
                        // Decode can publish under its queue lock. Never take
                        // the RPC monitor there: cancellation takes the locks
                        // in the opposite direction. Preserve frame order on
                        // the shared executor without one thread per request.
                        outputTail = outputTail.thenRunAsync(() -> emit(output), executor);
                        return true;
                    }
                })) {
                    throw Status.UNAVAILABLE.withDescription("Decode stopped before GENERATE")
                            .asRuntimeException();
                }
            } else {
                throw Status.INVALID_ARGUMENT.withDescription("Expected " + stage + ", got " + request.getStage())
                        .asRuntimeException();
            }
        } catch (RuntimeException error) {
            abort(error);
        }
    }

    private synchronized boolean emit(EngineRpcService.GenerateOutputsPB output) {
        if (stage != Stage.RUNNING) {
            return false;
        }
        if (output.getRequestId() != requestId) {
            abort(Status.DATA_LOSS.withDescription("Decode output identity mismatch").asRuntimeException());
            return false;
        }
        boolean terminal = output.hasErrorInfo()
                || output.getFlattenOutput().getFinishedList().contains(true);
        if (terminal) {
            stage = Stage.CLOSED;
            lease.completed();
        }
        response.onNext(output);
        if (terminal) {
            response.onCompleted();
        }
        return true;
    }

    private void abort(Throwable error) {
        if (stage != Stage.CLOSED) {
            cancel();
            response.onError(error);
        }
    }

    private synchronized void engineStopped() {
        abort(Status.CANCELLED.withDescription("Decode lease cancelled or engine stopped").asRuntimeException());
    }

    private synchronized void cancel() {
        if (stage != Stage.CLOSED) {
            stage = Stage.CLOSED;
            if (lease != null) {
                lease.cancel();
            }
        }
    }

    @Override
    public synchronized void onError(Throwable error) {
        cancel();
    }

    @Override
    public synchronized void onCompleted() {
        // A premature half-close is not successful completion of inference.
        abort(Status.CANCELLED.withDescription("Prefill closed RemoteGenerate before completion")
                .asRuntimeException());
    }
}
