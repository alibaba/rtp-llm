package org.flexlb.mockengine;

import io.grpc.Channel;
import io.grpc.Status;
import io.grpc.stub.ClientCallStreamObserver;
import io.grpc.stub.ClientResponseObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RpcServiceGrpc;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

/**
 * One P-to-D ownership lease on the production RemoteGenerate stream.
 * The caller supplies the master-selected channel; this class never discovers
 * or substitutes a decode engine. Closing the stream cancels its D-side lease.
 */
final class MockRemoteDecodeStream implements AutoCloseable {
    private enum Stage { ALLOCATE, ALLOCATED, LOAD, LOADED, GENERATE, CLOSED }

    private final long requestId;
    private final Consumer<EngineRpcService.GenerateOutputsPB> output;
    private final Consumer<Throwable> failure;
    private final Runnable completed;
    private final AtomicBoolean terminal = new AtomicBoolean();
    private final CompletableFuture<Void> allocated = new CompletableFuture<>();
    private final CompletableFuture<Void> loaded = new CompletableFuture<>();
    private ClientCallStreamObserver<EngineRpcService.GenerateRequestPB> request;
    private Stage stage = Stage.ALLOCATE;

    MockRemoteDecodeStream(Channel channel, EngineRpcService.GenerateInputPB input,
                           String clientId, long deadlineMs,
                           Consumer<EngineRpcService.GenerateOutputsPB> output,
                           Consumer<Throwable> failure, Runnable completed) {
        if (deadlineMs <= 0) {
            throw new IllegalArgumentException("RemoteGenerate deadline must be positive");
        }
        this.requestId = input.getRequestId();
        this.output = output;
        this.failure = failure;
        this.completed = completed;
        // Enqueue only acknowledges admission; its unary RPC context ends
        // before Fetch attaches. The P lease, not that admission RPC, owns D.
        io.grpc.Context leaseContext = io.grpc.Context.current().fork();
        io.grpc.Context previous = leaseContext.attach();
        try {
        RpcServiceGrpc.newStub(channel).withDeadlineAfter(deadlineMs, TimeUnit.MILLISECONDS)
                .remoteGenerate(new ClientResponseObserver<EngineRpcService.GenerateRequestPB,
                        EngineRpcService.GenerateOutputsPB>() {
                    @Override
                    public void beforeStart(ClientCallStreamObserver<EngineRpcService.GenerateRequestPB> stream) {
                        request = stream;
                    }

                    @Override
                    public void onNext(EngineRpcService.GenerateOutputsPB value) {
                        receive(value);
                    }

                    @Override
                    public void onError(Throwable error) {
                        fail(error);
                    }

                    @Override
                    public void onCompleted() {
                        if (!terminal.get()) {
                            fail(Status.DATA_LOSS.withDescription(
                                    "RemoteGenerate closed before a terminal output").asRuntimeException());
                        }
                    }
                });
        request.onNext(EngineRpcService.GenerateRequestPB.newBuilder()
                .setStage(EngineRpcService.RemoteStage.ALLOCATE)
                .setRequestId(requestId).setClientId(clientId).setInput(input).build());
        } finally {
            leaseContext.detach(previous);
        }
    }

    CompletableFuture<Void> allocated() {
        return allocated;
    }

    synchronized CompletableFuture<Void> load() {
        if (stage != Stage.ALLOCATED) {
            return CompletableFuture.failedFuture(new IllegalStateException("LOAD before ALLOCATE ack"));
        }
        stage = Stage.LOAD;
        request.onNext(EngineRpcService.GenerateRequestPB.newBuilder()
                .setRequestId(requestId).setStage(EngineRpcService.RemoteStage.LOAD).build());
        return loaded;
    }

    synchronized void generate(int firstToken) {
        if (stage != Stage.LOADED) {
            throw new IllegalStateException("GENERATE before LOAD ack");
        }
        stage = Stage.GENERATE;
        request.onNext(EngineRpcService.GenerateRequestPB.newBuilder()
                .setRequestId(requestId).setStage(EngineRpcService.RemoteStage.GENERATE)
                .setFirstGenerateTokenId(firstToken).build());
        // Keep the sending side open: its cancellation is the ownership signal
        // while D runs, just as the real P holds its RemoteGenerate context.
    }

    private synchronized void receive(EngineRpcService.GenerateOutputsPB value) {
        if (terminal.get()) {
            return;
        }
        boolean errorOutput = value.hasErrorInfo() && value.getErrorInfo().getErrorCodeValue() != 0;
        if (errorOutput && stage != Stage.GENERATE) {
            fail(Status.INTERNAL.withDescription(value.getErrorInfo().toString()).asRuntimeException());
            return;
        }
        if (stage == Stage.ALLOCATE) {
            stage = Stage.ALLOCATED;
            allocated.complete(null);
        } else if (stage == Stage.LOAD) {
            stage = Stage.LOADED;
            loaded.complete(null);
        } else if (stage == Stage.GENERATE) {
            if (value.getRequestId() != requestId) {
                fail(Status.DATA_LOSS.withDescription("RemoteGenerate request identity mismatch")
                        .asRuntimeException());
                return;
            }
            output.accept(value);
            if ((errorOutput || value.getFlattenOutput().getFinishedList().stream().anyMatch(Boolean::booleanValue))
                    && terminal.compareAndSet(false, true)) {
                stage = Stage.CLOSED;
                request.onCompleted();
                completed.run();
            }
        } else {
            fail(Status.DATA_LOSS.withDescription("Unexpected RemoteGenerate acknowledgement")
                    .asRuntimeException());
        }
    }

    private synchronized void fail(Throwable error) {
        if (!terminal.compareAndSet(false, true)) {
            return;
        }
        stage = Stage.CLOSED;
        allocated.completeExceptionally(error);
        loaded.completeExceptionally(error);
        if (request != null) {
            request.cancel("P-to-D stream failed", error);
        }
        failure.accept(error);
    }

    @Override
    public synchronized void close() {
        if (terminal.compareAndSet(false, true)) {
            stage = Stage.CLOSED;
            Throwable error = Status.CANCELLED.withDescription("Prefill context closed").asRuntimeException();
            allocated.completeExceptionally(error);
            loaded.completeExceptionally(error);
            request.cancel("Prefill context closed", error);
        }
    }
}
