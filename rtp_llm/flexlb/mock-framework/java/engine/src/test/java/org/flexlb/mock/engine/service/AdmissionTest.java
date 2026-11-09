package org.flexlb.mock.engine.service;

import org.flexlb.mock.engine.scheduler.TaskHistory;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.*;
import io.grpc.stub.StreamObserver;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.mock.protocol.Role;
import org.flexlb.mock.engine.*;
import org.flexlb.mock.engine.cache.KvCacheManager;
import org.flexlb.mock.engine.forward.ComputeModel;
import org.flexlb.mock.engine.forward.SimulatedCompute;
import org.flexlb.mock.engine.scheduler.EngineScheduler.QueueSnapshot;
import org.flexlb.mock.engine.scheduler.EngineScheduler.RequestFailure;
import org.flexlb.mock.engine.scheduler.RequestState;
import org.flexlb.mock.engine.scheduler.EngineScheduler;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Compiles and runs without any dependency on an Engine backend.
 */
class AdmissionTest {
    @Test void bothRolesUseTheSuppliedSchedulerWithoutABackend() throws Exception {
        var compute = ComputeModel.fromJson(new ObjectMapper().readTree("{\"forward\":{\"expression\":\"1\"}}"));
        for (var role : List.of(Role.prefill, Role.decode)) {
            var created = new AtomicReference<RejectingScheduler>();
            var lock = new Object();
            var history = new TaskHistory();
            try (var cache = new KvCacheManager(2, 8, 0);
                 var engine = new org.flexlb.mock.engine.execution.EngineLoop(lock, (domain, supplied, finished) -> {
                     assertSame(lock, domain);
                     assertSame(cache, supplied);
                     assertNull(created.get(), "One scheduler per instance");
                     var scheduler = new RejectingScheduler(domain); created.set(scheduler); return scheduler;
                 }, history, cache, compute, role, "test");
                 var service = new RtpRpcService(new RtpRpcService.Config("test", role, 2, 1, 1, 4096, 0, compute, -200, 0L),
                     cache, lock, history, engine, true)) {
                var server = io.grpc.netty.NettyServerBuilder.forPort(0).addService(service).build().start();
                var channel = ManagedChannelBuilder.forAddress("127.0.0.1", server.getPort()).usePlaintext().build();
                try {
                    var input = GenerateInputPB.newBuilder().setRequestId(1).addAllTokenIds(List.of(1, 2, 3))
                        .setGenerateConfig(GenerateConfigPB.newBuilder().setMaxNewTokens(role == Role.prefill ? 1 : 3)).build();
                    Throwable error;
                    if (role == Role.prefill) {
                        error = assertThrows(StatusRuntimeException.class, () -> RpcServiceGrpc.newBlockingStub(channel)
                            .withDeadlineAfter(2, TimeUnit.SECONDS).generateStreamCall(input).hasNext());
                    } else {
                        var ended = new CompletableFuture<Throwable>();
                        var stream = RpcServiceGrpc.newStub(channel).withDeadlineAfter(2, TimeUnit.SECONDS)
                            .remoteGenerate(new StreamObserver<GenerateOutputsPB>() {
                                public void onNext(GenerateOutputsPB output) { ended.completeExceptionally(new AssertionError("Unexpected admission")); }
                                public void onError(Throwable failure) { ended.complete(failure); }
                                public void onCompleted() { ended.completeExceptionally(new AssertionError("Unexpected completion")); }
                            });
                        stream.onNext(GenerateRequestPB.newBuilder().setStage(RemoteStage.ALLOCATE)
                            .setClientId("client").setRequestId(1).setInput(input).build());
                        error = ended.get(3, TimeUnit.SECONDS);
                    }
                    assertEquals(Status.Code.RESOURCE_EXHAUSTED, Status.fromThrowable(error).getCode());
                    assertTrue(error.getMessage().contains("injected scheduler admission"));
                    assertTrue(created.get().submitted);
                    service.close();
                    assertTrue(created.get().closed);
                } finally {
                    channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
                    server.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
                }
            }
        }
    }

    /**
     * Independent test policy: reject at admission so neither adapter can silently construct RTP's policy.
     */
    private static final class RejectingScheduler implements EngineScheduler {
        final Object lock;

        boolean submitted, closed;

        RejectingScheduler(Object lock) { this.lock = lock; }

        @Override public Request enqueue(Input input, Listener listener) {
            locked(); submitted = true;
            throw new RequestFailure(RequestFailure.Reason.OVERLOADED, "injected scheduler admission");
        }

        @Override public long nextDeadline() { locked(); return Long.MAX_VALUE; }

        @Override public List<Task> tasks() { locked(); return List.of(); }

        @Override public void transferCompleted(Request request, TransferResult result) { fail("No admitted transfer"); }

        @Override public void prepareKvAndCheckTimeouts() { locked(); }

        @Override public Work schedule() { locked(); return null; }

        @Override public List<Request> requests() { locked(); return List.of(); }

        @Override public QueueSnapshot queues() { locked(); return new QueueSnapshot(0, 0, 0, 0); }

        @Override public List<Runnable> takeNotifications() { locked(); return List.of(); }

        @Override public void close() { locked(); closed = true; }

        public List<InputResult> enqueue(List<InputItem> batch) { throw new UnsupportedOperationException(); }

        @Override public void control(Request request, Control event) { fail("No admitted request"); }

        @Override public void abort(Request request, RequestFailure failure) { fail("No admitted request"); }

        @Override public RequestState.Use beginTransfer(Request request) { throw new AssertionError("No admitted request"); }

        @Override public void processOutputs(Work work, SimulatedCompute.Outcome outcome) { fail("No admitted work"); }

        @Override public void failForward(Work work, RuntimeException error) { fail("No admitted work"); }

        @Override public boolean isForwardCancelled(Work work) { throw new AssertionError("No admitted work"); }

        private void locked() { assertTrue(Thread.holdsLock(lock)); }
    }
}
