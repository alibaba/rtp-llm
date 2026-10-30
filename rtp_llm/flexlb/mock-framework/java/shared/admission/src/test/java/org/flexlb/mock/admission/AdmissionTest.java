package org.flexlb.mock.admission;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.*;
import io.grpc.stub.StreamObserver;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.mock.api.model.Types.Role;
import org.flexlb.mock.cache.KvCache;
import org.flexlb.mock.compute.ComputeModel;
import org.flexlb.mock.compute.SimulatedCompute;
import org.flexlb.mock.engine.*;
import org.junit.jupiter.api.Test;

/** Compiles and runs without any dependency on an Engine backend. */
class AdmissionTest {
    /** Independent test policy: reject at admission so neither adapter can silently construct RTP's policy. */
    private static final class RejectingScheduler implements Scheduler {
        final Object lock;
        boolean submitted, closed;
        RejectingScheduler(Object lock) { this.lock = lock; }
        private void locked() { assertTrue(Thread.holdsLock(lock)); }
        @Override public Request input(Input input, Listener listener) {
            locked(); submitted = true;
            throw new RequestFailure(RequestFailure.Reason.OVERLOADED, "injected scheduler admission");
        }
        @Override public long nextDeadline() { locked(); return Long.MAX_VALUE; }
        @Override public Status status(long latest) { locked(); return new Status(List.of(), List.of(), latest, 0, 0); }
        @Override public void transferCompleted(Request request, TransferResult result) { fail("No admitted transfer"); }
        @Override public void maintain() { locked(); }
        @Override public Work schedule() { locked(); return null; }
        @Override public List<Request> requests() { locked(); return List.of(); }
        @Override public QueueSnapshot queues() { locked(); return new QueueSnapshot(0, 0, 0, 0); }
        @Override public String scrapeMetrics() { locked(); return "injected_scheduler 1\n"; }
        @Override public List<Runnable> takeNotifications() { locked(); return List.of(); }
        @Override public void close() { locked(); closed = true; }
        public List<InputResult> input(List<InputItem> batch) { throw new UnsupportedOperationException(); }
        @Override public void control(Request request, Control event) { fail("No admitted request"); }
        @Override public void finish(Request request, RequestFailure failure) { fail("No admitted request"); }
        @Override public RequestState.Use beginTransfer(Request request) { throw new AssertionError("No admitted request"); }
        @Override public void computeCompleted(Work work, SimulatedCompute.Outcome outcome) { fail("No admitted work"); }
        @Override public void failed(Work work, RuntimeException error) { fail("No admitted work"); }
        @Override public boolean cancelled(Work work) { throw new AssertionError("No admitted work"); }
    }


    @Test void bothRolesUseTheSuppliedSchedulerWithoutABackend() throws Exception {
        var compute = ComputeModel.fromJson(new ObjectMapper().readTree("{\"forward\":{\"expression\":\"1\"}}"));
        for (var role : List.of(Role.prefill, Role.decode)) {
            var created = new AtomicReference<RejectingScheduler>();
            try (var cache = new KvCache(2, 8, 0, KvCache.EvictionPolicy.LRU);
                 var service = new RpcService(new AdmissionConfig("test", role, 2, 1, 4096, 0, compute), cache,
                     (lock, supplied) -> {
                         assertSame(cache, supplied);
                         assertNull(created.get(), "One scheduler per instance");
                         var scheduler = new RejectingScheduler(lock); created.set(scheduler); return scheduler;
                     })) {
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
}
