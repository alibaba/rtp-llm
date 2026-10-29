package org.flexlb.mock.engine.rtp;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.*;
import io.grpc.stub.StreamObserver;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.mock.cache.DeviceCache;
import org.flexlb.mock.compute.SimulatedCompute;
import org.flexlb.mock.engine.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class SchedulerInjectionTest {
    @TempDir Path directory;

    private RtpConfig config(String role) throws Exception {
        var path = directory.resolve(role + ".json");
        var json = new ObjectMapper();
        var document = json.valueToTree(RtpNodeTest.configuration(role));
        ((com.fasterxml.jackson.databind.node.ObjectNode) document.path("component").path("config"))
            .put("max_batch_tokens_size", 1);
        json.writeValue(path.toFile(), document);
        return RtpConfig.load(path);
    }
    private static GenerateInputPB input(int output) {
        return GenerateInputPB.newBuilder().setRequestId(1).addAllTokenIds(List.of(1, 2, 3))
            .setGenerateConfig(GenerateConfigPB.newBuilder().setMaxNewTokens(output)).build();
    }

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
        @Override public void control(Request request, Control event) { fail("No admitted request"); }
        @Override public void finish(Request request, RequestFailure failure) { fail("No admitted request"); }
        @Override public RequestState.Use beginTransfer(Request request) { throw new AssertionError("No admitted request"); }
        @Override public void computeCompleted(Work work, SimulatedCompute.Outcome outcome) { fail("No admitted work"); }
        @Override public void failed(Work work, RuntimeException error) { fail("No admitted work"); }
        @Override public boolean cancelled(Work work) { throw new AssertionError("No admitted work"); }
    }

    @Test void bothNativeAdaptersUseTheInjectedSchedulerAndKeepTheirOwnLock() throws Exception {
        for (String role : List.of("prefill", "decode")) {
            var created = new AtomicReference<RejectingScheduler>();
            var resource = new AtomicReference<DeviceCache>();
            try (var node = RtpNode.start(config(role), (lock, cache) -> {
                assertNull(created.get(), "One scheduler per Engine instance");
                resource.set(cache);
                var scheduler = new RejectingScheduler(lock); created.set(scheduler); return scheduler;
            })) {
                assertSame(node.cache(), resource.get());
                node.checkHealthy();
                var channel = ManagedChannelBuilder.forTarget(node.rpcAddress()).usePlaintext().build();
                try {
                    if (role.equals("prefill")) {
                        var rpc = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(2, TimeUnit.SECONDS);
                        var error = assertThrows(StatusRuntimeException.class,
                            () -> rpc.generateStreamCall(input(1)).hasNext());
                        assertEquals(Status.Code.RESOURCE_EXHAUSTED, error.getStatus().getCode());
                        assertTrue(error.getMessage().contains("injected scheduler admission"));
                    } else {
                        var ended = new CompletableFuture<Throwable>();
                        var remote = RpcServiceGrpc.newStub(channel).withDeadlineAfter(2, TimeUnit.SECONDS)
                            .remoteGenerate(new StreamObserver<GenerateOutputsPB>() {
                                @Override public void onNext(GenerateOutputsPB output) { ended.completeExceptionally(new AssertionError("Unexpected admission")); }
                                @Override public void onError(Throwable error) { ended.complete(error); }
                                @Override public void onCompleted() { ended.completeExceptionally(new AssertionError("Unexpected completion")); }
                            });
                        remote.onNext(GenerateRequestPB.newBuilder().setStage(RemoteStage.ALLOCATE).setRequestId(1)
                            .setClientId("client").setInput(input(3)).build());
                        var error = ended.get(3, TimeUnit.SECONDS);
                        assertEquals(Status.Code.RESOURCE_EXHAUSTED, Status.fromThrowable(error).getCode());
                        assertTrue(error.getMessage().contains("injected scheduler admission"));
                    }
                    assertTrue(created.get().submitted);
                } finally { channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
            }
            assertTrue(created.get().closed);
            assertThrows(IllegalStateException.class, () -> resource.get().acquire(1, List.of()));
        }
    }


    @Test void defaultRtpPolicyStillRejectsOversizedPrefillWithNativeInputError() throws Exception {
        try (var node = RtpNode.start(config("prefill"))) {
            var channel = ManagedChannelBuilder.forTarget(node.rpcAddress()).usePlaintext().build();
            try {
                var rpc = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(2, TimeUnit.SECONDS);
                var error = assertThrows(StatusRuntimeException.class, () -> rpc.generateStreamCall(input(1)).hasNext());
                assertEquals(Status.Code.INVALID_ARGUMENT, error.getStatus().getCode());
                var metadata = error.getTrailers().get(Metadata.Key.of("grpc-status-details-bin", Metadata.BINARY_BYTE_MARSHALLER));
                assertEquals(605, ErrorDetailsPB.parseFrom(metadata).getErrorCode());
                assertEquals(128, rpc.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getAvailableKvCache());
            } finally { channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
        }
    }

    @Test void factoryFailureOrNullPolicyClosesTheNewInstancesCache() throws Exception {
        for (boolean returnsNull : List.of(false, true)) {
            var resource = new AtomicReference<DeviceCache>();
            var config = config("prefill");
            var error = assertThrows(RuntimeException.class,
                () -> RtpNode.start(config, (lock, cache) -> {
                    resource.set(cache);
                    if (returnsNull) return null;
                    throw new IllegalStateException("Cannot create scheduler");
                }));
            assertEquals(returnsNull ? NullPointerException.class : IllegalStateException.class, error.getClass());
            assertNotNull(resource.get());
            assertThrows(IllegalStateException.class, () -> resource.get().acquire(1, List.of()));
        }
    }
}
