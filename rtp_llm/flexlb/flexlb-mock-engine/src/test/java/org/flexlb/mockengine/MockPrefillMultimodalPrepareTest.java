package org.flexlb.mockengine;

import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.MultimodalRpcServiceGrpc;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.Path;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.jupiter.api.Assertions.*;
import static org.flexlb.mockengine.MockEngineTestSupport.*;

@Timeout(15)
class MockPrefillMultimodalPrepareTest {
    @TempDir Path directory;
    private MockPerformanceModel model() throws Exception {
        var model = performanceModel(directory, "1", .1);
        model.setPrefillBatchPolicy(new MockPrefillBatchPolicy(4, 10000, 10000, 4, false, 0, 0, false));
        model.pdPolicy = new MockPdPolicy(2, 5000, 10000, 1, 5000, 10000, 128, 2000);
        return model;
    }
    private static EngineRpcService.GenerateInputPB input(long id, int port, int decodePort) {
        var input = MockV41ImageTest.input(MockV41ImageTest.image("a".repeat(64)), port).toBuilder().setRequestId(id);
        input.getGenerateConfigBuilder().addRoleAddrs(EngineRpcService.RoleAddrPB.newBuilder()
                .setRoleStr("DECODE").setGrpcPort(decodePort));
        return input.build();
    }
    private static void success(StreamObserver<EngineRpcService.MultimodalOutputPB> observer) throws Exception {
        var computed = MockV41Image.compute(MockV41ImageTest.image("a".repeat(64)), MockV41ImageTest.config(), 1 << 20);
        observer.onNext(EngineRpcService.MultimodalOutputPB.newBuilder().addSplitSize(5)
                .setMultimodalEmbedding(computed.embedding()).setMultimodalFeatureHash(computed.hashes()).build());
        observer.onCompleted();
    }
    @Test void blockedMultimodalSlotDoesNotSerializeItsReadySibling() throws Exception {
        var entered = new CountDownLatch(2); var release = new CountDownLatch(1);
        Server vit = ServerBuilder.forPort(0).addService(new MultimodalRpcServiceGrpc.MultimodalRpcServiceImplBase() {
            @Override public void remoteMultimodalEmbedding(EngineRpcService.MultimodalInputsPB input,
                    StreamObserver<EngineRpcService.MultimodalOutputPB> observer) {
                entered.countDown();
                try { release.await(); success(observer); } catch (Exception error) { observer.onError(error); }
            }
        }).build().start();
        try (var cluster = MockEngineTestCluster.create(model(), 62880, 1, 1)) {
            var request = batch(1, slot(0, input(1, vit.getPort(), cluster.decode(0).getGrpcPort()),
                    input(2, vit.getPort(), cluster.decode(0).getGrpcPort())));
            var response = CompletableFuture.supplyAsync(() -> enqueueAndFetch(cluster.prefill(0), request));
            try {
                assertTrue(entered.await(3, TimeUnit.SECONDS), "both V calls must enter before either completes");
                assertEquals(0, cluster.decode(0).getOccupiedKvTokens(), "D ALLOCATE must follow successful MM");
            } finally { release.countDown(); }
            var ack = response.get(5, TimeUnit.SECONDS);
            assertEquals(2, ack.getSuccessesCount(), ack.toString()); assertEquals(0, ack.getErrorsCount());
        } finally { release.countDown(); vit.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
    }
    @Test void retryableMultimodalFailureRetriesWithinSlotBeforeAllocate() throws Exception {
        var calls = new AtomicInteger();
        Server vit = ServerBuilder.forPort(0).addService(new MultimodalRpcServiceGrpc.MultimodalRpcServiceImplBase() {
            @Override public void remoteMultimodalEmbedding(EngineRpcService.MultimodalInputsPB input,
                    StreamObserver<EngineRpcService.MultimodalOutputPB> observer) {
                if (calls.incrementAndGet() == 1) { observer.onError(Status.UNAVAILABLE.asRuntimeException()); return; }
                try { success(observer); } catch (Exception error) { observer.onError(error); }
            }
        }).build().start();
        try (var cluster = MockEngineTestCluster.create(model(), 62880, 1, 1)) {
            var ack = enqueueAndFetch(cluster.prefill(0), batch(1, slot(0, input(1, vit.getPort(), cluster.decode(0).getGrpcPort()))));
            assertEquals(1, ack.getSuccessesCount(), ack.toString()); assertEquals(2, calls.get());
            assertFalse(cluster.prefill(0).isLeakDetected());
        } finally { vit.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
    }
    @Test void permanentMultimodalFailureDoesNotRetryOrAllocateDecode() throws Exception {
        var calls = new AtomicInteger();
        Server vit = ServerBuilder.forPort(0).addService(new MultimodalRpcServiceGrpc.MultimodalRpcServiceImplBase() {
            @Override public void remoteMultimodalEmbedding(EngineRpcService.MultimodalInputsPB input,
                    StreamObserver<EngineRpcService.MultimodalOutputPB> observer) {
                calls.incrementAndGet(); observer.onError(Status.INVALID_ARGUMENT.withDescription("bad image").asRuntimeException());
            }
        }).build().start();
        try (var cluster = MockEngineTestCluster.create(model(), 62880, 1, 1)) {
            var ack = enqueueAndFetch(cluster.prefill(0), batch(1, slot(0, input(1, vit.getPort(), cluster.decode(0).getGrpcPort()))));
            assertEquals(1, ack.getErrorsCount(), ack.toString()); assertEquals(0, ack.getSuccessesCount());
            assertEquals(1, calls.get()); assertEquals(0, cluster.decode(0).getOccupiedKvTokens());
            assertFalse(cluster.prefill(0).hasInflightWork());
        } finally { vit.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
    }
    @Test void multimodalRetryClassificationMatchesRealTransportAndDomainErrors() {
        for (int code = 901; code <= 907; code++) {
            assertEquals(code == 903 || code == 904 || code == 906 || code == 907,
                    MockVitClient.retryable(MockRpcErrors.error(Status.INTERNAL, code, "image failure")));
        }
        assertTrue(MockVitClient.retryable(Status.UNAVAILABLE.asRuntimeException()));
        assertTrue(MockVitClient.retryable(Status.DEADLINE_EXCEEDED.asRuntimeException()));
        for (var status : java.util.List.of(Status.INTERNAL, Status.UNKNOWN, Status.DATA_LOSS,
                Status.CANCELLED, Status.RESOURCE_EXHAUSTED, Status.INVALID_ARGUMENT))
            assertFalse(MockVitClient.retryable(status.asRuntimeException()), status.toString());
        assertTrue(MockVitClient.retryable(Status.INTERNAL.withDescription("[MM_DOWNLOAD_FAILED] unavailable URL").asRuntimeException()));
        assertFalse(MockVitClient.retryable(Status.INTERNAL.withDescription("[MM_WRONG_FORMAT_ERROR] bad image").asRuntimeException()));
    }

    @Test void cancelDuringMultimodalPreparationCannotStartAllocateAfterTheReply() throws Exception {
        var entered = new CountDownLatch(1); var release = new CountDownLatch(1);
        Server vit = ServerBuilder.forPort(0).addService(new MultimodalRpcServiceGrpc.MultimodalRpcServiceImplBase() {
            @Override public void remoteMultimodalEmbedding(EngineRpcService.MultimodalInputsPB input,
                    StreamObserver<EngineRpcService.MultimodalOutputPB> observer) {
                entered.countDown();
                try { release.await(); success(observer); } catch (Exception error) { observer.onError(error); }
            }
        }).build().start();
        try (var cluster = MockEngineTestCluster.create(model(), 62880, 1, 1)) {
            var response = CompletableFuture.supplyAsync(() -> enqueueAndFetch(cluster.prefill(0), batch(1,
                    slot(0, input(1, vit.getPort(), cluster.decode(0).getGrpcPort())))));
            try {
                assertTrue(entered.await(3, TimeUnit.SECONDS));
                cluster.prefill(0).cancelRequest(1);
            } finally { release.countDown(); }
            var ack = response.get(5, TimeUnit.SECONDS);
            assertEquals(0, ack.getSuccessesCount(), ack.toString());
            assertEquals(1, ack.getErrorsCount());
            assertEquals(8429, ack.getErrors(0).getErrorInfo().getErrorCode());
            assertEquals(0, cluster.decode(0).getOccupiedKvTokens());
            assertFalse(cluster.prefill(0).hasInflightWork());
        } finally { release.countDown(); vit.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
    }

}
