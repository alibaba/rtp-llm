package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.*;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.engine.grpc.*;
import org.junit.jupiter.api.Test;
import java.util.concurrent.*;
import java.util.function.BooleanSupplier;
import static org.junit.jupiter.api.Assertions.*;

class MockVitTransportTest {
    static final ObjectMapper JSON = new ObjectMapper();
    static final class Worker implements AutoCloseable {
        final JavaMockEngineCluster.Config config = new JavaMockEngineCluster.Config();
        final NioEventLoopGroup boss = new NioEventLoopGroup(1), workers = new NioEventLoopGroup(1);
        final MockVitEngine vit;
        Server tcp;
        final ManagedChannel channel;
        Worker() throws Exception {
            config.engineConfig = MockEngineConfigFixture.load();
            config.rpcMaxCalls = 1;
            var model = MockPerformanceModel.loadCalibration(JSON.readTree("{\"mock\":{\"forward\":{\"vit\":{\"fixed_ms\":0}}}}"));
            vit = new MockVitEngine(config, model, "vit-0", "127.0.0.1", 0, boss, workers);
            tcp = vit.start();
            channel = InProcessChannelBuilder.forName(vit.localName()).directExecutor().build();
        }
        MultimodalRpcServiceGrpc.MultimodalRpcServiceBlockingStub stub() {
            return MultimodalRpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3, TimeUnit.SECONDS);
        }
        int inFlight() {
            return ((Number)((java.util.Map<?,?>)vit.admissionSnapshot().get("rpc_runtime")).get("in_flight")).intValue();
        }
        public void close() throws Exception {
            channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            tcp.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            vit.shutdown(); config.rpcExecutors(workers).close();
            boss.shutdownGracefully(0, 1, TimeUnit.SECONDS).sync();
            workers.shutdownGracefully(0, 1, TimeUnit.SECONDS).sync();
        }
    }
    static void await(BooleanSupplier condition) throws Exception {
        long end = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
        while (!condition.getAsBoolean() && System.nanoTime() < end) Thread.sleep(2);
        assertTrue(condition.getAsBoolean());
    }
    @Test void internalCallWorksWithoutTcpAndRetainsAdmissionAndLogicalAccounting() throws Exception {
        try (var worker = new Worker()) {
            worker.tcp.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            var image = MockVitServiceTest.image(224, 224, 0x123456);
            var request = MockVitServiceTest.inputs(image);
            var result = worker.stub().remoteMultimodalEmbedding(request);
            assertTrue(result.getMultimodalEmbedding().getFp32Data().isEmpty());
            assertEquals(java.util.List.of(64L, 64L), result.getMultimodalEmbedding().getShapeList());
            assertTrue(result.getSerializedSize() < 512);
            assertEquals(64 * 64 * 4, worker.vit.service.snapshot().get("embedding_bytes"));
            worker.vit.closeAdmission();
            assertEquals(Status.Code.UNAVAILABLE, assertThrows(StatusRuntimeException.class,
                    () -> worker.stub().remoteMultimodalEmbedding(request)).getStatus().getCode());
            assertFalse(worker.stub().getWorkerStatus(EngineRpcService.StatusVersionPB.getDefaultInstance()).getAlive());
            worker.vit.reopenAdmissionAfterFailedRemoval();
            assertEquals(result, worker.stub().remoteMultimodalEmbedding(request));
            await(() -> worker.inFlight() == 0);
        }
    }
    @Test void localAndNetworkCallsShareQuotaAndRequestDelayCancellationDoesNotCompute() throws Exception {
        try (var worker = new Worker()) {
            worker.vit.setPerformance(JSON.readTree("{\"mock\":{\"rpc\":{\"remote_multimodal_embedding\":{\"request\":{\"delay_ms\":2000}}}}}"));
            var request = MockVitServiceTest.inputs(MockVitServiceTest.image(56, 56, 0x123456));
            var context = Context.current().withCancellation();
            var pending = CompletableFuture.runAsync(() -> context.run(() -> worker.stub().remoteMultimodalEmbedding(request)));
            await(() -> worker.inFlight() == 1);
            var network = ManagedChannelBuilder.forAddress("127.0.0.1", worker.tcp.getPort()).usePlaintext().build();
            try {
                var error = assertThrows(StatusRuntimeException.class, () -> MultimodalRpcServiceGrpc.newBlockingStub(network)
                        .withDeadlineAfter(1, TimeUnit.SECONDS).remoteMultimodalEmbedding(request));
                assertEquals(Status.Code.RESOURCE_EXHAUSTED, error.getStatus().getCode());
                context.cancel(null);
                var cancelled = assertThrows(ExecutionException.class, () -> pending.get(1, TimeUnit.SECONDS));
                assertEquals(Status.Code.CANCELLED, Status.fromThrowable(cancelled.getCause()).getCode());
                await(() -> worker.inFlight() == 0);
                assertEquals(0, worker.vit.service.computations.get());
                var deadline = assertThrows(StatusRuntimeException.class, () -> worker.stub()
                        .withDeadlineAfter(40, TimeUnit.MILLISECONDS).remoteMultimodalEmbedding(request));
                assertEquals(Status.Code.DEADLINE_EXCEEDED, deadline.getStatus().getCode());
                await(() -> worker.inFlight() == 0);
                worker.vit.setPerformance(JSON.readTree("{\"mock\":{\"rpc\":{\"remote_multimodal_embedding\":{\"request\":{\"delay_ms\":0}}}}}"));
                assertTrue(worker.stub().remoteMultimodalEmbedding(request).hasMultimodalEmbedding());
                assertEquals(1, worker.vit.service.computations.get());
            } finally { context.cancel(null); network.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
        }
    }
    @Test void metadataConsumerRejectsMalformedShapePayloadHashAndOverflow() throws Exception {
        var tensor = EngineRpcService.TensorPB.newBuilder().addShape(1).addShape(8).build();
        var hash = EngineRpcService.TensorPB.newBuilder().setDataType(EngineRpcService.TensorPB.DataType.INT32)
                .addShape(1).setInt32Data(com.google.protobuf.ByteString.copyFrom(new byte[]{7,0,0,0})).build();
        var valid = EngineRpcService.MultimodalOutputPB.newBuilder().addSplitSize(1)
                .setMultimodalEmbedding(tensor).setMultimodalFeatureHash(hash).build();
        var response = new java.util.concurrent.atomic.AtomicReference<>(valid);
        String name = io.grpc.inprocess.InProcessServerBuilder.generateName();
        var server = io.grpc.inprocess.InProcessServerBuilder.forName(name).directExecutor()
                .addService(new MultimodalRpcServiceGrpc.MultimodalRpcServiceImplBase() {
                    @Override public void remoteMultimodalEmbedding(EngineRpcService.MultimodalInputsPB input,
                            io.grpc.stub.StreamObserver<EngineRpcService.MultimodalOutputPB> observer) {
                        observer.onNext(response.get()); observer.onCompleted();
                    }
                }).build().start();
        try (var client = new MockVitClient()) {
            client.localEndpoint = (host,port) -> name;
            var input = EngineRpcService.GenerateInputPB.newBuilder().addTokenIds(10).addTokenIds(7).addTokenIds(11)
                    .addMultimodalInputs(EngineRpcService.MultimodalInputPB.newBuilder().setMultimodalType(1).setMultimodalUrl("unused"));
            input.getMultimodalTokenLayoutBuilder().addSpansBuilder().setOffset(1).setLength(1);
            input.getGenerateConfigBuilder().addRoleAddrsBuilder().setRoleStr("VIT").setIp("127.0.0.1").setGrpcPort(1);
            assertEquals(input.build(), client.prepare(input.build(), 1000));
            for (var corrupt : java.util.List.of(
                    valid.toBuilder().setSplitSize(0, 0).build(),
                    valid.toBuilder().setMultimodalEmbedding(tensor.toBuilder().setShape(0, 2)).build(),
                    valid.toBuilder().setMultimodalEmbedding(tensor.toBuilder().setFp32Data(com.google.protobuf.ByteString.copyFromUtf8("x"))).build(),
                    valid.toBuilder().setMultimodalEmbedding(tensor.toBuilder().setShape(1, Long.MAX_VALUE)).build(),
                    valid.toBuilder().setMultimodalFeatureHash(hash.toBuilder().clearInt32Data()).build(),
                    valid.toBuilder().setMultimodalFeatureHash(hash.toBuilder().setInt32Data(com.google.protobuf.ByteString.copyFrom(new byte[]{8,0,0,0}))).build())) {
                response.set(corrupt);
                assertEquals(Status.Code.DATA_LOSS, assertThrows(StatusRuntimeException.class,
                        () -> client.prepare(input.build(), 1000)).getStatus().getCode());
            }
        } finally { server.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
    }

}
