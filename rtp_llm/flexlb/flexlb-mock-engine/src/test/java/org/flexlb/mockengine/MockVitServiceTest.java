package org.flexlb.mockengine;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.netty.NettyServerBuilder;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.MultimodalRpcServiceGrpc;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

class MockVitServiceTest {
    private MockVitService service;
    private ManagedChannel channel;
    private Server server;
    private MultimodalRpcServiceGrpc.MultimodalRpcServiceBlockingStub stub;

    void start(MockVitService.Config config) throws Exception {
        service = new MockVitService(config);
        server = NettyServerBuilder.forPort(0).addService(service).build().start();
        channel = ManagedChannelBuilder.forAddress("127.0.0.1", server.getPort()).usePlaintext().build();
        stub = MultimodalRpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(10, TimeUnit.SECONDS);
    }
    @AfterEach void close() throws Exception {
        if (channel != null) { channel.shutdownNow(); assertTrue(channel.awaitTermination(5, TimeUnit.SECONDS)); }
        if (server != null) { server.shutdownNow(); assertTrue(server.awaitTermination(5, TimeUnit.SECONDS)); }
        if (service != null) service.close();
    }
    static EngineRpcService.MultimodalInputPB image(int width, int height, int color) throws Exception {
        var image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) for (int x = 0; x < width; x++) image.setRGB(x, y, color);
        var bytes = new ByteArrayOutputStream(); ImageIO.write(image, "png", bytes);
        return EngineRpcService.MultimodalInputPB.newBuilder().setMultimodalType(1)
                .setMultimodalUrl("data:image/png;base64," + Base64.getEncoder().encodeToString(bytes.toByteArray()))
                .setMmPreprocessConfig(EngineRpcService.MMPreprocessConfigPB.newBuilder()
                        .setWidth(-1).setHeight(-1).setMinPixels(-1).setMaxPixels(-1)
                        .setFps(-1).setMinFrames(-1).setMaxFrames(-1).setMaxLongSidePixel(-1)).build();
    }
    static EngineRpcService.MultimodalInputsPB inputs(EngineRpcService.MultimodalInputPB... images) {
        return EngineRpcService.MultimodalInputsPB.newBuilder().addAllMultimodalInputs(List.of(images)).build();
    }
    @Test void coldHashSubmissionAndFeatureReadHaveIdenticalTokens() throws Exception {
        start(MockVitService.Config.defaults());
        var image = image(224, 224, 0xabcdef); String key = MockVitImage.key(image);
        var probe = EngineRpcService.MultimodalHashRequestPB.newBuilder().addKeys(key).build();
        assertFalse(stub.getMultimodalHashes(probe).getEntries(0).getHashHit());
        var hashes = stub.getMultimodalHashes(probe.toBuilder().setInputs(inputs(image)).build());
        assertTrue(hashes.getEntries(0).getHashHit());
        assertEquals(64, hashes.getEntries(0).getSplitSize(0));
        var features = stub.remoteMultimodalEmbedding(inputs(image));
        assertEquals(hashes.getEntries(0).getFeatureHashes(), features.getMultimodalFeatureHash().getInt32Data());
        assertEquals(List.of(64L, 64L), features.getMultimodalEmbedding().getShapeList());
        assertEquals(1, service.computations.get());
        assertEquals("VIT", stub.getWorkerStatus(EngineRpcService.StatusVersionPB.getDefaultInstance()).getRole());
        assertFalse(stub.getCacheStatus(EngineRpcService.CacheVersionPB.getDefaultInstance()).hasMultimodalCache());
        var directory = stub.getCacheStatus(EngineRpcService.CacheVersionPB.newBuilder().setNeedCacheKeys(true).build());
        assertEquals(List.of(key), directory.getMultimodalCache().getKeysList());
        assertEquals(hashes.getWorkerInstance(), directory.getMultimodalCache().getWorkerInstance());
    }
    @Test void distinctImagesAndPreprocessSettingsHaveDistinctHashes() throws Exception {
        start(MockVitService.Config.defaults());
        var a = image(224, 224, 0xaaaaaa); var b = image(224, 224, 0xbbbbbb);
        var first = stub.remoteMultimodalEmbedding(inputs(a));
        var second = stub.remoteMultimodalEmbedding(inputs(b));
        assertNotEquals(first.getMultimodalFeatureHash(), second.getMultimodalFeatureHash());
        var larger = a.toBuilder().setMmPreprocessConfig(a.getMmPreprocessConfig().toBuilder().setWidth(448).setHeight(224)).build();
        assertEquals(128, stub.remoteMultimodalEmbedding(inputs(larger)).getSplitSize(0));
        assertNotEquals(MockVitImage.key(a), MockVitImage.key(larger));
    }
    @Test void hashMetadataSurvivesEmbeddingEvictionAndRecomputationIsStable() throws Exception {
        var d = MockVitService.Config.defaults();
        start(new MockVitService.Config(8, 1, 4, 2400, 8192, d.maxInputBytes(), d.maxOutputBytes(), 0));
        var a = image(224, 224, 0xaaaaaa); var b = image(224, 224, 0xbbbbbb);
        var first = stub.remoteMultimodalEmbedding(inputs(a)); stub.remoteMultimodalEmbedding(inputs(b));
        var metadata = stub.getMultimodalHashes(EngineRpcService.MultimodalHashRequestPB.newBuilder().addKeys(MockVitImage.key(a)).build()).getEntries(0);
        assertTrue(metadata.getHashHit()); assertFalse(metadata.getEmbeddingHit());
        assertEquals(first, stub.remoteMultimodalEmbedding(inputs(a)));
        assertTrue(service.snapshot().get("embedding_bytes") <= 2400);
        assertTrue(service.snapshot().get("hash_bytes") <= 8192);
    }
    @Test void repeatedConcurrentRequestsShareComputationAndStatusRemainsAvailable() throws Exception {
        var d = MockVitService.Config.defaults();
        start(new MockVitService.Config(8, 1, 4, 1 << 20, 1 << 20, d.maxInputBytes(), d.maxOutputBytes(), 100));
        var image = image(224, 224, 0xabcdef);
        List<CompletableFuture<EngineRpcService.MultimodalOutputPB>> calls = new ArrayList<>();
        for (int i = 0; i < 8; i++) calls.add(CompletableFuture.supplyAsync(() -> stub.remoteMultimodalEmbedding(inputs(image))));
        assertTrue(stub.getWorkerStatus(EngineRpcService.StatusVersionPB.getDefaultInstance()).getAlive());
        CompletableFuture.allOf(calls.toArray(CompletableFuture[]::new)).get(5, TimeUnit.SECONDS);
        assertEquals(1, service.computations.get());
        for (var value : calls) assertEquals(calls.get(0).join(), value.join());
    }
    @Test void invalidMetadataAndImageErrorsDoNotProduceSuccessfulEmptyOutputs() throws Exception {
        start(MockVitService.Config.defaults());
        var input = image(28, 28, 0xabcdef);
        var mismatch = EngineRpcService.MultimodalHashRequestPB.newBuilder().addKeys("wrong").setInputs(inputs(input)).build();
        assertEquals(Status.Code.INVALID_ARGUMENT, assertThrows(StatusRuntimeException.class, () -> stub.getMultimodalHashes(mismatch)).getStatus().getCode());
        assertEquals(Status.Code.INVALID_ARGUMENT, assertThrows(StatusRuntimeException.class,
                () -> stub.remoteMultimodalEmbedding(inputs(input.toBuilder().setMultimodalUrl("data:image/png;base64,AAAA").build()))).getStatus().getCode());
    }
    @Test void oneTimedOutWaiterDoesNotCancelAnotherWaiterForSameImage() throws Exception {
        var d = MockVitService.Config.defaults();
        start(new MockVitService.Config(8, 1, 4, 1 << 20, 1 << 20, d.maxInputBytes(), d.maxOutputBytes(), 150));
        var image = image(224, 224, 0xabcdef);
        var surviving = CompletableFuture.supplyAsync(() -> stub.remoteMultimodalEmbedding(inputs(image)));
        var expired = stub.withDeadlineAfter(20, TimeUnit.MILLISECONDS);
        assertEquals(Status.Code.DEADLINE_EXCEEDED, assertThrows(StatusRuntimeException.class,
                () -> expired.remoteMultimodalEmbedding(inputs(image))).getStatus().getCode());
        assertEquals(64, surviving.get(5, TimeUnit.SECONDS).getSplitSize(0));
        assertEquals(1, service.computations.get());
    }
    @Test void prefillConsumesFeaturesAndRejectsWrongFrontendHashes() throws Exception {
        start(MockVitService.Config.defaults());
        var image = image(224, 224, 0xabcdef);
        var result = stub.remoteMultimodalEmbedding(inputs(image));
        var ids = result.getMultimodalFeatureHash().getInt32Data().asReadOnlyByteBuffer().order(java.nio.ByteOrder.LITTLE_ENDIAN);
        var input = EngineRpcService.GenerateInputPB.newBuilder().setRequestId(1).addTokenIds(10);
        while (ids.hasRemaining()) input.addTokenIds(ids.getInt());
        input.addTokenIds(11).addMultimodalInputs(image);
        input.getMultimodalTokenLayoutBuilder().addSpansBuilder().setOffset(1).setLength(64);
        input.getGenerateConfigBuilder().addRoleAddrsBuilder().setRoleStr("VIT").setIp("127.0.0.1").setGrpcPort(server.getPort());
        try (var client = new MockVitClient()) {
            assertEquals(input.build(), client.prepare(input.build(), 1000));
            assertEquals(input.build(), client.prepare(input.build(), 1000));
            assertEquals(1, client.channelCount());
            assertEquals(Status.Code.DATA_LOSS, assertThrows(StatusRuntimeException.class,
                    () -> client.prepare(input.setTokenIds(1, 0).build(), 1000)).getStatus().getCode());
        }
    }
    @Test void disabledHashCacheStillReturnsHashesAndEmbeddingOnlyHitCanRefillMetadata() throws Exception {
        var d = MockVitService.Config.defaults();
        start(new MockVitService.Config(8, 1, 4, 1 << 20, 0, d.maxInputBytes(), d.maxOutputBytes(), 0));
        var image = image(56, 56, 0x123456);
        var request = EngineRpcService.MultimodalHashRequestPB.newBuilder().addKeys(MockVitImage.key(image));
        var submitted = stub.getMultimodalHashes(request.setInputs(inputs(image)).build()).getEntries(0);
        assertTrue(submitted.getHashHit()); assertTrue(submitted.getEmbeddingHit());
        var probe = stub.getMultimodalHashes(request.clearInputs().build()).getEntries(0);
        assertFalse(probe.getHashHit()); assertTrue(probe.getEmbeddingHit());
        assertEquals(submitted.getFeatureHashes(), stub.getMultimodalHashes(request.setInputs(inputs(image)).build()).getEntries(0).getFeatureHashes());
        assertEquals(1, service.computations.get());
    }
    @Test void packedInputIsNeverSilentlyTreatedAsText() {
        var input = EngineRpcService.GenerateInputPB.newBuilder()
                .setV41Inputs(EngineRpcService.V41RequestInputsPB.getDefaultInstance()).build();
        try (var client = new MockVitClient()) {
            assertEquals(Status.Code.INVALID_ARGUMENT, assertThrows(StatusRuntimeException.class,
                    () -> client.prepare(input, 1000)).getStatus().getCode());
        }
    }
    @Test void duplicateImagesPreserveOutputSegmentsAndAsyncSubmissionErrorsAreTyped() throws Exception {
        start(MockVitService.Config.defaults());
        var image = image(56, 56, 0x123456);
        var result = stub.remoteMultimodalEmbedding(inputs(image, image));
        assertEquals(List.of(4, 4), result.getSplitSizeList());
        assertEquals(8, result.getMultimodalEmbedding().getShape(0));
        assertEquals(1, service.computations.get());
        assertEquals(Status.Code.INVALID_ARGUMENT, assertThrows(StatusRuntimeException.class,
                () -> stub.asyncSubmitEmbedding(EngineRpcService.MultimodalInputsPB.getDefaultInstance())).getStatus().getCode());
        stub.waitGreenNetVerdict(inputs(image));
        assertEquals(1, service.computations.get());
    }

}
