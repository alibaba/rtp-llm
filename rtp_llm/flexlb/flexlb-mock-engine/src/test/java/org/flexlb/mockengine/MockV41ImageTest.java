package org.flexlb.mockengine;

import com.google.protobuf.ByteString;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.netty.NettyServerBuilder;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.MultimodalRpcServiceGrpc;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

class MockV41ImageTest {
    static final String IDENTITY = "3c4495a0bb9a965190d72f80205fa78d38d66c9a7bcefd5d5ff38fa795683f12";
    static MockV41Image.Config config() { return new MockV41Image.Config(8, 14, 3, 1024, 1048576, IDENTITY); }
    static EngineRpcService.V41ImageInputPB image(String content) {
        return EngineRpcService.V41ImageInputPB.newBuilder().setStart(1).setNVitH(3).setNVitW(6)
                .setProcessorIdentity(IDENTITY).setContentSha256(content)
                .addAllTypes(List.of(0, 1, 1, 2, 3))
                .setPatches(EngineRpcService.TensorPB.newBuilder().setDataType(EngineRpcService.TensorPB.DataType.BF16)
                        .addAllShape(List.of(18L, 3L, 14L, 14L)).setBf16Data(ByteString.copyFrom(new byte[18 * 3 * 14 * 14 * 2]))).build();
    }
    static EngineRpcService.MultimodalInputsPB request(EngineRpcService.V41ImageInputPB image) {
        return EngineRpcService.MultimodalInputsPB.newBuilder().setTimeoutMs(1000).setV41Inputs(
                EngineRpcService.V41RequestInputsPB.newBuilder().setSchemaVersion(1).addImages(image)).build();
    }
    static EngineRpcService.GenerateInputPB input(EngineRpcService.V41ImageInputPB image, int port) {
        return EngineRpcService.GenerateInputPB.newBuilder().setRequestId(99)
                .addAllTokenIds(List.of(42, 129264, 129264, 129264, 129264, 129264, 43))
                .setV41Inputs(EngineRpcService.V41RequestInputsPB.newBuilder().setSchemaVersion(1).addImages(image)
                        .addAllTokenTypes(List.of(-1, 0, 1, 1, 2, 3, -1))
                        .addAllImageMask(List.of(false, true, true, true, true, true, false)))
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder().setMaxNewTokens(4)
                        .addRoleAddrs(EngineRpcService.RoleAddrPB.newBuilder().setRoleStr("VIT").setIp("127.0.0.1").setGrpcPort(port)))
                .build();
    }
    @Test void modelMetadataIdentityMatchesRealPythonProcessor(@TempDir Path model) throws Exception {
        Files.writeString(model.resolve("config.json"), """
                {"text_config":{"hidden_size":8,"vocab_size":129280,"max_position_embeddings":1048576},
                 "vision_config":{"patch_size":14,"downsample_ratio":3,"max_image_tokens":1024,"min_pixels":295936},
                 "image_token_id":129264}
                """);
        assertEquals(config(), MockV41Image.Config.fromModel(model));
    }
    @Test void typedValidationRejectsWrongProcessorGeometryAndPatchShapeBeforeForward() {
        var original = image("a".repeat(64));
        for (var bad : List.of(original.toBuilder().setProcessorIdentity("wrong").build(),
                original.toBuilder().setTypes(2, 2).build(), original.toBuilder().setNVitH(0).build(),
                original.toBuilder().setPatches(original.getPatches().toBuilder().setBf16Data(ByteString.EMPTY)).build())) {
            assertThrows(IllegalArgumentException.class, () -> MockV41Image.validate(request(bad), config(), 1 << 20));
        }
        assertThrows(IllegalArgumentException.class, () -> MockV41Image.validate(request(original), null, 1 << 20));
    }
    @Test void preparedRpcReturnsBf16AndPrefillPreservesCanonicalIds() throws Exception {
        var image = image("a".repeat(64));
        try (var service = MockVitTestSupport.service(MockVitService.Config.defaults(), config())) {
            Server server = NettyServerBuilder.forPort(0).addService(service).build().start();
            ManagedChannel channel = ManagedChannelBuilder.forAddress("127.0.0.1", server.getPort()).usePlaintext().build();
            try (var client = new MockVitClient()) {
                var stub = MultimodalRpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(5, TimeUnit.SECONDS);
                var first = stub.remoteMultimodalEmbedding(request(image));
                assertEquals(List.of(5L, 8L), first.getMultimodalEmbedding().getShapeList());
                assertEquals(EngineRpcService.TensorPB.DataType.BF16, first.getMultimodalEmbedding().getDataType());
                assertEquals(80, first.getMultimodalEmbedding().getBf16Data().size());
                var input = input(image, server.getPort());
                assertEquals(input, client.prepare(input, 1000));
                assertEquals(first, stub.remoteMultimodalEmbedding(request(image)));
                assertEquals(1, service.computations.get());
                var bad = input.toBuilder().setV41Inputs(input.getV41Inputs().toBuilder().setImageMask(1, false)).build();
                assertEquals(Status.Code.INVALID_ARGUMENT, assertThrows(StatusRuntimeException.class,
                        () -> client.prepare(bad, 1000)).getStatus().getCode());
                assertEquals(1, service.computations.get());
            } finally {
                channel.shutdownNow(); server.shutdownNow();
                assertTrue(channel.awaitTermination(5, TimeUnit.SECONDS));
                assertTrue(server.awaitTermination(5, TimeUnit.SECONDS));
            }
        }
    }
    @Test void emptyContentIdentityBypassesModelEncodeCache() throws Exception {
        try (var service = MockVitTestSupport.service(MockVitService.Config.defaults(), config())) {
            var image = image("");
            var work = MockV41Image.validate(request(image), config(), 1 << 20);
            assertNotEquals(work.getFirst().key(), MockV41Image.validate(request(image), config(), 1 << 20).getFirst().key());
            assertEquals(MockV41Image.compute(image, config(), 1 << 20), MockV41Image.compute(image, config(), 1 << 20));
        }
    }
    @Test void textOnlyPreparedMetadataDoesNotRequireVitAddress() {
        var input = EngineRpcService.GenerateInputPB.newBuilder().addTokenIds(42)
                .setV41Inputs(EngineRpcService.V41RequestInputsPB.newBuilder().setSchemaVersion(1)
                        .addTokenTypes(-1).addImageMask(false)).build();
        try (var client = new MockVitClient()) {
            assertEquals(input, client.prepare(input, 1000));
            assertEquals(0, client.channelCount());
        }
    }
    @Test void contentIdentitySeparatesSameCanonicalIdsAndReuseNeverSplitsImages() {
        var a = input(image("a".repeat(64)), 1); var b = input(image("b".repeat(64)), 1);
        var ka = MockImageCacheKeys.fullBlocks(a, 2, 0x4453563431494631L);
        var kb = MockImageCacheKeys.fullBlocks(b, 2, 0x4453563431494631L);
        assertNotEquals(ka, kb);
        assertEquals(0, MockImageCacheKeys.validHitBlocks(a, 2, 2));
        assertEquals(3, MockImageCacheKeys.validHitBlocks(a, 3, 2));
        var noIdentity = input(image(""), 1);
        assertEquals(List.of(), MockImageCacheKeys.fullBlocks(noIdentity, 2, 0));
    }
}
