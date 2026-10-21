package org.flexlb.mockengine;

import com.google.protobuf.ByteString;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.MultimodalRpcServiceGrpc;
import org.junit.jupiter.api.Test;
import java.util.concurrent.TimeUnit;
import static org.junit.jupiter.api.Assertions.*;

class LegacyMockVitClientTest {
    @Test void prefillExpandsReleasePlaceholderUsingPerImageFeatures() throws Exception {
        byte[] data = new byte[4 * 8 * 2];
        for (int i = 0; i < data.length; i++) data[i] = (byte)i;
        var server = start(data);
        var input = EngineRpcService.GenerateInputPB.newBuilder().setRequestId(10)
                .addTokenIds(1).addTokenIds(55).addTokenIds(3);
        input.addMultimodalInputsBuilder().setMultimodalType(1).setMultimodalUrl("data:image/mock");
        input.getGenerateConfigBuilder().addRoleAddrsBuilder().setRoleStr("VIT").setIp("127.0.0.1").setGrpcPort(server.getPort());
        try (var client = new LegacyMockVitClient(55)) {
            var prepared = client.prepare(input.build());
            assertEquals(6, prepared.getTokenIdsCount());
            for (int i = 0; i < 4; i++) assertEquals(MockFeatureHash.row(data, i * 16, 16), prepared.getTokenIds(1 + i));
            assertEquals(1, prepared.getTokenIds(0));
            assertEquals(3, prepared.getTokenIds(5));
            assertEquals(3, input.getTokenIdsCount());
            assertEquals(Status.Code.INVALID_ARGUMENT, assertThrows(StatusRuntimeException.class,
                    () -> client.prepare(input.setTokenIds(1, 99).build())).getStatus().getCode());
        } finally { server.shutdownNow(); assertTrue(server.awaitTermination(5, TimeUnit.SECONDS)); }
    }
    @Test void typedV41KeepsCanonicalIdsAndChecksFeatureRows() throws Exception {
        var server = start(new byte[4 * 8 * 2]);
        var input = EngineRpcService.GenerateInputPB.newBuilder().addTokenIds(10).addTokenIds(11)
                .addTokenIds(12).addTokenIds(13).addTokenIds(14).addTokenIds(15);
        var typed = input.getV41InputsBuilder().setSchemaVersion(1);
        for (int i = 0; i < 6; i++) { typed.addTokenTypes(i == 0 || i == 5 ? -1 : i - 1); typed.addImageMask(i > 0 && i < 5); }
        typed.addImagesBuilder().addTypes(0).addTypes(1).addTypes(2).addTypes(3);
        input.getGenerateConfigBuilder().addRoleAddrsBuilder().setRoleStr("VIT").setIp("127.0.0.1").setGrpcPort(server.getPort());
        try (var client = new LegacyMockVitClient(-1)) {
            assertEquals(input.build(), client.prepare(input.build()));
            input.getV41InputsBuilder().getImagesBuilder(0).addTypes(1);
            assertEquals(Status.Code.DATA_LOSS, assertThrows(StatusRuntimeException.class,
                    () -> client.prepare(input.build())).getStatus().getCode());
        } finally { server.shutdownNow(); assertTrue(server.awaitTermination(5, TimeUnit.SECONDS)); }
    }
    private static Server start(byte[] data) throws Exception {
        return NettyServerBuilder.forPort(0).addService(new MultimodalRpcServiceGrpc.MultimodalRpcServiceImplBase() {
            @Override public void remoteMultimodalEmbedding(EngineRpcService.MultimodalInputsPB request,
                    StreamObserver<EngineRpcService.MultimodalOutputsPB> observer) {
                assertFalse(request.getMetadataOnly());
                var result = EngineRpcService.MultimodalOutputsPB.newBuilder();
                result.addMultimodalOutputsBuilder().setMultimodalEmbedding(EngineRpcService.TensorPB.newBuilder()
                        .setDataType(EngineRpcService.TensorPB.DataType.BF16).addShape(4).addShape(8)
                        .setBf16Data(ByteString.copyFrom(data)));
                observer.onNext(result.build()); observer.onCompleted();
            }
        }).build().start();
    }
}
