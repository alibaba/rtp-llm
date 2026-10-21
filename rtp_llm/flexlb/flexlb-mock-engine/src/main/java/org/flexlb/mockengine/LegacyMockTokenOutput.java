package org.flexlb.mockengine;

import com.google.protobuf.ByteString;
import org.flexlb.engine.grpc.EngineRpcService;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Client-visible release frames: Python consumes [outputs, beams, delta tokens]. */
final class LegacyMockTokenOutput {
    static boolean enabled() {
        return "1".equals(System.getenv("RTP_LLM_MOCK_BUNDLE"));
    }

    static EngineRpcService.GenerateOutputsPB build(MockPerformanceModel.RequestShape shape,
                                                     boolean finished, boolean decode) {
        int total = shape.outputLen();
        int cumulative = finished ? total : 1;
        int delta = finished && decode && total > 1 ? total - 1 : cumulative;
        int count = shape.input().getTokenIdsCount();
        int token = count == 0 ? 0 : shape.input().getTokenIds(count - 1);
        ByteBuffer bytes = ByteBuffer.allocate(Math.multiplyExact(delta, Integer.BYTES))
                .order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < delta; i++) bytes.putInt(token);
        var tensor = EngineRpcService.TensorPB.newBuilder()
                .setDataType(EngineRpcService.TensorPB.DataType.INT32)
                .addShape(1).addShape(1).addShape(delta)
                .setInt32Data(ByteString.copyFrom(bytes.array()));
        return EngineRpcService.GenerateOutputsPB.newBuilder()
                .setRequestId(shape.input().getRequestId())
                .setFlattenOutput(EngineRpcService.FlattenOutputPB.newBuilder()
                        .addFinished(finished).setOutputIds(tensor)
                        .addAuxInfo(EngineRpcService.AuxInfoPB.newBuilder()
                                .setInputLen(shape.inputLen()).setPrefixLen((int) shape.hitTokens())
                                .setOutputLen(cumulative).setStepOutputLen(delta).setIterCount(1)))
                .build();
    }
}
