package org.flexlb.mock.admission;

import com.google.protobuf.ByteString;
import java.nio.*;
import org.flexlb.engine.grpc.EngineRpcService.*;

/** Synthetic token payload, real native tensor shape/encoding; never model inference. */
final class LocalResponse {
    static GenerateOutputsPB token(GenerateInputPB input, int tokenId, int generated, int stepTokens, int reused, long elapsedNanos, boolean finished, boolean pd) {
        int width = input.getGenerateConfig().getReturnIncremental() ? stepTokens : generated;
        var tokens = ByteBuffer.allocate(Math.multiplyExact(4, width)).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < width; i++) tokens.putInt(tokenId);
        var aux = AuxInfoPB.newBuilder().setInputLen(input.getTokenIdsCount()).setOutputLen(generated).setStepOutputLen(stepTokens)
            .setIterCount(generated).setPrefixLen(reused).setTotalReuseLen(reused).setLocalReuseLen(reused).setPdSep(pd)
            .setCostTimeUs((int) Math.min(Integer.MAX_VALUE, elapsedNanos / 1000))
            .setFirstTokenCostTimeUs((int) Math.min(Integer.MAX_VALUE, elapsedNanos / 1000));
        var output = FlattenOutputPB.newBuilder().addFinished(finished).addAuxInfo(aux).setOutputIds(TensorPB.newBuilder()
            .setDataType(TensorPB.DataType.INT32).addShape(1).addShape(1).addShape(width)
            .setInt32Data(ByteString.copyFrom(tokens.array())));
        return GenerateOutputsPB.newBuilder().setRequestId(input.getRequestId()).setFlattenOutput(output).build();
    }
}
