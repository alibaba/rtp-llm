package org.flexlb.mock.engine.rtp;

import com.google.protobuf.ByteString;
import java.nio.*;
import org.flexlb.engine.grpc.EngineRpcService.*;

/** Synthetic token payload, real native tensor shape/encoding; never model inference. */
final class LocalResponse {
    static GenerateOutputsPB oneToken(GenerateInputPB input, int reused, long elapsedNanos) {
        var token = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(input.getTokenIds(input.getTokenIdsCount() - 1));
        var aux = AuxInfoPB.newBuilder().setInputLen(input.getTokenIdsCount()).setOutputLen(1).setStepOutputLen(1)
            .setIterCount(1).setPrefixLen(reused).setTotalReuseLen(reused).setLocalReuseLen(reused)
            .setFirstTokenCostTimeUs((int) Math.min(Integer.MAX_VALUE, elapsedNanos / 1000));
        var output = FlattenOutputPB.newBuilder().addFinished(true).addAuxInfo(aux).setOutputIds(TensorPB.newBuilder()
            .setDataType(TensorPB.DataType.INT32).addShape(1).addShape(1).addShape(1)
            .setInt32Data(ByteString.copyFrom(token.array())));
        return GenerateOutputsPB.newBuilder().setRequestId(input.getRequestId()).setFlattenOutput(output).build();
    }
}
