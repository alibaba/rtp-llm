package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Map;

/** QueryConverter wire projection; owns no request, cache or scheduling state. */
final class MockResponseEncoding {
    private MockResponseEncoding() { }

    private static Map<Integer, Integer> multimodalLengths(EngineRpcService.GenerateInputPB input) {
        Map<Integer, Integer> lengths = new java.util.HashMap<>();
        if (input.hasV41Inputs()) {
            int rows = input.getV41Inputs().getImagesList().stream().mapToInt(EngineRpcService.V41ImageInputPB::getTypesCount).sum();
            if (rows > 0) lengths.put(1, rows);
            return lengths;
        }
        int count = Math.min(input.getMultimodalInputsCount(), input.getMultimodalTokenLayout().getSpansCount());
        for (int i = 0; i < count; i++) lengths.merge(input.getMultimodalInputs(i).getMultimodalType(),
                input.getMultimodalTokenLayout().getSpans(i).getLength(), Integer::sum);
        return lengths;
    }

    static EngineRpcService.GenerateOutputsPB output(MockPerformanceModel.RequestShape shape,
                                                           boolean finished, int outputLen,
                                                           int stepOutputLen, boolean tokenPayloads) {
        var flatten = EngineRpcService.FlattenOutputPB.newBuilder()
                .addFinished(finished)
                .addAuxInfo(EngineRpcService.AuxInfoPB.newBuilder()
                        .setInputLen(shape.inputLen())
                        .putAllMultimodalLengths(multimodalLengths(shape.input()))
                        .setPrefixLen((int) shape.hitTokens())
                        .setOutputLen(outputLen)
                        .setIterCount(1)
                        .setStepOutputLen(stepOutputLen)
                        .build());
        if (tokenPayloads) {
            // QueryConverter / Python trans_output require [outputs, beams, tokens]
            // and little-endian INT32 bytes. Repeat an input token, not model inference.
            int tokenCount = shape.input().getTokenIdsCount();
            int token = tokenCount == 0 ? 0 : shape.input().getTokenIds(tokenCount - 1);
            ByteBuffer ids = ByteBuffer.allocate(Math.multiplyExact(stepOutputLen, Integer.BYTES))
                    .order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < stepOutputLen; i++) ids.putInt(token);
            flatten.setOutputIds(EngineRpcService.TensorPB.newBuilder()
                    .setDataType(EngineRpcService.TensorPB.DataType.INT32)
                    .addShape(1).addShape(1).addShape(stepOutputLen)
                    .setInt32Data(com.google.protobuf.ByteString.copyFrom(ids.array())));
        }
        return EngineRpcService.GenerateOutputsPB.newBuilder()
                .setRequestId(shape.input().getRequestId())
                .setFlattenOutput(flatten)
                .build();
    }
}
