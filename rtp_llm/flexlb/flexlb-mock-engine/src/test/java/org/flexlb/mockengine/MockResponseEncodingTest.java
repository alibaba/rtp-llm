package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** Wire contract from cpp/model_rpc/QueryConverter.cc, independent of scheduling. */
class MockResponseEncodingTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void outputPreservesCumulativeAndStepLengths(boolean payloads) {
        var input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(17).addTokenIds(23).addTokenIds(42).build();
        var shape = new MockPerformanceModel.RequestShape(input, 512, 5, List.of(1L), 128, 1, true);
        var output = MockResponseEncoding.output(shape, true, 5, 4, payloads);
        assertEquals(17, output.getRequestId());
        var flattened = output.getFlattenOutput();
        assertEquals(List.of(true), flattened.getFinishedList());
        var aux = flattened.getAuxInfo(0);
        assertEquals(512, aux.getInputLen());
        assertEquals(128, aux.getPrefixLen());
        assertEquals(5, aux.getOutputLen());
        assertEquals(4, aux.getStepOutputLen());
        assertEquals(payloads, flattened.hasOutputIds());
        if (payloads) {
            var tensor = flattened.getOutputIds();
            assertEquals(List.of(1L, 1L, 4L), tensor.getShapeList());
            assertEquals(EngineRpcService.TensorPB.DataType.INT32, tensor.getDataType());
            var bytes = ByteBuffer.wrap(tensor.getInt32Data().toByteArray()).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < 4; i++) assertEquals(42, bytes.getInt());
            assertFalse(bytes.hasRemaining());
        }
    }
}
