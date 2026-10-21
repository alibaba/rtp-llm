package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import java.nio.ByteOrder;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class LegacyMockTokenOutputTest {
    private static MockPerformanceModel.RequestShape shape(int output) {
        var input = EngineRpcService.GenerateInputPB.newBuilder().setRequestId(12)
                .addTokenIds(10).addTokenIds(123456).build();
        return new MockPerformanceModel.RequestShape(input, 2, output, List.of(), 0, 0);
    }
    private static int tokens(EngineRpcService.GenerateOutputsPB response) {
        var tensor = response.getFlattenOutput().getOutputIds();
        assertEquals(List.of(1L, 1L, (long) tensor.getShape(2)), tensor.getShapeList());
        assertEquals(EngineRpcService.TensorPB.DataType.INT32, tensor.getDataType());
        assertEquals(tensor.getShape(2) * 4, tensor.getInt32Data().size());
        assertEquals(123456, tensor.getInt32Data().asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN).getInt());
        return (int) tensor.getShape(2);
    }
    @Test void prefillAndDecodeDeltasDoNotDuplicateFirstToken() {
        var first = LegacyMockTokenOutput.build(shape(4), false, false);
        var last = LegacyMockTokenOutput.build(shape(4), true, true);
        assertFalse(first.getFlattenOutput().getFinished(0));
        assertTrue(last.getFlattenOutput().getFinished(0));
        assertEquals(4, tokens(first) + tokens(last));
        assertEquals(4, last.getFlattenOutput().getAuxInfo(0).getOutputLen());
    }
    @Test void singleTokenTerminalIsNonempty() {
        assertEquals(1, tokens(LegacyMockTokenOutput.build(shape(1), true, true)));
    }
    @Test void standalonePrefillTerminalContainsAllTokens() {
        assertEquals(4, tokens(LegacyMockTokenOutput.build(shape(4), true, false)));
    }
}
