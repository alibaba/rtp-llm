package org.flexlb.mock.engine.service;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.ByteOrder;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.junit.jupiter.api.Test;

class LocalResponseTest {
    @Test void rpcChunksContainOnlyNewTokensForBothFrontendResponseModes() {
        for (boolean incremental : new boolean[] {false, true}) {
            var input = GenerateInputPB.newBuilder().setRequestId(7).addAllTokenIds(java.util.List.of(1, 2, 3, 4, 42))
                .setGenerateConfig(GenerateConfigPB.newBuilder().setReturnIncremental(incremental)).build();
            int received = 0;
            // Includes a multi-token step, as with a speculative decode commit.
            for (int committed : new int[] {1, 1, 3, 1, 2}) {
                received += committed;
                var response = RpcMessages.token(input, 223, received, committed, 4, 1000, received == 8, true);
                var chunk = response.getFlattenOutput();
                assertEquals(7, response.getRequestId());
                assertEquals(committed, chunk.getOutputIds().getShape(2));
                assertEquals(received, chunk.getAuxInfo(0).getOutputLen());
                assertEquals(committed, chunk.getAuxInfo(0).getStepOutputLen());
                assertEquals(4, chunk.getAuxInfo(0).getTotalReuseLen());
                assertEquals(4, chunk.getAuxInfo(0).getLocalReuseLen());
                assertEquals(0, chunk.getAuxInfo(0).getRemoteReuseLen());
                var data = chunk.getOutputIds().getInt32Data().asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
                assertEquals(committed * Integer.BYTES, data.remaining());
                while (data.hasRemaining()) assertEquals(223, data.getInt());
                assertEquals(received == 8, chunk.getFinished(0));
            }
            assertEquals(8, received);
        }
    }
}
