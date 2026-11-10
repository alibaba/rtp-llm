package org.flexlb.mock.engine;

import io.grpc.*;
import java.util.concurrent.TimeUnit;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.MultimodalRpcServiceGrpc;

/**
 * Sends real RPC work to exercise the service process failure path.
 */
public final class EngineFailureProbe {
    public static void main(String[] args) throws Exception {
        var channel = ManagedChannelBuilder.forTarget(args[0]).usePlaintext().build();
        try {
            var input = MultimodalInputsPB.newBuilder().setRequestId(1).addMultimodalInputs(
                MultimodalInputPB.newBuilder().setMultimodalType(1).setMultimodalUrl("test-image")
                    .setMmPreprocessConfig(MMPreprocessConfigPB.newBuilder().setWidth(28).setHeight(28))).build();
            try {
                MultimodalRpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3, TimeUnit.SECONDS).remoteMultimodalEmbedding(input);
                throw new AssertionError("Negative Forward must fail");
            } catch (StatusRuntimeException expected) {
                if (expected.getStatus().getCode() != Status.Code.INTERNAL
                        && expected.getStatus().getCode() != Status.Code.UNKNOWN) throw expected;
            }
        } finally { channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
    }
}
