package org.flexlb.mock.engine.rtp;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.*;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;

/** Test-only native traffic for Prometheus ingestion; this is not the production Client. */
public final class MetricsProbe {
    public static void main(String[] args) throws Exception {
        int length = Integer.parseInt(args[1]), count = Integer.parseInt(args[2]);
        var channel = ManagedChannelBuilder.forTarget(args[0]).usePlaintext().build();
        long reused = 0;
        try {
            for (int i = 0; i < count; i++) {
                var input = GenerateInputPB.newBuilder().setRequestId(i + 1)
                    .addAllTokenIds(IntStream.rangeClosed(1, length).boxed().toList())
                    .setGenerateConfig(GenerateConfigPB.newBuilder().setMaxNewTokens(1).setReuseCache(true)).build();
                var stream = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3, TimeUnit.SECONDS).generateStreamCall(input);
                var output = stream.next();
                if (stream.hasNext() || !output.getFlattenOutput().getFinished(0)) throw new IllegalStateException("Invalid native result");
                reused += output.getFlattenOutput().getAuxInfo(0).getTotalReuseLen();
            }
        } finally { channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
        new ObjectMapper().writeValue(Path.of(args[3]).toFile(), Map.of("requests", count, "input_tokens", (long) count * length, "hit_tokens", reused));
    }
}
