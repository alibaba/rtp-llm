package org.flexlb.mock.master;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.ManagedChannelBuilder;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.TimeUnit;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;

/** Test launcher: finish traffic on the real Mock RPC before starting the production Master. */
public final class LateMasterNode {
    public static void main(String[] args) throws Exception {
        var json = new ObjectMapper();
        var topology = json.readTree(Path.of(System.getenv("MOCK_TOPOLOGY_FILE")).toFile());
        String address = null;
        for (var node : topology) if (node.path("role").asText().equals("prefill")) address = node.path("rpc").asText();
        if (address == null) throw new IllegalStateException("Missing Prefill endpoint");
        var channel = ManagedChannelBuilder.forTarget(address).usePlaintext().build();
        try {
            var rpc = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(30, TimeUnit.SECONDS);
            for (int id = 1; id <= 1025; id++) {
                var input = GenerateInputPB.newBuilder().setRequestId(id).addAllTokenIds(List.of(1, 2))
                    .setGenerateConfig(GenerateConfigPB.newBuilder().setMaxNewTokens(1)).build();
                var outputs = rpc.generateStreamCall(input);
                int count = 0;
                while (outputs.hasNext()) {
                    if (!outputs.next().getFlattenOutput().getFinished(0)) throw new IllegalStateException("Incomplete warmup output");
                    count++;
                }
                if (count != 1) throw new IllegalStateException("Missing warmup output");
            }
            var state = rpc.getWorkerStatus(StatusVersionPB.newBuilder().setLatestFinishedVersion(1025).build());
            if (!state.getAlive() || state.getLatestFinishedVersion() != 1025 || state.getRunningTaskInfoCount() != 0)
                throw new IllegalStateException("Engine did not finish pre-Master traffic");
            json.writeValue(Path.of(System.getenv("MOCK_OUTPUT_DIR"), "before-master.json").toFile(),
                Map.of("finished_requests", 1025, "finished_version", state.getLatestFinishedVersion()));
        } finally { channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
        org.flexlb.MasterNode.main(args);
    }
}
