package org.flexlb.mock.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.*;
import java.nio.file.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.TimeUnit;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.*;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;

/** End-to-end probe against a separately launched, unmocked production Master. */
public final class MasterProbe {
    private static GenerateInputPB input(long id) {
        return GenerateInputPB.newBuilder().setRequestId(id).addAllTokenIds(List.of(1, 2, 3, 4))
            .setGenerateConfig(GenerateConfigPB.newBuilder().setMaxNewTokens(3).setReturnIncremental(true).setCanUsePdSeparation(true)).build();
    }
    private static FlexlbScheduleRequestPB request(long id) {
        return FlexlbScheduleRequestPB.newBuilder().setRequestId(id).setModel("mock").setSeqLen(4).setMaxNewTokens(3)
            .setCacheKeyBlockSize(16).setPriority(50).setRequestTimeMs(System.currentTimeMillis()).setGenerateInput(input(id).toByteString()).build();
    }
    public static void main(String[] args) throws Exception {
        var master = ManagedChannelBuilder.forTarget(args[0]).usePlaintext().build();
        try {
            var route = FlexlbServiceGrpc.newBlockingStub(master).withDeadlineAfter(8, TimeUnit.SECONDS).schedule(request(1600));
            if (!route.getSuccess() || route.getCode() != 200 || route.getEnqueuedByMaster() != Boolean.parseBoolean(args[1]))
                throw new IllegalStateException("Unexpected production Master route: " + route);
            var p = route.getServerStatusList().stream().filter(s -> s.getRole().equals("PREFILL")).findFirst().orElseThrow();
            var engine = ManagedChannelBuilder.forAddress(p.getServerIp(), p.getGrpcPort()).usePlaintext().build();
            try {
                var rpc = RpcServiceGrpc.newBlockingStub(engine).withDeadlineAfter(5, TimeUnit.SECONDS);
                var routed = input(1600).toBuilder();
                for (var server : route.getServerStatusList()) routed.getGenerateConfigBuilder().addRoleAddrs(RoleAddrPB.newBuilder()
                    .setRole(RoleAddrPB.RoleType.valueOf(server.getRole())).setIp(server.getServerIp()).setGrpcPort(server.getGrpcPort()));
                var stream = route.getEnqueuedByMaster() ? rpc.fetchResponse(FetchRequestPB.newBuilder().setRequestId(1600).build()) : rpc.generateStreamCall(routed.build());
                int tokens = 0; boolean finished = false;
                while (stream.hasNext()) {
                    var output = stream.next();
                    if (!output.hasFlattenOutput()) throw new IllegalStateException("Missing output: " + output);
                    tokens = output.getFlattenOutput().getAuxInfo(0).getOutputLen(); finished = output.getFlattenOutput().getFinished(0);
                }
                if (tokens != 3 || !finished) throw new IllegalStateException("Native routed stream incomplete");
            } finally { engine.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
            try (var client = new NativeClient(args[0])) {
                var result = client.send(request(1601), Duration.ofSeconds(8)).completion().toCompletableFuture().get(9, TimeUnit.SECONDS);
                if (result.outcome() != NativeClient.Outcome.success || result.outputTokens() != 3 || result.ttftNanos() == null)
                    throw new IllegalStateException("Client failed through production Master: " + result);
                new ObjectMapper().writeValue(Path.of(args[2]).toFile(), Map.of("enqueued_by_master", route.getEnqueuedByMaster(),
                    "outcome", result.outcome().name(), "tokens", result.outputTokens(), "ttft_ns", result.ttftNanos(), "elapsed_ns", result.elapsedNanos()));
            }
        } finally { master.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }
    }
}
