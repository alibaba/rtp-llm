package org.flexlb.mock.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.*;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import java.net.*;
import java.nio.file.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.*;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;

/** Network fixture for the real production protobuf; this is not the production Master. */
public final class ClientProbe {
    public static Map<String, Object> exercise(String prefill, String decode) throws Exception {
        var p = ManagedChannelBuilder.forTarget(prefill).usePlaintext().build();
        var pRoute = endpoint("PREFILL", prefill); var dRoute = endpoint("DECODE", decode);
        var schedules = new AtomicInteger();
        var master = NettyServerBuilder.forAddress(new InetSocketAddress("127.0.0.1", 0)).addService(new FlexlbServiceGrpc.FlexlbServiceImplBase() {
            @Override public void schedule(FlexlbScheduleRequestPB request, StreamObserver<FlexlbScheduleResponsePB> response) {
                boolean batch = schedules.incrementAndGet() == 2;
                try {
                    if (batch) {
                        var input = GenerateInputPB.parseFrom(request.getGenerateInput()).toBuilder();
                        input.getGenerateConfigBuilder().clearRoleAddrs().addRoleAddrs(RoleAddrPB.newBuilder().setRole(RoleAddrPB.RoleType.DECODE)
                            .setIp(dRoute.getServerIp()).setGrpcPort(dRoute.getGrpcPort()));
                        var ack = RpcServiceGrpc.newBlockingStub(p).withDeadlineAfter(2, TimeUnit.SECONDS).enqueueBatch(
                            EnqueueBatchRequestPB.newBuilder().setBatchId(91).setFetchAttachTimeoutMs(1500).addDpSlots(EnqueueBatchDpSlotPB.newBuilder()
                                .addRequests(EnqueueBatchExternalInputPB.newBuilder().setInput(input))).build());
                        if (ack.getSuccessesCount() != 1) throw Status.INTERNAL.withDescription("Fixture enqueue failed: " + ack).asRuntimeException();
                    }
                    response.onNext(FlexlbScheduleResponsePB.newBuilder().setSuccess(true).setCode(200).setEnqueuedByMaster(batch)
                        .addServerStatus(pRoute).addServerStatus(dRoute).setLifecycle(RequestLifecyclePB.newBuilder()
                            .setRequestId(request.getRequestId()).setBatchId(batch ? 91 : 0)).build()); response.onCompleted();
                } catch (Exception error) { response.onError(Status.INTERNAL.withCause(error).asRuntimeException()); }
            }
            @Override public void cancel(FlexlbCancelRequestPB request, StreamObserver<FlexlbCancelResponsePB> response) {
                RpcServiceGrpc.newBlockingStub(p).withDeadlineAfter(1, TimeUnit.SECONDS)
                    .cancel(CancelRequestPB.newBuilder().setRequestId(request.getRequestId()).build());
                response.onNext(FlexlbCancelResponsePB.newBuilder().setFound(true).build()); response.onCompleted();
            }
        }).build().start();
        try (var client = new NativeClient("127.0.0.1:" + master.getPort())) {
            var results = new LinkedHashMap<String, Object>();
            for (long id : new long[]{1501, 1502}) {
                var input = GenerateInputPB.newBuilder().setRequestId(id).addAllTokenIds(List.of(1, 2, 3, 4))
                    .setGenerateConfig(GenerateConfigPB.newBuilder().setMaxNewTokens(3).setCanUsePdSeparation(true).setReturnIncremental(true)).build();
                var request = FlexlbScheduleRequestPB.newBuilder().setRequestId(id).setSeqLen(4).setMaxNewTokens(3).setPriority(50)
                    .setGenerateInput(input.toByteString()).build();
                var call = client.send(request, Duration.ofSeconds(3));
                var result = call.completion().toCompletableFuture().get(4, TimeUnit.SECONDS);
                if (result.outcome() != NativeClient.Outcome.success || result.outputTokens() != 3 || result.ttftNanos() == null || result.ttftNanos() <= 0)
                    throw new IllegalStateException("Native Client failed: " + result);
                results.put(id == 1501 ? "generate" : "fetch", result);
            }
            results.put("schedule_calls", schedules.get()); return results;
        } finally { p.shutdownNow(); master.shutdownNow(); p.awaitTermination(3, TimeUnit.SECONDS); master.awaitTermination(3, TimeUnit.SECONDS); }
    }
    private static FlexlbServerStatusPB endpoint(String role, String address) {
        var parts = address.split(":");
        return FlexlbServerStatusPB.newBuilder().setRole(role).setServerIp(parts[0]).setGrpcPort(Integer.parseInt(parts[1])).build();
    }
    public static void main(String[] args) throws Exception {
        // Json has no Java time extension dependency: keep absolute times as ISO strings at this boundary.
        var mapper = new ObjectMapper(); var raw = exercise(args[0], args[1]);
        var output = new LinkedHashMap<String, Object>();
        for (var entry : raw.entrySet()) {
            if (entry.getValue() instanceof NativeClient.Result r) output.put(entry.getKey(), Map.of(
                "request_id", r.requestId(), "sent_at", r.sentAt().toString(), "ttft_ns", r.ttftNanos(), "elapsed_ns", r.elapsedNanos(),
                "output_tokens", r.outputTokens(), "outcome", r.outcome().name()));
            else output.put(entry.getKey(), entry.getValue());
        }
        mapper.writeValue(Path.of(args[2]).toFile(), output);
    }
}
