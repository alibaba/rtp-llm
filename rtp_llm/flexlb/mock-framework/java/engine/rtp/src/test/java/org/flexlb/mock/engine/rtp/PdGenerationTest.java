package org.flexlb.mock.engine.rtp;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.grpc.*;
import io.grpc.stub.StreamObserver;
import java.nio.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.BooleanSupplier;
import java.util.stream.IntStream;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class PdGenerationTest {
    @TempDir Path directory;
    private Harness node(String role, int blocks, int millis) throws Exception {
        var json = new ObjectMapper(); ObjectNode root = json.valueToTree(RtpNodeTest.configuration(role));
        ((ObjectNode) root.path("component").path("config")).put("seq_size_per_block", 2).put("reuse_cache", true);
        ((ObjectNode) root.path("component").path("config").path("capacity")).put("device_blocks", blocks);
        ((ObjectNode) root.path("environment").path("compute").path(role)).put("expression", Integer.toString(millis));
        var path = directory.resolve(role + ".json"); json.writeValue(path.toFile(), root);
        return new Harness(RtpNode.start(RtpConfig.load(path)));
    }
    private static final class Harness implements AutoCloseable {
        final RtpNode node; final ManagedChannel channel;
        Harness(RtpNode node) { this.node = node; channel = ManagedChannelBuilder.forTarget(node.rpcAddress()).usePlaintext().build(); }
        WorkerStatusPB status() { return RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3, TimeUnit.SECONDS).getWorkerStatus(StatusVersionPB.getDefaultInstance()); }
        @Override public void close() throws Exception { channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); node.close(); }
    }
    private static GenerateInputPB input(Harness d, long id, int length, int output, int timeout, boolean incremental) {
        int port = Integer.parseInt(d.node.rpcAddress().split(":")[1]);
        return GenerateInputPB.newBuilder().setRequestId(id).setPriority(7).addAllTokenIds(IntStream.rangeClosed(1, length).boxed().toList())
            .setGenerateConfig(GenerateConfigPB.newBuilder().setMaxNewTokens(output).setTimeoutMs(timeout).setReuseCache(true)
                .setCanUsePdSeparation(true).setReturnIncremental(incremental).addRoleAddrs(RoleAddrPB.newBuilder()
                    .setRole(RoleAddrPB.RoleType.DECODE).setIp("127.0.0.1").setGrpcPort(port))).build();
    }
    private static void await(BooleanSupplier condition) throws Exception {
        long end = System.nanoTime() + TimeUnit.SECONDS.toNanos(4);
        while (!condition.getAsBoolean()) { if (System.nanoTime() >= end) fail("Condition timed out"); Thread.sleep(5); }
    }
    private static final class Call implements AutoCloseable {
        final Context.CancellableContext context = Context.current().withCancellation();
        final BlockingQueue<Object> outputs = new LinkedBlockingQueue<>();
        Call(Harness p, GenerateInputPB input) { context.run(() -> RpcServiceGrpc.newStub(p.channel).generateStreamCall(input, new StreamObserver<>() {
            @Override public void onNext(GenerateOutputsPB value) { outputs.add(value); }
            @Override public void onError(Throwable error) { outputs.add(error); }
            @Override public void onCompleted() { outputs.add("completed"); }
        })); }
        Object next() throws Exception { var value = outputs.poll(4, TimeUnit.SECONDS); assertNotNull(value); return value; }
        GenerateOutputsPB output(int count, boolean finished, int width) throws Exception {
            var value = next(); assertInstanceOf(GenerateOutputsPB.class, value); var output = (GenerateOutputsPB) value;
            var flat = output.getFlattenOutput(); assertEquals(count, flat.getAuxInfo(0).getOutputLen());
            assertEquals(finished, flat.getFinished(0)); assertEquals(width, flat.getOutputIds().getShape(2));
            assertEquals(width * 4, flat.getOutputIds().getInt32Data().size()); assertTrue(flat.getAuxInfo(0).getPdSep());
            return output;
        }
        Throwable error(Status.Code expected) throws Exception { var value = next(); assertInstanceOf(Throwable.class, value);
            assertEquals(expected, Status.fromThrowable((Throwable) value).getCode()); return (Throwable) value; }
        @Override public void close() { context.cancel(null); }
    }
    private static long code(Throwable error) throws Exception {
        return ErrorDetailsPB.parseFrom(Status.trailersFromThrowable(error).get(Metadata.Key.of("grpc-status-details-bin", Metadata.BINARY_BYTE_MARSHALLER))).getErrorCode();
    }
    private static double metric(Harness node, String name, String label) throws Exception {
        try (var stream = java.net.URI.create((String) node.node.endpoints().get("metrics")).toURL().openStream()) {
            return LocalRpcTest.sample(new String(stream.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8), name, label);
        }
    }

    @Test void cumulativeNativeRequestTraversesPAndDWithActualPerTokenCompute() throws Exception {
        try (var p = node("prefill", 8, 15); var d = node("decode", 8, 15); var call = new Call(p, input(d, 1, 5, 4, 2000, false))) {
            long ttft = -1;
            for (int i = 1; i <= 4; i++) {
                var output = call.output(i, i == 4, i);
                var ids = output.getFlattenOutput().getOutputIds().getInt32Data().asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
                while (ids.hasRemaining()) assertEquals(5, ids.getInt());
                long current = output.getFlattenOutput().getAuxInfo(0).getFirstTokenCostTimeUs();
                if (ttft < 0) ttft = current; else assertEquals(ttft, current);
            }
            assertEquals("completed", call.next());
            await(() -> p.status().getAvailableKvCache() == 16 && d.status().getAvailableKvCache() == 16);
            assertEquals(1, metric(p, "mock_engine_forward_seconds_count", "role=\"prefill\""));
            assertEquals(3, metric(d, "mock_engine_forward_seconds_count", "role=\"decode\""));
            assertFalse(d.status().getFinishedTaskList(0).hasErrorInfo()); assertEquals(3, d.status().getFinishedTaskList(0).getIterateCount());
        }
    }

    @Test void incrementalStreamDoesNotRepeatFirstTokenOrReturnCumulativePayloads() throws Exception {
        try (var p = node("prefill", 8, 1); var d = node("decode", 8, 1); var call = new Call(p, input(d, 1, 3, 6, 2000, true))) {
            for (int i = 1; i <= 6; i++) call.output(i, i == 6, 1);
            assertEquals("completed", call.next()); assertEquals(5, metric(d, "mock_engine_forward_tokens_total", "kind=\"output\""));
        }
    }

    @Test void invalidRouteAndUnsupportedModesFailBeforeAllocatingResources() throws Exception {
        try (var p = node("prefill", 8, 1); var d = node("decode", 8, 1)) {
            var invalid = input(d, 1, 3, 4, 1000, false).toBuilder(); invalid.getGenerateConfigBuilder().clearRoleAddrs();
            try (var call = new Call(p, invalid.build())) { assertEquals(605, code(call.error(Status.Code.INVALID_ARGUMENT))); }
            invalid = input(d, 1, 3, 4, 1000, false).toBuilder(); invalid.getGenerateConfigBuilder().setCanUsePdSeparation(false);
            try (var call = new Call(p, invalid.build())) { call.error(Status.Code.UNIMPLEMENTED); }
            assertEquals(16, p.status().getAvailableKvCache()); assertEquals(16, d.status().getAvailableKvCache());
        }
    }

    @Test void cancellationDuringPComputeReleasesBothNodesWithoutOutput() throws Exception {
        try (var p = node("prefill", 8, 1000); var d = node("decode", 8, 1); var call = new Call(p, input(d, 1, 5, 4, 3000, true))) {
            await(() -> p.status().getRunningQueryLen() == 1 && d.status().getWaitingQueryLen() == 1);
            call.close(); await(() -> p.status().getAvailableKvCache() == 16 && d.status().getAvailableKvCache() == 16);
            assertEquals(0, metric(p, "mock_engine_forward_seconds_count", "role=\"prefill\""));
        }
    }

    @Test void cancellationDuringDComputeDoesNotProduceLateTokens() throws Exception {
        try (var p = node("prefill", 8, 1); var d = node("decode", 8, 1000); var call = new Call(p, input(d, 1, 5, 4, 3000, true))) {
            call.output(1, false, 1); await(() -> d.status().getRunningQueryLen() == 1);
            call.close(); await(() -> d.status().getAvailableKvCache() == 16);
            assertEquals(0, metric(d, "mock_engine_forward_seconds_count", "role=\"decode\""));
            assertEquals(16, p.status().getAvailableKvCache());
        }
    }

    @Test void prefillDoesNotStartBeforeDecodeAllocationAcknowledgement() throws Exception {
        try (var p = node("prefill", 8, 1); var d = node("decode", 8, 1); var holder = d.node.cache().acquire(6, List.of()).lease();
                var call = new Call(p, input(d, 1, 5, 4, 2500, true))) {
            await(() -> d.status().getWaitingQueryLen() == 1);
            assertEquals(0, p.status().getRunningTaskInfoCount()); assertEquals(16, p.status().getAvailableKvCache());
            assertEquals(0, metric(p, "mock_engine_requests_started_total", "")); holder.close();
            for (int n = 1; n <= 4; n++) call.output(n, n == 4, 1);
            assertEquals("completed", call.next());
        }
    }

    @Test void temporaryGrowthPressureWaitsAndResumesOnRelease() throws Exception {
        try (var p = node("prefill", 8, 1); var d = node("decode", 4, 10); var holder = d.node.cache().acquire(2, List.of()).lease();
                var call = new Call(p, input(d, 1, 3, 6, 2500, true))) {
            call.output(1, false, 1); call.output(2, false, 1);
            assertNull(call.outputs.poll(150, TimeUnit.MILLISECONDS)); holder.close();
            for (int i = 3; i <= 6; i++) call.output(i, i == 6, 1);
            assertEquals("completed", call.next()); assertEquals(8, d.status().getAvailableKvCache());
        }
    }

    @Test void growthBeyondPhysicalCapacityFailsWith602AndDoesNotPublishDCache() throws Exception {
        try (var p = node("prefill", 8, 1); var d = node("decode", 3, 1); var call = new Call(p, input(d, 1, 3, 6, 2000, true))) {
            for (int i = 1; i <= 4; i++) call.output(i, false, 1);
            assertEquals(602, code(call.error(Status.Code.RESOURCE_EXHAUSTED)));
            assertEquals(6, d.status().getAvailableKvCache()); assertTrue(d.node.cache().snapshot(-1, true).keys().isEmpty());
        }
    }

    @Test void deadlineDuringCapacityWaitReleasesRequestAndDoesNotWakeIntoGeneration() throws Exception {
        try (var p = node("prefill", 8, 1); var d = node("decode", 4, 1); var holder = d.node.cache().acquire(2, List.of()).lease();
                var call = new Call(p, input(d, 1, 3, 6, 300, true))) {
            call.output(1, false, 1); call.output(2, false, 1); call.error(Status.Code.DEADLINE_EXCEEDED);
            await(() -> d.status().getAvailableKvCache() == 4); holder.close();
            await(() -> d.status().getAvailableKvCache() == 8); assertEquals(1, metric(d, "mock_engine_forward_seconds_count", "role=\"decode\""));
        }
    }

    @Test void closingDecodeMidGenerationFailsPRequestAndReleasesP() throws Exception {
        try (var p = node("prefill", 8, 1); var d = node("decode", 8, 1000); var call = new Call(p, input(d, 1, 3, 4, 2000, true))) {
            call.output(1, false, 1); await(() -> d.status().getRunningQueryLen() == 1); d.node.close();
            call.error(Status.Code.UNAVAILABLE); assertEquals(16, p.status().getAvailableKvCache());
        }
    }

    @Test void cancellingOneRequestDoesNotCloseSharedChannelOrCancelAnother() throws Exception {
        try (var p = node("prefill", 8, 1); var d = node("decode", 8, 30);
                var first = new Call(p, input(d, 1, 3, 6, 2500, true)); var second = new Call(p, input(d, 2, 3, 4, 2500, true))) {
            first.output(1, false, 1); second.output(1, false, 1); first.close();
            for (int i = 2; i <= 4; i++) second.output(i, i == 4, 1);
            assertEquals("completed", second.next()); await(() -> d.status().getAvailableKvCache() == 16);
        }
    }

    @Test void unlimitedRequestIsOwnedByItsPCallAndCanCompleteNormally() throws Exception {
        try (var p = node("prefill", 8, 1); var d = node("decode", 8, 1); var call = new Call(p, input(d, 1, 3, 4, 0, true))) {
            for (int i = 1; i <= 4; i++) call.output(i, i == 4, 1);
            assertEquals("completed", call.next());
        }
    }

    @Test void duplicatePRequestCannotReplaceTheOriginalOwner() throws Exception {
        try (var p = node("prefill", 8, 1000); var d = node("decode", 8, 1); var first = new Call(p, input(d, 1, 3, 4, 2500, true))) {
            await(() -> p.status().getRunningQueryLen() == 1);
            try (var duplicate = new Call(p, input(d, 1, 3, 4, 2500, true))) { duplicate.error(Status.Code.ALREADY_EXISTS); }
            assertEquals(1, p.status().getRunningTaskInfoCount()); first.close();
            await(() -> p.status().getAvailableKvCache() == 16 && d.status().getAvailableKvCache() == 16);
        }
    }

    @Test void closingPAfterLoadStillCancelsTheRemoteGeneration() throws Exception {
        try (var p = node("prefill", 8, 1); var d = node("decode", 8, 1000); var call = new Call(p, input(d, 1, 3, 4, 2500, true))) {
            call.output(1, false, 1); await(() -> d.status().getRunningQueryLen() == 1); p.node.close();
            assertInstanceOf(Throwable.class, call.next()); await(() -> d.status().getAvailableKvCache() == 16);
        }
    }

    @Test void slowConsumerDoesNotBypassRequestDeadlineOrBufferTheEntireGeneration() throws Exception {
        try (var p = node("prefill", 8, 0); var d = node("decode", 2048, 0); var context = Context.current().withCancellation()) {
            var replies = new LinkedBlockingQueue<Object>();
            context.run(() -> RpcServiceGrpc.newStub(p.channel).generateStreamCall(input(d, 1, 3, 3000, 500, false),
                new io.grpc.stub.ClientResponseObserver<GenerateInputPB, GenerateOutputsPB>() {
                    @Override public void beforeStart(io.grpc.stub.ClientCallStreamObserver<GenerateInputPB> request) { request.disableAutoRequestWithInitial(1); }
                    @Override public void onNext(GenerateOutputsPB value) { replies.add(value); }
                    @Override public void onError(Throwable error) { replies.add(error); }
                    @Override public void onCompleted() { replies.add("completed"); }
                }));
            assertInstanceOf(GenerateOutputsPB.class, replies.poll(3, TimeUnit.SECONDS));
            await(() -> d.status().getFinishedTaskListCount() == 1 && p.status().getAvailableKvCache() == 16 && d.status().getAvailableKvCache() == 4096);
            assertTrue(d.status().getFinishedTaskList(0).hasErrorInfo());
            assertTrue(metric(d, "mock_engine_forward_seconds_count", "role=\"decode\"") < 2999);
        }
    }

    @Test void downstreamFailureBeforeFirstTokenCancelsRetainedPSource() throws Exception {
        try (var p = node("prefill", 8, 1000); var d = node("decode", 8, 1)) {
            var input = input(d, 1, 3, 4, 1000, true); d.node.close();
            try (var call = new Call(p, input)) {
                call.error(Status.Code.UNAVAILABLE); await(() -> p.status().getAvailableKvCache() == 16);
                assertEquals(0, metric(p, "mock_engine_forward_seconds_count", "role=\"prefill\""));
            }
        }
    }
}
