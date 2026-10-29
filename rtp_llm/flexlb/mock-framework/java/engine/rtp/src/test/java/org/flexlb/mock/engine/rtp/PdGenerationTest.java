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
    private Harness node(String role, int blocks, int millis) throws Exception { return node(role, blocks, millis, Map.of()); }
    private Harness node(String role, int blocks, int millis, Map<String, Object> models) throws Exception {
        var json = new ObjectMapper(); ObjectNode root = json.valueToTree(RtpNodeTest.configuration(role));
        ((ObjectNode) root.path("component").path("config")).put("seq_size_per_block", 2).put("reuse_cache", true);
        ((ObjectNode) root.path("component").path("config")).put("test_block_num", blocks + 1);
        ((ObjectNode) root.path("component").path("config").path("compute")).put("expression", Integer.toString(millis));
        models.forEach((key, value) -> ((ObjectNode) root.path("component").path("config")).set(key, json.valueToTree(value)));
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
        Call(Harness p, GenerateInputPB input) { this(p, input, false); }
        Call(Harness p, GenerateInputPB input, boolean fetch) { context.run(() -> {
            var observer = new StreamObserver<GenerateOutputsPB>() {
            @Override public void onNext(GenerateOutputsPB value) { outputs.add(value); }
            @Override public void onError(Throwable error) { outputs.add(error); }
            @Override public void onCompleted() { outputs.add("completed"); }
            };
            if (fetch) RpcServiceGrpc.newStub(p.channel).fetchResponse(FetchRequestPB.newBuilder().setRequestId(input.getRequestId()).build(), observer);
            else RpcServiceGrpc.newStub(p.channel).generateStreamCall(input, observer);
        }); }
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
    @Test void allGenerationPathsRejectTheSameUnsupportedInputsBeforeAdmission() throws Exception {
        try (var p = node("prefill", 16, 1); var d = node("decode", 16, 1)) {
            var modes = List.<java.util.function.Consumer<GenerateConfigPB.Builder>>of(
                g -> g.getStopWordsListBuilder().addRowsBuilder().addValues(1),
                g -> g.setSpEdit(true), g -> g.setSpInputLookup(true),
                g -> g.setReturnLogits(true), g -> g.setNumBeams(2), g -> g.setAdapterName(com.google.protobuf.StringValue.of("adapter")));
            long id = 1000;
            for (var mode : modes) {
                for (int length : new int[]{1, 2}) {
                    var request = input(d, id++, 4, length, 1000, true).toBuilder();
                    mode.accept(request.getGenerateConfigBuilder());
                    try (var call = new Call(p, request.build())) { call.error(Status.Code.UNIMPLEMENTED); }
                    var rejected = new LinkedBlockingQueue<Throwable>();
                    var remote = RpcServiceGrpc.newStub(d.channel).withDeadlineAfter(2, TimeUnit.SECONDS).remoteGenerate(new StreamObserver<GenerateOutputsPB>() {
                        public void onNext(GenerateOutputsPB value) { fail("Unsupported input was admitted"); }
                        public void onError(Throwable error) { rejected.add(error); }
                        public void onCompleted() { fail("Unsupported input completed"); }
                    });
                    remote.onNext(GenerateRequestPB.newBuilder().setStage(RemoteStage.ALLOCATE).setClientId("validation")
                        .setRequestId(request.getRequestId()).setInput(request).build());
                    var error = rejected.poll(3, TimeUnit.SECONDS);
                    assertNotNull(error); assertEquals(Status.Code.UNIMPLEMENTED, Status.fromThrowable(error).getCode());
                }
            }
            assertEquals(0, p.status().getRunningTaskInfoCount()); assertEquals(0, d.status().getRunningTaskInfoCount());
            assertEquals(32, p.status().getAvailableKvCache()); assertEquals(32, d.status().getAvailableKvCache());
        }
    }

    private static long code(Throwable error) throws Exception {
        return ErrorDetailsPB.parseFrom(Status.trailersFromThrowable(error).get(Metadata.Key.of("grpc-status-details-bin", Metadata.BINARY_BYTE_MARSHALLER))).getErrorCode();
    }
    private static double metric(Harness node, String name, String label) throws Exception {
        try (var stream = java.net.URI.create((String) node.node.endpoints().get("metrics")).toURL().openStream()) {
            return LocalRpcTest.sample(new String(stream.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8), name, label);
        }
    }

    private static EnqueueBatchResponsePB enqueue(Harness p, long ttl, GenerateInputPB... inputs) {
        var slot = EnqueueBatchDpSlotPB.newBuilder();
        for (var input : inputs) slot.addRequests(EnqueueBatchExternalInputPB.newBuilder().setInput(input));
        return RpcServiceGrpc.newBlockingStub(p.channel).withDeadlineAfter(4, TimeUnit.SECONDS).enqueueBatch(
            EnqueueBatchRequestPB.newBuilder().setBatchId(81).setFetchAttachTimeoutMs(ttl).addDpSlots(slot).build());
    }
    private static CancelStatusPB cancel(Harness p, long id) {
        return RpcServiceGrpc.newBlockingStub(p.channel).withDeadlineAfter(3, TimeUnit.SECONDS)
            .cancel(CancelRequestPB.newBuilder().setRequestId(id).build()).getStatus();
    }

    @Test void batchCompletesBothSchedulersWithoutFetchAndRetainsOutput() throws Exception {
        try (var p = node("prefill", 8, 100); var d = node("decode", 8, 1)) {
            var request = input(d, 101, 5, 4, 3000, true);
            var ack = enqueue(p, 2000, request); assertEquals(81, ack.getBatchId()); assertEquals(1, ack.getSuccessesCount());
            assertEquals(1, d.status().getRunningTaskInfoCount());
            await(() -> p.status().getRunningTaskInfoCount() == 1);
            assertEquals(81, p.status().getRunningTaskInfo(0).getBatchId());
            await(() -> d.status().getRunningTaskInfoCount() == 0 && p.status().getRunningTaskInfoCount() == 0);
            assertTrue(metric(d, "mock_engine_forward_seconds_count", "role=\"decode\"") > 0);
            assertTrue(p.status().getFinishedTaskListList().stream().anyMatch(t -> t.getRequestId() == 101));
            assertTrue(d.status().getFinishedTaskListList().stream().anyMatch(t -> t.getRequestId() == 101));
            assertEquals(16, p.status().getAvailableKvCache()); assertEquals(16, d.status().getAvailableKvCache());
            try (var fetch = new Call(p, request, true)) {
                for (int n = 1; n <= 4; n++) fetch.output(n, n == 4, 1);
                assertEquals("completed", fetch.next());
            }
            await(() -> d.status().getRunningTaskInfoCount() == 0 && p.status().getRunningTaskInfoCount() == 0);
            assertEquals(CancelStatusPB.CANCEL_STATUS_NOT_FOUND, cancel(p, 101));
        }
    }

    @Test void enqueueRpcDeadlineDoesNotOwnDeferredGeneration() throws Exception {
        try (var p = node("prefill", 8, 1); var d = node("decode", 8, 1)) {
            var context = Context.current().withCancellation();
            var request = input(d, 102, 5, 3, 2000, true);
            context.call(() -> enqueue(p, 1500, request)); context.cancel(null);
            Thread.sleep(60);
            try (var fetch = new Call(p, request, true)) {
                fetch.output(1, false, 1); fetch.output(2, false, 1); fetch.output(3, true, 1);
                assertEquals("completed", fetch.next());
            }
        }
    }

    @Test void unattachedOutputExpiresWithoutHoldingSchedulerResources() throws Exception {
        try (var p = node("prefill", 8, 1); var d = node("decode", 8, 1)) {
            var request = input(d, 103, 5, 4, 0, true);
            assertEquals(1, enqueue(p, 80, request).getSuccessesCount());
            await(() -> p.status().getRunningTaskInfoCount() == 0 && d.status().getRunningTaskInfoCount() == 0);
            assertEquals(16, p.status().getAvailableKvCache()); assertEquals(16, d.status().getAvailableKvCache());
            Thread.sleep(100);
            try (var fetch = new Call(p, request, true)) { fetch.error(Status.Code.NOT_FOUND); }
        }
    }

    @Test void requestTimeoutCompletesSchedulersAndRetainsErrorForFetch() throws Exception {
        try (var p = node("prefill", 8, 1); var d = node("decode", 8, 500)) {
            var request = input(d, 104, 5, 4, 100, true);
            assertEquals(1, enqueue(p, 3000, request).getSuccessesCount());
            await(() -> d.status().getRunningTaskInfoCount() == 0 && p.status().getRunningTaskInfoCount() == 0);
            try (var fetch = new Call(p, request, true)) { fetch.error(Status.Code.DEADLINE_EXCEEDED); }
        }
    }

    @Test void disconnectingFetchDoesNotCancelDecodeOrHoldResources() throws Exception {
        try (var p = node("prefill", 8, 1); var d = node("decode", 8, 30)) {
            var request = input(d, 110, 5, 4, 2000, true); enqueue(p, 1500, request);
            try (var fetch = new Call(p, request, true)) { fetch.output(1, false, 1); }
            await(() -> d.status().getRunningTaskInfoCount() == 0 && p.status().getRunningTaskInfoCount() == 0);
            assertTrue(d.status().getFinishedTaskListList().stream().anyMatch(t -> t.getRequestId() == 110 && !t.hasErrorInfo()));
            assertEquals(16, p.status().getAvailableKvCache()); assertEquals(16, d.status().getAvailableKvCache());
        }
    }

    @Test void cancelBeforeEnqueueFencesFutureAdmissionAndFetchWith8429() throws Exception {
        try (var p = node("prefill", 8, 1); var d = node("decode", 8, 1)) {
            assertEquals(CancelStatusPB.CANCEL_STATUS_TOMBSTONED, cancel(p, 105));
            var request = input(d, 105, 5, 4, 1000, true); var ack = enqueue(p, 1000, request);
            assertEquals(0, ack.getSuccessesCount()); assertEquals(8429, ack.getErrors(0).getErrorInfo().getErrorCode());
            assertEquals(0, d.status().getRunningTaskInfoCount());
            try (var fetch = new Call(p, request, true)) { assertEquals(8429, code(fetch.error(Status.Code.RESOURCE_EXHAUSTED))); }
            assertEquals(CancelStatusPB.CANCEL_STATUS_TOMBSTONED, cancel(p, 105));
            assertEquals(Status.Code.INVALID_ARGUMENT, assertThrows(StatusRuntimeException.class, () -> cancel(p, 0)).getStatus().getCode());
        }
    }

    @Test void cancelUnattachedBatchReleasesBothOwnersAndLeavesPreemptionFence() throws Exception {
        try (var p = node("prefill", 8, 100); var d = node("decode", 8, 1)) {
            var request = input(d, 106, 5, 4, 2000, true); enqueue(p, 1500, request);
            assertEquals(CancelStatusPB.CANCEL_STATUS_ACCEPTED, cancel(p, 106));
            await(() -> p.status().getRunningTaskInfoCount() == 0 && d.status().getRunningTaskInfoCount() == 0);
            assertEquals(16, p.status().getAvailableKvCache()); assertEquals(16, d.status().getAvailableKvCache());
            try (var fetch = new Call(p, request, true)) { assertEquals(8429, code(fetch.error(Status.Code.RESOURCE_EXHAUSTED))); }
            assertEquals(8429, enqueue(p, 1000, request).getErrors(0).getErrorInfo().getErrorCode());
        }
    }

    @Test void cancelAttachedFetchPropagatesPreemptionAndSecondFetchCannotStealIt() throws Exception {
        try (var p = node("prefill", 8, 1); var d = node("decode", 8, 500)) {
            var request = input(d, 107, 5, 4, 3000, true); enqueue(p, 100, request);
            try (var fetch = new Call(p, request, true)) {
                fetch.output(1, false, 1);
                try (var duplicate = new Call(p, request, true)) { duplicate.error(Status.Code.ALREADY_EXISTS); }
                assertEquals(CancelStatusPB.CANCEL_STATUS_ACCEPTED, cancel(p, 107));
                assertEquals(8429, code(fetch.error(Status.Code.RESOURCE_EXHAUSTED)));
            }
            await(() -> d.status().getRunningTaskInfoCount() == 0);
        }
    }

    @Test void duplicateBatchRejectsEveryEntryWithoutAnyAdmissionAndMixedErrorsStayPerRequest() throws Exception {
        try (var p = node("prefill", 8, 1); var d = node("decode", 8, 1)) {
            var request = input(d, 108, 5, 4, 2000, true);
            var duplicate = enqueue(p, 1000, request, request);
            assertEquals(2, duplicate.getErrorsCount()); assertEquals(0, duplicate.getSuccessesCount());
            assertEquals(Status.Code.ALREADY_EXISTS.value(), duplicate.getErrors(0).getErrorInfo().getErrorCode());
            assertEquals(0, d.status().getRunningTaskInfoCount());
            var invalid = request.toBuilder().setRequestId(109).clearTokenIds().build();
            var mixed = enqueue(p, 1000, request, invalid);
            assertEquals(1, mixed.getErrorsCount()); assertEquals(109, mixed.getErrors(0).getRequestId()); assertEquals(1, mixed.getSuccessesCount());
            cancel(p, 108);
        }
    }

    @Test void groupEnvelopeAndInvalidDpRankMissingInputAreHandledWithoutLostEntries() throws Exception {
        try (var p = node("prefill", 8, 1); var d = node("decode", 8, 1)) {
            var stub = RpcServiceGrpc.newBlockingStub(p.channel).withDeadlineAfter(3, TimeUnit.SECONDS);
            var request = input(d, 110, 5, 4, 2000, true);
            var invalid = stub.enqueueGroup(EnqueueGroupRequestPB.newBuilder().setBatchId(82).setDpRank(1)
                .addRequests(EnqueueGroupInputPB.newBuilder().setInput(request)).addRequests(EnqueueGroupInputPB.getDefaultInstance()).build());
            assertEquals(2, invalid.getErrorsCount()); assertEquals(82, invalid.getBatchId());
            var good = stub.enqueueGroup(EnqueueGroupRequestPB.newBuilder().setBatchId(83)
                .addRequests(EnqueueGroupInputPB.newBuilder().setInput(request)).build());
            assertEquals(1, good.getSuccessesCount()); cancel(p, 110);
        }
    }

    @Test void attachedFetchOutlivesAttachmentLeaseAndCancellingItReleasesDecode() throws Exception {
        try (var p = node("prefill", 8, 1); var d = node("decode", 8, 180)) {
            var request = input(d, 111, 5, 6, 3000, true); enqueue(p, 100, request);
            try (var fetch = new Call(p, request, true)) {
                fetch.output(1, false, 1); fetch.output(2, false, 1);
            }
            await(() -> d.status().getRunningTaskInfoCount() == 0);
            assertEquals(16, d.status().getAvailableKvCache());
        }
    }

    @Test void batchPreparationFailureCannotReturnSuccess() throws Exception {
        try (var p = node("prefill", 8, 1); var d = node("decode", 1, 1)) {
            var ack = enqueue(p, 1000, input(d, 112, 5, 4, 1000, true));
            assertEquals(0, ack.getSuccessesCount()); assertEquals(602, ack.getErrors(0).getErrorInfo().getErrorCode());
            assertEquals(0, p.status().getRunningTaskInfoCount());
        }
    }

    @Test void closingPrefillWithAnUnfetchedBatchCancelsDecodeReservation() throws Exception {
        try (var p = node("prefill", 8, 1); var d = node("decode", 8, 1)) {
            enqueue(p, 10000, input(d, 113, 5, 4, 0, true)); p.node.close();
            await(() -> d.status().getRunningTaskInfoCount() == 0);
            assertEquals(16, d.status().getAvailableKvCache());
        }
    }

    @Test void duplicateActiveIdDoesNotReplaceTheOriginalDeferredOwner() throws Exception {
        try (var p = node("prefill", 8, 1); var d = node("decode", 8, 1)) {
            var request = input(d, 114, 5, 3, 2000, true); enqueue(p, 1500, request);
            assertEquals(Status.Code.ALREADY_EXISTS.value(), enqueue(p, 1000, request).getErrors(0).getErrorInfo().getErrorCode());
            try (var fetch = new Call(p, request, true)) {
                fetch.output(1, false, 1); fetch.output(2, false, 1); fetch.output(3, true, 1); assertEquals("completed", fetch.next());
            }
        }
    }

    @Test void cancellingDuringDecodeAllocationRejectsAdmissionWithoutCancellingAnotherRequest() throws Exception {
        try (var p = node("prefill", 8, 1); var d = node("decode", 8, 1000);
                var busy = new Call(p, input(d, 115, 5, 4, 4000, true)); var workers = Executors.newVirtualThreadPerTaskExecutor()) {
            busy.output(1, false, 1);
            var pending = workers.submit(() -> enqueue(p, 2000, input(d, 116, 11, 4, 3000, true)));
            await(() -> d.status().getWaitingQueryLen() == 1);
            assertEquals(CancelStatusPB.CANCEL_STATUS_ACCEPTED, cancel(p, 116));
            var ack = pending.get(2, TimeUnit.SECONDS);
            assertEquals(0, ack.getSuccessesCount()); assertEquals(8429, ack.getErrors(0).getErrorInfo().getErrorCode());
            await(() -> d.status().getWaitingQueryLen() == 0);
            assertEquals(1, d.status().getRunningTaskInfoCount());
        }
    }

    @Test void acceptedDraftsAdvanceSeveralTokensWhileBudgetTailStaysExact() throws Exception {
        try (var p = node("prefill", 8, 1); var d = node("decode", 8, 1, Map.of("decode", Map.of("draft_tokens", 3, "acceptance_rate", 1)));
                var call = new Call(p, input(d, 1, 5, 10, 2000, true))) {
            call.output(1, false, 1); call.output(5, false, 4); call.output(9, false, 4); call.output(10, true, 1);
            assertEquals("completed", call.next()); assertEquals(3, metric(d, "mock_engine_forward_seconds_count", "role=\"decode\""));
            assertEquals(9, metric(d, "mock_engine_forward_tokens_total", "kind=\"output\""));
            assertEquals(3, d.status().getFinishedTaskList(0).getIterateCount());
        }
    }

    @Test void rejectedDraftsStillPayForAllVerificationWork() throws Exception {
        try (var p = node("prefill", 8, 1); var d = node("decode", 8, 1, Map.of("decode", Map.of("draft_tokens", 3, "acceptance_rate", 0),
                "compute", Map.of("expression", "totalStepTokens"))); var call = new Call(p, input(d, 1, 5, 10, 2000, true))) {
            for (int n = 1; n <= 10; n++) call.output(n, n == 10, 1);
            assertEquals("completed", call.next()); assertEquals(9, metric(d, "mock_engine_forward_seconds_count", "role=\"decode\""));
            assertEquals(30, metric(d, "mock_engine_forward_tokens_total", "kind=\"compute\""));
            assertTrue(metric(d, "mock_engine_forward_seconds_sum", "role=\"decode\"") >= .030);
        }
    }

    @Test void eosOnFirstTokenCompletesDWithoutADuplicateTokenOrForward() throws Exception {
        try (var p = node("prefill", 8, 1, Map.of("generation", Map.of("eos_probability", 1))); var d = node("decode", 8, 1);
                var call = new Call(p, input(d, 1, 5, 10, 2000, true))) {
            call.output(1, true, 1); assertEquals("completed", call.next());
            assertEquals(0, metric(d, "mock_engine_forward_seconds_count", "role=\"decode\""));
            assertFalse(d.status().getFinishedTaskList(0).hasErrorInfo()); assertEquals(16, d.status().getAvailableKvCache());
        }
    }

    @Test void eosLengthIsStableAcrossSpeculationAndNeverOvershootsTheLastRound() throws Exception {
        int expected = new org.flexlb.mock.compute.StoppingModel(new org.flexlb.mock.api.model.StoppingSpec(.2, 42L)).length("77", 3, 20, false);
        for (int draft : new int[]{0, 3}) {
            try (var p = node("prefill", 16, 1, Map.of("generation", Map.of("eos_probability", .2, "seed", 42)));
                    var d = node("decode", 16, 1, Map.of("decode", Map.of("draft_tokens", draft, "acceptance_rate", 1)))) {
                var input = input(d, 77, 5, 20, 2000, true).toBuilder(); input.getGenerateConfigBuilder().setMinNewTokens(3);
                try (var call = new Call(p, input.build())) {
                    int sum = 0; boolean finished = false;
                    for (Object value; !(value = call.next()).equals("completed");) {
                        assertInstanceOf(GenerateOutputsPB.class, value); var flat = ((GenerateOutputsPB) value).getFlattenOutput();
                        assertFalse(finished); sum += flat.getAuxInfo(0).getStepOutputLen(); assertEquals(sum, flat.getAuxInfo(0).getOutputLen());
                        assertEquals(flat.getAuxInfo(0).getStepOutputLen(), flat.getOutputIds().getShape(2)); finished = flat.getFinished(0);
                    }
                    assertTrue(finished); assertEquals(expected, sum);
                }
            }
        }
    }

    @Test void ignoreEosKeepsAnExplicitOutputLengthAuthoritative() throws Exception {
        try (var p = node("prefill", 8, 1, Map.of("generation", Map.of("eos_probability", 1))); var d = node("decode", 8, 1)) {
            var input = input(d, 1, 3, 4, 2000, true).toBuilder(); input.getGenerateConfigBuilder().setIgnoreEos(true);
            try (var call = new Call(p, input.build())) { for (int n = 1; n <= 4; n++) call.output(n, n == 4, 1); assertEquals("completed", call.next()); }
        }
    }

    @Test void averageDecodeProgressDoesNotInventSpeculativeCompute() throws Exception {
        try (var p = node("prefill", 16, 1); var d = node("decode", 16, 1,
                Map.of("decode", Map.of("tokens_per_step", 2.6)));
                var call = new Call(p, input(d, 1, 3, 8, 2000, true))) {
            call.output(1, false, 1);
            call.output(4, false, 3);
            call.output(7, false, 3);
            call.output(8, true, 1);
            assertEquals("completed", call.next());
            assertEquals(3, metric(d, "mock_engine_forward_seconds_count", "role=\"decode\""));
            assertEquals(3, metric(d, "mock_engine_forward_tokens_total", "kind=\"compute\""));
            assertEquals(7, metric(d, "mock_engine_forward_tokens_total", "kind=\"output\""));
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

    @Test void growthPressureFailsLikeRtpAndCannotResumeAfterRelease() throws Exception {
        try (var p = node("prefill", 8, 1); var d = node("decode", 4, 10); var holder = d.node.cache().acquire(2, List.of()).lease();
                var call = new Call(p, input(d, 1, 3, 6, 2500, true))) {
            call.output(1, false, 1); call.output(2, false, 1);
            assertEquals(602, code(call.error(Status.Code.RESOURCE_EXHAUSTED)));
            assertEquals(4, d.status().getAvailableKvCache()); holder.close();
            assertNull(call.outputs.poll(150, TimeUnit.MILLISECONDS));
            assertEquals(8, d.status().getAvailableKvCache());
            assertTrue(d.node.cache().snapshot(-1, true).keys().isEmpty());
            assertEquals(1, metric(d, "mock_engine_forward_seconds_count", "role=\"decode\""));
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
        try (var p = node("prefill", 8, 1); var d = node("decode", 4, 1); var holder = d.node.cache().acquire(3, List.of()).lease();
                var call = new Call(p, input(d, 1, 3, 6, 300, true))) {
            await(() -> d.status().getWaitingQueryLen() == 1);
            call.error(Status.Code.DEADLINE_EXCEEDED);
            await(() -> d.status().getWaitingQueryLen() == 0); holder.close();
            await(() -> d.status().getAvailableKvCache() == 8);
            assertEquals(0, metric(d, "mock_engine_forward_seconds_count", "role=\"decode\""));
            assertEquals(0, metric(p, "mock_engine_forward_seconds_count", "role=\"prefill\""));
            assertNull(call.outputs.poll(150, TimeUnit.MILLISECONDS));
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

    @Test void slowConsumerDoesNotHoldDecodeCompletionOrKv() throws Exception {
        try (var p = node("prefill", 8, 0); var d = node("decode", 2048, 0); var context = Context.current().withCancellation()) {
            var replies = new LinkedBlockingQueue<Object>();
            context.run(() -> RpcServiceGrpc.newStub(p.channel).generateStreamCall(input(d, 1, 3, 3000, 5000, false),
                new io.grpc.stub.ClientResponseObserver<GenerateInputPB, GenerateOutputsPB>() {
                    @Override public void beforeStart(io.grpc.stub.ClientCallStreamObserver<GenerateInputPB> request) { request.disableAutoRequestWithInitial(1); }
                    @Override public void onNext(GenerateOutputsPB value) { replies.add(value); }
                    @Override public void onError(Throwable error) { replies.add(error); }
                    @Override public void onCompleted() { replies.add("completed"); }
                }));
            assertInstanceOf(GenerateOutputsPB.class, replies.poll(3, TimeUnit.SECONDS));
            await(() -> d.status().getFinishedTaskListCount() == 1 && p.status().getAvailableKvCache() == 16 && d.status().getAvailableKvCache() == 4096);
            assertFalse(d.status().getFinishedTaskList(0).hasErrorInfo());
            assertEquals(2999, metric(d, "mock_engine_forward_seconds_count", "role=\"decode\""));
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
