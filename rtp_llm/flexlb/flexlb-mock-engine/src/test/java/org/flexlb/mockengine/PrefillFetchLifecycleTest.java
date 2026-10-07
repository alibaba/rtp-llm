package org.flexlb.mockengine;

import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.flexlb.mockengine.MockEngineTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;

/** Tests the engine protocol, independently of master routing or client timing. */
class PrefillFetchLifecycleTest {
    @TempDir Path tempDir;

    private MockEngineTestCluster cluster(String prefillMs) throws Exception {
        return MockEngineTestCluster.create(
                performanceModel(tempDir, prefillMs, 1, 5), 62100, 1, 1);
    }

    private void submit(MockEngineTestCluster c, long id, long ttlMs) {
        var ack = enqueue(c.prefill(0), batch(id, slot(0,
                inputWithDecode(id, 2048, c.decode(0).getGrpcPort(), 8)))
                .toBuilder().setFetchAttachTimeoutMs(ttlMs).build());
        assertEquals(1, ack.getSuccessesCount(), ack.toString());
        assertEquals(0, ack.getErrorsCount());
    }

    private static long value(JavaMockEngineCluster.FastRpcService s, String key) {
        return ((Number) s.getSnapshot().get(key)).longValue();
    }

    private static final class Output implements StreamObserver<EngineRpcService.GenerateOutputsPB> {
        final List<EngineRpcService.GenerateOutputsPB> frames = new CopyOnWriteArrayList<>();
        final CountDownLatch done = new CountDownLatch(1);
        volatile Throwable error;
        public void onNext(EngineRpcService.GenerateOutputsPB v) { frames.add(v); }
        public void onError(Throwable t) { error = t; done.countDown(); }
        public void onCompleted() { done.countDown(); }
        void success() throws Exception {
            assertTrue(done.await(4, TimeUnit.SECONDS), "Fetch never terminated");
            assertNull(error);
            assertFalse(frames.isEmpty());
            assertTrue(frames.stream().noneMatch(EngineRpcService.GenerateOutputsPB::hasErrorInfo));
            assertTrue(frames.get(frames.size() - 1).getFlattenOutput().getFinished(0));
        }
    }

    private Output fetch(MockEngineTestCluster c, long id) {
        Output output = new Output();
        c.prefill(0).fetchResponse(EngineRpcService.FetchRequestPB.newBuilder()
                .setRequestId(id).build(), output);
        return output;
    }

    private void released(MockEngineTestCluster c) throws Exception {
        c.awaitAllInflightZero(4000);
        for (var s : c.services().values()) {
            assertEquals(0, value(s, "prefill_contexts"));
            assertEquals(0, value(s, "decode_waiting_for_kv"));
            assertEquals(0, value(s, "held_blocks"));
            assertEquals(0, value(s, "referenced_blocks"));
        }
    }

    @Test void allocateIsReportedBeforePrefillAndLateFetchStartsDecode() throws Exception {
        try (var c = cluster("250")) {
            submit(c, 1, 3000);
            assertEquals(1, c.decode(0).getInflightCount());
            assertEquals(1, value(c.decode(0), "decode_waiting_for_kv"));
            var status = workerStatus(c.decode(0), 0);
            assertTrue(status.toString().contains("TASK_PHASE_KV_ALLOCATED"), status.toString());
            assertEquals(0, c.decode(0).getCompletedCount());
            c.awaitNoInflight(c.prefill(0), 2000);
            assertEquals(1, value(c.prefill(0), "prefill_contexts"));
            assertTrue(value(c.prefill(0), "held_blocks") > 0);
            assertEquals(1, value(c.decode(0), "decode_waiting_for_kv"));
            assertEquals(0, c.decode(0).getCompletedCount());
            fetch(c, 1).success();
            assertEquals(1, c.decode(0).getCompletedCount());
            released(c);
        }
    }

    @Test void earlyFetchAndDuplicateAttachment() throws Exception {
        try (var c = cluster("250")) {
            submit(c, 2, 3000);
            Output first = fetch(c, 2);
            Output duplicate = fetch(c, 2);
            assertEquals(Status.Code.NOT_FOUND, Status.fromThrowable(duplicate.error).getCode());
            assertEquals(0, c.decode(0).getCompletedCount());
            first.success();
            assertEquals(1, c.decode(0).getCompletedCount());
            released(c);
        }
    }

    @Test void missingFetchExpiresWithoutAnyDecodeExecution() throws Exception {
        try (var c = cluster("30")) {
            submit(c, 3, 200);
            c.awaitNoInflight(c.prefill(0), 1000);
            assertEquals(1, value(c.decode(0), "decode_waiting_for_kv"));
            released(c);
            assertEquals(1, value(c.prefill(0), "fetch_attach_expirations"));
            assertEquals(0, c.decode(0).getCompletedCount());
            Output late = fetch(c, 3);
            assertEquals(Status.Code.NOT_FOUND, Status.fromThrowable(late.error).getCode());
        }
    }

    @Test void automaticModeCompletesWithoutFetchOrResponseBuffers() throws Exception {
        try (var c = cluster("30")) {
            c.prefill(0).setAutoFetch(true);
            submit(c, 4, 1); // Missing-fetch TTL does not apply to automatic mode.
            c.awaitCompleted(1, 2000);
            released(c);
            assertEquals(0L, ((Number) ((Map<?, ?>) c.prefill(0).getSnapshot().get("rpc_counts")).get("fetch_response")).longValue());
            assertEquals(0, value(c.prefill(0), "fetch_attach_expirations"));
        }
    }

    @Test void cancelAfterPrefillReleasesPreparedDecode() throws Exception {
        try (var c = cluster("30")) {
            submit(c, 5, 3000);
            c.awaitNoInflight(c.prefill(0), 1000);
            c.prefill(0).cancel(5);
            released(c);
            assertEquals(0, c.decode(0).getCompletedCount());
        }
    }

    @Test void failedFetchAfterPrefillConsumesContextAndReleasesDecode() throws Exception {
        try (var c = cluster("30")) {
            submit(c, 13, 3000);
            c.awaitNoInflight(c.prefill(0), 1000);
            c.prefill(0).setFaultConfig(FaultInjectionConfig.builder().fetchError(true).build());
            Output output = fetch(c, 13);
            assertTrue(output.done.await(1, TimeUnit.SECONDS));
            assertNotNull(output.error);
            assertTrue(output.error.getMessage().contains("injected fetch_error"));
            released(c);
            assertEquals(0, value(c.prefill(0), "response_buffers"));
            assertEquals(0, value(c.prefill(0), "fetch_attach_expirations"));
            assertEquals(0, c.decode(0).getCompletedCount());
            c.prefill(0).clearFaultConfig();
            assertEquals(Status.Code.NOT_FOUND, Status.fromThrowable(fetch(c, 13).error).getCode());
        }
    }

    @Test void decodeCancellationBeforePrefillCannotResurrectRequest() throws Exception {
        try (var c = cluster("150")) {
            submit(c, 6, 3000);
            c.decode(0).cancel(6);
            Thread.sleep(250); // Observe the already-scheduled P completion callback.
            released(c);
            assertEquals(0, c.decode(0).getCompletedCount());
            assertEquals(0, c.prefill(0).getCompletedCount());
        }
    }

    @Test void nonBatchStreamAlreadyProvidesTheFetchContinuation() throws Exception {
        try (var c = cluster("30")) {
            Output output = new Output();
            c.prefill(0).generateStreamCall(inputWithDecode(7, 2048,
                    c.decode(0).getGrpcPort(), 8), output);
            output.success();
            released(c);
        }
    }

    @Test void localBatchAlsoRetainsItsDeferredContextUntilFetch() throws Exception {
        try (var c = cluster("30")) {
            var ack = enqueue(c.prefill(0), batch(8, slot(0, input(8, 2048))));
            assertEquals(1, ack.getSuccessesCount());
            c.awaitNoInflight(c.prefill(0), 1000);
            assertEquals(1, value(c.prefill(0), "prefill_contexts"));
            fetch(c, 8).success();
            released(c);
        }
    }
    @Test void decodeLackOfKvRejectsBeforePrefillExecutionAndRollsBack() throws Exception {
        try (var c = cluster("30")) {
            var input = inputWithDecode(9, 10, c.decode(0).getGrpcPort(), 8);
            input = input.toBuilder().setGenerateConfig(input.getGenerateConfig().toBuilder()
                    .setUniqueKey("{\"input_len\":7000000,\"block_cache_keys\":[91]}"))
                    .build();
            var ack = enqueue(c.prefill(0), batch(9, slot(0, input)));
            assertEquals(0, ack.getSuccessesCount());
            assertEquals(8211, ack.getErrors(0).getErrorInfo().getErrorCode());
            assertEquals(0, value(c.prefill(0), "prefill_batches"));
            assertEquals(0, c.decode(0).getAcceptedCount());
            released(c);
        }
    }

    @Test void ttlDuringPrefillAndRepeatedCancelCannotResurrectDecode() throws Exception {
        try (var c = cluster("150")) {
            submit(c, 10, 20);
            c.awaitAllInflightZero(1000);
            c.prefill(0).cancel(10);
            c.decode(0).cancel(10);
            Thread.sleep(200);
            released(c);
            assertEquals(0, c.decode(0).getCompletedCount());
            assertEquals(0, c.decode(0).getActiveDecodeCount());
        }
    }

    @Test void pendingFetchIsNotALeakAndDecodeCancelKeepsOriginalFailureFrame() throws Exception {
        try (var c = cluster("150")) {
            submit(c, 11, 3000);
            c.decode(0).checkLeakDrain(0);
            assertFalse(c.decode(0).isLeakDetected());
            Output output = fetch(c, 11);
            c.decode(0).cancel(11);
            assertTrue(output.done.await(2, TimeUnit.SECONDS));
            assertTrue(output.frames.stream().anyMatch(EngineRpcService.GenerateOutputsPB::hasErrorInfo));
            Thread.sleep(200);
            released(c);
        }
    }

    @Test void duplicatePrepareCannotOverwriteAnExecutingDecode() throws Exception {
        try (var c = cluster("30")) {
            Output original = new Output();
            c.decode(0).generateStreamCall(inputWithDecode(12, 2048,
                    c.decode(0).getGrpcPort(), 40), original);
            var ack = enqueue(c.prefill(0), batch(12, slot(0,
                    inputWithDecode(12, 2048, c.decode(0).getGrpcPort(), 8))));
            assertEquals(0, ack.getSuccessesCount());
            assertEquals(1, c.decode(0).getInflightCount());
            assertEquals(1, c.decode(0).getActiveDecodeCount());
            original.success();
            released(c);
        }
    }

}
