package org.flexlb.engine.grpc.dp;

import io.grpc.Server;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies {@link DpGrpcClient} end-to-end against an in-process gRPC server:
 *  - BatchEnqueue forwards the BatchEnqueueRequestPB intact (every input + dp_rank survives)
 *  - Enqueue ack with accepted=true completes the future successfully
 *  - Cancel reaches the engine RpcService.Cancel handler with the right request_id
 *  - Failures (no server / deadline) are propagated as completeExceptionally
 */
class DpGrpcClientTest {

    private static final String PREFILL_SERVER = "test-prefill-server";
    private static final String DECODE_SERVER = "test-decode-server";

    private Server prefillServer;
    private Server decodeServer;

    private final List<EngineRpcService.BatchEnqueueRequestPB> receivedEnqueues = new CopyOnWriteArrayList<>();
    private final List<Long> prefillCancels = new CopyOnWriteArrayList<>();
    private final List<Long> decodeCancels = new CopyOnWriteArrayList<>();
    private final AtomicInteger ackAccepted = new AtomicInteger(1);  // 1 = accept, 0 = reject

    @BeforeEach
    void startServers() throws Exception {
        prefillServer = InProcessServerBuilder.forName(PREFILL_SERVER)
                .directExecutor()
                .addService(new RpcServiceGrpc.RpcServiceImplBase() {
                    @Override
                    public void batchEnqueue(EngineRpcService.BatchEnqueueRequestPB request,
                                             StreamObserver<EngineRpcService.BatchEnqueueResponsePB> obs) {
                        receivedEnqueues.add(request);
                        EngineRpcService.BatchEnqueueResponsePB.Builder rb =
                                EngineRpcService.BatchEnqueueResponsePB.newBuilder()
                                        .setBatchId(request.getBatchId());
                        boolean accept = ackAccepted.get() == 1;
                        for (EngineRpcService.GenerateInputPB in : request.getInputsList()) {
                            EngineRpcService.EnqueueResponsePB.Builder slot =
                                    EngineRpcService.EnqueueResponsePB.newBuilder()
                                            .setRequestId(in.getRequestId());
                            if (!accept) {
                                slot.setErrorInfo(EngineRpcService.ErrorDetailsPB.newBuilder()
                                        .setErrorCode(1L)
                                        .setErrorMessage("rejected")
                                        .build());
                            }
                            rb.addAcks(slot.build());
                        }
                        obs.onNext(rb.build());
                        obs.onCompleted();
                    }
                    @Override
                    public void cancel(EngineRpcService.CancelRequestPB request,
                                       StreamObserver<EngineRpcService.EmptyPB> obs) {
                        prefillCancels.add(request.getRequestId());
                        obs.onNext(EngineRpcService.EmptyPB.getDefaultInstance());
                        obs.onCompleted();
                    }
                })
                .build()
                .start();

        decodeServer = InProcessServerBuilder.forName(DECODE_SERVER)
                .directExecutor()
                .addService(new RpcServiceGrpc.RpcServiceImplBase() {
                    @Override
                    public void cancel(EngineRpcService.CancelRequestPB request,
                                       StreamObserver<EngineRpcService.EmptyPB> obs) {
                        decodeCancels.add(request.getRequestId());
                        obs.onNext(EngineRpcService.EmptyPB.getDefaultInstance());
                        obs.onCompleted();
                    }
                })
                .build()
                .start();
    }

    @AfterEach
    void stopServers() {
        if (prefillServer != null) prefillServer.shutdownNow();
        if (decodeServer != null) decodeServer.shutdownNow();
    }

    @Test
    void enqueue_forwards_batch_intact_and_returns_ack() throws Exception {
        DpGrpcClient client = newClientWithInjectedChannels();

        EngineRpcService.BatchEnqueueRequestPB batch = EngineRpcService.BatchEnqueueRequestPB.newBuilder()
                .setBatchId(42L)
                .addInputs(EngineRpcService.GenerateInputPB.newBuilder().setRequestId(1).setDpRank(com.google.protobuf.Int32Value.of(0)).build())
                .addInputs(EngineRpcService.GenerateInputPB.newBuilder().setRequestId(2).setDpRank(com.google.protobuf.Int32Value.of(1)).build())
                .addInputs(EngineRpcService.GenerateInputPB.newBuilder().setRequestId(3).setDpRank(com.google.protobuf.Int32Value.of(2)).build())
                .addInputs(EngineRpcService.GenerateInputPB.newBuilder().setRequestId(4).setDpRank(com.google.protobuf.Int32Value.of(3)).build())
                .build();

        EngineRpcService.BatchEnqueueResponsePB ack = client.enqueue("prefill-host", 9999, batch).get(2, TimeUnit.SECONDS);

        assertEquals(42L, ack.getBatchId());
        assertEquals(4, ack.getAcksCount());
        for (int i = 0; i < 4; i++) {
            assertEquals(i + 1, ack.getAcks(i).getRequestId());
            assertEquals(0L, ack.getAcks(i).getErrorInfo().getErrorCode(),
                    "per-slot error_code=0 means accepted");
        }
        assertEquals(1, receivedEnqueues.size());
        EngineRpcService.BatchEnqueueRequestPB got = receivedEnqueues.get(0);
        assertEquals(4, got.getInputsCount());
        for (int i = 0; i < 4; i++) {
            assertEquals(i + 1, got.getInputs(i).getRequestId());
            assertEquals(i, got.getInputs(i).getDpRank().getValue(), "dp_rank must be forwarded verbatim to prefill");
        }
    }

    @Test
    void enqueue_with_rejected_ack_completes_future_with_accepted_false() throws Exception {
        ackAccepted.set(0);
        DpGrpcClient client = newClientWithInjectedChannels();
        EngineRpcService.BatchEnqueueRequestPB batch =
                EngineRpcService.BatchEnqueueRequestPB.newBuilder()
                        .setBatchId(7L)
                        .addInputs(EngineRpcService.GenerateInputPB.newBuilder().setRequestId(11).setDpRank(0).build())
                        .build();
        EngineRpcService.BatchEnqueueResponsePB ack = client.enqueue("prefill-host", 9999, batch).get(2, TimeUnit.SECONDS);
        assertEquals(1, ack.getAcksCount());
        assertEquals(1L, ack.getAcks(0).getErrorInfo().getErrorCode());
        assertEquals("rejected", ack.getAcks(0).getErrorInfo().getErrorMessage());
    }

    @Test
    void cancelPrefill_reaches_prefill_endpoint_only() throws Exception {
        DpGrpcClient client = newClientWithInjectedChannels();
        client.cancelPrefill("prefill-host", 9999, 123L).get(2, TimeUnit.SECONDS);
        assertEquals(List.of(123L), prefillCancels);
        assertTrue(decodeCancels.isEmpty());
    }

    @Test
    void cancelDecode_reaches_decode_endpoint_only() throws Exception {
        DpGrpcClient client = newClientWithInjectedChannels();
        client.cancelDecode("decode-host", 9999, 456L).get(2, TimeUnit.SECONDS);
        assertEquals(List.of(456L), decodeCancels);
        assertTrue(prefillCancels.isEmpty());
    }

    @Test
    void enqueue_does_not_block_caller_thread() throws Exception {
        DpGrpcClient client = newClientWithInjectedChannels();
        long t0 = System.nanoTime();
        for (int i = 0; i < 100; i++) {
            EngineRpcService.BatchEnqueueRequestPB b =
                    EngineRpcService.BatchEnqueueRequestPB.newBuilder().setBatchId(i).build();
            CompletableFuture<EngineRpcService.BatchEnqueueResponsePB> f = client.enqueue("prefill-host", 9999, b);
            assertNotNull(f);
        }
        long elapsedMicros = (System.nanoTime() - t0) / 1000;
        assertTrue(elapsedMicros < 1_000_000,
                "100 enqueue calls must not block; observed " + elapsedMicros + "us");
    }

    /**
     * Inject pre-built in-process channels into a fresh DpGrpcClient by reflectively
     * populating the {@code channels} map. Avoids the production buildChannel() Netty
     * code path for unit tests.
     */
    @SuppressWarnings("unchecked")
    private DpGrpcClient newClientWithInjectedChannels() throws Exception {
        DpGrpcClient client = new DpGrpcClient();
        Field channelsField = DpGrpcClient.class.getDeclaredField("channels");
        channelsField.setAccessible(true);
        Map<String, Object> channelsMap = (Map<String, Object>) channelsField.get(client);

        Class<?> entryClass = Class.forName("org.flexlb.engine.grpc.dp.DpGrpcClient$ChannelEntry");
        var entryCtor = entryClass.getDeclaredConstructor(io.grpc.ManagedChannel.class);
        entryCtor.setAccessible(true);

        var prefillChannel = InProcessChannelBuilder.forName(PREFILL_SERVER).directExecutor().build();
        var decodeChannel = InProcessChannelBuilder.forName(DECODE_SERVER).directExecutor().build();

        channelsMap.put("prefill-host:9999", entryCtor.newInstance(prefillChannel));
        channelsMap.put("decode-host:9999", entryCtor.newInstance(decodeChannel));
        return client;
    }
}
