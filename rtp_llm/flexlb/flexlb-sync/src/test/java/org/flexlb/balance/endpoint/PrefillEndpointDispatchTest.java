package org.flexlb.balance.endpoint;

import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import org.flexlb.balance.scheduler.BatchIdGenerator;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.DispatchMeta;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Dispatch-pipeline tests for {@link PrefillEndpoint#submitBatch}.
 *
 * <p>Migrated from the deleted {@code DefaultBatchDispatcherTest}: the async
 * gRPC dispatch now lives on the endpoint, and per-item outcomes are settled
 * through {@link BatchItem} terminal transitions instead of scheduler
 * callbacks, so assertions check the item futures directly.
 */
class PrefillEndpointDispatchTest {

    private FlexlbConfig config;
    private EngineGrpcClient grpcClient;
    private BatchDispatchExecutor dispatchExecutor;
    private PrefillEndpoint endpoint;
    private final List<EngineRpcService.EnqueueBatchRequestPB> sentBatches = new CopyOnWriteArrayList<>();

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
        config.setCostFormula("sum(computeTokens)");
        config.setFlexlbBatchEnqueueDeadlineMs(5_000L);

        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        dispatchExecutor = new BatchDispatchExecutor(configService, null);

        grpcClient = mock(EngineGrpcClient.class);
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(),
                any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    EngineRpcService.EnqueueBatchRequestPB request = inv.getArgument(2);
                    sentBatches.add(request);
                    return CompletableFuture.completedFuture(ackFor(request));
                });

        WorkerStatus status = new WorkerStatus();
        status.setIp("127.0.0.1");
        status.setPort(8080);
        status.setGrpcPort(8090);
        status.setRole(RoleType.PREFILL);

        endpoint = new PrefillEndpoint(status, config, grpcClient, dispatchExecutor,
                new BatchIdGenerator("127.0.0.1", 7001), () -> 3,
                mock(BatchSchedulerReporter.class), null);
    }

    @AfterEach
    void tearDown() {
        endpoint.close();
        dispatchExecutor.shutdown();
    }

    // ==================== success path ====================

    @Test
    void ackSuccessCompletesAllItemFutures() throws Exception {
        BatchItem first = createItem(1L);
        BatchItem second = createItem(2L);

        endpoint.submitBatch(List.of(first, second), meta());

        Response firstResp = first.future().get(2, TimeUnit.SECONDS);
        Response secondResp = second.future().get(2, TimeUnit.SECONDS);
        assertTrue(firstResp.isSuccess());
        assertTrue(secondResp.isSuccess());
        assertTrue(firstResp.isEnqueuedByMaster());
        assertEquals(3, firstResp.getQueueLength());

        assertEquals(1, sentBatches.size());
        // Batch stays inflight until worker-status calibration removes it
        assertEquals(1, trackedEntryCount());
        // submitBatch assigned the generated batch ID to every item
        assertEquals(first.assignedBatchId(), sentBatches.getFirst().getBatchId());
    }

    @Test
    void alreadyCompletedItemsAreNotDispatched() {
        BatchItem item = createItem(1L);
        // Settle through the item's own terminal method (queue-expiry path)
        // instead of completing the future externally.
        item.failExpired();

        endpoint.submitBatch(List.of(item), meta());

        verify(grpcClient, never()).batchEnqueueAsync(anyString(), anyInt(), any(), anyLong());
        assertEquals(0, trackedEntryCount());
    }

    // ==================== gRPC failure paths ====================

    @Test
    void grpcErrorFailsAllItemsAndReleasesBatch() throws Exception {
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(),
                any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenReturn(CompletableFuture.failedFuture(new RuntimeException("connection reset")));

        BatchItem item = createItem(1L);
        endpoint.submitBatch(List.of(item), meta());

        Response response = item.future().get(2, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertTrue(response.getErrorMessage().contains("gRPC dispatch failed"));
        assertEquals(0, trackedEntryCount());
    }

    @Test
    void deadlineExceededFailsItemsAsTimeout() throws Exception {
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(),
                any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenReturn(CompletableFuture.failedFuture(
                        new StatusRuntimeException(Status.DEADLINE_EXCEEDED)));

        BatchItem item = createItem(1L);
        endpoint.submitBatch(List.of(item), meta());

        Response response = item.future().get(2, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertTrue(response.getErrorMessage().contains("EnqueueBatch deadline exceeded"));
        assertEquals(0, trackedEntryCount());
    }

    @Test
    void nullResponseFailsAllItems() throws Exception {
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(),
                any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(null));

        BatchItem item = createItem(1L);
        endpoint.submitBatch(List.of(item), meta());

        Response response = item.future().get(2, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertTrue(response.getErrorMessage().contains("null response"));
        assertEquals(0, trackedEntryCount());
    }

    // ==================== response parsing paths ====================

    @Test
    void batchIdMismatchFailsAllItems() throws Exception {
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(),
                any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    EngineRpcService.EnqueueBatchRequestPB request = inv.getArgument(2);
                    EngineRpcService.EnqueueBatchResponsePB response =
                            EngineRpcService.EnqueueBatchResponsePB.newBuilder()
                                    .setBatchId(request.getBatchId() + 1)
                                    .build();
                    return CompletableFuture.completedFuture(response);
                });

        BatchItem item = createItem(1L);
        endpoint.submitBatch(List.of(item), meta());

        Response response = item.future().get(2, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertTrue(response.getErrorMessage().contains("batch_id mismatch"));
        assertEquals(0, trackedEntryCount());
    }

    @Test
    void responseErrorListFailsOnlyRejectedRequest() throws Exception {
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(),
                any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    EngineRpcService.EnqueueBatchRequestPB request = inv.getArgument(2);
                    EngineRpcService.EnqueueBatchResponsePB.Builder response =
                            EngineRpcService.EnqueueBatchResponsePB.newBuilder()
                                    .setBatchId(request.getBatchId());
                    response.addSuccesses(EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                            .setRequestId(1L).build());
                    response.addErrors(EngineRpcService.EnqueueBatchErrorPB.newBuilder()
                            .setRequestId(2L)
                            .setErrorInfo(EngineRpcService.ErrorDetailsPB.newBuilder()
                                    .setErrorCode(13)
                                    .setErrorMessage("decode alloc failed")
                                    .build())
                            .build());
                    return CompletableFuture.completedFuture(response.build());
                });

        BatchItem first = createItem(1L);
        BatchItem second = createItem(2L);
        endpoint.submitBatch(List.of(first, second), meta());

        assertTrue(first.future().get(2, TimeUnit.SECONDS).isSuccess());
        Response rejected = second.future().get(2, TimeUnit.SECONDS);
        assertFalse(rejected.isSuccess());
        assertTrue(rejected.getErrorMessage().contains("decode alloc failed"));
        // Rejected item repacked out of the batch; survivor keeps the entry inflight
        assertEquals(1, trackedEntryCount());
        assertEquals(1, endpoint.prefillPendingRequestCount());
    }

    @Test
    void missingAckFailsMissingRequest() throws Exception {
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(),
                any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    EngineRpcService.EnqueueBatchRequestPB request = inv.getArgument(2);
                    EngineRpcService.EnqueueBatchResponsePB response =
                            EngineRpcService.EnqueueBatchResponsePB.newBuilder()
                                    .setBatchId(request.getBatchId())
                                    .addSuccesses(EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                                            .setRequestId(1L).build())
                                    .build();
                    return CompletableFuture.completedFuture(response);
                });

        BatchItem first = createItem(1L);
        BatchItem second = createItem(2L);
        endpoint.submitBatch(List.of(first, second), meta());

        assertTrue(first.future().get(2, TimeUnit.SECONDS).isSuccess());
        Response missing = second.future().get(2, TimeUnit.SECONDS);
        assertFalse(missing.isSuccess());
        assertTrue(missing.getErrorMessage().contains("EnqueueBatch missing ack for request 2"));
    }

    // ==================== executor shutdown ====================

    @Test
    void executorShutdownRejectsDispatchAndFailsItems() throws Exception {
        dispatchExecutor.shutdown();

        BatchItem item = createItem(1L);
        endpoint.submitBatch(List.of(item), meta());

        Response response = item.future().get(2, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(0, trackedEntryCount());
        verify(grpcClient, never()).batchEnqueueAsync(anyString(), anyInt(), any(), anyLong());
    }

    @Test
    void endpointCloseDrainsBatcherQueueWithFailure() throws Exception {
        // Prevent immediate dispatch so the item stays queued in the batcher
        config.setFlexlbBatchSizeMax(1000);
        config.setFlexlbBatchFixedWaitMs(60_000);

        BatchItem item = createItem(1L);
        endpoint.getBatcher().offer(item);

        endpoint.close();

        Response response = item.future().get(2, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        verify(grpcClient, never()).batchEnqueueAsync(anyString(), anyInt(), any(), anyLong());
    }

    // ==================== helpers ====================

    private int trackedEntryCount() {
        return endpoint.prefillInflightCount() + endpoint.prefillEngineWorkCount();
    }

    private static DispatchMeta meta() {
        return new DispatchMeta("batch_full");
    }

    private static EngineRpcService.EnqueueBatchResponsePB ackFor(
            EngineRpcService.EnqueueBatchRequestPB request) {
        EngineRpcService.EnqueueBatchResponsePB.Builder response =
                EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(request.getBatchId());
        request.getDpSlotsList().stream()
                .flatMap(slot -> slot.getRequestsList().stream())
                .map(external -> external.getInput().getRequestId())
                .forEach(requestId -> response.addSuccesses(
                        EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                                .setRequestId(requestId)
                                .build()));
        return response.build();
    }

    private BatchItem createItem(long requestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(128);
        request.setMaxNewTokens(8);

        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setConfig(new FlexlbConfig());
        ctx.setGenerateInputPbBytes(generateInputBytes(requestId));

        Response routeResponse = new Response();
        routeResponse.setSuccess(true);

        ServerStatus prefill = new ServerStatus();
        prefill.setRole(RoleType.PREFILL);
        prefill.setServerIp("127.0.0.1");
        prefill.setHttpPort(8080);
        prefill.setGrpcPort(8090);
        prefill.setDpRank(0);
        prefill.setRequestId(requestId);

        ServerStatus decode = new ServerStatus();
        decode.setRole(RoleType.DECODE);
        decode.setServerIp("127.0.0.2");
        decode.setHttpPort(8081);
        decode.setGrpcPort(8091);
        decode.setRequestId(requestId);

        return new BatchItem(ctx, new CompletableFuture<>(), routeResponse,
                prefill, decode, endpoint, null, System.currentTimeMillis());
    }

    private static byte[] generateInputBytes(long requestId) {
        EngineRpcService.GenerateInputPB input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .addTokenIds(101)
                .addTokenIds(102)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(8)
                        .build())
                .build();
        return input.toByteArray();
    }
}
