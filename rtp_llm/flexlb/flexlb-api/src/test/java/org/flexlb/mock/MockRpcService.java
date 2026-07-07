package org.flexlb.mock;

import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RoleTypeProtoConverter;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Mock implementation of {@link RpcServiceGrpc.RpcServiceImplBase}.
 *
 * <p>Implements only the RPCs needed by FlexLB's batch scheduler:
 * <ul>
 *   <li>{@code enqueueBatch()} — records requests, applies configured delay, returns success/error</li>
 *   <li>{@code cancel()} — records cancel calls; may skip response if {@code ignoreCancel=true}</li>
 *   <code>getWorkerStatus()} — returns configurable {@link EngineRpcService.WorkerStatusPB}</code>
 *   <li>{@code getCacheStatus()} — returns configurable {@link EngineRpcService.CacheStatusPB}</li>
 *   <li>{@code checkHealth()} — always returns healthy</li>
 * </ul>
 * All other RPCs fall through to the default (UNIMPLEMENTED) behavior.
 *
 * <p>All call records use thread-safe collections ({@link CopyOnWriteArrayList},
 * {@link java.util.concurrent.ConcurrentHashMap}) so they can be read from test threads
 * for assertion without additional synchronization.
 */
public class MockRpcService extends RpcServiceGrpc.RpcServiceImplBase {

    private static final Logger log = LoggerFactory.getLogger(MockRpcService.class);

    // ==================== Call records (thread-safe, for assertions) ====================

    /** All EnqueueBatch requests received, in arrival order. */
    final CopyOnWriteArrayList<EngineRpcService.EnqueueBatchRequestPB> enqueuedRequests = new CopyOnWriteArrayList<>();

    /** All Cancel requests received, in arrival order. */
    final CopyOnWriteArrayList<Long> cancelledRequests = new CopyOnWriteArrayList<>();

    /** Counter for GetWorkerStatus calls. */
    final AtomicLong workerStatusCallCount = new AtomicLong(0);

    /** Counter for GetCacheStatus calls. */
    final AtomicLong cacheStatusCallCount = new AtomicLong(0);

    /** Counter for CheckHealth calls. */
    final AtomicLong healthCheckCount = new AtomicLong(0);

    // ==================== Behavior (volatile, hot-swappable) ====================

    private volatile MockWorkerBehavior behavior = MockWorkerBehavior.builder().build();

    /**
     * Update behavior at runtime.  Thread-safe — the next RPC will see the new config.
     */
    public void setBehavior(MockWorkerBehavior behavior) {
        this.behavior = behavior;
    }

    public MockWorkerBehavior getBehavior() {
        return behavior;
    }

    // ==================== Accessors for test assertions ====================

    public List<EngineRpcService.EnqueueBatchRequestPB> getEnqueuedRequests() {
        return List.copyOf(enqueuedRequests);
    }

    public List<Long> getCancelledRequests() {
        return List.copyOf(cancelledRequests);
    }

    public int getEnqueueCount() {
        return enqueuedRequests.size();
    }

    public int getCancelCount() {
        return cancelledRequests.size();
    }

    public long getWorkerStatusCallCount() {
        return workerStatusCallCount.get();
    }

    public long getCacheStatusCallCount() {
        return cacheStatusCallCount.get();
    }

    public long getHealthCheckCount() {
        return healthCheckCount.get();
    }

    /** Clear all call records (useful between test cases sharing a worker). */
    public void resetRecords() {
        enqueuedRequests.clear();
        cancelledRequests.clear();
        workerStatusCallCount.set(0);
        cacheStatusCallCount.set(0);
        healthCheckCount.set(0);
    }

    // ==================== RPC implementations ====================

    @Override
    public void enqueueBatch(EngineRpcService.EnqueueBatchRequestPB request,
                            StreamObserver<EngineRpcService.EnqueueBatchResponsePB> responseObserver) {
        enqueuedRequests.add(request);
        MockWorkerBehavior beh = behavior;
        log.info("MockRpcService enqueueBatch: batch_id={}, dp_slots={}, delay_ms={}, fail={}",
                request.getBatchId(), request.getDpSlotsCount(), beh.getEnqueueDelayMs(), beh.isFailOnEnqueue());

        if (beh.getEnqueueDelayMs() > 0) {
            try {
                Thread.sleep(beh.getEnqueueDelayMs());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        EngineRpcService.EnqueueBatchResponsePB.Builder responseBuilder =
                EngineRpcService.EnqueueBatchResponsePB.newBuilder()
                        .setBatchId(request.getBatchId());

        if (beh.isFailOnEnqueue()) {
            for (EngineRpcService.EnqueueBatchDpSlotPB slot : request.getDpSlotsList()) {
                for (EngineRpcService.EnqueueBatchExternalInputPB ext : slot.getRequestsList()) {
                    long reqId = ext.getInput().getRequestId();
                    responseBuilder.addErrors(EngineRpcService.EnqueueBatchErrorPB.newBuilder()
                            .setRequestId(reqId)
                            .setErrorInfo(EngineRpcService.ErrorDetailsPB.newBuilder()
                                    .setErrorCode(beh.getEnqueueErrorCode())
                                    .setErrorMessage(beh.getEnqueueErrorMessage())
                                    .build())
                            .build());
                }
            }
        } else {
            for (EngineRpcService.EnqueueBatchDpSlotPB slot : request.getDpSlotsList()) {
                for (EngineRpcService.EnqueueBatchExternalInputPB ext : slot.getRequestsList()) {
                    long reqId = ext.getInput().getRequestId();
                    responseBuilder.addSuccesses(EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                            .setRequestId(reqId)
                            .build());
                }
            }
        }

        responseObserver.onNext(responseBuilder.build());
        responseObserver.onCompleted();
    }

    @Override
    public void cancel(EngineRpcService.CancelRequestPB request,
                       StreamObserver<EngineRpcService.EmptyPB> responseObserver) {
        cancelledRequests.add(request.getRequestId());
        MockWorkerBehavior beh = behavior;
        log.info("MockRpcService cancel: request_id={}, ignore={}",
                request.getRequestId(), beh.isIgnoreCancel());

        if (beh.isIgnoreCancel()) {
            // Don't respond — simulate a worker that never acks cancel.
            // The scheduler's TTL or gRPC deadline will eventually time out.
            return;
        }

        responseObserver.onNext(EngineRpcService.EmptyPB.getDefaultInstance());
        responseObserver.onCompleted();
    }

    @Override
    public void getWorkerStatus(EngineRpcService.StatusVersionPB request,
                                StreamObserver<EngineRpcService.WorkerStatusPB> responseObserver) {
        workerStatusCallCount.incrementAndGet();
        MockWorkerBehavior beh = behavior;

        EngineRpcService.WorkerStatusPB.Builder builder = EngineRpcService.WorkerStatusPB.newBuilder()
                .setAlive(true)
                .setRole(RoleTypeProtoConverter.fromProto(beh.getRoleType()).getCode())
                .setRoleType(beh.getRoleType())
                .setAvailableConcurrency(beh.getAvailableConcurrency())
                .setAvailableKvCache(beh.getAvailableKvCache())
                .setTotalKvCache(beh.getTotalKvCache())
                .setStatusVersion(request.getLatestCacheVersion() + 1)
                .setLatestFinishedVersion(request.getLatestFinishedVersion())
                .setDpSize(1)
                .setTpSize(1)
                .setDpRank(0);

        responseObserver.onNext(builder.build());
        responseObserver.onCompleted();
    }

    @Override
    public void getCacheStatus(EngineRpcService.CacheVersionPB request,
                               StreamObserver<EngineRpcService.CacheStatusPB> responseObserver) {
        cacheStatusCallCount.incrementAndGet();
        MockWorkerBehavior beh = behavior;

        EngineRpcService.CacheStatusPB.Builder builder = EngineRpcService.CacheStatusPB.newBuilder()
                .setAvailableKvCache(beh.getAvailableKvCache())
                .setTotalKvCache(beh.getTotalKvCache())
                .setBlockSize(1024)
                .setVersion(request.getLatestCacheVersion() + 1);

        responseObserver.onNext(builder.build());
        responseObserver.onCompleted();
    }

    @Override
    public void checkHealth(EngineRpcService.EmptyPB request,
                            StreamObserver<EngineRpcService.CheckHealthResponsePB> responseObserver) {
        healthCheckCount.incrementAndGet();
        responseObserver.onNext(EngineRpcService.CheckHealthResponsePB.newBuilder()
                .setHealth("ok")
                .build());
        responseObserver.onCompleted();
    }
}
