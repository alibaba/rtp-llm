package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillActiveIndex;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.DecodeCapacityAcquirer;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.master.WorkerStatusTestSupport;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/** Delivery and queued withdrawal arbitrate through the real exact Decode resource ledger. */
class QueuedDeliveryWithdrawalRaceTest {

    @Test
    @Timeout(15)
    void preparedDispatchExcludesWithdrawalUntilTheQueueCommitGetsItsSender() throws Exception {
        try (var fixture = new Fixture(); var executor = Executors.newSingleThreadExecutor()) {
            var prepared = new CountDownLatch(1);
            var allowCommit = new CountDownLatch(1);
            var selection = executor.submit(() -> fixture.select(evaluator(() -> {
                prepared.countDown();
                await(allowCommit);
            })));
            AdmissionHandle withdrawal = null;
            try {
                assertTrue(prepared.await(5, TimeUnit.SECONDS));
                assertEquals(1, fixture.decode.resourceSnapshot().activeDispatchPermits(),
                        "pause after the real delivery lease is acquired and before queue commit");
                withdrawal = fixture.owner.claimQueuedRoute(fixture.decode, fixture.reservation, 100);
                assertNull(withdrawal, "a prepared sender must exclude a concurrent queued withdrawal");
                assertEquals(ContextResult.Phase.OPEN, fixture.context.snapshot().phase());
                assertNotNull(fixture.context.snapshot().route());
                assertNull(fixture.context.snapshot().delivery());
                allowCommit.countDown();
                selection.get(5, TimeUnit.SECONDS);
                fixture.runtime.deliveries().awaitDeliveries();
                fixture.assertDelivered();
            } finally {
                allowCommit.countDown();
                if (withdrawal != null) { fixture.owner.completeWithdrawal(withdrawal, false); }
            }
        }
    }

    @Test
    @Timeout(15)
    void queuedWithdrawalExcludesPreparationAndItsRollbackRestoresDelivery() throws Exception {
        try (var fixture = new Fixture()) {
            var withdrawal = fixture.owner.claimQueuedRoute(fixture.decode, fixture.reservation, 100);
            assertNotNull(withdrawal, "an exact queued reservation without a dispatch lease remains withdrawable");
            try {
                assertNotNull(fixture.context.snapshot().placement());
                fixture.select(evaluator(() -> { throw new AssertionError("withdrawal must exclude preparation before prediction"); }));
                assertNull(fixture.context.snapshot().delivery());
                assertFalse(fixture.future.isDone());
                assertEquals(1, fixture.prefill.ownership.progress().queueDepth(), "the losing preparation must leave the route queued");
                assertEquals(0, fixture.prefill.ownership.stats().batchCount());
                assertEquals(0, fixture.decode.resourceSnapshot().activeDispatchPermits());
                verify(fixture.client, never()).batchEnqueueAsync(anyString(), anyInt(), any());
            } finally {
                fixture.owner.completeWithdrawal(withdrawal, false);
            }
            assertEquals(ContextResult.Phase.OPEN, fixture.context.snapshot().phase());
            assertNotNull(fixture.context.snapshot().route());
            assertNull(fixture.context.snapshot().delivery());
            fixture.select(evaluator(() -> { }));
            fixture.runtime.deliveries().awaitDeliveries();
            fixture.assertDelivered();
        }
    }

    private static PrefillTimePredictor.Evaluator evaluator(Runnable predictionBarrier) {
        return new PrefillTimePredictor.Evaluator() {
            @Override public long estimateMs(long total, long hit) { return 1L; }
            @Override public double predictBatchMs(PrefillBatchFeatures features) {
                predictionBarrier.run();
                return 1.0;
            }
        };
    }

    private static void await(CountDownLatch latch) {
        try { assertTrue(latch.await(5, TimeUnit.SECONDS), "preparation barrier was not released"); }
        catch (InterruptedException interrupted) { throw new AssertionError(interrupted); }
    }

    private static final class Fixture implements AutoCloseable {
        final EngineGrpcClient client = mock(EngineGrpcClient.class);
        final SchedulerRuntime runtime;
        final QueuedRequestScheduler owner;
        final PrefillEndpoint prefill;
        final DecodeEndpoint decode;
        final WorkerBatcher batcher;
        final RequestContext context;
        final CompletableFuture<Response> future;
        final DecodeResources.ReservationHandle reservation;
        final RequestRoute route;

        Fixture() throws Exception {
            var config = SchedulingTestConfig.batchConfig();
            SchedulingTestConfig.usePriorityQueue(config);
            SchedulingTestConfig.useSingleDecision(config);
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            when(client.batchEnqueueAsync(anyString(), anyInt(), any())).thenAnswer(call -> {
                EngineRpcService.EnqueueBatchRequestPB request = call.getArgument(2);
                var reply = EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(request.getBatchId());
                for (var slot : request.getDpSlotsList()) {
                    for (var input : slot.getRequestsList()) { reply.addSuccessesBuilder().setRequestId(input.getInput().getRequestId()); }
                }
                return CompletableFuture.completedFuture(reply.build());
            });
            var reporter = mock(DeliveryMetricsReporter.class);
            var dispatcher = new DeliveryDispatcher(client, service, null, reporter, null);
            var repository = new RequestRepository();
            var availability = new PlacementAvailability();
            var directory = new EndpointRegistry(service, repository, reporter, dispatcher, availability);
            runtime = new SchedulerRuntime(repository, directory, reporter,
                    mock(RequestSchedulerReporter.class), dispatcher, service, mock(EngineCancelChannel.class));
            owner = new QueuedRequestScheduler(config, mock(RequestWorkerSelector.class), reporter,
                    mock(DecodeCapacityAcquirer.class), runtime, availability,
                    new DecodeResources.AdmissionCapacity(64L, 90), mock(RecentCacheKeyTraceReporter.class));
            runtime.registerScheduler(owner);
            var pStatus = WorkerStatus.createDiscovered(RoleType.PREFILL, "g", "127.0.0.1", 8080, 8081, "test");
            var dStatus = WorkerStatus.createDiscovered(RoleType.DECODE, "g", "127.0.0.2", 8080, 8081, "test");
            prefill = construct(PrefillEndpoint.class,
                    new Class<?>[]{WorkerStatus.class, org.flexlb.config.FlexlbConfig.class, DeliveryDispatcher.class,
                            DeliveryMetricsReporter.class, PlacementAvailability.class},
                    pStatus, config, dispatcher, reporter, availability);
            decode = construct(DecodeEndpoint.class,
                    new Class<?>[]{WorkerStatus.class, RequestRepository.class, PlacementAvailability.class},
                    dStatus, repository, availability);
            applyStatus(prefill, status(RoleType.PREFILL));
            applyStatus(decode, status(RoleType.DECODE));
            var state = new PrefillState(new ReentrantLock(), PrefillActiveIndex.ordered(16, WorkerBatcher.PRIORITY_QUEUE_ORDER), System::currentTimeMillis);
            ReflectionTestUtils.setField(prefill, "ownership", state);
            batcher = new WorkerBatcher("withdrawal-race", prefill, QueueExecutionSettings.capture(config), dispatcher, state);
            ReflectionTestUtils.setField(prefill, "batcher", batcher);
            var runtimeState = Class.forName("org.flexlb.balance.scheduler.WorkerBatcher$RuntimeState");
            @SuppressWarnings({"rawtypes", "unchecked"})
            Object running = Enum.valueOf((Class)runtimeState, "RUNNING");
            // Drive the real selection directly so the regression controls both transaction boundaries.
            ReflectionTestUtils.setField(batcher, "runtimeState", running);
            context = new RequestContext(config);
            context.data().setRequest(Request.builder().requestId(909001L).rawPriority(50).seqLen(16L).build());
            context.data().setSchedulingMetadata(SchedulingMetadata.explicit(50, Long.MAX_VALUE));
            context.data().setGenerateInputPb(EngineRpcService.GenerateInputPB.newBuilder()
                    .setRequestId(context.data().getRequestId()).addTokenIds(1).build().toByteString());
            future = owner.register(context, () -> {}, StrategyErrorType.RESOURCE_EXHAUSTED);
            try (var pin = decode.tryPinGeneration()) {
                reservation = decode.tryReserveQueuedRequest(pin, context.data().getRequestId(), 16L, 16L, 50, null).handle();
            }
            assertNotNull(reservation);
            var decodeMetadata = new ServerStatus();
            decodeMetadata.setRequestId(context.data().getRequestId()); decodeMetadata.setRole(RoleType.DECODE);
            decodeMetadata.setServerIp(decode.getIp()); decodeMetadata.setHttpPort(decode.getHttpPort());
            decodeMetadata.setGrpcPort(decode.getGrpcPort()); decodeMetadata.setSuccess(true);
            var responseTemplate = new Response(); responseTemplate.setSuccess(true);
            route = SchedulingTestConfig.createRoute(context, responseTemplate, null, decodeMetadata,
                    prefill, decode, reservation, System.currentTimeMillis());
            responseTemplate.setServerStatus(List.of(route.prefill(), route.decode()));
            var admission = owner.claimAdmissionHandle(context.data().getRequestId(), future);
            assertNotNull(admission);
            assertEquals(org.flexlb.balance.PlacementResult.Status.SUCCESS, owner.commitRoute(route, new AbstractRequestScheduler.Publication() {
                private boolean published;
                @Override public void publish() { published = batcher.offer(route); }
                @Override public boolean published() { return published; }
            }));
            admission.finish();
        }

        void select(PrefillTimePredictor.Evaluator evaluator) {
            batcher.processSelection(List.of(route), "single_request", evaluator, OptionalLong.empty());
        }

        void assertDelivered() throws Exception {
            assertNotNull(context.snapshot().delivery(), "a committed route must retain its sender");
            assertTrue(future.get(5, TimeUnit.SECONDS).isSuccess());
            assertEquals(0, prefill.ownership.progress().queueDepth());
            assertEquals(0, decode.resourceSnapshot().activeDispatchPermits());
            verify(client, times(1)).batchEnqueueAsync(anyString(), anyInt(), any());
        }

        @Override public void close() {
            runtime.deliveries().awaitDeliveries();
            ReflectionTestUtils.invokeMethod(runtime, "closeRequestExecutors");
            ((ExpirationTimer)ReflectionTestUtils.getField(runtime, "timer")).close();
            for (String field : List.of("dispatchExecutor", "completionExecutor")) {
                assertTrue(((ExecutorService)ReflectionTestUtils.getField(runtime.deliveries(), field)).isTerminated(),
                        field + " must terminate after the execution registrations drain");
            }
            for (String field : List.of("requestWorkers", "requestRecovery", "cleanupExecutor", "deliveryCleanupExecutor")) {
                assertTrue(((ExecutorService)ReflectionTestUtils.getField(runtime, field)).isTerminated(),
                        field + " must terminate after the execution registrations drain");
            }
        }
    }

    private static <T> T construct(Class<T> type, Class<?>[] signature, Object... args) throws Exception {
        var constructor = type.getDeclaredConstructor(signature);
        constructor.setAccessible(true);
        return constructor.newInstance(args);
    }

    private static WorkerStatusResponse status(RoleType role) {
        var response = new WorkerStatusResponse();
        response.setRole(role); response.setAlive(true); response.setStatusVersion(1L);
        response.setLatestFinishedVersion(0L); response.setRunningTaskInfo(Map.of()); response.setFinishedTaskInfo(Map.of());
        response.setRunningQueryLen(0L); response.setTotalKvCacheTokens(100000L); response.setAvailableKvCacheTokens(100000L);
        return response;
    }

    private static void applyStatus(WorkerEndpoint endpoint, WorkerStatusResponse response) {
        WorkerStatus worker = endpoint.getStatus();
        org.flexlb.balance.endpoint.WorkerEndpoint.StatusChange projection;
        worker.lock.lock();
        try { projection = endpoint.applyPreparedStatus(worker, worker.prepareNewStatus(WorkerStatusTestSupport.freezeStatusResponse(worker, response))); }
        finally { worker.lock.unlock(); }
        endpoint.publishStatus(projection);
    }
}
