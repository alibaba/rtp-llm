package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.*;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.*;
import org.junit.jupiter.api.Test;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.jupiter.api.Assertions.*;

class MockRpcTimingTest {
    private static final class Fixture implements AutoCloseable {
        final MockRpcExecutors runtime;
        final MockRpcExecutors.Engine engine;
        final AtomicInteger handled = new AtomicInteger();
        final AtomicInteger version = new AtomicInteger(1);
        final java.util.concurrent.atomic.AtomicBoolean fail = new java.util.concurrent.atomic.AtomicBoolean();
        final Server server; final ManagedChannel channel;
        Fixture(String configuration) throws Exception { this(configuration, 8); }
        Fixture(String configuration, int maxCalls) throws Exception {
            runtime = new MockRpcExecutors(1, maxCalls); engine=runtime.engine();
            engine.timing(new MockTiming(new ObjectMapper().readTree(configuration)),ignored->{});
            server=ServerBuilder.forPort(0).executor(runtime.controlExecutor()).callExecutor(engine).intercept(engine)
                .addService(new RpcServiceGrpc.RpcServiceImplBase(){
                    @Override public void getWorkerStatus(EngineRpcService.StatusVersionPB request, StreamObserver<EngineRpcService.WorkerStatusPB> response) {
                        handled.incrementAndGet();
                        if (fail.get()) throw new IllegalStateException("business failure");
                        response.onNext(EngineRpcService.WorkerStatusPB.newBuilder().setLatestFinishedVersion(version.get()).build());
                        response.onCompleted();
                    }
                    @Override public void fetchResponse(EngineRpcService.FetchRequestPB request, StreamObserver<EngineRpcService.GenerateOutputsPB> response) {
                        for(int i=0;i<3;i++) response.onNext(EngineRpcService.GenerateOutputsPB.newBuilder().setRequestId(i).build());
                        response.onCompleted();
                    }
                }).build().start();
            channel=ManagedChannelBuilder.forAddress("127.0.0.1",server.getPort()).usePlaintext().build();
        }
        CompletableFuture<EngineRpcService.WorkerStatusPB> status(long deadline) {
            var result=new CompletableFuture<EngineRpcService.WorkerStatusPB>();
            RpcServiceGrpc.newStub(channel).withDeadlineAfter(deadline,TimeUnit.MILLISECONDS).getWorkerStatus(
                EngineRpcService.StatusVersionPB.getDefaultInstance(),new StreamObserver<>(){
                    public void onNext(EngineRpcService.WorkerStatusPB value){result.complete(value);}
                    public void onError(Throwable error){result.completeExceptionally(error);}
                    public void onCompleted(){}
                });return result;
        }
        public void close() throws Exception {
            channel.shutdownNow().awaitTermination(3,TimeUnit.SECONDS);
            server.shutdownNow().awaitTermination(3,TimeUnit.SECONDS);runtime.close();
        }
    }
    @Test void responseDelayFreezesSnapshotAndDoesNotBlockWorkerOrUnrelatedMethod() throws Exception {
        try(var f=new Fixture("{\"rpc\":{\"get_worker_status\":{\"response\":{\"delay_ms\":250}}}}")) {
            var response=f.status(3000);
            long end=System.nanoTime()+TimeUnit.SECONDS.toNanos(2);
            while(f.handled.get()==0&&System.nanoTime()<end)Thread.sleep(1);
            assertEquals(1,f.handled.get()); assertFalse(response.isDone()); f.version.set(2);
            var outputs=RpcServiceGrpc.newBlockingStub(f.channel).withDeadlineAfter(1,TimeUnit.SECONDS)
                .fetchResponse(EngineRpcService.FetchRequestPB.getDefaultInstance());
            for(int i=0;i<3;i++){assertTrue(outputs.hasNext());assertEquals(i,outputs.next().getRequestId());}
            assertFalse(outputs.hasNext());
            assertEquals(1,response.get(2,TimeUnit.SECONDS).getLatestFinishedVersion());
        }
    }
    @Test void deadlineDuringRequestDelayPreventsLateAdmission() throws Exception {
        try(var f=new Fixture("{\"rpc\":{\"get_worker_status\":{\"request\":{\"delay_ms\":200}}}}")) {
            var error=assertThrows(ExecutionException.class,()->f.status(40).get(1,TimeUnit.SECONDS));
            assertEquals(Status.Code.DEADLINE_EXCEEDED,Status.fromThrowable(error.getCause()).getCode());
            Thread.sleep(250);assertEquals(0,f.handled.get());
        }
    }
    @Test void streamResponseDelayPreservesMessagesAndTerminalOrder() throws Exception {
        try(var f=new Fixture("{\"rpc\":{\"fetch_response\":{\"response\":{\"pause\":{\"probability\":1,\"duration_ms\":50}}}}}")) {
            var outputs=RpcServiceGrpc.newBlockingStub(f.channel).withDeadlineAfter(2,TimeUnit.SECONDS)
                .fetchResponse(EngineRpcService.FetchRequestPB.getDefaultInstance());
            for(int i=0;i<3;i++)assertEquals(i,outputs.next().getRequestId());assertFalse(outputs.hasNext());
        }
    }
    private static void await(java.util.function.BooleanSupplier condition) throws Exception {
        long end=System.nanoTime()+TimeUnit.SECONDS.toNanos(2);
        while(!condition.getAsBoolean()&&System.nanoTime()<end)Thread.sleep(1);
        assertTrue(condition.getAsBoolean());
    }
    @Test void responseDelayHoldsCallQuotaAndDeadlineReleasesTimersAndAdmission() throws Exception {
        try(var f=new Fixture("{\"rpc\":{\"get_worker_status\":{\"response\":{\"delay_ms\":1000}}}}",1)) {
            var pending=f.status(300);
            await(()->f.handled.get()==1);
            assertEquals(1,f.engine.inFlight());
            assertEquals(1,f.engine.snapshot().get("delayed_calls"));
            var rejected=assertThrows(ExecutionException.class,()->f.status(2000).get(1,TimeUnit.SECONDS));
            assertEquals(Status.Code.RESOURCE_EXHAUSTED,Status.fromThrowable(rejected.getCause()).getCode());
            var expired=assertThrows(ExecutionException.class,()->pending.get(1,TimeUnit.SECONDS));
            assertEquals(Status.Code.DEADLINE_EXCEEDED,Status.fromThrowable(expired.getCause()).getCode());
            await(()->f.engine.inFlight()==0 && f.engine.snapshot().get("delayed_calls").equals(0));
        }
    }
    @Test void shutdownDropsDelayedRequestWithoutBusinessAdmission() throws Exception {
        try(var f=new Fixture("{\"rpc\":{\"get_worker_status\":{\"request\":{\"delay_ms\":1000}}}}")) {
            var pending=f.status(3000);
            await(()->f.engine.snapshot().get("delayed_calls").equals(1));
            f.engine.stop();
            var error=assertThrows(ExecutionException.class,()->pending.get(1,TimeUnit.SECONDS));
            assertEquals(Status.Code.UNAVAILABLE,Status.fromThrowable(error.getCause()).getCode());
            assertEquals(0,f.handled.get()); assertEquals(0,f.engine.inFlight());
            assertEquals(0,f.engine.snapshot().get("delayed_calls"));
        }
    }

    @Test void delayedHandlerFailureTerminatesCallAndCleansAdmission() throws Exception {
        try(var f=new Fixture("{\"rpc\":{\"get_worker_status\":{\"request\":{\"delay_ms\":10}}}}")) {
            f.fail.set(true);
            var error=assertThrows(ExecutionException.class,()->f.status(2000).get(1,TimeUnit.SECONDS));
            assertEquals(Status.Code.UNKNOWN,Status.fromThrowable(error.getCause()).getCode());
            await(()->f.engine.inFlight()==0 && f.engine.snapshot().get("delayed_calls").equals(0));
        }
    }

}
