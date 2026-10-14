package org.flexlb.diagnostics;
import org.junit.jupiter.api.Test;
import java.util.concurrent.*;
import static org.junit.jupiter.api.Assertions.*;
class ObservedExecutorTest {
    @Test void queuedAgeAndShutdownPreserveOriginalTasks() throws Exception {
        ObservedExecutor pool=new ObservedExecutor(1,1,1,TimeUnit.SECONDS,new LinkedBlockingQueue<>(2),Executors.defaultThreadFactory(),new ThreadPoolExecutor.AbortPolicy(),"test");
        CountDownLatch entered=new CountDownLatch(1),release=new CountDownLatch(1);
        try {
            pool.execute(()->{entered.countDown();try{release.await();}catch(InterruptedException e){Thread.currentThread().interrupt();}});
            assertTrue(entered.await(2,TimeUnit.SECONDS));
            FutureTask<Integer> queued=new FutureTask<>(()->42);pool.execute(queued);
            assertTrue(pool.oldestPendingMs()>0);assertTrue(pool.oldestRunningMs()>0);
            assertEquals(java.util.List.of(queued),pool.shutdownNow());
            assertFalse(queued.isDone());
        }finally{release.countDown();pool.shutdownNow();assertTrue(pool.awaitTermination(2,TimeUnit.SECONDS));}
    }
    @Test void rejectedTaskIsNotRunAndCanBeCounted() throws Exception {
        ObservedExecutor pool=new ObservedExecutor(1,1,1,TimeUnit.SECONDS,new SynchronousQueue<>(),Executors.defaultThreadFactory(),new ThreadPoolExecutor.AbortPolicy(),"test");
        CountDownLatch entered=new CountDownLatch(1),release=new CountDownLatch(1);
        try{
            pool.execute(()->{entered.countDown();try{release.await();}catch(InterruptedException ignored){}});assertTrue(entered.await(2,TimeUnit.SECONDS));
            assertThrows(RejectedExecutionException.class,()->pool.execute(()->fail("rejected task ran")));
            assertEquals(1,pool.rejectionCount());assertEquals(0,pool.callerRunCount());
        }finally{release.countDown();pool.shutdownNow();assertTrue(pool.awaitTermination(2,TimeUnit.SECONDS));}
    }
    @Test void diagnosticFailureCannotPreventCommitAndDoneIsIdempotent(){
        SyncObservation s=new SyncObservation();java.util.concurrent.atomic.AtomicInteger completed=new java.util.concurrent.atomic.AtomicInteger();
        s.started();s.received(1024,30,0);s.conversion=v->{throw new IllegalStateException("telemetry");};s.ready(true);s.applying();
        s.apply=v->{completed.incrementAndGet();throw new IllegalStateException("telemetry");};
        s.done();s.done();assertEquals(1,completed.get());assertEquals(0,s.phase);assertEquals(0,s.payloadBytes);assertTrue(s.lastAliveResponseUs>0);
    }
}
