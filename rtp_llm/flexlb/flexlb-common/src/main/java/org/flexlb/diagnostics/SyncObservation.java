package org.flexlb.diagnostics;

import java.util.concurrent.atomic.AtomicLong;
import java.util.function.DoubleConsumer;

/** Small generation-local observation. No payload/key retention or hot-path maps. */
public final class SyncObservation {
    private static final DoubleConsumer NOOP = value -> { };
    public volatile DoubleConsumer conversion = NOOP, queue = NOOP, apply = NOOP;
    public volatile long rpcStartUs, responseUs, readyUs, applyStartUs, lastAliveResponseUs;
    public volatile long payloadBytes, entryCount, finishedTaskCount;
    public volatile int phase; // 0 idle, 1 RPC, 2 conversion, 3 pending callback, 4 applying
    public volatile long maxQueueUs, maxApplyUs, conversionUs;
    public final AtomicLong lastSlowLogUs = new AtomicLong(), lastSnapshotLogUs = new AtomicLong();
    public final AtomicLong suppressedSnapshots = new AtomicLong(), suppressedSlow = new AtomicLong();
    private static void record(DoubleConsumer consumer,double value){try{consumer.accept(value);}catch(RuntimeException ignored){}}
    public void started() { rpcStartUs = System.nanoTime()/1000; responseUs=readyUs=applyStartUs=0; payloadBytes=0;entryCount=finishedTaskCount=0; phase=1; }
    public void received(long bytes, long entries, long finished) {
        responseUs=System.nanoTime()/1000; payloadBytes=bytes; entryCount=entries; finishedTaskCount=finished; phase=2;
    }
    public void ready(boolean alive) {
        readyUs=System.nanoTime()/1000; conversionUs=responseUs>0?readyUs-responseUs:0;
        if(responseUs>0)record(conversion,conversionUs/1000.0);
        if(alive)lastAliveResponseUs=responseUs>0?responseUs:readyUs;
        phase=3;
    }
    public void applying() {
        applyStartUs=System.nanoTime()/1000;
        long wait=readyUs>0?applyStartUs-readyUs:0;
        maxQueueUs=Math.max(maxQueueUs,wait);if(readyUs>0)record(queue,wait/1000.0);phase=4;
    }
    public void done() {
        if(phase==0)return;
        long elapsed=applyStartUs>0?System.nanoTime()/1000-applyStartUs:0;
        maxApplyUs=Math.max(maxApplyUs,elapsed);if(applyStartUs>0)record(apply,elapsed/1000.0);
        phase=0;payloadBytes=0;
    }
    public long phaseAgeUs(long now) {
        long start=switch(phase){case 1->rpcStartUs;case 2->responseUs;case 3->readyUs;case 4->applyStartUs;default->0;};
        return start>0?Math.max(0,now-start):0;
    }
    public static boolean logNow(AtomicLong last, AtomicLong suppressed, long intervalUs) {
        long now=System.nanoTime()/1000,old=last.get();
        if(now-old<intervalUs||!last.compareAndSet(old,now)){suppressed.incrementAndGet();return false;}
        return true;
    }
}
