package org.flexlb.mock.vit;

import io.grpc.Status;
import java.util.*;
import java.util.concurrent.TimeUnit;
import org.flexlb.mock.api.model.Types.Role;
import org.flexlb.mock.compute.*;

/** One Forward lane; only prepared, deduplicated media enter this queue. */
final class VitScheduler implements AutoCloseable {
    private record Job(VitCache.Claim claim,long enqueued,long deadline) {}
    private final VitConfig config;
    private final VitCache cache;
    private final VitMetrics metrics;
    private final ArrayDeque<Job> waiting=new ArrayDeque<>();
    private final SimulatedCompute compute=new SimulatedCompute();
    private final Thread worker;
    private boolean closed;
    private RuntimeException failure;
    private int running;
    private long sequence;

    VitScheduler(VitConfig config,VitCache cache) { this(config,cache,new VitMetrics()); }
    VitScheduler(VitConfig config,VitCache cache,VitMetrics metrics) {
        this.config=config; this.cache=cache; this.metrics=metrics;
        worker=Thread.ofVirtual().name("vision-"+config.id()).start(this::run);
    }
    void submit(VitCache.Claim claim) { submit(claim,Long.MAX_VALUE); }
    synchronized void submit(VitCache.Claim claim,long deadline) {
        if(closed || failure!=null) throw Status.UNAVAILABLE.withDescription("V scheduler stopped").asRuntimeException();
        if(waiting.size()>=config.queueSize()) {
            metrics.events.increment("py_rtp_vit_embedding_overload_qps");
            throw Status.RESOURCE_EXHAUSTED.withDescription("V waiting queue full").asRuntimeException();
        }
        waiting.add(new Job(claim,System.nanoTime(),deadline)); notifyAll();
    }
    synchronized boolean cancel(VitCache.Claim claim) {
        boolean removed=waiting.removeIf(job->job.claim.result()==claim.result());
        if(removed) notifyAll(); return removed;
    }
    private synchronized List<Job> next() throws InterruptedException {
        while(!closed && waiting.isEmpty()) wait();
        if(closed) return null;
        long until=waiting.getFirst().enqueued+TimeUnit.MILLISECONDS.toNanos(config.batchSize()==1?0:config.waitMs());
        while(!closed && !waiting.isEmpty() && waiting.size()<config.batchSize() && until>System.nanoTime())
            TimeUnit.NANOSECONDS.timedWait(this,until-System.nanoTime());
        if(closed) return null;
        var jobs=new ArrayList<Job>();
        while(!waiting.isEmpty() && jobs.size()<config.batchSize()) jobs.add(waiting.remove());
        running=jobs.size(); return jobs;
    }
    private void run() {
        List<Job> batch=List.of();
        try {
            while((batch=next())!=null) {
                var selected=new ArrayList<Job>();
                for(var job:batch) {
                    if(System.nanoTime()>=job.deadline) cache.complete(job.claim,Status.DEADLINE_EXCEEDED.withDescription("V ready work expired").asRuntimeException());
                    else if(job.claim.users().get()==0) cache.complete(job.claim,Status.CANCELLED.withDescription("No V consumers remain").asRuntimeException());
                    else selected.add(job);
                }
                batch=selected;
                synchronized(this) { running=batch.size(); }
                if(batch.isEmpty()) continue;
                long started=System.nanoTime();
                for(var job:batch) metrics.events.observe("py_rtp_vit_embedding_queue_wait_rt",(started-job.enqueued)/1e6);
                metrics.events.observe("py_rtp_vit_embedding_batch_size",batch.size());
                var requests=new ArrayList<ForwardBatch.Request>();
                for(var job:batch) {
                    var image=job.claim.image(); long rows=image.rows();
                    requests.add(new ForwardBatch.Request(Long.toString(++sequence),rows,0,rows,0,0,50,
                        List.of(new ForwardBatch.Image(image.key(),image.width(),image.height(),3))));
                }
                var work=new ForwardBatch(Long.toString(++sequence),List.of(new ForwardBatch.Partition(0,requests)));
                var sample=config.compute().plan(Role.vit,config.id(),work);
                var result=compute.submit(sample).completion().toCompletableFuture().get();
                metrics.events.observe("py_rtp_vit_embedding_forward_rt",result.elapsedNanos()/1e6);
                metrics.events.observe("mock_vit_theoretical_forward_rt",sample.planned().toNanos()/1e6);
                synchronized(this) { running=0; }
                Throwable error=result.state()==SimulatedCompute.State.COMPLETED?null:Status.UNAVAILABLE.withDescription("V forward cancelled").asRuntimeException();
                for(var job:batch) cache.complete(job.claim,error!=null?error:System.nanoTime()>=job.deadline
                    ?Status.DEADLINE_EXCEEDED.withDescription("V media computation expired").asRuntimeException():null);
            }
        } catch(InterruptedException error) { Thread.currentThread().interrupt(); fail(batch,Status.UNAVAILABLE.withCause(error).asRuntimeException()); }
        catch(Exception error) { fail(batch,new IllegalStateException("V forward failed",error)); }
    }
    private void fail(List<Job> batch,RuntimeException error) {
        List<Job> rest;
        synchronized(this) { failure=error;closed=true;running=0;rest=new ArrayList<>(waiting);waiting.clear();notifyAll(); }
        if(batch!=null) for(var job:batch) cache.complete(job.claim,error);
        for(var job:rest) cache.complete(job.claim,error);
    }
    synchronized void checkHealthy() { if(failure!=null) throw failure; if(closed) throw new IllegalStateException("V scheduler closed"); }
    synchronized Map<String,Double> snapshot() {
        checkHealthy();
        return Map.of("py_rtp_vit_embedding_queue_size",(double)waiting.size(),"mock_vit_running_image_count",(double)running);
    }
    @Override public void close() {
        List<Job> rest;
        synchronized(this) { if(closed) { compute.close();worker.interrupt();return; }closed=true;rest=new ArrayList<>(waiting);waiting.clear();notifyAll(); }
        var error=Status.UNAVAILABLE.withDescription("V stopped").asRuntimeException();
        for(var job:rest) cache.complete(job.claim,error);
        compute.close();worker.interrupt();
    }
}
