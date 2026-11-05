package org.flexlb.mock.engine.vit;

import io.grpc.Status;
import java.util.*;
import java.util.concurrent.*;
import org.flexlb.mock.protocol.Role;
import org.flexlb.mock.engine.forward.*;

/** One Forward lane; only prepared, deduplicated media enter this queue. */
final class VitScheduler implements AutoCloseable {
    record Work(VitGeometry.Image image,long enqueued,long deadline,CompletableFuture<VitGeometry.Image> result) {}
    private final VitConfig config;
    private final VitMetrics metrics;
    private final ArrayDeque<Work> waiting=new ArrayDeque<>();
    private final SimulatedCompute compute=new SimulatedCompute();
    private final Thread worker;
    private boolean closed;
    private RuntimeException failure;
    private int running;
    private long sequence;

    VitScheduler(VitConfig config) { this(config,new VitMetrics()); }
    VitScheduler(VitConfig config,VitMetrics metrics) {
        this.config=config; this.metrics=metrics;
        worker=Thread.ofVirtual().name("vision-"+config.id()).start(this::run);
    }
    Work submit(VitGeometry.Image image) { return submit(image,Long.MAX_VALUE); }
    synchronized Work submit(VitGeometry.Image image,long deadline) {
        if(closed || failure!=null) throw Status.UNAVAILABLE.withDescription("V scheduler stopped").asRuntimeException();
        if(waiting.size()>=config.queueSize()) {
            metrics.events.increment("py_rtp_vit_embedding_overload_qps");
            throw Status.RESOURCE_EXHAUSTED.withDescription("V waiting queue full").asRuntimeException();
        }
        var work=new Work(image,System.nanoTime(),deadline,new CompletableFuture<>());
        waiting.add(work); notifyAll(); return work;
    }
    boolean cancel(Work work) {
        synchronized(this) { if(!waiting.remove(work)) return false; notifyAll(); }
        work.result.completeExceptionally(Status.CANCELLED.withDescription("V ready work cancelled").asRuntimeException());
        return true;
    }
    private synchronized List<Work> next() throws InterruptedException {
        while(!closed && waiting.isEmpty()) wait();
        if(closed) return null;
        long until=waiting.getFirst().enqueued+TimeUnit.MILLISECONDS.toNanos(config.batchSize()==1?0:config.waitMs());
        while(!closed && !waiting.isEmpty() && waiting.size()<config.batchSize() && until>System.nanoTime())
            TimeUnit.NANOSECONDS.timedWait(this,until-System.nanoTime());
        if(closed) return null;
        var jobs=new ArrayList<Work>();
        while(!waiting.isEmpty() && jobs.size()<config.batchSize()) jobs.add(waiting.remove());
        running=jobs.size(); return jobs;
    }
    private void run() {
        List<Work> batch=List.of();
        try {
            while((batch=next())!=null) {
                var selected=new ArrayList<Work>();
                for(var job:batch) {
                    if(System.nanoTime()>=job.deadline) complete(job,Status.DEADLINE_EXCEEDED.withDescription("V ready work expired").asRuntimeException());
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
                    var image=job.image; long rows=image.rows();
                    requests.add(new ForwardBatch.Request(Long.toString(++sequence),rows,0,rows,0,0,50,
                        List.of(new ForwardBatch.Image(image.key(),image.width(),image.height(),3))));
                }
                var work=new ForwardBatch(Long.toString(++sequence),List.of(new ForwardBatch.Partition(0,requests)));
                var sample=config.compute().plan(Role.vit,config.id(),work);
                var result=compute.submit(sample).completion().toCompletableFuture().get();
                boolean completedForward=result.state()==SimulatedCompute.State.COMPLETED;
                if(completedForward) {
                    metrics.events.observe("py_rtp_vit_embedding_forward_rt",result.elapsedNanos()/1e6);
                    metrics.events.observe("mock_vit_theoretical_forward_rt",sample.planned().toNanos()/1e6);
                }
                synchronized(this) { running=0; }
                Throwable error=completedForward?null:Status.UNAVAILABLE.withDescription("V forward cancelled").asRuntimeException();
                long completed=System.nanoTime();
                for(var job:batch) {
                    Throwable failure=error!=null?error:completed>=job.deadline
                        ?Status.DEADLINE_EXCEEDED.withDescription("V media computation expired").asRuntimeException():null;
                    if(failure==null) {
                        metrics.events.observe("py_rtp_vit_embedding_rt",(completed-job.enqueued)/1e6);
                        metrics.events.observe("py_rtp_vit_embedding_tps",job.image.rows());
                    }
                    complete(job,failure);
                }
            }
        } catch(InterruptedException error) { Thread.currentThread().interrupt(); fail(batch,Status.UNAVAILABLE.withCause(error).asRuntimeException()); }
        catch(Exception error) { fail(batch,new IllegalStateException("V forward failed",error)); }
    }
    private static void complete(Work work,Throwable error) {
        if(error==null) work.result.complete(work.image); else work.result.completeExceptionally(error);
    }
    private void fail(List<Work> batch,RuntimeException error) {
        List<Work> rest;
        synchronized(this) { failure=error;closed=true;running=0;rest=new ArrayList<>(waiting);waiting.clear();notifyAll(); }
        if(batch!=null) for(var job:batch) complete(job,error);
        for(var job:rest) complete(job,error);
    }
    synchronized void checkHealthy() { if(failure!=null) throw failure; if(closed) throw new IllegalStateException("V scheduler closed"); }
    synchronized Map<String,Double> snapshot() {
        checkHealthy();
        return Map.of("py_rtp_vit_embedding_queue_size",(double)waiting.size(),"mock_vit_running_image_count",(double)running);
    }
    @Override public void close() {
        List<Work> rest;
        synchronized(this) { if(closed) { compute.close();worker.interrupt();return; }closed=true;rest=new ArrayList<>(waiting);waiting.clear();notifyAll(); }
        var error=Status.UNAVAILABLE.withDescription("V stopped").asRuntimeException();
        for(var job:rest) complete(job,error);
        compute.close();worker.interrupt();
    }
}
