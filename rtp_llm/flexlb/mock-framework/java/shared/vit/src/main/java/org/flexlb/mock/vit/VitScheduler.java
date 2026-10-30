package org.flexlb.mock.vit;

import org.flexlb.mock.multimodal.MediaInput;

import io.grpc.Status;
import java.util.*;
import java.util.concurrent.*;
import org.flexlb.mock.compute.*;
import org.flexlb.mock.api.model.Types.Role;

/** One serial forward executor; concurrent callers are merged within a bounded collection window. */
final class VitScheduler implements AutoCloseable {
    private record Job(List<VitCache.Claim> claims, long enqueued) {}
    private final VitConfig config;
    private final VitCache cache;
    private final ArrayDeque<Job> waiting = new ArrayDeque<>();
    private final SimulatedCompute compute = new SimulatedCompute();
    private final Thread worker;
    private boolean closed;
    private RuntimeException failure;
    private int running;
    private long forwards, batchItems, sequence;
    private double forwardMs, theoreticalMs;
    VitScheduler(VitConfig config, VitCache cache) {
        this.config=config; this.cache=cache; worker=Thread.ofVirtual().name("vision-"+config.id()).start(this::run);
    }
    synchronized void submit(List<VitCache.Claim> claims) {
        if (closed || failure != null) throw Status.UNAVAILABLE.withDescription("V scheduler stopped").asRuntimeException();
        if (claims.isEmpty()) return;
        if (config.batchSize() > 1 && claims.size() > config.batchImages()) throw MediaInput.invalid("Image count exceeds V forward budget");
        if (waiting.size() >= config.queueSize()) throw Status.RESOURCE_EXHAUSTED.withDescription("V waiting queue full").asRuntimeException();
        waiting.add(new Job(List.copyOf(claims),System.nanoTime())); notifyAll();
    }
    private List<Job> next() throws InterruptedException {
        synchronized (this) {
            while (!closed && waiting.isEmpty()) wait();
            if (closed) return null;
            long until=waiting.getFirst().enqueued+TimeUnit.MILLISECONDS.toNanos(config.batchSize()==1?0:config.waitMs());
            while (!closed && waiting.size()<config.batchSize() && until>System.nanoTime())
                TimeUnit.NANOSECONDS.timedWait(this,until-System.nanoTime());
            if (closed) return null;
            var jobs=new ArrayList<Job>(); int images=0;
            int limit=config.batchSize()==1?Integer.MAX_VALUE:config.batchImages();
            while (!waiting.isEmpty() && jobs.size()<config.batchSize() && images+waiting.getFirst().claims.size()<=limit) {
                var job=waiting.remove();jobs.add(job);images+=job.claims.size();
            }
            running=images; return jobs;
        }
    }
    private void run() {
        List<Job> batch=List.of();
        try {
            while ((batch=next())!=null) {
                var selected=new ArrayList<Job>();
                for(var job:batch) {
                    var wanted=new ArrayList<VitCache.Claim>();
                    for(var claim:job.claims) {
                        if(claim.users().get()>0) wanted.add(claim);
                        else cache.complete(claim,Status.CANCELLED.withDescription("No V consumers remain").asRuntimeException());
                    }
                    if(!wanted.isEmpty()) selected.add(new Job(wanted,job.enqueued));
                }
                batch=selected;
                if(batch.isEmpty()) { synchronized(this) { running=0; }continue; }
                var requests=new ArrayList<ForwardBatch.Request>();
                for (var job:batch) {
                    var images=job.claims.stream().map(c -> new ForwardBatch.Image(c.image().key(),c.image().width(),c.image().height(),3)).toList();
                    long rows=job.claims.stream().mapToLong(c->c.image().rows()).sum();
                    requests.add(new ForwardBatch.Request(Long.toString(++sequence),rows,0,rows,0,0,50,images));
                }
                var work=new ForwardBatch(Long.toString(++sequence),List.of(new ForwardBatch.Partition(0,requests)));
                var sample=config.compute().plan(Role.vit,config.id(),work);
                var result=compute.submit(sample).completion().toCompletableFuture().get();
                synchronized(this) { forwards++;batchItems+=requests.size();forwardMs+=result.elapsedNanos()/1e6;theoreticalMs+=sample.base().toNanos()/1e6;running=0; }
                Throwable error=result.state()==SimulatedCompute.State.COMPLETED?null:Status.UNAVAILABLE.withDescription("V forward cancelled").asRuntimeException();
                for(var job:batch) for(var claim:job.claims) cache.complete(claim,error);
            }
        } catch (InterruptedException error) { Thread.currentThread().interrupt(); fail(batch,Status.UNAVAILABLE.withCause(error).asRuntimeException()); }
        catch (Exception error) { fail(batch,new IllegalStateException("V forward failed",error)); }
    }
    private void fail(List<Job> batch,RuntimeException error) {
        List<Job> rest;
        synchronized(this) { failure=error;closed=true;running=0;rest=new ArrayList<>(waiting);waiting.clear();notifyAll(); }
        if(batch!=null) for(var job:batch) for(var claim:job.claims) cache.complete(claim,error);
        for(var job:rest) for(var claim:job.claims) cache.complete(claim,error);
    }
    synchronized void checkHealthy() { if(failure!=null) throw failure; if(closed) throw new IllegalStateException("V scheduler closed"); }
    synchronized String metrics() {
        checkHealthy();return "mock_engine_requests{role=\"vit\",state=\"waiting\"} "+waiting.size()+"\nmock_engine_requests{role=\"vit\",state=\"running\"} "+running
            +"\npy_rtp_vit_embedding_forward_rt_sum "+forwardMs+"\npy_rtp_vit_embedding_forward_rt_count "+forwards
            +"\nmock_vit_theoretical_forward_rt_sum "+theoreticalMs+"\nmock_vit_theoretical_forward_rt_count "+forwards
            +"\npy_rtp_vit_embedding_batch_size_sum "+batchItems+"\npy_rtp_vit_embedding_batch_size_count "+forwards+"\n";
    }
    @Override public void close() {
        List<Job> rest;
        synchronized(this) { if(closed) { compute.close();worker.interrupt();return; }closed=true;rest=new ArrayList<>(waiting);waiting.clear();notifyAll(); }
        var error=Status.UNAVAILABLE.withDescription("V stopped").asRuntimeException();
        for(var job:rest) for(var claim:job.claims) cache.complete(claim,error);
        compute.close();worker.interrupt();
    }
}
